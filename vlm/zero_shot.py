"""Zero-shot VLM check: does a small vision-language model spot concealment/shoplifting?

Score = logit("Yes") - logit("No") for the first answer token, so ROC/PR can be computed
without parsing text. Two evaluation sets:

  ucf     UCF-Crime Shoplifting *test* videos (real CCTV, 320x240). Windows of 5 s inside the
          annotated event are positives; windows >= 5 s away from any event are negatives.
  kaggle  The 182 staged clips (92 shoplifting / 90 normal), whole clip per sample.

Usage: python -m vlm.zero_shot --set ucf [--limit N]
"""

from __future__ import annotations

import argparse
import json
import re
import time
from pathlib import Path

import cv2
import numpy as np
import torch
from PIL import Image
from sklearn.metrics import average_precision_score, roc_auc_score

from eval.cross_dataset import bootstrap_ci

MODEL_ID = "Qwen/Qwen3-VL-2B-Instruct"
N_FRAMES = 8
FRAME_SIZE = (448, 336)  # (w, h)
PROMPTS = {
    "base": (
        "These {n} frames, in time order, come from a store CCTV camera. "
        "Is a person concealing or stealing merchandise (for example hiding an item in a bag, "
        "pocket or under clothing instead of paying)? Answer with only Yes or No."
    ),
    "stages": (
        "These {n} frames, in time order, come from a store CCTV camera. Look carefully at the "
        "customers' hands. Shoplifting looks like: a person picks up a product from a shelf or "
        "display, then puts it into their own bag, pocket, jacket or waistband instead of a "
        "basket or trolley, often glancing around. Normal shopping looks like: browsing, "
        "looking at products, putting items back, or placing them in a store basket or cart. "
        "Does the footage show shoplifting? Answer with only Yes or No."
    ),
}
PROMPT = PROMPTS["base"]


_DETECTOR = None


def _person_union_box(bgr_frames: list[np.ndarray], pad: float = 0.25):
    """Union box of all confident people across a few frames, padded; None if nobody found."""
    global _DETECTOR
    if _DETECTOR is None:
        from ultralytics import YOLO

        _DETECTOR = YOLO("yolov8n-pose.pt")
    boxes = []
    for f in bgr_frames:
        r = _DETECTOR.predict(f, classes=[0], conf=0.25, verbose=False)[0]
        boxes += r.boxes.xyxy.cpu().numpy().tolist()
    if not boxes:
        return None
    b = np.array(boxes)
    x1, y1, x2, y2 = b[:, 0].min(), b[:, 1].min(), b[:, 2].max(), b[:, 3].max()
    w, h = x2 - x1, y2 - y1
    H, W = bgr_frames[0].shape[:2]
    return (max(0, x1 - pad * w), max(0, y1 - pad * h), min(W, x2 + pad * w), min(H, y2 + pad * h))


def read_frames(path: str, start: int, end: int, n: int = N_FRAMES, size=None, crop: bool = False) -> list[Image.Image]:
    size = size or FRAME_SIZE
    cap = cv2.VideoCapture(path)
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    end = min(end, total - 1)
    idx = np.linspace(start, max(start, end), n).astype(int)
    raw = []
    for i in idx:
        cap.set(cv2.CAP_PROP_POS_FRAMES, int(i))
        ok, bgr = cap.read()
        if ok:
            raw.append(bgr)
    cap.release()
    if crop and raw:
        box = _person_union_box(raw[:: max(1, len(raw) // 4)])
        if box is not None:
            x1, y1, x2, y2 = (int(v) for v in box)
            if x2 - x1 > 20 and y2 - y1 > 20:
                raw = [f[y1:y2, x1:x2] for f in raw]
    return [
        Image.fromarray(cv2.cvtColor(cv2.resize(f, size, interpolation=cv2.INTER_CUBIC), cv2.COLOR_BGR2RGB))
        for f in raw
    ]


def ucf_samples(root: Path, ann_path: Path, win: int = 150, max_neg: int = 3) -> list[dict]:
    samples = []
    for line in ann_path.read_text().splitlines():
        parts = line.split()
        if len(parts) < 6 or parts[1] != "Shoplifting":
            continue
        name = parts[0]
        path = root / name
        if not path.exists():
            continue
        a = [int(x) for x in parts[2:6]]
        events = [(a[0], a[1])] + ([(a[2], a[3])] if a[2] >= 0 else [])
        total = int(cv2.VideoCapture(str(path)).get(cv2.CAP_PROP_FRAME_COUNT))
        for s, e in events:
            mid = (s + e) // 2
            lo = max(s, min(mid - win // 2, e - win)) if e - s > win else s
            samples.append({"clip": name, "path": str(path), "start": lo, "end": min(lo + win, e) if e - s > win else e, "label": 1})
        # negatives: evenly spaced windows far from every event
        cands = [p for p in range(0, total - win, win)
                 if all(p + win < s - win or p > e + win for s, e in events)]
        if cands:
            pick = np.linspace(0, len(cands) - 1, min(max_neg, len(cands))).astype(int)
            for j in pick:
                samples.append({"clip": name, "path": str(path), "start": cands[j], "end": cands[j] + win, "label": 0})
    return samples


def kaggle_samples(root: Path) -> list[dict]:
    out = []
    for folder, label in (("shoplifting", 1), ("normal", 0)):
        for p in sorted((root / folder).glob("*.mp4")):
            n = int(cv2.VideoCapture(str(p)).get(cv2.CAP_PROP_FRAME_COUNT))
            out.append({"clip": p.stem, "path": str(p), "start": 0, "end": n - 1, "label": label})
    return out


def load_model(model_id: str = MODEL_ID, quant: str = "fp16"):
    from transformers import AutoProcessor, Qwen3VLForConditionalGeneration

    processor = AutoProcessor.from_pretrained(model_id)
    kwargs = {}
    if quant == "4bit":
        from transformers import BitsAndBytesConfig

        kwargs["quantization_config"] = BitsAndBytesConfig(
            load_in_4bit=True, bnb_4bit_compute_dtype=torch.float16, bnb_4bit_quant_type="nf4"
        )
    model = Qwen3VLForConditionalGeneration.from_pretrained(
        model_id, torch_dtype=torch.float16, device_map="cuda" if torch.cuda.is_available() else "cpu", **kwargs
    ).eval()
    tok = processor.tokenizer
    yes = [tok.encode(t, add_special_tokens=False)[0] for t in ("Yes", "yes", " Yes")]
    no = [tok.encode(t, add_special_tokens=False)[0] for t in ("No", "no", " No")]
    return processor, model, sorted(set(yes)), sorted(set(no))


@torch.no_grad()
def vlm_score(processor, model, yes_ids, no_ids, frames: list[Image.Image], prompt: str = PROMPT) -> dict:
    content = [{"type": "image"} for _ in frames] + [{"type": "text", "text": prompt.format(n=len(frames))}]
    text = processor.apply_chat_template([{"role": "user", "content": content}], add_generation_prompt=True, tokenize=False)
    inputs = processor(text=[text], images=frames, return_tensors="pt").to(model.device)
    logits = model(**inputs).logits[0, -1].float()
    lp = torch.log_softmax(logits, dim=-1)
    s_yes = torch.logsumexp(lp[yes_ids], 0).item()
    s_no = torch.logsumexp(lp[no_ids], 0).item()
    return {"score": s_yes - s_no, "p_yes": float(np.exp(s_yes) / (np.exp(s_yes) + np.exp(s_no)))}


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--set", choices=["ucf", "kaggle"], required=True)
    ap.add_argument("--limit", type=int, default=None, help="evaluate only the first N samples (smoke test)")
    ap.add_argument("--out", default=None)
    ap.add_argument("--frames", type=int, default=N_FRAMES)
    ap.add_argument("--prompt", choices=list(PROMPTS), default="base")
    ap.add_argument("--size", default="448x336", help="WxH each frame is resized to")
    ap.add_argument("--crop", action="store_true", help="crop to the union box of detected people")
    ap.add_argument("--model", default=MODEL_ID)
    ap.add_argument("--quant", choices=["fp16", "4bit"], default="fp16")
    a = ap.parse_args()
    size = tuple(int(v) for v in a.size.split("x"))

    if a.set == "ucf":
        samples = ucf_samples(Path("data/external/ucf_shoplifting"), Path("data/Temporal_Anomaly_Annotation_for_Testing_Videos.txt"))
    else:
        samples = kaggle_samples(Path("data"))
    if a.limit:
        rng = np.random.RandomState(0)
        samples = [samples[i] for i in sorted(rng.choice(len(samples), min(a.limit, len(samples)), replace=False))]
    print(f"{len(samples)} samples, {sum(s['label'] for s in samples)} positive", flush=True)

    out_path = Path(a.out or f"artifacts/results/vlm_zero_shot_{a.set}.json")
    done = {}
    if out_path.exists():
        done = {(r["clip"], r["start"]): r for r in json.loads(out_path.read_text())["rows"]}
    processor, model, yes_ids, no_ids = load_model(a.model, a.quant)

    rows, t0 = [], time.time()
    for i, s in enumerate(samples, 1):
        key = (s["clip"], s["start"])
        if key in done:
            rows.append(done[key])
            continue
        frames = read_frames(s["path"], s["start"], s["end"], a.frames, size, a.crop)
        if len(frames) < 4:
            continue
        r = vlm_score(processor, model, yes_ids, no_ids, frames, PROMPTS[a.prompt])
        rows.append({**{k: s[k] for k in ("clip", "start", "end", "label")}, **r})
        if i % 20 == 0:
            print(f"  {i}/{len(samples)}  {(time.time() - t0) / i:.2f}s/sample", flush=True)

    y = np.array([r["label"] for r in rows])
    sc = np.array([r["score"] for r in rows])
    clips = np.array([r["clip"] for r in rows])
    res = {"model": a.model, "quant": a.quant, "frames": a.frames, "prompt": a.prompt, "size": a.size, "crop": a.crop,
           "n": len(rows), "pos": int(y.sum())}
    if len(np.unique(y)) > 1:
        res.update(roc_auc=float(roc_auc_score(y, sc)), pr_auc=float(average_precision_score(y, sc)))
        res.update(bootstrap_ci(y, sc, clips))
        pred = sc > 0
        res.update(
            precision_at_0=float((y[pred] == 1).mean()) if pred.any() else float("nan"),
            recall_at_0=float(pred[y == 1].mean()),
            false_alarm_rate_at_0=float(pred[y == 0].mean()),
        )
    print(json.dumps(res, indent=2))
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps({"summary": res, "rows": rows}, indent=1), encoding="utf-8")


if __name__ == "__main__":
    main()
