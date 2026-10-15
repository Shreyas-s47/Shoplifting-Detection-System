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
PROMPT = (
    "These {n} frames, in time order, come from a store CCTV camera. "
    "Is a person concealing or stealing merchandise (for example hiding an item in a bag, "
    "pocket or under clothing instead of paying)? Answer with only Yes or No."
)


def read_frames(path: str, start: int, end: int, n: int = N_FRAMES) -> list[Image.Image]:
    cap = cv2.VideoCapture(path)
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    end = min(end, total - 1)
    idx = np.linspace(start, max(start, end), n).astype(int)
    frames = []
    for i in idx:
        cap.set(cv2.CAP_PROP_POS_FRAMES, int(i))
        ok, bgr = cap.read()
        if not ok:
            continue
        rgb = cv2.cvtColor(cv2.resize(bgr, FRAME_SIZE, interpolation=cv2.INTER_CUBIC), cv2.COLOR_BGR2RGB)
        frames.append(Image.fromarray(rgb))
    cap.release()
    return frames


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


def load_model():
    from transformers import AutoProcessor, Qwen3VLForConditionalGeneration

    processor = AutoProcessor.from_pretrained(MODEL_ID)
    model = Qwen3VLForConditionalGeneration.from_pretrained(
        MODEL_ID, torch_dtype=torch.float16, device_map="cuda" if torch.cuda.is_available() else "cpu"
    ).eval()
    tok = processor.tokenizer
    yes = [tok.encode(t, add_special_tokens=False)[0] for t in ("Yes", "yes", " Yes")]
    no = [tok.encode(t, add_special_tokens=False)[0] for t in ("No", "no", " No")]
    return processor, model, sorted(set(yes)), sorted(set(no))


@torch.no_grad()
def vlm_score(processor, model, yes_ids, no_ids, frames: list[Image.Image], explain: bool = False) -> dict:
    content = [{"type": "image"} for _ in frames] + [{"type": "text", "text": PROMPT.format(n=len(frames))}]
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
    a = ap.parse_args()

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
    processor, model, yes_ids, no_ids = load_model()

    rows, t0 = [], time.time()
    for i, s in enumerate(samples, 1):
        key = (s["clip"], s["start"])
        if key in done:
            rows.append(done[key])
            continue
        frames = read_frames(s["path"], s["start"], s["end"])
        if len(frames) < 4:
            continue
        r = vlm_score(processor, model, yes_ids, no_ids, frames)
        rows.append({**{k: s[k] for k in ("clip", "start", "end", "label")}, **r})
        if i % 20 == 0:
            print(f"  {i}/{len(samples)}  {(time.time() - t0) / i:.2f}s/sample", flush=True)

    y = np.array([r["label"] for r in rows])
    sc = np.array([r["score"] for r in rows])
    clips = np.array([r["clip"] for r in rows])
    res = {"model": MODEL_ID, "n": len(rows), "pos": int(y.sum())}
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
