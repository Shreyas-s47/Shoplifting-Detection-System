"""Fast pixel-based stage: frozen SigLIP2 frame embeddings + a small classifier.

Each 5 s window -> 8 frames -> SigLIP2 image embeddings (L2-normalised) -> window feature
[mean, std, last - first] -> logistic regression. Trained on Kaggle clips and/or UCF train
windows pseudo-labelled by the VLM teacher, tested on the UCF test windows (true labels).

Usage:
  python -m vlm.fast_probe embed      # cache embeddings (GPU, once)
  python -m vlm.fast_probe evaluate   # train/test experiments + speed
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
import torch
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, roc_auc_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from eval.cross_dataset import bootstrap_ci
from vlm.pseudo_label import train_windows
from vlm.zero_shot import kaggle_samples, read_frames, ucf_samples

ENCODER = "google/siglip2-base-patch16-384"
CACHE = Path("data/features_vlm")
N_FRAMES = 8


class Encoder:
    def __init__(self, model_id: str = ENCODER):
        from transformers import AutoImageProcessor, AutoModel

        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.proc = AutoImageProcessor.from_pretrained(model_id)
        self.model = AutoModel.from_pretrained(model_id, torch_dtype=torch.float16).to(self.device).eval()

    @torch.no_grad()
    def frames(self, imgs) -> np.ndarray:
        px = self.proc(images=imgs, return_tensors="pt")["pixel_values"].to(self.device, torch.float16)
        out = self.model.get_image_features(pixel_values=px)
        feats = out if isinstance(out, torch.Tensor) else out.pooler_output
        f = torch.nn.functional.normalize(feats.float(), dim=-1)
        return f.cpu().numpy()


def window_feature(frame_emb: np.ndarray) -> np.ndarray:
    return np.concatenate([frame_emb.mean(0), frame_emb.std(0), frame_emb[-1] - frame_emb[0]])


def embed_set(enc: Encoder, samples: list[dict], name: str) -> None:
    feats, keep, t0 = [], [], time.time()
    for i, s in enumerate(samples):
        frames = read_frames(s["path"], s["start"], s["end"], N_FRAMES, (384, 384))
        if len(frames) < 4:
            continue
        feats.append(window_feature(enc.frames(frames)))
        keep.append({k: s[k] for k in s if k != "path"})
        if (i + 1) % 100 == 0:
            print(f"  {name}: {i + 1}/{len(samples)} ({(time.time() - t0) / (i + 1) * 1000:.0f} ms/window incl. decode)", flush=True)
    CACHE.mkdir(parents=True, exist_ok=True)
    np.save(CACHE / f"{name}_X.npy", np.stack(feats).astype(np.float32))
    (CACHE / f"{name}_meta.json").write_text(json.dumps(keep), encoding="utf-8")
    print(f"{name}: {len(feats)} windows cached", flush=True)


def kaggle_windows() -> list[dict]:
    out = []
    for s in kaggle_samples(Path("data")):
        mid = (s["start"] + s["end"]) // 2
        out.append({**s, "end": mid})
        out.append({**s, "start": mid})
    return out


def cmd_embed() -> None:
    enc = Encoder()
    embed_set(enc, ucf_samples(Path("data/external/ucf_shoplifting"), Path("data/Temporal_Anomaly_Annotation_for_Testing_Videos.txt")), "ucf_test")
    embed_set(enc, train_windows(Path("data/external/ucf_shoplifting"), Path("data/Anomaly_Train.txt"), 20), "ucf_train")
    embed_set(enc, kaggle_windows(), "kaggle")


def load(name: str):
    return np.load(CACHE / f"{name}_X.npy"), json.loads((CACHE / f"{name}_meta.json").read_text())


def kaggle_split() -> dict:
    """Reuse the dedupe-aware clip split from data/features so clips never cross train/test."""
    split = {}
    for sp in ("train", "val", "test"):
        for m in json.load(open(f"data/features/meta_{sp}.json", encoding="utf-8")):
            split[m["clip_id"]] = sp
    return split


def teacher_labels(meta: list[dict], hi: float, lo: float) -> np.ndarray:
    """1 = teacher confident theft (or the top window of its video), 0 = confident normal, -1 = dropped."""
    t = {(r["clip"], r["start"]): r["p_yes"] for r in json.loads(Path("artifacts/results/ucf_train_teacher_labels.json").read_text())}
    p = np.array([t.get((m["clip"], m["start"]), np.nan) for m in meta])
    y = np.full(len(meta), -1)
    y[p >= hi] = 1
    y[p < lo] = 0
    clips = np.array([m["clip"] for m in meta])
    for c in np.unique(clips):  # every train video contains a theft: force its top window positive
        idx = np.where(clips == c)[0]
        if np.isfinite(p[idx]).any():
            y[idx[np.nanargmax(p[idx])]] = 1
    return y


def fit_predict(Xtr, ytr, Xte, C: float = 0.5):
    clf = make_pipeline(StandardScaler(), LogisticRegression(C=C, class_weight="balanced", max_iter=3000))
    clf.fit(Xtr, ytr)
    return clf, clf.predict_proba(Xte)[:, 1]


def report(name: str, y, p, clips) -> dict:
    r = {"roc_auc": float(roc_auc_score(y, p)), "pr_auc": float(average_precision_score(y, p)), "n": int(len(y)), "pos": int(y.sum())}
    r.update(bootstrap_ci(y, p, clips))
    print(f"{name:52s} ROC {r['roc_auc']:.3f} {np.round(r['roc_ci'], 3).tolist()}  PR {r['pr_auc']:.3f}  (n={r['n']}, pos={r['pos']})", flush=True)
    return r


def cmd_evaluate(hi: float, lo: float, C: float) -> None:
    Xu, mu = load("ucf_test")
    yu = np.array([m["label"] for m in mu])
    cu = np.array([m["clip"] for m in mu])
    Xt, mt = load("ucf_train")
    yt = teacher_labels(mt, hi, lo)
    Xk, mk = load("kaggle")
    yk = np.array([m["label"] for m in mk])
    ck = np.array([m["clip"] for m in mk])
    sp = kaggle_split()
    ktr = np.array([sp.get(c, "train") != "test" for c in ck])
    kte = ~ktr
    keep_t = yt >= 0
    print(f"UCF train windows: {keep_t.sum()} used ({(yt == 1).sum()} pos, {(yt == 0).sum()} neg, {(~keep_t).sum()} dropped)")

    res = {"hi": hi, "lo": lo, "C": C}
    _, p = fit_predict(Xk[ktr], yk[ktr], Xu, C)
    res["kaggle_to_ucf_test"] = report("P1 Kaggle -> UCF test", yu, p, cu)
    _, p = fit_predict(Xk[ktr], yk[ktr], Xk[kte], C)
    res["kaggle_to_kaggle_test"] = report("P1 Kaggle -> Kaggle test", yk[kte], p, ck[kte])
    _, p = fit_predict(Xt[keep_t], yt[keep_t], Xu, C)
    res["ucf_teacher_to_ucf_test"] = report("P2 UCF train (teacher labels) -> UCF test", yu, p, cu)
    Xc = np.concatenate([Xk[ktr], Xt[keep_t]])
    yc = np.concatenate([yk[ktr], yt[keep_t]])
    clf, p = fit_predict(Xc, yc, Xu, C)
    res["combined_to_ucf_test"] = report("P3 Kaggle + UCF teacher -> UCF test", yu, p, cu)
    res["combined_to_kaggle_test"] = report("P3 Kaggle + UCF teacher -> Kaggle test", yk[kte], clf.predict_proba(Xk[kte])[:, 1], ck[kte])

    # speed: encoder + classifier on one 8-frame window (excludes video decode)
    enc = Encoder()
    from PIL import Image

    dummy = [Image.new("RGB", (384, 384)) for _ in range(N_FRAMES)]
    for _ in range(3):
        enc.frames(dummy)
    torch.cuda.synchronize() if torch.cuda.is_available() else None
    t0 = time.time()
    for _ in range(20):
        f = window_feature(enc.frames(dummy))
        clf.predict_proba(f[None])
    torch.cuda.synchronize() if torch.cuda.is_available() else None
    res["ms_per_window"] = (time.time() - t0) / 20 * 1000
    print(f"speed: {res['ms_per_window']:.1f} ms per 8-frame window (encoder + classifier, GPU fp16)")
    Path("artifacts/results/fast_probe.json").write_text(json.dumps(res, indent=2), encoding="utf-8")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("cmd", choices=["embed", "evaluate"])
    ap.add_argument("--hi", type=float, default=0.5)
    ap.add_argument("--lo", type=float, default=0.15)
    ap.add_argument("--C", type=float, default=0.5)
    a = ap.parse_args()
    cmd_embed() if a.cmd == "embed" else cmd_evaluate(a.hi, a.lo, a.C)


if __name__ == "__main__":
    main()
