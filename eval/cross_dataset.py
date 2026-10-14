"""Cross-dataset / cross-camera evaluation with clip-level bootstrap confidence intervals.

Datasets (all 32-frame x 77-feature windows at 15 FPS):
  kaggle        original YOLO-pose clips (weak clip-level labels)
  rs_train      RetailS normal-only training footage (subsampled)
  rs_staged     RetailS staged shoplifting test set (frame-level labels, 6 cameras)
  rs_real       RetailS real-world incidents (== PoseLift test set, frame-level labels)
  pl_train      PoseLift normal-only training footage

A window is positive if >= ``pos_frac`` of its frames are theft; windows with
0 < frac < pos_frac are dropped as ambiguous.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Callable

import numpy as np
from sklearn.metrics import average_precision_score, roc_auc_score

from model.baseline import aggregate_sequence_features, train_baseline_model

UNI = Path("data/features_unified")
KAGGLE = Path("data/features")


def load_unified(name: str, pos_frac: float = 0.5) -> dict:
    X = np.load(UNI / f"{name}_X.npy").astype(np.float32)
    meta = json.load(open(UNI / f"{name}_meta.json", encoding="utf-8"))
    frac = np.array([m["theft_frac"] for m in meta])
    keep = (frac == 0) | (frac >= pos_frac)
    y = (frac >= pos_frac).astype(int)
    return {
        "X": X[keep],
        "y": y[keep],
        "clip": np.array([f"{name}:{m['clip_id']}" for m in meta])[keep],
        "cam": np.array([m["camera_id"] for m in meta])[keep],
    }


def load_kaggle(split: str) -> dict:
    X = np.load(KAGGLE / f"X_{split}.npy").astype(np.float32)
    y = np.load(KAGGLE / f"y_{split}.npy").astype(int)
    meta = json.load(open(KAGGLE / f"meta_{split}.json", encoding="utf-8"))
    return {
        "X": X,
        "y": y,
        "clip": np.array([f"kaggle:{m['clip_id']}" for m in meta]),
        "cam": np.array(["k"] * len(y)),
    }


def concat(*parts: dict) -> dict:
    return {k: np.concatenate([p[k] for p in parts]) for k in ("X", "y", "clip", "cam")}


def bootstrap_ci(y: np.ndarray, p: np.ndarray, clips: np.ndarray, n: int = 300, seed: int = 0) -> dict:
    """Cluster bootstrap over clips (windows of one clip are not independent)."""
    rng = np.random.RandomState(seed)
    uniq = np.unique(clips)
    idx_by = {c: np.where(clips == c)[0] for c in uniq}
    roc, pr = [], []
    for _ in range(n):
        pick = rng.choice(uniq, len(uniq), replace=True)
        idx = np.concatenate([idx_by[c] for c in pick])
        if len(np.unique(y[idx])) < 2:
            continue
        roc.append(roc_auc_score(y[idx], p[idx]))
        pr.append(average_precision_score(y[idx], p[idx]))
    lo = lambda a: float(np.percentile(a, 2.5)) if a else float("nan")
    hi = lambda a: float(np.percentile(a, 97.5)) if a else float("nan")
    return {"roc_ci": [lo(roc), hi(roc)], "pr_ci": [lo(pr), hi(pr)]}


def score(model, test: dict) -> dict:
    p = model.predict_proba(aggregate_sequence_features(test["X"]))[:, 1]
    out = {
        "n_windows": int(len(p)),
        "n_pos": int(test["y"].sum()),
        "roc_auc": float(roc_auc_score(test["y"], p)),
        "pr_auc": float(average_precision_score(test["y"], p)),
        "pos_rate": float(test["y"].mean()),
    }
    out.update(bootstrap_ci(test["y"], p, test["clip"]))
    return out


def fit(train: dict, model_type: str):
    return train_baseline_model(train["X"], train["y"], model_type=model_type)


def run(model_type: str, out_path: Path) -> dict:
    rs_train, rs_staged, rs_real = load_unified("retails_train"), load_unified("retails_test_staged"), load_unified("retails_test_realworld")
    pl_train = load_unified("poselift_train")
    kag_train, kag_val, kag_test = load_kaggle("train"), load_kaggle("val"), load_kaggle("test")
    kag_trainval = concat(kag_train, kag_val)
    normals = concat(rs_train, pl_train)

    results: dict = {"model": model_type}

    def report(name: str, train: dict, test: dict):
        r = score(fit(train, model_type), test)
        r["train_windows"] = int(len(train["y"]))
        results[name] = r
        print(f"{name:55s} ROC {r['roc_auc']:.3f} {r['roc_ci']}  PR {r['pr_auc']:.3f} {r['pr_ci']}  (n={r['n_windows']}, pos={r['pos_rate']:.2f})", flush=True)

    print(f"== {model_type} ==")
    report("A  RetailS staged+normals -> RetailS real-world", concat(rs_staged, normals), rs_real)
    report("B  Kaggle -> RetailS real-world", kag_trainval, rs_real)
    report("C  RetailS staged+normals -> Kaggle test", concat(rs_staged, normals), kag_test)
    report("D  Kaggle -> Kaggle test (same domain)", kag_trainval, kag_test)
    report("E  RetailS staged+normals+Kaggle -> RetailS real", concat(rs_staged, normals, kag_trainval), rs_real)

    # Leave-one-camera-out on staged data (train on other cameras + normals)
    for cam in sorted(np.unique(rs_staged["cam"])):
        # test = held-out camera's staged windows + its normal-only windows (negatives)
        te = concat(
            {k: v[rs_staged["cam"] == cam] for k, v in rs_staged.items()},
            {k: v[normals["cam"] == cam] for k, v in normals.items()},
        )
        tr_s = {k: v[rs_staged["cam"] != cam] for k, v in rs_staged.items()}
        tr_n = {k: v[normals["cam"] != cam] for k, v in normals.items()}
        if len(np.unique(te["y"])) < 2:
            continue
        report(f"F  leave-camera-{cam}-out (staged+normals)", concat(tr_s, tr_n), te)

    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(results, indent=2), encoding="utf-8")
    return results


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--model", default="logistic", choices=["logistic", "random_forest"])
    ap.add_argument("--out", default=None)
    a = ap.parse_args()
    run(a.model, Path(a.out or f"artifacts/results/cross_dataset_{a.model}.json"))


if __name__ == "__main__":
    main()
