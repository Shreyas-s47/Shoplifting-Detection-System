"""Train the normal-behaviour autoencoder on normal-only footage and evaluate it.

Training data : RetailS + PoseLift normal-only windows (80% of normal clips).
Held-out      : the remaining 20% of normal clips (negatives for the staged set).
Test sets     : RetailS staged (+ held-out normals) and RetailS real-world (== PoseLift test).
Per-camera calibration: scores are standardised with that camera's *training-normal*
score mean/std (no theft labels used), a label-free form of per-store adaptation.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
from sklearn.metrics import average_precision_score, roc_auc_score

from eval.cross_dataset import bootstrap_ci, concat, load_unified
from model import anomaly


def split_by_clip(d: dict, frac_test: float, seed: int) -> tuple[dict, dict]:
    clips = np.unique(d["clip"])
    rng = np.random.RandomState(seed)
    rng.shuffle(clips)
    held = set(clips[: int(len(clips) * frac_test)])
    mask = np.array([c in held for c in d["clip"]])
    return {k: v[~mask] for k, v in d.items()}, {k: v[mask] for k, v in d.items()}


def calibrate(s: np.ndarray, cam: np.ndarray, stats: dict, fallback: tuple[float, float]) -> np.ndarray:
    out = np.empty_like(s)
    for c in np.unique(cam):
        m, sd = stats.get(c, fallback)
        out[cam == c] = (s[cam == c] - m) / sd
    return out


def metrics(y, s, clip) -> dict:
    r = {
        "roc_auc": float(roc_auc_score(y, s)),
        "pr_auc": float(average_precision_score(y, s)),
        "n": int(len(y)),
        "pos_rate": float(np.mean(y)),
    }
    r.update(bootstrap_ci(y, s, clip))
    return r


def run(epochs: int, seed: int, out: Path) -> dict:
    normals = concat(load_unified("retails_train"), load_unified("poselift_train"))
    train, held = split_by_clip(normals, 0.2, seed)
    staged, real = load_unified("retails_test_staged"), load_unified("retails_test_realworld")
    print(f"train normals {len(train['y'])} | held-out normals {len(held['y'])} | staged {len(staged['y'])} | real {len(real['y'])}")

    model = anomaly.fit(train["X"], epochs=epochs, seed=seed)

    s_train = np.log(anomaly.score(model, train["X"]) + 1e-8)
    stats = {c: (float(s_train[train["cam"] == c].mean()), float(s_train[train["cam"] == c].std() + 1e-6))
             for c in np.unique(train["cam"])}
    fallback = (float(s_train.mean()), float(s_train.std() + 1e-6))

    def scored(d):
        raw = np.log(anomaly.score(model, d["X"]) + 1e-8)
        return raw, calibrate(raw, d["cam"], stats, fallback)

    results: dict = {"epochs": epochs, "seed": seed}
    s_real_raw, s_real_cal = scored(real)
    results["real_world_raw"] = metrics(real["y"], s_real_raw, real["clip"])
    results["real_world_camera_calibrated"] = metrics(real["y"], s_real_cal, real["clip"])

    st = concat(staged, held)
    s_raw, s_cal = scored(st)
    results["staged_plus_heldout_normals_raw"] = metrics(st["y"], s_raw, st["clip"])
    results["staged_plus_heldout_normals_camera_calibrated"] = metrics(st["y"], s_cal, st["clip"])

    for name, r in results.items():
        if isinstance(r, dict):
            print(f"{name:50s} ROC {r['roc_auc']:.3f} {np.round(r['roc_ci'], 3).tolist()}  PR {r['pr_auc']:.3f}  (n={r['n']}, pos={r['pos_rate']:.2f})")
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(results, indent=2), encoding="utf-8")
    return results


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--epochs", type=int, default=25)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", default="artifacts/results/anomaly_autoencoder.json")
    a = ap.parse_args()
    run(a.epochs, a.seed, Path(a.out))


if __name__ == "__main__":
    main()
