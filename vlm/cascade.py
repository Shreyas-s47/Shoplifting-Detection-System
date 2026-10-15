"""Step 2: cascade simulation on UCF-Crime test windows.

Stage 1 = cheap pose pre-filter (YOLOv8n-pose on the 8 frames of a window, no training):
  presence * (hand-near-torso + wrist motion). Stage 2 = Qwen3-VL-4B 'stages' scores (cached).
Only the top `pass_rate` of windows by the pre-filter reach the VLM; the rest are scored -inf.

Usage: python -m vlm.cascade
"""

from __future__ import annotations

import json
import time
from pathlib import Path

import numpy as np
from sklearn.metrics import roc_auc_score

from eval.cross_dataset import bootstrap_ci
from vlm.zero_shot import read_frames, ucf_samples

VLM_JSON = Path("artifacts/results/vlm_ucf_4b_stages.json")
OUT = Path("artifacts/results/cascade.json")
CACHE = Path("data/features_vlm/ucf_test_prefilter.json")


def prefilter_features(model, frames) -> dict:
    """Pose cues over one window: presence, hands near torso, wrist motion (all height-normalised)."""
    import cv2

    near, wrist_tracks, n_people = 0.0, [], 0
    for f in frames:
        bgr = cv2.cvtColor(np.asarray(f), cv2.COLOR_RGB2BGR)
        r = model.predict(bgr, conf=0.25, verbose=False)[0]
        if r.keypoints is None or len(r.boxes) == 0:
            wrist_tracks.append(None)
            continue
        kp = r.keypoints.xy.cpu().numpy()
        cf = r.keypoints.conf.cpu().numpy()
        n_people = max(n_people, len(kp))
        best = 0.0
        for k, c in zip(kp, cf):
            h = max(k[:, 1].max() - k[:, 1].min(), 1.0)
            if c[[5, 6, 11, 12]].min() < 0.3:
                continue
            torso = k[[5, 6, 11, 12]].mean(0)
            for w in (9, 10):
                if c[w] > 0.3:
                    best = max(best, 1.0 - min(1.0, np.linalg.norm(k[w] - torso) / (0.6 * h)))
        near = max(near, best)
        wrist_tracks.append(kp[0, [9, 10]] / max(kp[0, :, 1].max() - kp[0, :, 1].min(), 1.0))
    ok = [w for w in wrist_tracks if w is not None]
    motion = float(np.mean([np.linalg.norm(b - a, axis=1).mean() for a, b in zip(ok, ok[1:])])) if len(ok) > 1 else 0.0
    return {"people": n_people, "near": float(near), "motion": motion}


def main() -> None:
    from ultralytics import YOLO

    samples = ucf_samples(Path("data/external/ucf_shoplifting"), Path("data/Temporal_Anomaly_Annotation_for_Testing_Videos.txt"))
    model = YOLO("yolov8n-pose.pt")
    feats, t0 = {}, time.time()
    for s in samples:
        frames = read_frames(s["path"], s["start"], s["end"])
        feats[(s["clip"], s["start"])] = prefilter_features(model, frames)
    ms = (time.time() - t0) / len(samples) * 1000
    print(f"pre-filter: {ms:.0f} ms per window (decode + 8 pose passes)")

    vlm = {(r["clip"], r["start"]): r for r in json.loads(VLM_JSON.read_text())["rows"]}
    rows = [(k, vlm[k], feats[k]) for k in feats if k in vlm]
    y = np.array([r["label"] for _, r, _ in rows])
    clips = np.array([k[0] for k, _, _ in rows])
    v = np.array([r["score"] for _, r, _ in rows])
    pres = np.array([f["people"] > 0 for _, _, f in rows], float)
    near = np.array([f["near"] for _, _, f in rows])
    mot = np.array([f["motion"] for _, _, f in rows])
    mot_n = mot / (np.percentile(mot, 95) + 1e-9)
    pre = pres * (near + np.clip(mot_n, 0, 1))

    res = {"n": len(y), "pos": int(y.sum()), "prefilter_ms_per_window": ms,
           "prefilter_roc": {"combined": float(roc_auc_score(y, pre)), "near": float(roc_auc_score(y, near)),
                             "motion": float(roc_auc_score(y, mot))},
           "vlm_alone_roc": float(roc_auc_score(y, v)), "cascade": []}
    print(f"n={len(y)} pos={int(y.sum())}  VLM alone ROC {res['vlm_alone_roc']:.3f}")
    print("pre-filter ROC:", {k: round(x, 3) for k, x in res["prefilter_roc"].items()})
    for rate in (0.2, 0.3, 0.5, 0.7, 1.0):
        thr = np.quantile(pre, 1 - rate) if rate < 1 else -np.inf
        passed = pre >= thr
        casc = np.where(passed, v, v.min() - 10)
        r = {"pass_rate": rate, "recall_kept": float(passed[y == 1].mean()), "roc": float(roc_auc_score(y, casc)),
             "pred": {}}
        pred = casc > 0
        r["recall_at_0"] = float(pred[y == 1].mean())
        r["false_alarm_at_0"] = float(pred[y == 0].mean())
        r["precision_at_0"] = float((y[pred] == 1).mean()) if pred.any() else None
        r["vlm_calls_per_min_per_camera"] = rate * 12  # one 5 s window every 5 s
        res["cascade"].append(r)
        print(f"pass {rate:.0%}: theft windows kept {r['recall_kept']:.2f}  ROC {r['roc']:.3f}  "
              f"recall@0 {r['recall_at_0']:.2f}  FA@0 {r['false_alarm_at_0']:.2f}  VLM calls/min/cam {r['vlm_calls_per_min_per_camera']:.1f}")
    OUT.write_text(json.dumps(res, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
