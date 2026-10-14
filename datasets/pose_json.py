"""Convert PoseLift / RetailS pose JSON files into 32-frame feature windows.

Both datasets store ``{person_id: {frame_index: {"keypoints": [x, y, c] * 17}}}``
at 15 FPS with per-frame ``test_frame_mask`` ground truth (0 normal, 1 shoplifting).

Tracks are split at gaps longer than ``max_gap`` frames and shorter gaps are
linearly interpolated, so a window never silently spans a missing stretch.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np

from features.sequence_builder import extract_features_for_track

DATASETS = {
    "retails": {
        "train": ("data/external/retails/RetailS/RetailS_train/pose/train", None),
        "test_staged": (
            "data/external/retails/RetailS/RetailS_test_staged/pose/test",
            "data/external/retails/RetailS/RetailS_test_staged/gt/test_frame_mask",
        ),
        "test_realworld": (
            "data/external/retails/RetailS/RetailS_test_realworld/pose/test",
            "data/external/retails/RetailS/RetailS_test_realworld/gt/test_frame_mask",
        ),
    },
    "poselift": {
        "train": ("data/external/poselift_full/PoseLift/Json_files/data/PoseLift/pose/train", None),
        "test": (
            "data/external/poselift_full/PoseLift/Json_files/data/PoseLift/pose/test",
            "data/external/poselift_full/PoseLift/Json_files/data/PoseLift/gt/test_frame_mask",
        ),
    },
}


def camera_of(stem: str) -> str:
    """Camera id from names like ``1_041812_1`` or ``01_0222_alphapose_tracked_person``."""
    return str(int(stem.split("_")[0]))


def gt_stem(pose_stem: str) -> str:
    return pose_stem.replace("_alphapose_tracked_person", "")


def load_segments(path: Path, min_len: int = 32, max_gap: int = 5) -> list[dict[str, Any]]:
    """Return gap-free, interpolated person segments of at least ``min_len`` frames."""
    with open(path, "r", encoding="utf-8") as fh:
        data = json.load(fh)
    segments = []
    for person, frames in data.items():
        idx = sorted(int(k) for k in frames)
        if not idx:
            continue
        start = 0
        for i in range(1, len(idx) + 1):
            if i == len(idx) or idx[i] - idx[i - 1] > max_gap:
                chunk = idx[start:i]
                start = i
                if chunk[-1] - chunk[0] + 1 < min_len:
                    continue
                full = np.arange(chunk[0], chunk[-1] + 1)
                raw = np.array(
                    [np.asarray(frames[str(f)]["keypoints"], dtype=np.float32).reshape(17, 3) for f in chunk]
                )
                # Some PoseLift joints are NaN: treat them as missing (confidence 0) and
                # fill coordinates from the joint's finite frames.
                finite = np.isfinite(raw).all(axis=2)  # (T, 17)
                kp = np.zeros((len(full), 17, 3), dtype=np.float32)
                chunk_arr = np.asarray(chunk)
                for j in range(17):
                    ok = finite[:, j]
                    conf = np.where(ok, raw[:, j, 2], 0.0)
                    kp[:, j, 2] = np.interp(full, chunk_arr, conf)
                    if ok.any():
                        for c in range(2):
                            kp[:, j, c] = np.interp(full, chunk_arr[ok], raw[ok, j, c])
                segments.append({"person": person, "frames": full, "keypoints": kp})
    return segments


def _records(seg: dict[str, Any], lo: int, hi: int, camera: str) -> list[dict[str, Any]]:
    out = []
    for f, kp in zip(seg["frames"][lo:hi], seg["keypoints"][lo:hi]):
        vis = kp[kp[:, 2] > 0.05]
        pts = vis if len(vis) else kp
        bbox = [float(pts[:, 0].min()), float(pts[:, 1].min()), float(pts[:, 0].max()), float(pts[:, 1].max())]
        out.append(
            {
                "frame_index": int(f),
                "timestamp_ms": int(f) * 1000.0 / 15.0,
                "track_id": int(seg["person"]),
                "camera_id": camera,
                "bbox_xyxy": bbox,
                "keypoints_xyc": kp.tolist(),
            }
        )
    return out


def clip_windows(
    path: Path,
    gt_path: Path | None,
    window: int = 32,
    stride: int = 8,
    max_segments: int | None = None,
    windows_per_segment: int | None = None,
    rng: np.random.RandomState | None = None,
) -> tuple[np.ndarray, list[dict[str, Any]]]:
    """Feature windows for one clip with the fraction of theft frames in each window."""
    camera = camera_of(path.stem)
    segs = load_segments(path, min_len=window)
    rng = rng or np.random.RandomState(0)
    if max_segments is not None and len(segs) > max_segments:
        segs = [segs[i] for i in rng.choice(len(segs), max_segments, replace=False)]
    gt = np.load(gt_path).astype(np.float32).ravel() if gt_path is not None and gt_path.exists() else None

    xs, metas = [], []
    for seg in segs:
        lo, hi = 0, len(seg["frames"])
        if windows_per_segment is not None:
            span = window + stride * (windows_per_segment - 1)
            if hi - lo > span:
                lo = int(rng.randint(0, hi - span + 1))
                hi = lo + span
        wins, meta = extract_features_for_track(_records(seg, lo, hi, camera), window, stride)
        for w, m in zip(wins, meta):
            s, e = int(m["start_frame"]), int(m["end_frame"])
            if gt is None:
                frac = 0.0
            else:
                frac = float(gt[s : min(e + 1, len(gt))].mean()) if s < len(gt) else 0.0
            xs.append(w)
            metas.append(
                {
                    "clip_id": path.stem,
                    "camera_id": camera,
                    "track_id": int(seg["person"]),
                    "start_frame": s,
                    "end_frame": e,
                    "theft_frac": frac,
                }
            )
    if not xs:
        return np.empty((0, window, 77), dtype=np.float32), []
    return np.stack(xs).astype(np.float32), metas


def build(
    dataset: str,
    split: str,
    out_dir: Path,
    max_clips: int | None,
    max_segments: int | None,
    windows_per_segment: int | None,
    seed: int = 0,
) -> dict[str, Any]:
    pose_dir, gt_dir = DATASETS[dataset][split]
    files = sorted(Path(pose_dir).glob("*.json"))
    rng = np.random.RandomState(seed)
    if max_clips is not None and len(files) > max_clips:
        files = [files[i] for i in sorted(rng.choice(len(files), max_clips, replace=False))]
    X, M = [], []
    for i, f in enumerate(files):
        g = Path(gt_dir) / f"{gt_stem(f.stem)}.npy" if gt_dir else None
        x, m = clip_windows(f, g, max_segments=max_segments, windows_per_segment=windows_per_segment, rng=rng)
        if len(x):
            X.append(x)
            M.extend(m)
        if (i + 1) % 25 == 0:
            print(f"  {dataset}/{split}: {i + 1}/{len(files)} clips, {sum(len(a) for a in X)} windows", flush=True)
    out_dir.mkdir(parents=True, exist_ok=True)
    X = np.concatenate(X) if X else np.empty((0, 32, 77), dtype=np.float32)
    np.save(out_dir / f"{dataset}_{split}_X.npy", X.astype(np.float16))
    with open(out_dir / f"{dataset}_{split}_meta.json", "w", encoding="utf-8") as fh:
        json.dump(M, fh)
    frac = np.array([m["theft_frac"] for m in M]) if M else np.array([])
    return {
        "dataset": dataset,
        "split": split,
        "clips": len(files),
        "windows": int(len(X)),
        "windows_with_any_theft": int((frac > 0).sum()),
        "windows_theft_ge_0.5": int((frac >= 0.5).sum()),
    }


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--dataset", choices=list(DATASETS), required=True)
    ap.add_argument("--split", required=True)
    ap.add_argument("--out-dir", default="data/features_unified")
    ap.add_argument("--max-clips", type=int, default=None)
    ap.add_argument("--max-segments", type=int, default=None)
    ap.add_argument("--windows-per-segment", type=int, default=None)
    a = ap.parse_args()
    print(build(a.dataset, a.split, Path(a.out_dir), a.max_clips, a.max_segments, a.windows_per_segment))


if __name__ == "__main__":
    main()
