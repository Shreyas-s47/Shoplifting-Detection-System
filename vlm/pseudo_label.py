"""VLM-teacher pseudo-labels for UCF-Crime Shoplifting *train* videos.

UCF train videos only carry a video-level label ("a theft happens somewhere"), so we score
evenly spaced 5 s windows with the zero-shot teacher (Qwen3-VL-4B, 'stages' prompt) and keep
its P(yes) per window. Training code turns these into labels (see vlm.fast_probe).

Usage: python -m vlm.pseudo_label [--per-video 20]
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import cv2
import numpy as np

from vlm.zero_shot import PROMPTS, load_model, read_frames, vlm_score

TEACHER = "Qwen/Qwen3-VL-4B-Instruct"
WIN = 150  # 5 s at 30 FPS


def train_windows(root: Path, train_list: Path, per_video: int) -> list[dict]:
    out = []
    for line in train_list.read_text().splitlines():
        if not line.startswith("Shoplifting/"):
            continue
        path = root / Path(line.strip()).name
        if not path.exists():
            continue
        total = int(cv2.VideoCapture(str(path)).get(cv2.CAP_PROP_FRAME_COUNT))
        starts = np.linspace(0, max(0, total - WIN - 1), min(per_video, max(1, total // WIN))).astype(int)
        out += [{"clip": path.name, "path": str(path), "start": int(s), "end": int(s) + WIN} for s in starts]
    return out


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--per-video", type=int, default=20)
    ap.add_argument("--out", default="artifacts/results/ucf_train_teacher_labels.json")
    a = ap.parse_args()

    windows = train_windows(Path("data/external/ucf_shoplifting"), Path("data/Anomaly_Train.txt"), a.per_video)
    print(f"{len(windows)} windows from {len({w['clip'] for w in windows})} train videos", flush=True)

    out_path = Path(a.out)
    done = {}
    if out_path.exists():
        done = {(r["clip"], r["start"]): r for r in json.loads(out_path.read_text())}
    processor, model, yes_ids, no_ids = load_model(TEACHER, "4bit")

    rows, t0 = [], time.time()
    for i, w in enumerate(windows, 1):
        key = (w["clip"], w["start"])
        if key in done:
            rows.append(done[key])
            continue
        frames = read_frames(w["path"], w["start"], w["end"])
        if len(frames) < 4:
            continue
        r = vlm_score(processor, model, yes_ids, no_ids, frames, PROMPTS["stages"])
        rows.append({"clip": w["clip"], "start": w["start"], "end": w["end"], **r})
        if i % 25 == 0:
            print(f"  {i}/{len(windows)}  {(time.time() - t0) / i:.2f}s/window", flush=True)
            out_path.parent.mkdir(parents=True, exist_ok=True)
            out_path.write_text(json.dumps(rows, indent=1), encoding="utf-8")

    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(rows, indent=1), encoding="utf-8")
    p = np.array([r["p_yes"] for r in rows])
    print(f"done: {len(rows)} windows, mean P(yes) {p.mean():.2f}, >0.5: {(p > 0.5).mean():.2f}")


if __name__ == "__main__":
    main()
