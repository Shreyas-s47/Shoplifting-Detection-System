# Lightweight Shoplifting-Behaviour Alert System

## Purpose

This project builds a lightweight, privacy-aware CCTV system that flags suspicious concealment-like behaviour for human review. It does not verify theft or make automatic accusations: it observes body pose and motion, not the item itself.

The Phase 2 demonstration is designed for laptop CPU execution. Raspberry Pi or Jetson deployment is a future target; any efficiency claim must be based on measured laptop CPU benchmarks.

## Finalized architecture

CCTV video at 15 FPS -> YOLOv8n-Pose -> ByteTrack -> normalized 17-joint pose sequence -> shelf-zone and body-relative interaction features -> baseline and small GRU/LSTM classifier -> smoothing/cooldown -> human-review alert -> FastAPI/WebSocket dashboard.

### Phase 2 architecture revision

The earlier dual-pose proposal (YOLOv8-Pose plus BlazePose) is revised to a single YOLOv8n-Pose pipeline. PoseLift is the primary dataset and one canonical pose representation prevents incompatible training and runtime features, reduces inference cost, and makes the pipeline easier to evaluate. BlazePose is not retained for finger detail: it is a pose model, and distant CCTV does not provide dependable hand articulation.

## System components

| Component | Method | Responsibility |
|---|---|---|
| Person pose | YOLOv8n-Pose | Detect people and obtain pose keypoints. |
| Tracking | ByteTrack | Maintain person IDs across video frames. |
| Scene interaction | Shelf polygons | Detect wrist approach to store shelf areas. |
| Body interaction | Shoulder/hip/bounding-box-relative regions | Estimate hand-to-body/waist interaction. |
| Behaviour model | Baseline then GRU/LSTM | Score a short movement sequence as normal or suspicious. |
| Alert policy | Smoothing, consecutive positives, cooldown | Control false alerts and create clips for review. |

The project does not claim hand-object detection. It models hand-to-shelf and hand-to-body interactions; detecting bags or objects would require separately labelled object data.

## Dataset policy

| Dataset | Usage |
|---|---|
| PoseLift | Primary labelled pose dataset. Inspect the downloaded JSON/Pickle schema and preserve clip/camera IDs. |
| UCF-Crime | Separate external/generalization check. Do not blindly combine it with PoseLift training data. |
| Consent-based staged clips | Required calibration data for normal actions and the actual demo camera setup. |

The supplied PoseLift data belongs in `data/external/poselift/`. Do not commit raw videos, downloaded datasets, model weights, or generated artifacts. Commit manifests, source code, configs and small non-sensitive samples only.

## Data and model rules

- Standardize input to 15 FPS and use overlapping approximately two-second windows (initial setting: 32 frames).
- Normalize poses by bounding-box scale and torso/shoulder size.
- Use angles, wrist velocity and acceleration, wrist-to-hip distance, wrist-to-shelf distance, dwell time, confidence, and interaction-state flags.
- Train logistic regression or XGBoost as a baseline before the recurrent model.
- Use class weighting or focal loss for class imbalance.
- Split by clip before window generation. Windows from one clip must never appear in both train and test data.
- Tune the alert threshold only using validation data. Require smoothed/consecutive positives and apply a per-person cooldown.

## Evaluation

Report PR-AUC, ROC-AUC, precision, recall, F1, event-level recall, false alerts per person-hour, time-to-alert, CPU FPS, p95 latency, RAM, model size and ONNX INT8 results. Accuracy by itself is not sufficient.

## Intended structure

- `data/external/poselift/` — downloaded PoseLift data, not committed
- `data/raw/`, `data/keypoints/`, `data/manifests/` — source clips, generated poses and metadata
- `configs/` — camera zones, thresholds and model settings
- `detection/`, `features/`, `model/`, `pipeline/` — inference and ML code
- `backend/`, `dashboard/` — review-alert API and interface
- `docs/`, `tests/`, `artifacts/` — plans, testing and generated models

## Build order

1. Implement video ingestion, YOLOv8n-Pose and ByteTrack.
2. Implement pose normalization, feature extraction and dataset manifests.
3. Train and compare a baseline classifier and a small GRU/LSTM.
4. Add offline alert smoothing and evaluation.
5. Export and benchmark ONNX/INT8 on the laptop CPU.
6. Add the FastAPI backend and minimal dashboard.

See `docs/module_01_02_implementation_plan.md` for the first implementation milestone.

## Status

### Module 1 — Video ingestion and preprocessing (done)

- `pipeline/video_reader.py` validates video metadata, samples deterministically at 15 FPS, and resizes frames to 640x360.
- Dataset manifest at `data/manifests/clips.csv`: 182 usable clips (92 shoplifting, 90 normal), all sourced at 640x480 / 30 FPS.
- Tests: `tests/test_video_reader.py`.

### Module 2 — Person pose and tracking (done)

- `detection/person_pose.py` + `detection/tracker.py`: YOLOv8n-Pose (COCO-17 keypoints) + ByteTrack, restricted to the `person` class, with a per-track `track_age` field.
- `pipeline/extract_keypoints.py` writes one JSONL record per tracked person per sampled frame — camera ID, frame index, timestamp, track ID, bbox, keypoints, detection confidence, track age — plus an optional annotated debug video (box, skeleton, track ID overlay).
- `pipeline/batch_extract.py` runs extraction across the whole dataset, reusing one loaded model instead of reloading it per clip.
- `pipeline/benchmark.py` measures mean FPS, p95 frame latency, peak RAM, and active-track count.
- `pipeline/live_demo.py` opens a real-time window running the full pose+tracking pipeline on a clip or a webcam.
- `configs/default.yaml` is wired into every entry point above — detector confidence, IoU threshold, image size, tracker config, and device are read from it, not hardcoded.
- Full-dataset run: all 182 clips processed, 0 failures, 29,925 sampled frames, 41,455 person-records — one pose artifact per clip under `data/keypoints/`.
- Benchmarks (`docs/module_02_benchmark_notes.md`): ~15-24 FPS on CPU, ~34 FPS average across the full batch on GPU (RTX 4050 laptop GPU).
- Tests: `tests/test_tracker.py`.

Next: Module 3 — pose normalization and feature extraction.

## How to run

Install once: `python -m pip install -r requirements.txt`

- Rebuild the clip manifest: `python -m pipeline.video_reader --raw-root data --manifest data/manifests/clips.csv`
- Extract pose+tracking for one clip: `python -m pipeline.extract_keypoints <video> --output <out.jsonl> [--debug-video <out.mp4>]`
- Extract for the whole dataset: `python -m pipeline.batch_extract`
- Benchmark throughput: `python -m pipeline.benchmark <clip> ...`
- Live demo window: `python -m pipeline.live_demo [--source <video-path-or-webcam-index>]`
- Run tests: `python -m pytest tests/`

All commands above read `configs/default.yaml` by default (target FPS, resize, detector/tracker settings, device); pass `--device cpu` to force CPU on a machine without a CUDA GPU.

## Responsible use

- Every output is a review alert, never a conclusion of theft.
- Save clips only on alerts and define retention/deletion rules.
- Restrict dashboard access and log reviews.
- A real deployment needs a dedicated privacy and legal assessment plus clear CCTV notices.
