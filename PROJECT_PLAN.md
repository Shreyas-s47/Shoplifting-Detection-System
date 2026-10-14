# Project Plan — Real-World Shoplifting Detection System

This is the single reference document for the project: the goal, current state, research basis, target architecture, datasets, storage budget, phases, evaluation protocol and decisions. Update the **Decision log** and **Progress tracker** as work proceeds.

---

## 1. Goal

Build a shoplifting-behaviour detection system that:
- works on **real supermarket CCTV**, including Indian stores, under varied cameras, lighting, crowding and occlusion;
- is **accurate**: high event recall with few false alerts per camera-hour;
- is **real-time**: live multi-camera inference on a single GPU, with a CPU/edge path later;
- gives an **explainable review alert** (stage and reason), never an automatic accusation;
- works **without a large Indian labelled dataset**, through cross-dataset training, augmentation, VLM pseudo-labelling and per-store adaptation.

Out of scope: face recognition, automatic punishment or blacklisting, cashier/POS integration.

---

## 2. Current state (October 2026)

| Module | Status | Notes |
|---|---|---|
| 1. Video ingestion | Done | 15 FPS, 640x360, manifest of 182 clips (92 shoplifting / 90 normal, Kaggle) |
| 2. Pose + tracking | Done | YOLOv8n-Pose + ByteTrack, JSONL per clip under `data/keypoints/` |
| 3. Features | Done (merged) | 77-dim per-frame features, 32-frame windows, clip-level split |
| 4. Classifier | Done (merged), weak | GRU test ROC-AUC 0.76; logistic/RF baseline 0.81 |
| 5+. Alerts, real-time classifier, ONNX, dashboard | Not started | `pipeline/live_demo.py` is pose-only |

Known issues to fix in Phase 0:
- Threshold tuned on the test set (`pipeline/train_model.py`).
- Every window in a theft clip is labelled positive (label noise).
- Shelf features are constant (no polygons configured).
- `model/train.py` defaults `input_dim=76`; the data is 77.
- Tiny single split (about 14 theft clips in test), so the numbers are unstable.
- The GRU checkpoint comes from epoch 2 and overfits after that.
- `FINAL_PROJECT_REPORT.md` is a draft with unreal claims. **Ignore it.**

---

## 3. Research basis (why this design)

| Finding | Implication |
|---|---|
| PoseLift benchmark: best pose anomaly model (STG-NF) reaches only 67.5 AUROC | Pose alone cannot see the item, so add visual hand-region evidence |
| RetailS (2026): periodic unsupervised adaptation per camera beats offline models in 91.6% of evaluations | Add a per-store adaptation loop |
| Paza (2026): zero-shot VLM verifier reaches 89.5% precision, with 240x fewer calls via a cheap pre-filter | Use a VLM only as a gated Tier-3 verifier and explainer |
| RTMO beats YOLO11-Pose and YOLO26-Pose on COCO at similar scale | Benchmark pose models on our footage before choosing |
| Video foundation models (VideoMAE V2, V-JEPA 2) transfer to small data as frozen backbones with a small head | Use them frozen on hand/torso crops |
| No large public real shoplifting dataset exists | Combine datasets, augment, pseudo-label with a VLM, and adapt on-site |

---

## 4. Target architecture: 3-tier cascade

```
CCTV streams
   |
   v
TIER 1  every frame (fast)
   person detection + pose (RTMO / YOLO26-Pose / YOLO11-Pose, chosen by benchmark)
   tracking (ByteTrack / BoT-SORT) + optional body Re-ID (OSNet), no faces
   object cues: bag / basket / trolley / handheld product (YOLO-World / YOLOE or fine-tuned YOLO)
   |
   v
TIER 2  every 32-frame window per person
   A  skeleton transformer / GCN  (pretrained on large skeleton action data, fine-tuned)
   B  pose-guided hand+torso crops -> frozen video foundation model -> small head
   C  normal-behaviour anomaly model (STG-NF style), adapted per camera without labels
   -> fusion head -> calibrated suspicion score + stage
      (approach, reach, pick, conceal, leave)
   |
   v  only when the score and stage logic fire
TIER 3  small VLM (Qwen3-VL-2B/4B, SmolVLM2, Gemma) checks the ~5 s clip
   -> verdict + one-line reason
   |
   v
ALERT POLICY  smoothing, stage-sequence check, per-track cooldown, clip save
   |
   v
FastAPI + WebSocket -> review dashboard (clip, stage timeline, reason, accept/reject)
   |
   v
feedback log -> threshold tuning and retraining
```

### Key innovations
1. **Pose-guided hand-region fusion.** Skeleton dynamics are combined with foundation-model features of small hand and torso crops. This lets the model see concealment cheaply.
2. **VLM teacher, small student.** A VLM pseudo-labels large unlabelled retail footage offline, and the fast Tier-2 model is distilled from it.
3. **Zero-label per-store adaptation.** Each camera's normal-behaviour model and thresholds update periodically from its own traffic.
4. **Stage-aware explainable alerts.** Each alert carries a stage sequence plus the VLM reason.
5. **Cascade efficiency.** Expensive models run only when cheap ones fire.
6. **Robustness data engine.** Skeleton and video augmentation, cross-dataset training, and a staged Indian-context test set.

---

## 5. Datasets

| Dataset | Content | Labels | Use | Est. size | Access / licence |
|---|---|---|---|---|---|
| Kaggle shoplifting (current) | 182 clips, 640x480 @30 FPS | clip-level | train/val/test | **0.73 GB** (already on disk) | Kaggle, kipshidze |
| PoseLift | 153 real-store videos, 6 cameras, 43 theft; **pose only, no video** | frame-level | pose train/test, anomaly benchmark | **~0.1–0.5 GB** (pose PKL/JSON) | Apache-2.0, Google Drive; **current download incomplete** (permissions) |
| UCF-Crime: Shoplifting class only | 50 videos (29 train / 21 test), 320x240, real CCTV | temporal labels for test | cross-dataset test, pretraining | **~1–2 GB** for the shoplifting videos; full UCF-Crime is ~96 GB (avoid) | Official site / mirrors |
| UCF-Crime: Normal videos (subset) | real CCTV normal footage | normal | anomaly-model normal pool | ~2–5 GB for a few hundred clips | same |
| DCSASS | UCF-Crime-derived short clips, incl. shoplifting | clip-level | extra training clips | **~1–3 GB** (to verify on Kaggle) | Kaggle |
| MNNIT Allahabad shoplifting (India) | normal vs concealment, 640x480 @30 FPS | clip-level | **Indian-context train/test** | unknown, est. **1–5 GB** (verify before download) | Mendeley, CC BY 4.0 |
| Simuletic CCTV sample | synthetic skeleton + video, normal/theft | sequence | augmentation / sanity | **< 1 GB** (open sample) | Kaggle; full set is paid |
| RetailS | multi-day, 6-camera real retail store, pose | frame-level | best real-world benchmark | unknown | **Email the authors** for access |
| Own staged clips (India) | 30–100 consented clips in a shop or mock aisle | frame-level (CVAT) | **held-out Indian test** | ~2–5 GB raw | own |
| Unlabelled retail footage | public store walkthroughs / CCTV | none | VLM pseudo-labelling, anomaly pretraining | ~10–30 GB (capped) | must respect licences |

All raw data lives under `data/external/<name>/` (gitignored). Each dataset gets a converter to the unified schema (section 8).

---

## 6. Storage budget

Disk status at planning time: **D: 146 GB free**, **C: only 11 GB free**.

> **Important:** pip, Hugging Face, Torch and Ultralytics caches default to C:. Before downloading models, set:
> `HF_HOME=D:\ml-cache\huggingface`, `TORCH_HOME=D:\ml-cache\torch`, `PIP_CACHE_DIR=D:\ml-cache\pip`, `ULTRALYTICS` weights into the project folder.

| Item | Estimate |
|---|---|
| Datasets (Phase 1, without unlabelled pool) | 6–15 GB |
| Unlabelled footage pool (Phase 4, optional, capped) | 10–30 GB |
| Re-extracted keypoints + features (all datasets, several pose models) | 2–5 GB |
| Hand-crop clips / cached foundation-model embeddings | 5–15 GB |
| Pose models (RTMO-S/M, YOLO26/11-Pose n/s) | < 0.5 GB |
| Video foundation model (VideoMAE V2-S/B, V-JEPA 2 ViT-L) | 0.1–1.5 GB |
| VLMs (Qwen3-VL-2B + 4B, SmolVLM2-2.2B; fp16 + 4-bit) | 10–20 GB |
| Python env (torch CUDA, transformers, mmpose/rtmlib, vllm/ollama) | 8–12 GB |
| Training checkpoints, ONNX/TensorRT engines, logs | 3–8 GB |
| Alert clips / demo recordings | 1–3 GB |
| **Total, core (no unlabelled pool)** | **~35–60 GB** |
| **Total, with unlabelled pool** | **~50–90 GB** |

This fits on D:. Download order: small datasets first (PoseLift, MNNIT, DCSASS, UCF shoplifting), then models, then the unlabelled pool only when Phase 4 starts. Check each dataset's real size before downloading and record it in the Decision log.

---

## 7. Hardware and compute

- Local: **RTX 4050 Laptop GPU, 6 GB VRAM.** Good for pose extraction, Tier-2 training, frozen-backbone embedding extraction, and 4-bit 2–4B VLM inference.
- For heavier jobs (VLM pseudo-labelling at scale, larger backbones), use Colab or Kaggle GPU (T4/P100/A100) if needed.
- Deployment target: single GPU for live multi-camera inference. CPU/edge (ONNX INT8, Jetson) is a later optimisation.

---

## 8. Unified data schema

Per clip, after conversion:
```
data/unified/<dataset>/<clip_id>/
  meta.json        dataset, clip_id, camera_id, scene_id, fps, resolution, source_path, licence
  poses.jsonl      frame_idx, ts_ms, track_id, bbox_xyxy, keypoints_xyc (COCO-17), det_conf, pose_model
  labels.npy       per-frame 0/1 (clip-level datasets: all frames of a theft clip = 1, flagged weak=true)
  objects.jsonl    (Phase 2) frame_idx, class, bbox, conf
data/manifests/unified.csv   one row per clip: dataset, clip_id, scene_id, label, weak_label, split_group
```
Splits are grouped by `scene_id`/camera so that the same scene never appears in both train and test.

---

## 9. Phases

Each phase has deliverables and an exit criterion. Do not start the next phase until the exit criterion is met, except where marked parallel.

### Phase 0 — Trustworthy foundation
- Commit the merge; delete `updated project to copy/`.
- Fix `input_dim`, make the threshold validation-only, and add early stopping.
- Evaluation harness: grouped k-fold by clip/scene, bootstrap confidence intervals.
- Metrics: ROC-AUC, PR-AUC, F1, event recall, false alerts per hour, time-to-alert.
- Experiment tracking (MLflow, local) and a seed-controlled run of every baseline.
- **Exit:** one command (`python -m eval.run_all`) reproduces a results table for logistic, XGBoost, GRU and LSTM with CIs.

### Phase 1 — Data engine
- Fix the PoseLift download and fetch MNNIT, DCSASS, UCF-Crime shoplifting and the Simuletic sample (sizes checked first).
- Write converters to the unified schema; build `unified.csv`.
- Write a skeleton augmentation pack: viewpoint rotation, scale, joint dropout (occlusion), jitter, frame drop, speed change, mirror.
- Write a video augmentation pack for crops: low light, blur, JPEG/H.264 compression, noise, colour shift.
- Record and label staged Indian-context clips (CVAT); they are held out for testing.
- **Exit:** all datasets in the unified schema; cross-dataset baseline table (train on X, test on Y).

### Phase 2 — Perception upgrade (Tier 1)
- Benchmark RTMO-S/M, YOLO26-Pose and YOLO11-Pose: keypoint stability, missed people under occlusion, and FPS on the RTX 4050.
- Tracker comparison: ByteTrack vs BoT-SORT (+Re-ID), with ID switches on our clips.
- Object cues: test YOLO-World/YOLOE prompts (bag, backpack, handbag, basket, trolley, bottle, packet). Fine-tune if they are weak.
- Re-extract all datasets with the chosen pose model.
- **Exit:** chosen Tier-1 stack with measured FPS and a short justification table.

### Phase 3 — Behaviour models (Tier 2)
- Branch A: small skeleton transformer (Shopformer-style tokens) and ST-GCN/BlockGCN, optionally pretrained on public skeleton action data.
- Branch B: wrist/torso crop extraction, frozen VideoMAE V2-S / V-JEPA 2 embeddings, and a small attentive head.
- Branch C: STG-NF-style normal-behaviour model, trained on normal clips only.
- Fusion head with temperature calibration, plus a stage head (weak stage labels from rules plus a small hand-labelled set).
- Ablations: A, B, C, A+B, A+B+C, each with and without augmentation; within-dataset and cross-dataset.
- **Exit:** the best Tier-2 model beats the Phase 0 baseline with non-overlapping CIs on cross-dataset tests.

### Phase 4 — VLM tier and distillation (Tier 3)
- Zero-shot VLM verifier (Qwen3-VL-2B/4B, SmolVLM2), prompt design, structured JSON output (`verdict`, `stage`, `reason`, `confidence`).
- Measure the precision/false-alert reduction when Tier 3 is added on top of Tier 2.
- VLM pseudo-labelling of the unlabelled footage pool, retraining Tier 2 on the pseudo-labels, and measuring the gain.
- **Exit:** Tier 3 reduces false alerts per hour without losing more than a few points of event recall; distillation result recorded either way.

### Phase 5 — Real-time system
- Streaming orchestrator: multi-camera input (RTSP/files), per-track window buffers, batched Tier-2 inference, async Tier-3 queue.
- Export: ONNX/TensorRT (FP16/INT8) for pose and Tier-2.
- Alert policy: EMA smoothing, stage sequence check, cooldown, pre/post-event clip save.
- Per-camera adaptation: daily unsupervised update of Branch C and the threshold to a target false-alert rate.
- **Exit:** measured end-to-end FPS, p95 latency, GPU memory and the number of concurrent streams on the RTX 4050.

### Phase 6 — Product
- FastAPI backend: REST for alerts/clips/feedback and WebSocket for live alerts.
- Dashboard: live camera grid, alert queue, clip player with skeleton overlay, stage timeline, VLM reason, accept/reject.
- Privacy: face blur on saved clips, clips saved only on alert, retention policy, access log.
- **Exit:** end-to-end demo from video or webcam to an alert on the dashboard within a few seconds.

### Phase 7 — Final evaluation and write-up
- Final tables: within-dataset, cross-dataset, leave-one-scene-out, staged Indian test, false alerts per camera-hour on long normal footage, time-to-alert, throughput, ablations.
- Error analysis with example clips (true positives, false positives, misses).
- Write the report only from measured numbers.

Dependency order: 0 → 1 → 2 → 3 → 4 → 5 → 6 → 7 (Phase 5 export work can start in parallel once the Tier-2 architecture is fixed).

---

## 10. Evaluation protocol

- **Splits:** grouped by scene/camera and clip, never by window. k-fold (k=5) for small datasets; fixed held-out test sets for cross-dataset and staged Indian data.
- **Threshold and calibration:** tuned on validation only.
- **Window metrics:** ROC-AUC, PR-AUC, precision, recall, F1.
- **Event metrics:** event recall (a theft clip/segment is detected if any alert overlaps it), false alerts per camera-hour, time-to-alert.
- **System metrics:** FPS, p95 latency, GPU/CPU memory, streams per GPU, model sizes.
- **Uncertainty:** bootstrap 95% CIs on every reported number.
- **Ablation rule:** a component stays only if it improves the cross-dataset or Indian-test result.

---

## 11. Tooling

| Purpose | Tool |
|---|---|
| Core | Python 3.12, PyTorch (CUDA), NumPy, pandas, scikit-learn, XGBoost |
| Pose / detection / tracking | Ultralytics (YOLO11/26-Pose, YOLO-World/YOLOE), rtmlib or MMPose (RTMO), BoxMOT (BoT-SORT, OSNet Re-ID) |
| Video foundation models / VLMs | Hugging Face transformers, bitsandbytes (4-bit); Ollama or vLLM for serving |
| Experiments | MLflow (local), Optuna for tuning |
| Labelling | CVAT or Label Studio |
| Export / runtime | ONNX Runtime, TensorRT |
| Serving / UI | FastAPI, WebSocket, a lightweight web dashboard |
| Testing | pytest; every module gets unit tests plus a small end-to-end smoke test |

---

## 12. Target repository layout

```
configs/              default.yaml, cameras/<cam>.yaml (zones), experiments/*.yaml
data/                 external/, unified/, manifests/, keypoints/, features/   (gitignored except manifests)
datasets/             converters per dataset -> unified schema
detection/            pose, tracking, re-id, object cues
features/             skeleton features, augmentation, crop extraction
model/                skeleton (A), visual (B), anomaly (C), fusion, stage head, training
vlm/                  verifier, prompts, pseudo-labelling
eval/                 metrics, grouped CV, run_all, reports
pipeline/             ingestion, extraction, streaming orchestrator, live demo
backend/              FastAPI app
dashboard/            web UI
artifacts/            models, exports, results (small files committed, large ones ignored)
docs/                 design notes, benchmark notes
tests/
```

---

## 13. Risks and mitigations

| Risk | Mitigation |
|---|---|
| Datasets are small and staged | Cross-dataset training, augmentation, VLM pseudo-labels, per-store adaptation |
| PoseLift / RetailS access blocked | Proceed with other sets; email authors; PoseLift requires fixed Drive permissions |
| 6 GB VRAM limits | Frozen backbones, cached embeddings, 4-bit VLMs, small batches, Colab/Kaggle for big jobs |
| VLM hallucination | VLM is a verifier only; structured output; measured precision; a human always reviews |
| False alerts in crowded stores | Stage-sequence logic, smoothing, cooldown, per-camera thresholds by target alerts/hour |
| Domain shift to Indian stores | Staged Indian test set, MNNIT data, adaptation loop |
| Privacy / legal | No face recognition, face blur, alert-only clip storage, retention rules, CCTV notices |
| C: drive almost full | All caches redirected to D: |

---

## 14. Responsible use

- Every output is a **review alert** for a human, never proof of theft.
- No face recognition or identity databases. Re-ID is body-appearance only, within a session.
- Store clips only on alert, with defined retention, deletion and access logging.
- A real deployment needs a privacy and legal assessment plus clear CCTV notices.

---

## 15. Progress tracker

| Phase | Status | Date | Notes |
|---|---|---|---|
| Merge friend's Modules 3–4 | Done | 2026-10-14 | 18/18 tests pass; checkpoint reproduces ROC-AUC 0.765 |
| 0 Foundation | Not started | | |
| 1 Data engine | Not started | | |
| 2 Perception | Not started | | |
| 3 Behaviour models | Not started | | |
| 4 VLM tier | Not started | | |
| 5 Real-time | Not started | | |
| 6 Product | Not started | | |
| 7 Final evaluation | Not started | | |

## 16. Decision log

- 2026-10-14: Friend's Modules 3–4 merged into the main repo.
- 2026-10-14: `FINAL_PROJECT_REPORT.md` declared non-authoritative; ignore it.
- 2026-10-14: Logistic/RF baseline beats the GRU on the current split, so model complexity is treated as unproven until Phase 0 CV.
- 2026-10-14: Architecture changed from pose-only GRU to a 3-tier cascade (pose + hand-region visual + anomaly, gated VLM verifier).
- 2026-10-14: Hardware is no longer restricted to CPU; real-time on a single GPU is the target.

---

## 17. References

- PoseLift (WACV-W 2025): https://arxiv.org/pdf/2501.06591 · https://github.com/TeCSAR-UNCC/PoseLift
- Shopformer (2025): https://arxiv.org/pdf/2504.19970
- RetailS / periodic adaptation (2026): https://arxiv.org/pdf/2603.04723
- Paza zero-shot VLM theft detection (2026): https://arxiv.org/pdf/2604.14846
- RTMO (CVPR 2024): https://arxiv.org/pdf/2312.07526 · GTR pose comparison: https://arxiv.org/pdf/2609.26590
- VideoMAE V2: https://arxiv.org/pdf/2303.16727 · InternVideo-Next: https://arxiv.org/pdf/2512.01342
- LAVAD (CVPR 2024): https://lucazanella.github.io/lavad/ · Vad-R1: https://arxiv.org/pdf/2505.19877 · TRACES: https://arxiv.org/pdf/2511.00580
- MoCoDAD (ICCV 2023): https://arxiv.org/pdf/2307.07205 · Action Hints (2025): https://arxiv.org/pdf/2509.11058
- ProtoGCN (CVPR 2025): https://arxiv.org/html/2411.18941v1 · BlockGCN: https://arxiv.org/html/2305.11468v3
- Real-time hand-object interaction cascade (2025): https://arxiv.org/pdf/2507.13326
- MNNIT shoplifting dataset: https://data.mendeley.com/datasets/r3yjf35hzr/1
- Simuletic CCTV dataset: https://www.kaggle.com/datasets/simuletic/cctv-shoplifting-detection-dataset-yolo-and-vlm
- Current dataset: https://www.kaggle.com/datasets/kipshidze/shoplifting-video-dataset
