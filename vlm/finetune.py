"""LoRA fine-tune of the Qwen3-VL verifier (binary Yes/No logit objective).

Train: UCF train windows (teacher labels, negatives subsampled) + Kaggle train/val clips.
Test : UCF test windows (real CCTV) and Kaggle test clips. Zero-shot is scored first at the
same settings so the comparison is like for like.

Usage: python -m vlm.finetune [--epochs 2] [--size 384x288]
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
import torch
from sklearn.metrics import average_precision_score, roc_auc_score

from eval.cross_dataset import bootstrap_ci
from vlm.fast_probe import kaggle_split, kaggle_windows, teacher_labels
from vlm.pseudo_label import train_windows
from vlm.zero_shot import PROMPTS, load_model, read_frames, ucf_samples

OUT = Path("artifacts/results/vlm_finetune.json")


def inputs_for(processor, frames, prompt, device):
    content = [{"type": "image"} for _ in frames] + [{"type": "text", "text": prompt.format(n=len(frames))}]
    text = processor.apply_chat_template([{"role": "user", "content": content}], add_generation_prompt=True, tokenize=False)
    return processor(text=[text], images=frames, return_tensors="pt").to(device)


def margin(model, inp, yes, no):
    lg = model(**inp, logits_to_keep=1).logits[0, -1].float()
    lp = torch.log_softmax(lg, -1)
    return torch.logsumexp(lp[yes], 0) - torch.logsumexp(lp[no], 0)


@torch.no_grad()
def evaluate(model, processor, yes, no, data, name):
    model.eval()
    sc = []
    for fr, _, _ in data:
        with torch.autocast("cuda", dtype=torch.float16):
            sc.append(margin(model, inputs_for(processor, fr, PROMPTS["stages"], "cuda"), yes, no).item())
    sc = np.array(sc)
    y = np.array([d[1] for d in data])
    clips = np.array([d[2] for d in data])
    r = {"n": len(y), "pos": int(y.sum()), "roc_auc": float(roc_auc_score(y, sc)), "pr_auc": float(average_precision_score(y, sc))}
    r.update(bootstrap_ci(y, sc, clips))
    pred = sc > 0
    r.update(recall_at_0=float(pred[y == 1].mean()), false_alarm_at_0=float(pred[y == 0].mean()),
             precision_at_0=float((y[pred] == 1).mean()) if pred.any() else None)
    print(f"{name}: ROC {r['roc_auc']:.3f} {np.round(r['roc_ci'], 3).tolist()} PR {r['pr_auc']:.3f} "
          f"recall@0 {r['recall_at_0']:.2f} FA@0 {r['false_alarm_at_0']:.2f}", flush=True)
    return r


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--epochs", type=int, default=2)
    ap.add_argument("--size", default="384x288")
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--accum", type=int, default=8)
    ap.add_argument("--model", default="Qwen/Qwen3-VL-2B-Instruct")
    a = ap.parse_args()
    size = tuple(int(v) for v in a.size.split("x"))
    rng = np.random.RandomState(0)

    # ---- data (frames preloaded to RAM) ----
    ucf_root = Path("data/external/ucf_shoplifting")
    tw = train_windows(ucf_root, Path("data/Anomaly_Train.txt"), 20)
    yt = teacher_labels([{"clip": w["clip"], "start": w["start"]} for w in tw], 0.5, 0.15)
    pos = [i for i in range(len(tw)) if yt[i] == 1]
    neg = [i for i in range(len(tw)) if yt[i] == 0]
    neg = list(rng.choice(neg, min(len(neg), 2 * len(pos)), replace=False))
    sp = kaggle_split()

    def build(samples):
        out = []
        for s in samples:
            fr = read_frames(s["path"], s["start"], s["end"], 8, size)
            if len(fr) >= 4:
                out.append((fr, s["label"], s["clip"]))
        return out

    t0 = time.time()
    train = build([{**tw[i], "label": int(yt[i])} for i in pos + neg])
    n_ucf = len(train)
    kag = [s for s in kaggle_windows() if s["start"] == 0]  # first half of each clip
    train += build([s for s in kag if sp.get(s["clip"], "train") != "test"])
    print(f"train: {n_ucf} UCF + {len(train) - n_ucf} Kaggle windows ({time.time() - t0:.0f}s)", flush=True)
    test_ucf = build(ucf_samples(ucf_root, Path("data/Temporal_Anomaly_Annotation_for_Testing_Videos.txt")))
    test_kag = build([s for s in kag if sp.get(s["clip"], "train") == "test"])

    # ---- model ----
    processor, model, yes, no = load_model(a.model, "fp16")
    res = {"model": a.model, "size": a.size, "epochs": a.epochs, "lr": a.lr, "n_train": len(train)}
    res["zero_shot_ucf"] = evaluate(model, processor, yes, no, test_ucf, "zero-shot  UCF test")
    res["zero_shot_kaggle"] = evaluate(model, processor, yes, no, test_kag, "zero-shot  Kaggle test")

    from peft import LoraConfig, get_peft_model

    model.gradient_checkpointing_enable()
    model.enable_input_require_grads()
    cfg = LoraConfig(r=16, lora_alpha=32, lora_dropout=0.05,
                     target_modules=r".*language_model.*\.(q_proj|k_proj|v_proj|o_proj|gate_proj|up_proj|down_proj)")
    model = get_peft_model(model, cfg)
    for p in model.parameters():
        if p.requires_grad:
            p.data = p.data.float()
    model.print_trainable_parameters()
    params = [p for p in model.parameters() if p.requires_grad]
    opt = torch.optim.AdamW(params, lr=a.lr, weight_decay=0.01)
    scaler = torch.amp.GradScaler("cuda")
    n_pos = sum(d[1] for d in train)
    pw = torch.tensor(max(1.0, (len(train) - n_pos) / max(1, n_pos)) ** 0.5, device="cuda")
    total = a.epochs * len(train)
    step, t0, run = 0, time.time(), []
    for ep in range(a.epochs):
        model.train()
        for i in rng.permutation(len(train)):
            fr, y, _ = train[i]
            with torch.autocast("cuda", dtype=torch.float16):
                m = margin(model, inputs_for(processor, fr, PROMPTS["stages"], "cuda"), yes, no)
            loss = torch.nn.functional.binary_cross_entropy_with_logits(
                m.clamp(-12, 12), torch.tensor(float(y), device="cuda"), pos_weight=pw)
            scaler.scale(loss / a.accum).backward()
            run.append(loss.item())
            step += 1
            if step % a.accum == 0:
                scaler.unscale_(opt)
                torch.nn.utils.clip_grad_norm_(params, 1.0)
                scaler.step(opt)
                scaler.update()
                opt.zero_grad(set_to_none=True)
            if step % 40 == 0:
                print(f"  ep{ep} {step}/{total} loss {np.mean(run[-40:]):.3f} {(time.time() - t0) / step:.1f}s/step", flush=True)
        res[f"ucf_after_ep{ep + 1}"] = evaluate(model, processor, yes, no, test_ucf, f"fine-tuned ep{ep + 1} UCF test")
        res[f"kaggle_after_ep{ep + 1}"] = evaluate(model, processor, yes, no, test_kag, f"fine-tuned ep{ep + 1} Kaggle test")
        OUT.write_text(json.dumps(res, indent=2), encoding="utf-8")
    model.save_pretrained("artifacts/models/vlm_lora")


if __name__ == "__main__":
    main()
