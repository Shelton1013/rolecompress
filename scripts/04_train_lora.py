# -*- coding: utf-8 -*-
"""Stage 4: LoRA adaptation of the frozen backbone to the role-allocated token stream.

For each training QA, we:
  1. segment the video + get per-segment ASR,
  2. predict roles with the trained RoleRouter (query-agnostic; one pass per video, cached),
  3. build the role-allocated multimodal prompt (backbone.build_answer_inputs),
  4. compute the QA loss (teacher-forced gold answer) and update ONLY the LoRA params.

Backbone weights are frozen; only LoRA (attn/mlp proj) trains. Multi-GPU via accelerate/
torchrun + gradient checkpointing. This is the only stage that needs real GPU memory,
and rank-16 LoRA on 7B fits one A6000; we shard data across the 8.

NOTE: this is a training *scaffold*. The exact target-module names for LoRA and the loss
masking of the answer span are marked [VERIFY] for the chosen backbone/transformers.
"""
import argparse
import json
import os

import torch
from torch.utils.data import DataLoader, Dataset

from rolecompress import segment as seg_mod
from rolecompress import asr as asr_mod
from rolecompress.backbone import BackboneConfig, RoleCompressBackbone
from rolecompress.data import read_jsonl
from rolecompress.roles import Role, RoleBudget
from rolecompress.router import RoleRouter, RouterConfig


class QADataset(Dataset):
    def __init__(self, path):
        self.rows = list(read_jsonl(path))

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, i):
        return self.rows[i]


def load_router(path, device):
    ckpt = torch.load(path, map_location=device)
    cfg = RouterConfig(**ckpt["cfg"])
    model = RoleRouter(cfg).to(device).eval()
    model.load_state_dict(ckpt["state_dict"])
    return model


def add_lora(model):
    from peft import LoraConfig, get_peft_model
    # [VERIFY] LLM-decoder proj names (same for Qwen3-VL / Qwen2.5-VL dense; MoE uses expert names)
    target_modules = ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]
    lcfg = LoraConfig(r=16, lora_alpha=32, lora_dropout=0.05, bias="none",
                      target_modules=target_modules, task_type="CAUSAL_LM")
    return get_peft_model(model, lcfg)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--qa_train", required=True)
    ap.add_argument("--manifest", required=True)
    ap.add_argument("--router_ckpt", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--model_id", default="Qwen/Qwen3-VL-8B-Instruct")
    ap.add_argument("--model_path", default=None, help="local dir to a downloaded model; overrides --model_id")
    ap.add_argument("--epochs", type=int, default=2)
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--grad_accum", type=int, default=8)
    ap.add_argument("--n_low", type=int, default=1)
    ap.add_argument("--n_high", type=int, default=4)
    ap.add_argument("--win", type=float, default=6.0)
    ap.add_argument("--fps", type=float, default=1.0)
    args = ap.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    manifest = {r["video_id"]: r["path"] for r in read_jsonl(args.manifest)}
    budget = RoleBudget(n_unique_visual=args.n_low, n_synergistic=args.n_high)

    backbone = RoleCompressBackbone(BackboneConfig(model_id=(args.model_path or args.model_id)))
    backbone.model = add_lora(backbone.model)
    backbone.model.train()
    backbone.model.enable_input_require_grads()
    try:
        backbone.model.gradient_checkpointing_enable()
    except Exception:
        pass
    router = load_router(args.router_ckpt, device)

    ds = QADataset(args.qa_train)
    dl = DataLoader(ds, batch_size=1, shuffle=True, num_workers=2, collate_fn=lambda x: x[0])
    opt = torch.optim.AdamW([p for p in backbone.model.parameters() if p.requires_grad], lr=args.lr)

    role_cache = {}

    def roles_for_video(vid, segs, frames_per_seg, seg_asr):
        if vid in role_cache:
            return role_cache[vid]
        with torch.no_grad():
            vis, txt, scal = backbone.pooled_segment_features(frames_per_seg, seg_asr)
            logits = router(vis.unsqueeze(0).to(device), txt.unsqueeze(0).to(device),
                            scal.unsqueeze(0).to(device), torch.ones(1, len(segs), dtype=torch.bool, device=device))
            roles = [Role(int(x)) for x in logits.argmax(-1)[0].tolist()]
        role_cache[vid] = roles
        return roles

    step = 0
    for ep in range(args.epochs):
        for row in dl:
            vid = row["video_id"] if "video_id" in row else row.get("video")
            path = manifest.get(vid) or row.get("path")
            try:
                segs, frames_per_seg, dur = seg_mod.segment_video(path, win=args.win, fps=args.fps)
                utt = asr_mod.transcribe(path) if not row.get("asr_cached") else []
                seg_asr = asr_mod.align_to_segments(utt, segs) if utt else row.get("seg_asr", [""] * len(segs))
                roles = roles_for_video(vid, segs, frames_per_seg, seg_asr)
                inputs, vtok, _ = backbone.build_answer_inputs(
                    row["question"], row.get("choices"), frames_per_seg, roles, seg_asr, budget)
            except Exception as e:
                print(f"[skip] {vid}: {e}"); continue

            # teacher-forced answer loss  [VERIFY] label masking to answer span only
            gold = row["answer"]
            gold_ids = backbone.tokenizer(gold, add_special_tokens=False, return_tensors="pt").input_ids.to(device)
            full = torch.cat([inputs["input_ids"], gold_ids], dim=1)
            labels = full.clone()
            labels[:, :inputs["input_ids"].shape[1]] = -100
            model_kwargs = {k: v for k, v in inputs.items() if k not in ("input_ids", "attention_mask")}
            out = backbone.model(input_ids=full, attention_mask=torch.ones_like(full), labels=labels, **model_kwargs)
            loss = out.loss / args.grad_accum
            loss.backward()
            step += 1
            if step % args.grad_accum == 0:
                torch.nn.utils.clip_grad_norm_([p for p in backbone.model.parameters() if p.requires_grad], 1.0)
                opt.step(); opt.zero_grad()
            if step % 50 == 0:
                print(f"ep{ep} step{step} loss {out.loss.item():.4f} vtok {vtok}")

    os.makedirs(args.out, exist_ok=True)
    backbone.model.save_pretrained(args.out)  # saves LoRA adapter only
    print(f"saved LoRA adapter to {args.out}")


if __name__ == "__main__":
    main()
