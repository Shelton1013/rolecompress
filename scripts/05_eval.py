# -*- coding: utf-8 -*-
"""Stage 5: efficiency-accuracy evaluation + synergy-subset crossover.

Evaluates a chosen allocation POLICY on a long-video QA benchmark at matched budgets:
  --policy rolecompress|uniform|remo|query|tokenmerge|random_role|oracle_role

Outputs a jsonl of per-item results and a summary json (accuracy, mean visual tokens,
FLOPs proxy, synergy-subset accuracy, role fractions). Sweep --n_high to trace the Pareto.

The synergy subset is precomputed by --tag_synergy (needs gold; uses the 3 frozen passes).
"""
import argparse
import json
import os
import random

import torch

from rolecompress import segment as seg_mod
from rolecompress import asr as asr_mod
from rolecompress import baselines as bl
from rolecompress.backbone import BackboneConfig, RoleCompressBackbone
from rolecompress.data import read_jsonl, write_jsonl
from rolecompress.metrics import EvalAccumulator, mcq_correct, is_synergy_required
from rolecompress.roles import Role, RoleBudget
from rolecompress.router import RoleRouter, RouterConfig

# Policy families:
#   ROLE_POLICIES     -> produce a role per segment, then allocate_frames by budget.
#   KEEP_POLICIES     -> produce explicit kept frames directly (query/saliency).
#   tokenmerge        -> uniform frames + spatial downscale (fewer visual tokens/frame).
ROLE_POLICIES = {"rolecompress", "uniform", "remo", "random_role", "oracle_role"}
KEEP_POLICIES = {"query", "saliency"}


def load_router(path, device):
    ckpt = torch.load(path, map_location=device)
    model = RoleRouter(RouterConfig(**ckpt["cfg"])).to(device).eval()
    model.load_state_dict(ckpt["state_dict"])
    return model


def _cached_asr(vid, path, cache_dir):
    """Transcribe once, cache to disk, reuse across policies/budgets (ASR on long videos is slow)."""
    if not cache_dir:
        return asr_mod.transcribe(path)
    os.makedirs(cache_dir, exist_ok=True)
    cp = os.path.join(cache_dir, str(vid).replace("/", "_").replace("\\", "_") + ".json")
    if os.path.exists(cp):
        try:
            return json.load(open(cp, encoding="utf-8"))
        except Exception:
            pass
    utt = asr_mod.transcribe(path)
    try:
        json.dump(utt, open(cp, "w", encoding="utf-8"), ensure_ascii=False)
    except Exception:
        pass
    return utt


def remo_roles(backbone, frames_per_seg, seg_asr):
    """ReMo-style baseline: a segment is REDUNDANT if its visual mean feature is highly
    similar to something already covered (here: to its own ASR text embedding, cross-modal),
    else UNIQUE_VISUAL. No synergy branch (that's the point of the baseline)."""
    import torch.nn.functional as F
    vis, txt, _ = backbone.pooled_segment_features(frames_per_seg, seg_asr)
    roles = []
    for i in range(vis.shape[0]):
        if seg_asr[i].strip():
            sim = F.cosine_similarity(vis[i:i+1].float(), txt[i:i+1].float()).item()
            roles.append(Role.REDUNDANT if sim > 0.25 else Role.UNIQUE_VISUAL)
        else:
            roles.append(Role.UNIQUE_VISUAL)
    return roles


def policy_roles(policy, backbone, router, device, segs, frames_per_seg, seg_asr,
                 oracle_margins=None, tau=(0.5, 0.0)):
    n = len(segs)
    if policy == "uniform":
        return [Role.UNIQUE_VISUAL] * n            # every segment same budget = uniform
    if policy == "random_role":
        return [random.choice(list(Role)) for _ in range(n)]
    if policy == "remo":
        return remo_roles(backbone, frames_per_seg, seg_asr)
    if policy == "oracle_role" and oracle_margins is not None:
        from rolecompress.roles import assign_role_from_margins
        return [assign_role_from_margins(*oracle_margins.get(i, (0, 0, 0)), tau[0], tau[1]) for i in range(n)]
    # rolecompress (default)
    with torch.no_grad():
        vis, txt, scal = backbone.pooled_segment_features(frames_per_seg, seg_asr)
        logits = router(vis.unsqueeze(0).to(device), txt.unsqueeze(0).to(device),
                        scal.unsqueeze(0).to(device), torch.ones(1, n, dtype=torch.bool, device=device))
        return [Role(int(x)) for x in logits.argmax(-1)[0].tolist()]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--qa_eval", required=True)
    ap.add_argument("--manifest", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--policy", default="rolecompress",
                    choices=["rolecompress", "rolecompress_tf", "uniform", "remo", "random_role",
                             "oracle_role", "query", "saliency", "tokenmerge"])
    ap.add_argument("--keep_total", type=int, default=8,
                    help="total kept frames for query/saliency/tokenmerge (budget knob; sweep for Pareto)")
    ap.add_argument("--downscale", type=float, default=2.0, help="tokenmerge spatial downscale factor")
    ap.add_argument("--router_ckpt", default=None)
    ap.add_argument("--lora", default=None)
    ap.add_argument("--model_id", default="Qwen/Qwen2.5-VL-7B-Instruct")
    ap.add_argument("--model_path", default=None, help="local dir to a downloaded model; overrides --model_id")
    ap.add_argument("--n_low", type=int, default=1)
    ap.add_argument("--n_high", type=int, default=4)
    ap.add_argument("--uniform_frames", type=int, default=2, help="frames/seg for uniform policy at this budget")
    ap.add_argument("--win", type=float, default=6.0)
    ap.add_argument("--fps", type=float, default=1.0)
    ap.add_argument("--tag_synergy", action="store_true",
                    help="also run 3 frozen passes per item to mark the synergy subset")
    ap.add_argument("--tau_hi", type=float, default=0.5)
    ap.add_argument("--tau_lo", type=float, default=0.0)
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--asr_cache", default=None,
                    help="dir to cache ASR per video (reused across policies/budgets). "
                         "Default: <out_dir>/asr_cache. Set '' to disable.")
    ap.add_argument("--no_asr", action="store_true", help="skip ASR (fast; tests video-only path)")
    args = ap.parse_args()
    if args.asr_cache is None:
        args.asr_cache = os.path.join(os.path.dirname(args.out) or ".", "asr_cache")

    device = "cuda" if torch.cuda.is_available() else "cpu"
    manifest = {r["video_id"]: r["path"] for r in read_jsonl(args.manifest)}
    backbone = RoleCompressBackbone(BackboneConfig(model_id=(args.model_path or args.model_id)), lora_adapter_path=args.lora)
    router = load_router(args.router_ckpt, device) if (args.policy == "rolecompress" and args.router_ckpt) else None

    budget = RoleBudget(n_unique_visual=args.n_low, n_synergistic=args.n_high)
    if args.policy == "uniform":
        budget = RoleBudget(n_redundant=args.uniform_frames, n_unique_text=args.uniform_frames,
                            n_unique_visual=args.uniform_frames, n_synergistic=args.uniform_frames)

    acc = EvalAccumulator()
    per_item = []
    rows = list(read_jsonl(args.qa_eval))
    if args.limit:
        rows = rows[:args.limit]

    for r in rows:
        vid = r.get("video_id") or r.get("video")
        path = manifest.get(vid) or r.get("path")
        try:
            segs, frames_per_seg, dur = seg_mod.segment_video(path, win=args.win, fps=args.fps)
            if args.no_asr:
                seg_asr = [""] * len(segs)
            elif r.get("seg_asr"):
                seg_asr = r["seg_asr"]
            else:
                seg_asr = asr_mod.align_to_segments(_cached_asr(vid, path, args.asr_cache), segs)
        except Exception as e:
            print(f"[skip] {vid}: {e}"); continue

        in_syn = bool(r.get("synergy_required", False))
        oracle_margins = None
        if args.tag_synergy or args.policy == "oracle_role":
            # 3 frozen passes at the QA level (whole-video probe)
            from rolecompress.pid_labels import SegmentProbe
            probe = SegmentProbe(vid, -1, 0, dur, r["question"], r["answer"], r.get("choices"),
                                 " ".join(seg_asr))
            allframes = [f for fs in frames_per_seg for f in fs][:32]
            m = backbone.score_probe(probe, allframes)
            in_syn = is_synergy_required(m.m_text, m.m_vision, m.m_joint, args.tau_hi, args.tau_lo)

        # dispatch by policy family
        roles_for_log = None
        if args.policy == "rolecompress_tf":
            # TRAINING-FREE ablation: roles from per-segment answer-confidence probes (no router,
            # no LoRA). Detects synergy via confidence divergence but is query-dependent and costs
            # 3 forward passes per segment -> motivates the learned query-agnostic router.
            from rolecompress.roles import assign_role_from_margins
            if not r.get("choices"):
                print(f"[skip] {vid}: rolecompress_tf needs MCQ choices"); continue
            roles = []
            for frames, asr in zip(frames_per_seg, seg_asr):
                m = backbone.segment_confidence_margins(r["question"], r["choices"], frames, asr)
                roles.append(assign_role_from_margins(m.m_text, m.m_vision, m.m_joint, args.tau_hi, args.tau_lo))
            roles_for_log = roles
            inputs, vtok, _ = backbone.build_answer_inputs(r["question"], r.get("choices"),
                                                           frames_per_seg, roles, seg_asr, budget)
        elif args.policy in ROLE_POLICIES:
            roles = policy_roles(args.policy, backbone, router, device, segs, frames_per_seg, seg_asr,
                                 oracle_margins, (args.tau_hi, args.tau_lo))
            roles_for_log = roles
            inputs, vtok, _ = backbone.build_answer_inputs(r["question"], r.get("choices"),
                                                           frames_per_seg, roles, seg_asr, budget)
        elif args.policy in KEEP_POLICIES:
            if args.policy == "query":
                keep = bl.query_frame_keep(backbone, r["question"], frames_per_seg, args.keep_total)
            else:  # saliency (~FastV input-side)
                keep = bl.saliency_frame_keep(frames_per_seg, args.keep_total)
            inputs, vtok, _ = backbone.build_answer_inputs(r["question"], r.get("choices"),
                                                           frames_per_seg, None, seg_asr, keep_override=keep)
        else:  # tokenmerge: uniform frames + spatial downscale
            per_seg = max(1, args.keep_total // max(1, len(segs)))
            keep = bl.uniform_frame_keep(frames_per_seg, per_seg)
            inputs, vtok, _ = backbone.build_answer_inputs(r["question"], r.get("choices"),
                                                           frames_per_seg, None, seg_asr,
                                                           keep_override=keep, downscale=args.downscale)
        pred = backbone.generate_answer(inputs, max_new_tokens=16 if r.get("choices") else 64)
        correct = mcq_correct(pred, r["answer"], r.get("choices")) if r.get("choices") else (r["answer"].lower() in pred.lower())
        acc.add(correct, vtok, in_synergy_subset=in_syn, roles=roles_for_log)
        per_item.append({"video_id": vid, "pred": pred, "gold": r["answer"], "correct": correct,
                         "visual_tokens": vtok, "synergy_required": in_syn,
                         "roles": [rr.short for rr in roles_for_log] if roles_for_log else None})

    summ = acc.summary()
    summ.update({"policy": args.policy, "n_low": args.n_low, "n_high": args.n_high})
    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    write_jsonl(args.out, per_item)
    json.dump(summ, open(args.out + ".summary.json", "w"), indent=2)
    print(json.dumps(summ, indent=2))


if __name__ == "__main__":
    main()
