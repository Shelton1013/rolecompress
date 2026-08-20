# -*- coding: utf-8 -*-
"""End-to-end smoke test on ONE video and ONE GPU. Run this BEFORE any large-scale stage —
it exercises every backbone integration point (the [VERIFY] lines) in ~1-2 min so you catch
processor/API mismatches on a single example instead of a 1000-video run.

Usage:
  python scripts/smoke_test.py --video /path/to/one_video.mp4 \
      --model_id Qwen/Qwen3-VL-4B-Instruct   # 4B loads fast; use 8B if you prefer
Optionally --question "..." for an open-ended check.

It prints, for each step, what it validated; any step that throws tells you exactly which
integration point to fix. Prints SMOKE TEST PASSED at the end if all steps succeed.
"""
import argparse
import traceback

import numpy as np


def step(name):
    print(f"\n=== {name} ===", flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--video", required=True)
    ap.add_argument("--model_id", default="Qwen/Qwen2.5-VL-7B-Instruct")
    ap.add_argument("--model_path", default=None, help="local dir to a downloaded model; overrides --model_id")
    ap.add_argument("--question", default=None)
    ap.add_argument("--win", type=float, default=6.0)
    ap.add_argument("--fps", type=float, default=1.0)
    ap.add_argument("--no_asr", action="store_true", help="skip ASR (faster; tests video-only path)")
    args = ap.parse_args()

    from rolecompress import segment as seg_mod
    from rolecompress import asr as asr_mod
    from rolecompress import baselines as bl
    from rolecompress.backbone import BackboneConfig, RoleCompressBackbone
    from rolecompress.pid_labels import SegmentProbe, margins_to_label
    from rolecompress.roles import Role, RoleBudget, assign_role_from_margins

    step("1. load backbone")
    backbone = RoleCompressBackbone(BackboneConfig(model_id=(args.model_path or args.model_id)))
    print(f"  family={backbone.family}  llm_hidden={backbone.llm_hidden}  device={backbone.device}")

    step("2. segment video + sample frames")
    segs, frames_per_seg, dur = seg_mod.segment_video(args.video, win=args.win, fps=args.fps)
    print(f"  duration={dur:.1f}s  segments={len(segs)}  frames/seg~{np.mean([len(f) for f in frames_per_seg]):.1f}")
    assert len(segs) > 0 and any(len(f) for f in frames_per_seg), "no frames sampled"

    step("3. ASR + align to segments")
    if args.no_asr:
        seg_asr = [""] * len(segs); print("  (skipped)")
    else:
        utt = asr_mod.transcribe(args.video)
        seg_asr = asr_mod.align_to_segments(utt, segs)
        n_speech = sum(1 for a in seg_asr if a.strip())
        print(f"  utterances={len(utt)}  segments_with_speech={n_speech}")

    step("4. router features (pooled, LLM space)")
    vis, txt, scal = backbone.pooled_segment_features(frames_per_seg[:4], seg_asr[:4])
    print(f"  vis{tuple(vis.shape)}  txt{tuple(txt.shape)}  scal{tuple(scal.shape)}  (d must == llm_hidden={backbone.llm_hidden})")
    assert vis.shape[-1] == backbone.llm_hidden, "feature dim != llm_hidden"

    step("5. score_probe: text / vision / joint margins (self-sup label signal)")
    s0 = segs[0]
    choices = ["something happens", "nothing happens", "a person appears", "a landscape"]
    probe = SegmentProbe(video_id="smoke", seg_index=0, seg_start=s0.start, seg_end=s0.end,
                         question="Which best describes this segment?", gold="A", choices=choices,
                         asr_text=seg_asr[0])
    m = backbone.score_probe(probe, frames_per_seg[0])
    print(f"  m_text={m.m_text:.3f}  m_vision={m.m_vision:.3f}  m_joint={m.m_joint:.3f}")
    role0 = assign_role_from_margins(m.m_text, m.m_vision, m.m_joint)
    lbl = margins_to_label(probe, m, tau_hi=0.5, tau_lo=0.0, temperature=1.0, keep_threshold=-10)
    print(f"  -> role={role0.name}  soft={[round(x,2) for x in lbl.soft_role]}")

    step("6. role-allocated QA build + generate (frame-budget path)")
    # exercise all role branches: speech->REDUNDANT, else UNIQUE_VISUAL, first->SYNERGISTIC
    roles = []
    for i, a in enumerate(seg_asr):
        roles.append(Role.SYNERGISTIC if i == 0 else (Role.REDUNDANT if a.strip() else Role.UNIQUE_VISUAL))
    budget = RoleBudget(n_unique_visual=1, n_synergistic=4)
    q = args.question or "What is happening in this video?"
    ch = None if args.question else choices
    inputs, vtok, _ = backbone.build_answer_inputs(q, ch, frames_per_seg, roles, seg_asr, budget)
    print(f"  visual_tokens={vtok}  input_ids={tuple(inputs['input_ids'].shape)}")
    out = backbone.generate_answer(inputs, max_new_tokens=32)
    print(f"  generated: {out!r}")

    step("7. baseline path (keep_override) — query/saliency share this code path")
    keep = bl.saliency_frame_keep(frames_per_seg, keep_total=6)
    inputs2, vtok2, _ = backbone.build_answer_inputs(q, ch, frames_per_seg, None, seg_asr, keep_override=keep)
    print(f"  saliency baseline visual_tokens={vtok2}")

    print("\nSMOKE TEST PASSED ✅  (backbone integration OK; safe to run the full pipeline)")


if __name__ == "__main__":
    try:
        main()
    except Exception:
        print("\nSMOKE TEST FAILED ❌  — the traceback below points at the integration step to fix "
              "(check the [VERIFY] markers in rolecompress/backbone.py):\n")
        traceback.print_exc()
        raise SystemExit(1)
