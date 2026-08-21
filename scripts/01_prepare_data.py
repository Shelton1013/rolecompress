# -*- coding: utf-8 -*-
"""Stage 1: ASR + segmentation + segment-probe generation + router-feature caching.

Input: a jsonl manifest of videos and (optionally) timestamped captions/QA:
  {"video_id": "...", "path": "/data/videos/x.mp4",
   "captions": [{"start":..,"end":..,"text":..}, ...]  (optional, for probe generation),
   "qa": [{"question":..,"choices":[..],"answer":"A","start":..,"end":..}, ...] (optional) }

Outputs (under --out):
  probes.jsonl        segment probes for PID labeling
  asr/<video_id>.json ASR utterances
  seg_feats/<video_id>.pt  cached router features
  segments/<video_id>.json  segment spans (for reuse in eval)

Run distributed with --shard i/N across GPUs; merge the probes.jsonl afterwards.
"""
import argparse
import json
import os
import random

from rolecompress import asr as asr_mod
from rolecompress import segment as seg_mod
from rolecompress.backbone import BackboneConfig, RoleCompressBackbone
from rolecompress.data import read_jsonl, write_jsonl
import torch


def make_cloze_probe(caption: str, distractor_pool, question: str = "Which best describes what happens in this segment?"):
    """Turn a segment caption (or ASR line) into a 4-way MCQ: gold=this segment, distractors=other segments."""
    gold = caption.strip()
    pool = [d for d in distractor_pool if d.strip() != gold]
    if len(gold) < 8 or len(pool) < 3:
        return None
    distractors = random.sample(pool, 3)
    choices = [gold] + distractors
    random.shuffle(choices)
    letter = chr(ord("A") + choices.index(gold))
    return {"question": question, "choices": choices, "answer": letter}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--manifest", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--model_id", default="Qwen/Qwen2.5-VL-7B-Instruct")
    ap.add_argument("--model_path", default=None, help="local dir to a downloaded model; overrides --model_id")
    ap.add_argument("--win", type=float, default=6.0)
    ap.add_argument("--fps", type=float, default=1.0)
    ap.add_argument("--max_per_seg", type=int, default=8)
    ap.add_argument("--asr_size", default="large-v3")
    ap.add_argument("--make_feats", action="store_true", help="also cache router features (loads the backbone)")
    ap.add_argument("--shard", default="0/1", help="i/N for distributed sharding")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()
    random.seed(args.seed)

    i, N = map(int, args.shard.split("/"))
    os.makedirs(args.out, exist_ok=True)
    for sub in ("asr", "seg_feats", "segments"):
        os.makedirs(os.path.join(args.out, sub), exist_ok=True)

    rows = [r for k, r in enumerate(read_jsonl(args.manifest)) if k % N == i]
    backbone = None
    if args.make_feats:
        backbone = RoleCompressBackbone(BackboneConfig(model_id=(args.model_path or args.model_id)))

    probe_rows = []
    for r in rows:
        vid, path = r["video_id"], r["path"]
        try:
            segs, frames_per_seg, duration = seg_mod.segment_video(path, win=args.win, fps=args.fps, max_per_seg=args.max_per_seg)
        except Exception as e:
            print(f"[skip] {vid}: segmentation failed: {e}")
            continue
        # ASR
        try:
            utt = asr_mod.transcribe(path, size=args.asr_size)
        except Exception as e:
            print(f"[warn] {vid}: ASR failed ({e}); empty transcript")
            utt = []
        json.dump(utt, open(os.path.join(args.out, "asr", f"{vid}.json"), "w", encoding="utf-8"), ensure_ascii=False)
        seg_asr = asr_mod.align_to_segments(utt, segs)
        json.dump([{"index": s.index, "start": s.start, "end": s.end} for s in segs],
                  open(os.path.join(args.out, "segments", f"{vid}.json"), "w", encoding="utf-8"))

        # probes from captions (or fall back to the video's own qa list)
        caps = r.get("captions")
        if caps:
            pool = [c["text"].strip() for c in caps if c.get("text")]
            for s, c in zip(segs, caps):
                mcq = make_cloze_probe(c.get("text", ""), pool)
                if mcq:
                    probe_rows.append({"video_id": vid, "seg_index": s.index, "seg_start": s.start, "seg_end": s.end,
                                       "question": mcq["question"], "gold": mcq["answer"], "choices": mcq["choices"],
                                       "asr_text": seg_asr[s.index]})
        elif r.get("qa"):
            for qa in r["qa"]:
                # attach to the segment containing the qa time (or all segments if untimed)
                s_idx = next((s.index for s in segs if s.start <= qa.get("start", 0) < s.end), 0)
                probe_rows.append({"video_id": vid, "seg_index": s_idx, "seg_start": segs[s_idx].start,
                                   "seg_end": segs[s_idx].end, "question": qa["question"],
                                   "gold": qa["answer"], "choices": qa.get("choices"), "asr_text": seg_asr[s_idx]})
        else:
            # fallback: build probes from ASR speech (any video with speech works -> good for small
            # validation runs with no caption/qa annotation). Gold = this segment's line; distractors
            # = other segments' lines. Needs >=4 distinct speech segments.
            pool = [a.strip() for a in seg_asr if len(a.strip()) >= 8]
            if len(pool) >= 4:
                for s in segs:
                    line = seg_asr[s.index].strip()
                    mcq = make_cloze_probe(line, pool, question="Which line is spoken in this segment?")
                    if mcq:
                        probe_rows.append({"video_id": vid, "seg_index": s.index, "seg_start": s.start, "seg_end": s.end,
                                           "question": mcq["question"], "gold": mcq["answer"], "choices": mcq["choices"],
                                           "asr_text": seg_asr[s.index]})

        # cache router features
        if backbone is not None:
            vis, txt, scal = backbone.pooled_segment_features(frames_per_seg, seg_asr)
            torch.save({"vis": vis.cpu(), "txt": txt.cpu(), "scal": scal.cpu()},
                       os.path.join(args.out, "seg_feats", f"{vid}.pt"))
        print(f"[ok] {vid}: {len(segs)} segs, {len(utt)} utts")

    write_jsonl(os.path.join(args.out, f"probes.shard{i}of{N}.jsonl"), probe_rows)
    print(f"wrote {len(probe_rows)} probes for shard {i}/{N}")


if __name__ == "__main__":
    main()
