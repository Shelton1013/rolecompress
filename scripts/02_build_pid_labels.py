# -*- coding: utf-8 -*-
"""Stage 2: self-supervised role labels via single- vs joint-modality head margins.

Reads probes.jsonl, runs the 3 frozen passes per probe (text/vision/joint), computes
margins, calibrates thresholds on a dev split, and writes labels.jsonl.

Distributed: --shard i/N; each GPU scores a slice. Merge labels afterwards, then run
--calibrate_only once on the merged margins to fix tau_hi/tau_lo and re-emit soft labels.
"""
import argparse
import json
import os
import random

import numpy as np

from rolecompress import segment as seg_mod
from rolecompress.backbone import BackboneConfig, RoleCompressBackbone
from rolecompress.data import read_jsonl, write_jsonl, label_to_row
from rolecompress.pid_labels import (ProbeMargins, SegmentProbe, build_labels,
                                     calibrate_thresholds, margins_to_label)


def load_probe_objs(path):
    for row in read_jsonl(path):
        yield SegmentProbe(**{k: v for k, v in row.items() if k in SegmentProbe.__annotations__}), row


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--probes", required=True)
    ap.add_argument("--data_root", required=True, help="dir with segments/ and video paths manifest")
    ap.add_argument("--manifest", required=True, help="video_id -> path")
    ap.add_argument("--out", required=True)
    ap.add_argument("--model_id", default="Qwen/Qwen3-VL-8B-Instruct")
    ap.add_argument("--model_path", default=None, help="local dir to a downloaded model; overrides --model_id")
    ap.add_argument("--shard", default="0/1")
    ap.add_argument("--fps", type=float, default=1.0)
    ap.add_argument("--max_per_seg", type=int, default=8)
    ap.add_argument("--tau_hi", type=float, default=None, help="if None, calibrate from data")
    ap.add_argument("--tau_lo", type=float, default=None)
    ap.add_argument("--temperature", type=float, default=1.0)
    ap.add_argument("--calibrate_only", action="store_true",
                    help="skip scoring; read a merged *.margins.jsonl and (re)emit labels")
    args = ap.parse_args()

    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    i, N = map(int, args.shard.split("/"))

    if args.calibrate_only:
        rows = list(read_jsonl(args.probes))  # here --probes points at merged margins
        margins = [ProbeMargins(r["m_text"], r["m_vision"], r["m_joint"]) for r in rows]
        thr = ({"tau_hi": args.tau_hi, "tau_lo": args.tau_lo}
               if args.tau_hi is not None else calibrate_thresholds(margins))
        print("thresholds:", thr)
        out = []
        for r, m in zip(rows, margins):
            probe = SegmentProbe(video_id=r["video_id"], seg_index=r["seg_index"], seg_start=r.get("seg_start", 0),
                                 seg_end=r.get("seg_end", 0), question=r.get("question", ""), gold=r.get("gold", ""),
                                 choices=r.get("choices"), asr_text=r.get("asr_text", ""))
            lbl = margins_to_label(probe, m, thr["tau_hi"], thr["tau_lo"], args.temperature, thr["tau_lo"])
            out.append(label_to_row(lbl))
        write_jsonl(args.out, out)
        json.dump(thr, open(args.out + ".thresholds.json", "w"))
        print(f"wrote {len(out)} labels")
        return

    # scoring path
    manifest = {r["video_id"]: r["path"] for r in read_jsonl(args.manifest)}
    backbone = RoleCompressBackbone(BackboneConfig(model_id=(args.model_path or args.model_id)))
    readers = {}

    def frames_for(video_id, start, end):
        if video_id not in readers:
            readers[video_id] = seg_mod.VideoReader(manifest[video_id])
        vr = readers[video_id]
        n = max(1, min(args.max_per_seg, int(round((end - start) * args.fps))))
        step = (end - start) / n
        times = [start + step * (k + 0.5) for k in range(n)]
        return vr.frames_at(times)

    margin_rows = []
    all_rows = [(p, r) for k, (p, r) in enumerate(load_probe_objs(args.probes)) if k % N == i]
    for k, (probe, raw) in enumerate(all_rows):
        try:
            frames = frames_for(probe.video_id, probe.seg_start, probe.seg_end)
            m = backbone.score_probe(probe, frames)
        except Exception as e:
            print(f"[skip] {probe.video_id}#{probe.seg_index}: {e}")
            continue
        margin_rows.append({"video_id": probe.video_id, "seg_index": probe.seg_index,
                            "seg_start": probe.seg_start, "seg_end": probe.seg_end,
                            "question": probe.question, "gold": probe.gold, "choices": probe.choices,
                            "asr_text": probe.asr_text,
                            "m_text": m.m_text, "m_vision": m.m_vision, "m_joint": m.m_joint})
        if (k + 1) % 200 == 0:
            print(f"scored {k+1}/{len(all_rows)}")

    out_margins = args.out.replace(".jsonl", "") + f".margins.shard{i}of{N}.jsonl"
    write_jsonl(out_margins, margin_rows)
    print(f"wrote {len(margin_rows)} margins to {out_margins}. "
          f"Merge shards then run --calibrate_only on the merged file.")


if __name__ == "__main__":
    main()
