# -*- coding: utf-8 -*-
"""Build a training MANIFEST (for 01_prepare_data.py) from LLaVA-Video-178K annotations.

Input: one or more *_qa_processed.json (MC preferred; OE optional) with records shaped:
  {"id": "RW587", "data_source": "...",
   "video": "academic_source/Charades/RW587.mp4",
   "conversations": [{"from":"human","value":"<image>\\nQ?\\nA. ..\\nB. ..\\nPlease.."},
                     {"from":"gpt","value":"D. ..."}, ...]}

Output: manifest.jsonl of {"video_id","path","qa":[{"question","choices","answer"}], "data_source"}
grouped per video (multiple QA turns -> one qa list), matching 01_prepare_data's r["qa"] path.

DATA-LEAKAGE GUARD: rows whose source (2nd component of the `video` path, e.g. "Charades")
is in --exclude_sources are DROPPED, because several LLaVA-Video academic sources feed the
eval benchmarks (MVBench/VideoMME/LongVideoBench draw from NExT-QA/STAR/PerceptionTest/
Ego4D/ActivityNet/...). The full source histogram + kept/dropped counts are printed so the
choice is explicit, never silent.
"""
import argparse
import json
import os
import re
from collections import Counter, defaultdict

# Sources known to feed the eval benchmarks -> exclude from TRAINING by default.
# (MVBench is built from STAR/PerceptionTest/NExTQA/CLEVRER/Ego4D/TVQA/...; VideoMME & LongVideoBench
#  aggregate ActivityNet/YouCook2/etc.) Adjust via --exclude_sources after seeing the histogram.
DEFAULT_EXCLUDE = [
    "NExTQA", "NextQA", "nextqa", "STAR", "PerceptionTest", "Perception_Test", "perception",
    "Ego4D", "ego4d", "ActivityNet", "activitynet", "CLEVRER", "clevrer", "TVQA", "tvqa",
    "YouCook2", "youcook2", "VLNCE", "MovieChat", "moviechat",
]

_OPT = re.compile(r"^\s*([A-H])[\.\)]\s*(.+?)\s*$")
_ANS = re.compile(r"^\s*([A-H])\b")


def parse_mc(human_value, gpt_value):
    """Parse an embedded MCQ turn -> (question, choices, answer_letter) or None."""
    text = human_value.replace("<image>", "").strip()
    lines = [ln for ln in text.split("\n") if ln.strip()]
    q_lines, choices = [], []
    for ln in lines:
        m = _OPT.match(ln)
        if m:
            choices.append(m.group(2).strip())
        elif choices:
            # a non-option line after options started = the trailing "Please answer..." instruction
            continue
        else:
            q_lines.append(ln.strip())
    question = " ".join(q_lines).strip()
    am = _ANS.match(gpt_value.strip())
    if not question or len(choices) < 2 or not am:
        return None
    ans_idx = ord(am.group(1)) - ord("A")
    if ans_idx < 0 or ans_idx >= len(choices):
        return None
    return question, choices, am.group(1)


def source_of(video_path):
    parts = video_path.replace("\\", "/").split("/")
    # e.g. academic_source/Charades/RW587.mp4 -> "Charades"
    return parts[1] if len(parts) >= 3 else (parts[0] if parts else "?")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--anno", nargs="+", required=True, help="one or more *_qa_processed.json")
    ap.add_argument("--video_root", required=True,
                    help="dir the `video` field is relative to (its abs path = video_root/<video>)")
    ap.add_argument("--out", required=True)
    ap.add_argument("--exclude_sources", default=",".join(DEFAULT_EXCLUDE),
                    help="comma list of source dir names to DROP (leakage guard). '' = keep all.")
    ap.add_argument("--include_oe", action="store_true",
                    help="also emit open-ended turns (choices=null); default MC-only for clean labels")
    ap.add_argument("--require_video", action="store_true",
                    help="only keep rows whose mp4 actually exists on disk")
    ap.add_argument("--limit", type=int, default=0)
    args = ap.parse_args()

    exclude = {s.strip() for s in args.exclude_sources.split(",") if s.strip()}
    by_video = defaultdict(lambda: {"video_id": None, "path": None, "data_source": None, "qa": []})
    src_hist = Counter()
    dropped_src = Counter()
    n_missing = 0

    for anno in args.anno:
        is_mc = "mc_" in os.path.basename(anno)
        data = json.load(open(anno, encoding="utf-8"))
        for r in data:
            vid = r.get("id")
            vpath = r.get("video")
            if not vid or not vpath:
                continue
            src = source_of(vpath)
            src_hist[src] += 1
            if src in exclude:
                dropped_src[src] += 1
                continue
            abspath = os.path.join(args.video_root, vpath)
            if args.require_video and not os.path.exists(abspath):
                n_missing += 1
                continue
            conv = r.get("conversations", [])
            slot = by_video[vid]
            slot["video_id"] = vid
            slot["path"] = abspath
            slot["data_source"] = r.get("data_source") or src
            # conversations come in (human, gpt) pairs
            for i in range(0, len(conv) - 1, 2):
                if conv[i].get("from") != "human" or conv[i + 1].get("from") != "gpt":
                    continue
                hv, gv = conv[i]["value"], conv[i + 1]["value"]
                if is_mc:
                    parsed = parse_mc(hv, gv)
                    if parsed:
                        q, ch, a = parsed
                        slot["qa"].append({"question": q, "choices": ch, "answer": a})
                elif args.include_oe:
                    q = hv.replace("<image>", "").strip()
                    if q and gv.strip():
                        slot["qa"].append({"question": q, "choices": None, "answer": gv.strip()})

    rows = [v for v in by_video.values() if v["qa"]]
    if args.limit:
        rows = rows[:args.limit]

    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    with open(args.out, "w", encoding="utf-8") as f:
        for v in rows:
            f.write(json.dumps(v, ensure_ascii=False) + "\n")

    n_qa = sum(len(v["qa"]) for v in rows)
    print("=== source histogram (all records) ===")
    for s, c in src_hist.most_common():
        tag = "  [EXCLUDED]" if s in exclude else ""
        print(f"  {s:20s} {c:7d}{tag}")
    print(f"\nexcluded sources: {sorted(exclude)}")
    print(f"dropped by leakage guard: {sum(dropped_src.values())} records "
          f"({dict(dropped_src)})")
    if args.require_video:
        print(f"dropped (video missing on disk): {n_missing}")
    print(f"\nwrote {len(rows)} videos, {n_qa} QA to {args.out}")
    print("Next: run 01_prepare_data.py --manifest <this> --make_feats --shard i/N")


if __name__ == "__main__":
    main()
