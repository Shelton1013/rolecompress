# -*- coding: utf-8 -*-
"""Stage 0: convert a public long-video QA benchmark to RoleCompress's unified jsonl.

Writes:
  <out>/<benchmark>_eval.jsonl   unified eval rows (video_id, question, choices, answer, path)
  <out>/<benchmark>_manifest.jsonl  one row per unique video (video_id, path)

Videos must already be downloaded to --video_dir (benchmarks ship QA, not always the videos).
Field mappings live in rolecompress/benchmarks.py; verify against the HF dataset viewer if a
benchmark's schema changed. Use --benchmark generic --field_map '...json...' for custom data.

Examples:
  python scripts/00_convert_benchmarks.py --benchmark videomme --split test \
      --video_dir /data/videomme/videos --out /data/rolecompress
  python scripts/00_convert_benchmarks.py --benchmark egoschema --hf_config Subset \
      --video_dir /data/egoschema/videos --out /data/rolecompress
"""
import argparse
import json
import os
import re

from rolecompress.benchmarks import ADAPTERS, HF_IDS, generic
from rolecompress.data import write_jsonl


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--benchmark", required=True,
                    choices=list(ADAPTERS.keys()) + ["generic"])
    ap.add_argument("--video_dir", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--hf_id", default=None, help="override the HF dataset id")
    ap.add_argument("--hf_config", default=None, help="dataset config/subset name if needed")
    ap.add_argument("--split", default="test")
    ap.add_argument("--field_map", default=None, help='JSON for --benchmark generic')
    ap.add_argument("--from_jsonl", default=None, help="skip HF; read rows from a local jsonl instead")
    ap.add_argument("--anno_dir", default=None,
                    help="LOCAL annotation dir (recurse *.json/*.jsonl); best for already-downloaded "
                         "benchmarks like MLVU (points at its json/ folder). Skips the HF Hub.")
    ap.add_argument("--limit", type=int, default=0)
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)

    adapt = generic(json.loads(args.field_map)) if args.benchmark == "generic" else ADAPTERS[args.benchmark]

    # load rows: local annotation dir  >  local jsonl  >  HF Hub
    if args.anno_dir:
        import glob
        rows = []
        files = sorted(glob.glob(os.path.join(args.anno_dir, "**", "*.json"), recursive=True)) + \
                sorted(glob.glob(os.path.join(args.anno_dir, "**", "*.jsonl"), recursive=True))
        for fp in files:
            task = re.sub(r"^\d+[_\-]?", "", os.path.splitext(os.path.basename(fp))[0])  # e.g. 4_count -> count
            try:
                if fp.endswith(".jsonl"):
                    items = [json.loads(l) for l in open(fp, encoding="utf-8") if l.strip()]
                else:
                    data = json.load(open(fp, encoding="utf-8"))
                    items = data if isinstance(data, list) else \
                            next((v for v in data.values() if isinstance(v, list)), [])
            except Exception as e:
                print(f"[skip anno] {fp}: {e}"); continue
            for it in items:
                if isinstance(it, dict):
                    it.setdefault("question_type", task)      # infer task from filename (MLVU's 9 tasks)
                    rows.append(it)
        print(f"loaded {len(rows)} rows from {len(files)} annotation files under {args.anno_dir}")
    elif args.from_jsonl:
        from rolecompress.data import read_jsonl
        rows = list(read_jsonl(args.from_jsonl))
    else:
        from datasets import load_dataset
        hf_id = args.hf_id or HF_IDS[args.benchmark]
        ds = load_dataset(hf_id, args.hf_config, split=args.split) if args.hf_config \
            else load_dataset(hf_id, split=args.split)
        rows = list(ds)
    if args.limit:
        rows = rows[:args.limit]

    eval_rows, seen_videos, dropped = [], {}, 0
    for row in rows:
        try:
            u = adapt(dict(row), args.video_dir)
        except Exception as e:
            dropped += 1; continue
        if u is None:
            dropped += 1; continue
        eval_rows.append(u)
        seen_videos[u["video_id"]] = u["path"]

    name = args.benchmark
    write_jsonl(os.path.join(args.out, f"{name}_eval.jsonl"), eval_rows)
    write_jsonl(os.path.join(args.out, f"{name}_manifest.jsonl"),
                [{"video_id": v, "path": p} for v, p in seen_videos.items()])
    print(f"[{name}] wrote {len(eval_rows)} eval rows, {len(seen_videos)} unique videos "
          f"(dropped {dropped}: missing video file or unparsable). "
          f"-> {args.out}/{name}_eval.jsonl  +  {name}_manifest.jsonl")
    if not eval_rows:
        print("WARNING: 0 rows kept. Check --video_dir (videos must be downloaded) and the "
              "field mapping in rolecompress/benchmarks.py against the HF dataset viewer.")


if __name__ == "__main__":
    main()
