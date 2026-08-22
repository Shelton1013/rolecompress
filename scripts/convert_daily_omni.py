# -*- coding: utf-8 -*-
"""Convert Daily-Omni annotations -> eval qa jsonl + video manifest for 05_eval.py.

Daily-Omni record shape:
  {"Question": "...", "Choice": ["A. ...","B. ...","C. ...","D. ..."], "Answer": "B",
   "video_id": "Ec_lQgZ9wlg", "Type": "Event Sequence", "video_duration": "30s", ...}

Outputs:
  --out_qa       : {"video_id","question","choices"(letters stripped),"answer"(letter),"task_type"}
  --out_manifest : {"video_id","path"}  (path resolved by scanning --video_root for <video_id>.*)

`task_type` carries Daily-Omni's `Type` so the synergy fraction can later be broken down by
question type (audio-visual sync / event sequence / ...).
"""
import argparse
import glob
import json
import os
import re

_LETTER = re.compile(r"^\s*([A-H])[\.\):]\s*")
_VID_EXT = (".mp4", ".mkv", ".webm", ".avi", ".mov")


def strip_letter(opt):
    return _LETTER.sub("", opt).strip()


def index_videos(video_root):
    """Map many id-forms -> abspath for every video file under video_root, so a qa `video_id`
    resolves whether the file is <id>.mp4, <id>_video.mp4, or Videos/<id>/<id>_video.mp4
    (Daily-Omni layout keys the clip by its PARENT DIR name = the video_id)."""
    idx = {}
    for ext in _VID_EXT:
        for p in glob.glob(os.path.join(video_root, "**", "*" + ext), recursive=True):
            stem = os.path.splitext(os.path.basename(p))[0]
            keys = {stem, os.path.basename(os.path.dirname(p))}  # filename stem + parent dir name
            for suf in ("_video", "-video", "_v"):
                if stem.endswith(suf):
                    keys.add(stem[: -len(suf)])
            for k in keys:
                idx.setdefault(k, p)
    return idx


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--anno", nargs="+", required=True, help="one or more Daily-Omni json files")
    ap.add_argument("--video_root", required=True, help="dir to scan for <video_id>.<ext>")
    ap.add_argument("--out_qa", required=True)
    ap.add_argument("--out_manifest", required=True)
    ap.add_argument("--require_video", action="store_true", help="drop QA whose video is missing")
    args = ap.parse_args()

    vindex = index_videos(args.video_root)
    print(f"indexed {len(vindex)} video files under {args.video_root}")

    qa_rows, manifest = [], {}
    n_missing = 0
    for anno in args.anno:
        data = json.load(open(anno, encoding="utf-8"))
        for r in (data if isinstance(data, list) else data.get("data", [])):
            vid = r.get("video_id") or r.get("video")
            q = r.get("Question") or r.get("question")
            choices = r.get("Choice") or r.get("choices")
            ans = r.get("Answer") or r.get("answer")
            if not (vid and q and choices and ans):
                continue
            am = _LETTER.match(str(ans) + ".") if len(str(ans).strip()) == 1 else _LETTER.match(str(ans))
            ans_letter = (am.group(1) if am else str(ans).strip()[:1]).upper()
            path = vindex.get(vid)
            if args.require_video and not path:
                n_missing += 1
                continue
            if path:
                manifest[vid] = path
            qa_rows.append({
                "video_id": vid,
                "question": q,
                "choices": [strip_letter(c) for c in choices],
                "answer": ans_letter,
                "task_type": r.get("Type") or r.get("type"),
            })

    os.makedirs(os.path.dirname(args.out_qa) or ".", exist_ok=True)
    with open(args.out_qa, "w", encoding="utf-8") as f:
        for x in qa_rows:
            f.write(json.dumps(x, ensure_ascii=False) + "\n")
    with open(args.out_manifest, "w", encoding="utf-8") as f:
        for vid, path in manifest.items():
            f.write(json.dumps({"video_id": vid, "path": path}, ensure_ascii=False) + "\n")

    import collections
    types = collections.Counter(x["task_type"] for x in qa_rows)
    print(f"wrote {len(qa_rows)} QA ({len(manifest)} unique videos) "
          f"| missing videos: {n_missing}")
    print("question types:", dict(types))
    print(f"qa -> {args.out_qa}\nmanifest -> {args.out_manifest}")


if __name__ == "__main__":
    main()
