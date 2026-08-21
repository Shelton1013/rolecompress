# -*- coding: utf-8 -*-
"""Benchmark adapters: map public long-video QA datasets to RoleCompress's unified format.

Unified eval row:
  {"video_id": str, "question": str, "choices": [str,...], "answer": "A"/"B"/..., "path": str}
Unified manifest row (one per unique video):
  {"video_id": str, "path": str}

Each adapter is a function (hf_row, video_dir) -> unified_row or None. Field names come from
each dataset's HF schema; because schemas drift, every adapter is small and easy to tweak, and
there is a `generic` adapter driven by a --field_map for anything not covered. Verify field
names against the HF dataset viewer if a benchmark updates.

Datasets (HF ids used by scripts/00_convert_benchmarks.py):
  videomme       -> lmms-lab/Video-MME               (filter duration == 'long')
  mlvu           -> MLVU/MLVU  (or sy1998/MLVU)       (multiple-choice tasks)
  longvideobench -> longvideobench/LongVideoBench
  egoschema      -> lmms-lab/egoschema  (subset/full)
"""
from __future__ import annotations

import os
import re
from typing import Callable, Dict, List, Optional


def _letter(idx: int) -> str:
    return chr(ord("A") + int(idx))


def _strip_letter_prefix(opt: str) -> str:
    return re.sub(r"^\s*[A-H][\.\):]\s*", "", str(opt)).strip()


_VID_EXT = (".mp4", ".mkv", ".webm", ".avi", ".mov", ".ts")
_VIDEO_INDEX: Dict[str, Dict[str, str]] = {}


def _index_videos(video_dir: str) -> Dict[str, str]:
    """Walk video_dir RECURSIVELY once and map both filename and stem -> full path, so nested
    layouts like MLVU's video/<task>/<name>.mp4 resolve by basename."""
    if video_dir in _VIDEO_INDEX:
        return _VIDEO_INDEX[video_dir]
    idx: Dict[str, str] = {}
    for root, _dirs, files in os.walk(video_dir):
        for f in files:
            if f.lower().endswith(_VID_EXT):
                full = os.path.join(root, f)
                idx.setdefault(f, full)                       # by filename
                idx.setdefault(os.path.splitext(f)[0], full)  # by stem
    _VIDEO_INDEX[video_dir] = idx
    return idx


def _resolve_path(video_dir: str, *candidates: str) -> Optional[str]:
    idx = _index_videos(video_dir)
    for c in candidates:
        if not c:
            continue
        c = str(c)
        for key in (c, os.path.basename(c), os.path.splitext(os.path.basename(c))[0]):
            if key in idx:
                return idx[key]
        for ext in ("",) + _VID_EXT:                          # flat fallback
            p = os.path.join(video_dir, c + ext)
            if os.path.exists(p):
                return p
    return None


# ------------------------------------------------------------------ adapters
def videomme(row: Dict, video_dir: str) -> Optional[Dict]:
    if str(row.get("duration", "")).lower() not in ("long", ""):  # keep long split (or unlabeled)
        return None
    vid = str(row.get("videoID") or row.get("video_id") or row.get("video"))
    options = row.get("options") or row.get("candidates") or []
    choices = [_strip_letter_prefix(o) for o in options]
    ans = str(row.get("answer", "")).strip()
    ans = ans if ans[:1].isalpha() else _letter(int(ans))
    path = _resolve_path(video_dir, vid, os.path.basename(str(row.get("video", ""))))
    if not (choices and path):
        return None
    return {"video_id": vid, "question": row["question"], "choices": choices, "answer": ans[:1].upper(), "path": path}


def mlvu(row: Dict, video_dir: str) -> Optional[Dict]:
    vid = str(row.get("video") or row.get("video_name") or row.get("video_id"))
    choices = [_strip_letter_prefix(o) for o in (row.get("candidates") or row.get("options") or [])]
    ans_raw = row.get("answer")
    if isinstance(ans_raw, int):
        ans = _letter(ans_raw)
    else:
        a = str(ans_raw).strip()
        # answer may be the full option text or a letter
        ans = a[:1].upper() if a[:1].isalpha() and len(a) <= 2 else \
              (_letter(choices.index(_strip_letter_prefix(a))) if _strip_letter_prefix(a) in choices else a[:1].upper())
    path = _resolve_path(video_dir, vid, os.path.basename(str(vid)))
    if not (choices and path):
        return None
    task = row.get("question_type") or row.get("task_type") or ""
    return {"video_id": vid, "question": row["question"], "choices": choices, "answer": ans,
            "path": path, "task_type": task}


def longvideobench(row: Dict, video_dir: str) -> Optional[Dict]:
    vid = str(row.get("video_id") or row.get("video_path") or row.get("video"))
    choices = [_strip_letter_prefix(o) for o in (row.get("candidates") or row.get("options") or [])]
    cc = row.get("correct_choice", row.get("answer"))
    ans = _letter(cc) if isinstance(cc, int) else str(cc)[:1].upper()
    path = _resolve_path(video_dir, vid, os.path.basename(str(row.get("video_path", ""))))
    if not (choices and path):
        return None
    return {"video_id": vid, "question": row["question"], "choices": choices, "answer": ans, "path": path}


def egoschema(row: Dict, video_dir: str) -> Optional[Dict]:
    vid = str(row.get("video_idx") or row.get("q_uid") or row.get("video"))
    # options either as a list or option_0..option_4
    opts = row.get("options")
    if not opts:
        opts = [row.get(f"option {i}", row.get(f"option_{i}")) for i in range(5)]
        opts = [o for o in opts if o is not None]
    choices = [_strip_letter_prefix(o) for o in opts]
    ans_raw = row.get("answer", row.get("correct_answer"))
    ans = _letter(ans_raw) if isinstance(ans_raw, int) else str(ans_raw)[:1].upper()
    path = _resolve_path(video_dir, vid, str(row.get("q_uid", "")))
    if not (choices and path):
        return None
    return {"video_id": vid, "question": row["question"], "choices": choices, "answer": ans, "path": path}


def generic(field_map: Dict[str, str]) -> Callable:
    """Build an adapter from an explicit field map, e.g.
    {"video_id":"vid","question":"q","choices":"opts","answer":"ans"}."""
    def _adapt(row: Dict, video_dir: str) -> Optional[Dict]:
        vid = str(row.get(field_map["video_id"]))
        choices = [_strip_letter_prefix(o) for o in row.get(field_map["choices"], [])]
        ans_raw = row.get(field_map["answer"])
        ans = _letter(ans_raw) if isinstance(ans_raw, int) else str(ans_raw)[:1].upper()
        path = _resolve_path(video_dir, vid)
        if not (choices and path):
            return None
        return {"video_id": vid, "question": row[field_map["question"]], "choices": choices, "answer": ans, "path": path}
    return _adapt


ADAPTERS: Dict[str, Callable] = {
    "videomme": videomme, "mlvu": mlvu, "longvideobench": longvideobench, "egoschema": egoschema,
}

HF_IDS = {
    "videomme": "lmms-lab/Video-MME",
    "mlvu": "MLVU/MLVU",
    "longvideobench": "longvideobench/LongVideoBench",
    "egoschema": "lmms-lab/egoschema",
}
