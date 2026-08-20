# -*- coding: utf-8 -*-
"""Datasets & IO.

Formats (jsonl):
  probes.jsonl    : one SegmentProbe per line (fields match pid_labels.SegmentProbe)
  labels.jsonl    : {video_id, seg_index, m_text, m_vision, m_joint, hard_role, soft_role[4], ok}
  qa_train.jsonl  : {video, question, choices?, answer, segments:[{start,end,asr}], ...}
  qa_eval.jsonl   : same as qa_train + optional {m_text,m_vision,m_joint} for synergy subset
  seg_feats/<video_id>.pt : cached router features {vis,txt,scal} for a video's segments

This module only handles (de)serialization + a torch Dataset for router training over
cached features. The heavy multimodal collation for LoRA training lives in scripts/04.
"""
from __future__ import annotations

import json
import os
from dataclasses import asdict
from typing import Dict, Iterator, List, Optional

import torch
from torch.utils.data import Dataset

from .pid_labels import ProbeMargins, RoleLabel, SegmentProbe


def read_jsonl(path: str) -> Iterator[dict]:
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                yield json.loads(line)


def write_jsonl(path: str, rows) -> None:
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")


def load_probes(path: str) -> List[SegmentProbe]:
    return [SegmentProbe(**{k: v for k, v in row.items() if k in SegmentProbe.__annotations__}) for row in read_jsonl(path)]


def label_to_row(lbl: RoleLabel) -> dict:
    return {
        "video_id": lbl.video_id, "seg_index": lbl.seg_index,
        "m_text": lbl.margins.m_text, "m_vision": lbl.margins.m_vision, "m_joint": lbl.margins.m_joint,
        "hard_role": lbl.hard_role, "soft_role": lbl.soft_role, "ok": lbl.ok,
    }


def row_to_label(row: dict) -> RoleLabel:
    return RoleLabel(
        video_id=row["video_id"], seg_index=row["seg_index"],
        margins=ProbeMargins(row["m_text"], row["m_vision"], row["m_joint"]),
        hard_role=row["hard_role"], soft_role=row["soft_role"], ok=row["ok"],
    )


class RouterFeatureDataset(Dataset):
    """One item = one video: cached (vis, txt, scal) features + soft/hard role targets per segment.
    Router trains at video granularity so its temporal-context transformer sees all segments."""

    def __init__(self, labels_path: str, feats_dir: str, max_segments: int = 256, only_ok: bool = True):
        self.feats_dir = feats_dir
        self.max_segments = max_segments
        by_video: Dict[str, List[dict]] = {}
        for row in read_jsonl(labels_path):
            if only_ok and not row.get("ok", True):
                continue
            by_video.setdefault(row["video_id"], []).append(row)
        self.videos = []
        for vid, rows in by_video.items():
            feat_path = os.path.join(feats_dir, f"{vid}.pt")
            if os.path.exists(feat_path):
                rows.sort(key=lambda r: r["seg_index"])
                self.videos.append((vid, rows))

    def __len__(self):
        return len(self.videos)

    def __getitem__(self, i):
        vid, rows = self.videos[i]
        feats = torch.load(os.path.join(self.feats_dir, f"{vid}.pt"), map_location="cpu")
        T = min(self.max_segments, feats["vis"].shape[0], len(rows))
        vis, txt, scal = feats["vis"][:T], feats["txt"][:T], feats["scal"][:T]
        soft = torch.tensor([rows[j]["soft_role"] for j in range(T)], dtype=torch.float32)
        hard = torch.tensor([rows[j]["hard_role"] for j in range(T)], dtype=torch.long)
        mask = torch.ones(T, dtype=torch.bool)
        return {"vis": vis, "txt": txt, "scal": scal, "soft": soft, "hard": hard, "mask": mask, "video_id": vid}


def collate_router(batch, max_segments: int = 256):
    """Pad variable-length segment sequences to the batch max."""
    T = min(max_segments, max(b["vis"].shape[0] for b in batch))
    dv, dt, ds = batch[0]["vis"].shape[1], batch[0]["txt"].shape[1], batch[0]["scal"].shape[1]
    B = len(batch)
    R = batch[0]["soft"].shape[1]
    vis = torch.zeros(B, T, dv); txt = torch.zeros(B, T, dt); scal = torch.zeros(B, T, ds)
    soft = torch.zeros(B, T, R); hard = torch.zeros(B, T, dtype=torch.long); mask = torch.zeros(B, T, dtype=torch.bool)
    for i, b in enumerate(batch):
        t = min(T, b["vis"].shape[0])
        vis[i, :t], txt[i, :t], scal[i, :t] = b["vis"][:t], b["txt"][:t], b["scal"][:t]
        soft[i, :t], hard[i, :t], mask[i, :t] = b["soft"][:t], b["hard"][:t], b["mask"][:t]
    return {"vis": vis, "txt": txt, "scal": scal, "soft": soft, "hard": hard, "mask": mask}
