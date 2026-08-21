# -*- coding: utf-8 -*-
"""Video segmentation + frame sampling utilities (decord/ffmpeg based).

A "segment" is a short span (default: shot boundaries merged to ~4-8s, or fixed windows).
We keep it simple and robust: fixed-length windows by default, with an optional shot-based
mode. Downstream only needs, per segment: (start, end, list of raw frames).
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional, Sequence, Tuple

import numpy as np


@dataclass
class Segment:
    index: int
    start: float
    end: float
    frame_times: List[float]   # timestamps (s) of raw sampled frames in this segment


def make_windows(duration: float, win: float = 6.0, min_win: float = 2.0) -> List[Tuple[float, float]]:
    if duration <= 0:
        return []
    spans = []
    t = 0.0
    while t < duration - 1e-3:
        end = min(duration, t + win)
        if end - t < min_win and spans:
            spans[-1] = (spans[-1][0], end)
        else:
            spans.append((t, end))
        t = end
    return spans


def sample_frame_times(spans: Sequence[Tuple[float, float]], fps: float = 1.0, max_per_seg: int = 8) -> List[Segment]:
    segs = []
    for i, (a, b) in enumerate(spans):
        n = max(1, min(max_per_seg, int(round((b - a) * fps))))
        if n == 1:
            times = [(a + b) / 2.0]
        else:
            step = (b - a) / n
            times = [a + step * (k + 0.5) for k in range(n)]
        segs.append(Segment(i, a, b, times))
    return segs


class VideoReader:
    """Thin decord wrapper. Returns HxWx3 uint8 frames at requested timestamps."""

    def __init__(self, path: str):
        import decord  # noqa
        from decord import VideoReader as _VR, cpu
        self.vr = _VR(path, ctx=cpu(0))
        self.fps = float(self.vr.get_avg_fps()) or 25.0
        self.n = len(self.vr)
        self.duration = self.n / self.fps

    def frames_at(self, times: Sequence[float]) -> List[np.ndarray]:
        idx = [min(self.n - 1, max(0, int(round(t * self.fps)))) for t in times]
        if not idx:
            return []
        batch = self.vr.get_batch(idx).asnumpy()  # (k,H,W,3)
        return [batch[i] for i in range(batch.shape[0])]


def segment_video(path: str, win: float = 6.0, fps: float = 1.0, max_per_seg: int = 8,
                  max_segments: int = 64):
    """High-level: path -> (segments, frames_per_segment).

    `max_segments` caps the number of segments regardless of video length by adaptively
    enlarging the window (segment ops are O(#segments) per video, so this keeps cost bounded
    on hour-long videos)."""
    vr = VideoReader(path)
    w = win
    if max_segments and vr.duration > 0 and (vr.duration / win) > max_segments:
        w = vr.duration / max_segments
    spans = make_windows(vr.duration, win=w)
    segs = sample_frame_times(spans, fps=fps, max_per_seg=max_per_seg)
    frames_per_seg = [vr.frames_at(s.frame_times) for s in segs]
    return segs, frames_per_seg, vr.duration


# --------- shot-based (optional) ---------
def shot_boundaries_ffmpeg(path: str, threshold: float = 0.30) -> List[float]:
    """Return shot-cut timestamps via ffmpeg scdet. Requires ffmpeg on PATH."""
    import re
    import subprocess
    cmd = ["ffmpeg", "-hide_banner", "-i", path, "-an",
           "-vf", f"select='gt(scene,{threshold})',showinfo", "-f", "null", "-"]
    p = subprocess.run(cmd, capture_output=True, text=True, encoding="utf-8", errors="replace")
    cuts = [float(m) for m in re.findall(r"pts_time:\s*([0-9.]+)", p.stderr)]
    return sorted(set(cuts))
