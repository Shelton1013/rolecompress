# -*- coding: utf-8 -*-
"""Baseline allocation policies, on the SAME input-frame-budget axis as RoleCompress so the
accuracy-vs-visual-token Pareto is directly comparable.

Each baseline returns per-segment *kept local frame indices* (the common currency). The eval
then measures the realized visual-token count, so policies are compared by the x-axis (budget),
not by forcing identical knobs.

Honesty note on names:
  - `query`      : query-aware frame selection (keep frames most similar to the question).
                   This is the query-CONDITIONED contrast to our query-agnostic roles (SeViLA /
                   keyframe-selector family), implemented at input granularity.
  - `saliency`   : content-saliency frame selection (keep highest temporal-change / detail
                   frames). Input-side analogue of FastV's "keep high-attention visual tokens";
                   a true intra-LLM FastV (attention hook after layer K) is provided separately
                   as an optional appendix point (see fastv_hook.py) and is [VERIFY]-heavy.
  - `tokenmerge` : spatial token reduction via frame downscaling at matched frame count
                   (ToMe-video analogue). Handled in backbone.build_answer_inputs(downscale=...).
"""
from __future__ import annotations

from typing import List, Sequence

import numpy as np

from .roles import _even_indices


def _global_topk_to_local(scores_per_seg: Sequence[Sequence[float]], keep_total: int) -> List[List[int]]:
    """Given a score per raw frame (grouped by segment), keep the globally top `keep_total`
    frames; return per-segment local indices. Guarantees >=1 frame overall."""
    flat = [(s, si, li) for si, seg in enumerate(scores_per_seg) for li, s in enumerate(seg)]
    if not flat:
        return [[] for _ in scores_per_seg]
    flat.sort(key=lambda x: -x[0])
    keep_total = max(1, min(keep_total, len(flat)))
    chosen = flat[:keep_total]
    out: List[List[int]] = [[] for _ in scores_per_seg]
    for _, si, li in chosen:
        out[si].append(li)
    for si in range(len(out)):
        out[si].sort()
    return out


def frame_saliency_scores(frames: Sequence[np.ndarray]) -> List[float]:
    """Per-frame saliency = temporal change (L1 diff vs previous frame) + spatial detail (gradient)."""
    scores: List[float] = []
    prev = None
    for f in frames:
        g = f.astype(np.float32).mean(-1)
        detail = float(np.abs(np.diff(g, axis=0)).mean() + np.abs(np.diff(g, axis=1)).mean())
        change = float(np.abs(g - prev).mean()) if prev is not None else detail
        scores.append(0.5 * detail + 0.5 * change)
        prev = g
    return scores


def top_salient_local(frames: Sequence[np.ndarray], k: int) -> List[int]:
    """Local indices of the top-`k` most salient frames within one segment (sorted)."""
    if k <= 0 or not frames:
        return []
    if k >= len(frames):
        return list(range(len(frames)))
    sc = frame_saliency_scores(frames)
    return sorted(sorted(range(len(frames)), key=lambda j: -sc[j])[:k])


def saliency_frame_keep(seg_frames: Sequence[Sequence[np.ndarray]], keep_total: int) -> List[List[int]]:
    """Score each frame by saliency; keep the globally top `keep_total`."""
    scores = [frame_saliency_scores(frames) for frames in seg_frames]
    return _global_topk_to_local(scores, keep_total)


def query_frame_keep(backbone, question: str, seg_frames: Sequence[Sequence[np.ndarray]], keep_total: int) -> List[List[int]]:
    """Query-aware selection: cosine similarity between each frame's pooled LLM-space embedding
    and the question's pooled embedding; keep the globally top `keep_total`. Query-CONDITIONED
    (must recompute if the question changes) — the contrast to our query-agnostic roles."""
    import torch
    import torch.nn.functional as F
    H = backbone.llm_hidden
    q = backbone._encode_text_mean(question, H).float()
    scores: List[List[float]] = []
    for frames in seg_frames:
        seg = []
        for f in frames:
            fv = backbone._encode_frames_mean([f], H).float()
            seg.append(float(F.cosine_similarity(fv[None], q[None]).item()))
        scores.append(seg)
    return _global_topk_to_local(scores, keep_total)


def uniform_frame_keep(seg_frames: Sequence[Sequence[np.ndarray]], per_seg: int) -> List[List[int]]:
    """Uniform: keep `per_seg` evenly-spaced frames from every segment."""
    return [_even_indices(len(f), min(per_seg, len(f))) for f in seg_frames]
