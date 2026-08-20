# -*- coding: utf-8 -*-
"""Core definitions: information roles, role assignment from head-margins, and the
role -> frame/token budget policy. This module is pure Python/torch-free where possible
so it can be unit-tested locally without a GPU."""
from __future__ import annotations

import enum
import math
from dataclasses import dataclass
from typing import Dict, List, Sequence, Tuple


class Role(enum.IntEnum):
    """Cross-modal information role of a video segment w.r.t. answering.

    REDUNDANT      : text/ASR already conveys it -> drop visual tokens.
    UNIQUE_VISUAL  : only vision carries it        -> keep sparse visual.
    SYNERGISTIC    : needs BOTH text and vision    -> keep dense visual.
    UNIQUE_TEXT    : only text carries it (speech)  -> keep text, minimal visual (treated like REDUNDANT for visual budget).
    """
    REDUNDANT = 0
    UNIQUE_VISUAL = 1
    SYNERGISTIC = 2
    UNIQUE_TEXT = 3

    @property
    def short(self) -> str:
        return {0: "R", 1: "Uv", 2: "S", 3: "Ut"}[int(self)]


ROLE_NAMES = [r.name for r in Role]
NUM_ROLES = len(Role)


@dataclass
class RoleBudget:
    """Frame budget per role for the primary (frame-level) allocation.

    n_redundant/n_unique_text: usually 0 (rely on ASR text tokens).
    n_unique_visual: a few frames. n_synergistic: dense frames.
    merge_factor_*: optional spatial token-merge factor for the token-level path
        (1 = no merge, 2 = 2x2 pool -> 1/4 tokens, etc.).
    """
    n_redundant: int = 0
    n_unique_text: int = 0
    n_unique_visual: int = 1
    n_synergistic: int = 4
    merge_redundant: int = 4
    merge_unique_visual: int = 2
    merge_synergistic: int = 1
    min_total_frames: int = 4          # never send an empty video to the backbone
    max_total_frames: int = 64         # hard cap for memory safety

    def frames_for(self, role: "Role") -> int:
        return {
            Role.REDUNDANT: self.n_redundant,
            Role.UNIQUE_TEXT: self.n_unique_text,
            Role.UNIQUE_VISUAL: self.n_unique_visual,
            Role.SYNERGISTIC: self.n_synergistic,
        }[role]

    def merge_for(self, role: "Role") -> int:
        return {
            Role.REDUNDANT: self.merge_redundant,
            Role.UNIQUE_TEXT: self.merge_redundant,
            Role.UNIQUE_VISUAL: self.merge_unique_visual,
            Role.SYNERGISTIC: self.merge_synergistic,
        }[role]


def assign_role_from_margins(
    m_text: float,
    m_vision: float,
    m_joint: float,
    tau_hi: float = 0.5,
    tau_lo: float = 0.0,
) -> Role:
    """Hard role assignment from single- vs joint-modality head margins.

    Margins are "confidence of the gold answer" (e.g. logit margin for MCQ, or
    length-normalized log-likelihood for open-ended). Higher = the head can answer.

    Priority order matters:
      1. text alone suffices                       -> REDUNDANT (cheapest, drop visual)
      2. only joint works (both singles weak)      -> SYNERGISTIC
      3. vision alone works, text can't            -> UNIQUE_VISUAL
      4. text works but weakly, vision doesn't     -> UNIQUE_TEXT
      5. fallback                                  -> UNIQUE_VISUAL (keep some visual, safe)
    """
    text_ok = m_text >= tau_hi
    vis_ok = m_vision >= tau_hi
    joint_ok = m_joint >= tau_hi
    text_weak = m_text < tau_lo
    vis_weak = m_vision < tau_lo

    if text_ok and m_text >= m_vision:
        return Role.REDUNDANT
    if joint_ok and text_weak and vis_weak:
        return Role.SYNERGISTIC
    if vis_ok and text_weak:
        return Role.UNIQUE_VISUAL
    if text_ok:
        return Role.UNIQUE_TEXT
    if joint_ok and m_joint - max(m_text, m_vision) >= (tau_hi - tau_lo) * 0.5:
        # joint clearly beats both singles -> synergy even if a single is middling
        return Role.SYNERGISTIC
    return Role.UNIQUE_VISUAL


def soft_role_target(
    m_text: float,
    m_vision: float,
    m_joint: float,
    temperature: float = 1.0,
) -> List[float]:
    """Soft 4-way role distribution for KL training of the router.

    We turn margins into per-role scores, then softmax. This keeps the training
    signal graded instead of a brittle argmax.
      REDUNDANT     score ~ m_text                         (text alone)
      UNIQUE_VISUAL score ~ m_vision - m_text              (vision beyond text)
      SYNERGISTIC   score ~ m_joint - max(m_text, m_vision)(joint beyond both)
      UNIQUE_TEXT   score ~ m_text - m_vision              (text beyond vision)
    """
    s_red = m_text
    s_uv = m_vision - m_text
    s_syn = m_joint - max(m_text, m_vision)
    s_ut = m_text - m_vision
    scores = [s_red, s_uv, s_syn, s_ut]
    mx = max(scores)
    exps = [math.exp((s - mx) / max(1e-6, temperature)) for s in scores]
    z = sum(exps)
    return [e / z for e in exps]


def allocate_frames(
    roles: Sequence["Role"],
    seg_frame_counts: Sequence[int],
    budget: RoleBudget,
) -> List[List[int]]:
    """Given a role per segment and how many raw frames each segment has,
    return, per segment, the list of *local frame indices* to keep.

    Frames are chosen evenly within the segment. Enforces min/max total frames:
      - if total kept < min_total_frames, promote the highest-count segments to keep 1 more.
      - if total kept > max_total_frames, demote synergy->unique budget proportionally.
    Returns local indices (0-based within each segment).
    """
    assert len(roles) == len(seg_frame_counts)
    keep: List[List[int]] = []
    for role, nf in zip(roles, seg_frame_counts):
        k = min(budget.frames_for(role), nf)
        keep.append(_even_indices(nf, k))

    total = sum(len(x) for x in keep)

    # enforce minimum: add frames to segments that currently have the fewest but are non-empty
    if total < budget.min_total_frames:
        order = sorted(range(len(keep)), key=lambda i: (len(keep[i]), -seg_frame_counts[i]))
        i = 0
        while total < budget.min_total_frames and any(len(keep[j]) < seg_frame_counts[j] for j in range(len(keep))):
            idx = order[i % len(order)]
            if len(keep[idx]) < seg_frame_counts[idx]:
                keep[idx] = _even_indices(seg_frame_counts[idx], len(keep[idx]) + 1)
                total += 1
            i += 1

    # enforce maximum: trim from the segments with the most frames first
    if total > budget.max_total_frames:
        order = sorted(range(len(keep)), key=lambda i: -len(keep[i]))
        i = 0
        while total > budget.max_total_frames and any(len(x) > 0 for x in keep):
            idx = order[i % len(order)]
            if len(keep[idx]) > 0:
                keep[idx] = _even_indices(seg_frame_counts[idx], len(keep[idx]) - 1)
                total -= 1
            i += 1
    return keep


def _even_indices(n: int, k: int) -> List[int]:
    """k evenly-spaced indices in [0, n). Returns [] for k<=0, [n//2] for k==1."""
    if k <= 0 or n <= 0:
        return []
    if k == 1:
        return [n // 2]
    if k >= n:
        return list(range(n))
    step = (n - 1) / (k - 1)
    return sorted({int(round(i * step)) for i in range(k)})


def role_stats(roles: Sequence["Role"]) -> Dict[str, float]:
    """Distribution of roles (fractions) for logging."""
    n = max(1, len(roles))
    out = {r.name: 0.0 for r in Role}
    for r in roles:
        out[Role(r).name] += 1.0 / n
    return out
