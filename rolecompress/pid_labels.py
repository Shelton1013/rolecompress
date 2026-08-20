# -*- coding: utf-8 -*-
"""Self-supervised role labels from single- vs joint-modality head disagreement.

For each segment probe (q, gold, distractors), we score three *frozen* passes of the
SAME backbone: text-only, vision-only, joint. We turn each into a margin (higher = the
head can answer) and derive hard + soft role targets via rolecompress.roles.

This module defines the margin computation given a scoring backend, plus the batching
logic. The scoring backend is provided by backbone.RoleCompressBackbone (see .score_*).

No extra models are used — labels come from the backbone itself. This mirrors SPICE's
signal source but is used here to produce per-segment compression labels.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence

from .roles import Role, assign_role_from_margins, soft_role_target


@dataclass
class SegmentProbe:
    video_id: str
    seg_index: int
    seg_start: float
    seg_end: float
    question: str
    gold: str                       # gold answer text (open) or gold letter (mcq)
    choices: Optional[List[str]] = None   # for MCQ; None for open-ended
    asr_text: str = ""              # ASR transcript of THIS segment (the cheap dense modality)


@dataclass
class ProbeMargins:
    m_text: float
    m_vision: float
    m_joint: float


@dataclass
class RoleLabel:
    video_id: str
    seg_index: int
    margins: ProbeMargins
    hard_role: int                  # Role int
    soft_role: List[float]          # length-4 distribution
    ok: bool                        # whether ANY head could answer (else discard)


def margins_to_label(
    probe: SegmentProbe,
    m: ProbeMargins,
    tau_hi: float,
    tau_lo: float,
    temperature: float,
    keep_threshold: float,
) -> RoleLabel:
    """Convert raw margins to a role label. `keep_threshold`: if no head reaches it,
    the probe is degenerate (unanswerable/ambiguous) and flagged ok=False for filtering."""
    ok = max(m.m_text, m.m_vision, m.m_joint) >= keep_threshold
    hard = assign_role_from_margins(m.m_text, m.m_vision, m.m_joint, tau_hi, tau_lo)
    soft = soft_role_target(m.m_text, m.m_vision, m.m_joint, temperature)
    return RoleLabel(probe.video_id, probe.seg_index, m, int(hard), soft, ok)


def calibrate_thresholds(
    margins: Sequence[ProbeMargins],
    hi_quantile: float = 0.6,
    lo_quantile: float = 0.4,
) -> Dict[str, float]:
    """Pick tau_hi/tau_lo from the empirical margin distribution on a dev split so the
    role assignment is not sensitive to absolute margin scale across backbones/datasets."""
    allm = sorted(v for m in margins for v in (m.m_text, m.m_vision, m.m_joint))
    if not allm:
        return {"tau_hi": 0.5, "tau_lo": 0.0}
    def q(p: float) -> float:
        i = min(len(allm) - 1, max(0, int(p * (len(allm) - 1))))
        return allm[i]
    return {"tau_hi": q(hi_quantile), "tau_lo": q(lo_quantile)}


def build_labels(
    probes: Sequence[SegmentProbe],
    margins: Sequence[ProbeMargins],
    tau_hi: float,
    tau_lo: float,
    temperature: float = 1.0,
    keep_threshold: Optional[float] = None,
) -> List[RoleLabel]:
    """Vectorized wrapper: probes and their precomputed margins -> role labels."""
    assert len(probes) == len(margins)
    if keep_threshold is None:
        keep_threshold = tau_lo
    return [
        margins_to_label(p, m, tau_hi, tau_lo, temperature, keep_threshold)
        for p, m in zip(probes, margins)
    ]


def label_agreement(pred_roles: Sequence[int], human_roles: Sequence[int]) -> float:
    """Top-1 agreement between self-sup roles and a human-labeled subset (sanity check)."""
    assert len(pred_roles) == len(human_roles)
    if not pred_roles:
        return 0.0
    return sum(int(a == b) for a, b in zip(pred_roles, human_roles)) / len(pred_roles)
