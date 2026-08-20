# -*- coding: utf-8 -*-
"""Evaluation metrics: accuracy, visual budget, FLOPs proxy, synergy-subset, role stats."""
from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence


def mcq_correct(pred: str, gold_letter: str, choices: Optional[List[str]] = None) -> bool:
    """Robust MCQ scoring: accept a leading letter, or exact/loose choice-text match."""
    pred = (pred or "").strip()
    m = re.match(r"\s*\(?([A-H])\)?[\.\):]?", pred)
    if m:
        return m.group(1).upper() == gold_letter.upper()
    if choices:
        gi = ord(gold_letter.upper()) - ord("A")
        if 0 <= gi < len(choices):
            gt = choices[gi].strip().lower()
            return gt and gt in pred.lower()
    return False


def flops_proxy(visual_tokens: int, text_tokens: int = 0) -> float:
    """Prefill attention cost proxy ∝ N^2 over the full sequence (visual dominates on long video)."""
    n = visual_tokens + text_tokens
    return float(n) * float(n)


@dataclass
class EvalAccumulator:
    n: int = 0
    correct: int = 0
    vtok_sum: float = 0.0
    flops_sum: float = 0.0
    per_role_used: Dict[str, float] = field(default_factory=dict)
    # bookkeeping for the synergy subset
    syn_n: int = 0
    syn_correct: int = 0

    def add(self, is_correct: bool, visual_tokens: int, text_tokens: int = 0,
            in_synergy_subset: bool = False, roles: Optional[Sequence] = None):
        self.n += 1
        self.correct += int(is_correct)
        self.vtok_sum += visual_tokens
        self.flops_sum += flops_proxy(visual_tokens, text_tokens)
        if in_synergy_subset:
            self.syn_n += 1
            self.syn_correct += int(is_correct)
        if roles:
            for r in roles:
                key = getattr(r, "name", str(r))
                self.per_role_used[key] = self.per_role_used.get(key, 0.0) + 1.0

    def summary(self) -> Dict[str, float]:
        n = max(1, self.n)
        out = {
            "accuracy": self.correct / n,
            "mean_visual_tokens": self.vtok_sum / n,
            "mean_flops_proxy": self.flops_sum / n,
            "n": self.n,
        }
        if self.syn_n:
            out["synergy_subset_accuracy"] = self.syn_correct / self.syn_n
            out["synergy_subset_n"] = self.syn_n
        if self.per_role_used:
            tot = sum(self.per_role_used.values()) or 1.0
            for k, v in self.per_role_used.items():
                out[f"role_frac_{k}"] = v / tot
        return out


def is_synergy_required(m_text: float, m_vision: float, m_joint: float, tau_hi: float, tau_lo: float) -> bool:
    """A QA item is 'synergy-required' if neither single-modality frozen pass answers it
    but the joint pass does. Used to construct the crossover-experiment subset."""
    return (m_text < tau_lo) and (m_vision < tau_lo) and (m_joint >= tau_hi)
