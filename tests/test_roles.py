# -*- coding: utf-8 -*-
"""GPU-free unit tests for the novel logic. Run locally BEFORE the server run:
    pip install pytest && pytest -q tests/
These validate role assignment, budget allocation, soft targets, and metric math —
i.e. everything that determines correctness of the method independent of the backbone.
"""
import math

from rolecompress.roles import (Role, RoleBudget, assign_role_from_margins, soft_role_target,
                                 allocate_frames, _even_indices, role_stats)
from rolecompress.metrics import flops_proxy, is_synergy_required, mcq_correct


def test_role_assignment_redundant():
    # text alone strong -> redundant (drop visual)
    assert assign_role_from_margins(2.0, 0.1, 2.1, tau_hi=0.5, tau_lo=0.0) == Role.REDUNDANT


def test_role_assignment_synergy():
    # neither single works, joint works -> synergistic
    assert assign_role_from_margins(-1.0, -1.0, 1.5, tau_hi=0.5, tau_lo=0.0) == Role.SYNERGISTIC


def test_role_assignment_unique_visual():
    assert assign_role_from_margins(-1.0, 2.0, 2.0, tau_hi=0.5, tau_lo=0.0) == Role.UNIQUE_VISUAL


def test_soft_target_sums_to_one():
    t = soft_role_target(0.3, 0.9, 1.2)
    assert abs(sum(t) - 1.0) < 1e-6
    assert len(t) == 4


def test_soft_target_synergy_peaks_when_joint_dominates():
    t = soft_role_target(-1.0, -1.0, 2.0)
    assert t[int(Role.SYNERGISTIC)] == max(t)


def test_even_indices():
    assert _even_indices(10, 0) == []
    assert _even_indices(10, 1) == [5]
    assert _even_indices(4, 4) == [0, 1, 2, 3]
    assert len(_even_indices(10, 3)) == 3
    assert _even_indices(5, 10) == [0, 1, 2, 3, 4]  # k>=n


def test_allocate_frames_budget():
    b = RoleBudget(n_redundant=0, n_unique_visual=1, n_synergistic=4, min_total_frames=4, max_total_frames=64)
    roles = [Role.REDUNDANT, Role.UNIQUE_VISUAL, Role.SYNERGISTIC]
    counts = [8, 8, 8]
    keep = allocate_frames(roles, counts, b)
    assert len(keep[0]) == 0      # redundant dropped
    assert len(keep[1]) == 1      # unique visual sparse
    assert len(keep[2]) == 4      # synergy dense


def test_allocate_frames_min_enforced():
    b = RoleBudget(n_redundant=0, n_unique_visual=0, n_synergistic=0, min_total_frames=4)
    roles = [Role.REDUNDANT, Role.REDUNDANT]
    counts = [8, 8]
    keep = allocate_frames(roles, counts, b)
    assert sum(len(x) for x in keep) >= 4  # min enforced even if all redundant


def test_allocate_frames_max_enforced():
    b = RoleBudget(n_synergistic=50, max_total_frames=10)
    roles = [Role.SYNERGISTIC, Role.SYNERGISTIC]
    counts = [50, 50]
    keep = allocate_frames(roles, counts, b)
    assert sum(len(x) for x in keep) <= 10


def test_flops_proxy_quadratic():
    assert flops_proxy(10) == 100.0
    assert flops_proxy(20) == 400.0


def test_synergy_required():
    assert is_synergy_required(-1, -1, 1.0, tau_hi=0.5, tau_lo=0.0) is True
    assert is_synergy_required(1.0, -1, 1.0, tau_hi=0.5, tau_lo=0.0) is False  # text alone works


def test_mcq_correct():
    assert mcq_correct("B", "B")
    assert mcq_correct("(C) because ...", "C")
    assert not mcq_correct("A", "D")
    assert mcq_correct("the cat", "A", choices=["the cat", "a dog", "a bird", "none"])


def test_role_stats():
    s = role_stats([Role.REDUNDANT, Role.REDUNDANT, Role.SYNERGISTIC, Role.UNIQUE_VISUAL])
    assert abs(s["REDUNDANT"] - 0.5) < 1e-6
    assert abs(sum(s.values()) - 1.0) < 1e-6
