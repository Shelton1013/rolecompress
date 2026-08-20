# -*- coding: utf-8 -*-
"""Shape/smoke tests for the router (CPU, tiny). Validates forward + loss run and shapes."""
import torch

from rolecompress.router import RoleRouter, RouterConfig, router_loss
from rolecompress.roles import NUM_ROLES


def _tiny_cfg():
    return RouterConfig(d_visual=16, d_text=24, d_scalar=4, d_hidden=32,
                        n_layers=2, use_context=True, n_ctx_layers=1, n_heads=2, max_segments=32)


def test_router_forward_shapes():
    cfg = _tiny_cfg()
    m = RoleRouter(cfg)
    B, T = 2, 5
    vis = torch.randn(B, T, cfg.d_visual)
    txt = torch.randn(B, T, cfg.d_text)
    scal = torch.randn(B, T, cfg.d_scalar)
    mask = torch.ones(B, T, dtype=torch.bool)
    logits = m(vis, txt, scal, mask)
    assert logits.shape == (B, T, NUM_ROLES)


def test_router_loss_backward():
    cfg = _tiny_cfg()
    m = RoleRouter(cfg)
    B, T = 2, 5
    vis = torch.randn(B, T, cfg.d_visual, requires_grad=False)
    txt = torch.randn(B, T, cfg.d_text)
    scal = torch.randn(B, T, cfg.d_scalar)
    mask = torch.ones(B, T, dtype=torch.bool)
    soft = torch.softmax(torch.randn(B, T, NUM_ROLES), dim=-1)
    hard = soft.argmax(-1)
    logits = m(vis, txt, scal, mask)
    loss = router_loss(logits, soft, mask, hard)
    loss.backward()
    assert torch.isfinite(loss)
    assert any(p.grad is not None for p in m.parameters())


def test_router_predict_roles():
    cfg = _tiny_cfg()
    m = RoleRouter(cfg)
    vis = torch.randn(1, 4, cfg.d_visual)
    txt = torch.randn(1, 4, cfg.d_text)
    scal = torch.randn(1, 4, cfg.d_scalar)
    mask = torch.ones(1, 4, dtype=torch.bool)
    roles = m.predict_roles(vis, txt, scal, mask)
    assert roles.shape == (1, 4)
    assert roles.min() >= 0 and roles.max() < NUM_ROLES
