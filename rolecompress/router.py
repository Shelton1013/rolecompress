# -*- coding: utf-8 -*-
"""RoleRouter: a lightweight per-segment role classifier.

Input per segment: a cheap feature vector built from
  - pooled visual features of the segment (mean/CLS of the vision encoder, few frames)
  - pooled text (ASR) embedding of the segment
  - a couple of scalar priors (segment length, has_speech, visual motion/variance)
Output: 4-way role logits (REDUNDANT, UNIQUE_VISUAL, SYNERGISTIC, UNIQUE_TEXT).

Trained with KL to the soft role target from pid_labels (see roles.soft_role_target).
This is ~5-15M params and trains on cached features in minutes-hours, no backbone grad.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

from .roles import NUM_ROLES


@dataclass
class RouterConfig:
    d_visual: int = 1152      # vision encoder hidden (e.g. Qwen2.5-VL ViT ~1280; set per backbone)
    d_text: int = 3584        # LLM hidden for pooled ASR text embedding (Qwen2.5-VL-7B = 3584)
    d_scalar: int = 4         # [seg_seconds, has_speech, visual_var, asr_len_norm]
    d_hidden: int = 512
    n_layers: int = 3
    dropout: float = 0.1
    use_context: bool = True  # a small temporal transformer over segments (video-level context)
    n_ctx_layers: int = 2
    n_heads: int = 4
    max_segments: int = 256


class RoleRouter(nn.Module):
    def __init__(self, cfg: RouterConfig):
        super().__init__()
        self.cfg = cfg
        self.vis_proj = nn.Linear(cfg.d_visual, cfg.d_hidden)
        self.txt_proj = nn.Linear(cfg.d_text, cfg.d_hidden)
        self.scal_proj = nn.Linear(cfg.d_scalar, cfg.d_hidden)
        self.in_norm = nn.LayerNorm(cfg.d_hidden * 3)
        self.mixer = _mlp(cfg.d_hidden * 3, cfg.d_hidden, cfg.d_hidden, cfg.n_layers, cfg.dropout)

        if cfg.use_context:
            self.pos = nn.Parameter(torch.zeros(1, cfg.max_segments, cfg.d_hidden))
            nn.init.trunc_normal_(self.pos, std=0.02)
            enc_layer = nn.TransformerEncoderLayer(
                d_model=cfg.d_hidden, nhead=cfg.n_heads, dim_feedforward=cfg.d_hidden * 2,
                dropout=cfg.dropout, batch_first=True, activation="gelu", norm_first=True,
            )
            self.ctx = nn.TransformerEncoder(enc_layer, num_layers=cfg.n_ctx_layers)
        self.head = nn.Linear(cfg.d_hidden, NUM_ROLES)

    def forward(
        self,
        vis: torch.Tensor,      # (B, T, d_visual)  pooled per-segment visual feature
        txt: torch.Tensor,      # (B, T, d_text)    pooled per-segment ASR-text feature
        scal: torch.Tensor,     # (B, T, d_scalar)
        seg_mask: Optional[torch.Tensor] = None,  # (B, T) bool, True = valid segment
    ) -> torch.Tensor:
        """Returns role logits (B, T, NUM_ROLES)."""
        h = torch.cat([self.vis_proj(vis), self.txt_proj(txt), self.scal_proj(scal)], dim=-1)
        h = self.in_norm(h)
        h = self.mixer(h)  # (B, T, d_hidden)
        if self.cfg.use_context:
            T = h.size(1)
            h = h + self.pos[:, :T]
            key_padding = None if seg_mask is None else ~seg_mask
            h = self.ctx(h, src_key_padding_mask=key_padding)
        return self.head(h)

    @torch.no_grad()
    def predict_roles(self, *args, **kwargs) -> torch.Tensor:
        """Argmax role ids (B, T)."""
        logits = self.forward(*args, **kwargs)
        return logits.argmax(dim=-1)


def _mlp(d_in: int, d_hidden: int, d_out: int, n_layers: int, dropout: float) -> nn.Module:
    layers = [nn.Linear(d_in, d_hidden), nn.GELU(), nn.Dropout(dropout)]
    for _ in range(max(0, n_layers - 2)):
        layers += [nn.Linear(d_hidden, d_hidden), nn.GELU(), nn.Dropout(dropout)]
    layers += [nn.Linear(d_hidden, d_out)]
    return nn.Sequential(*layers)


def router_loss(
    logits: torch.Tensor,        # (B, T, R)
    soft_target: torch.Tensor,   # (B, T, R) soft role distribution
    seg_mask: torch.Tensor,      # (B, T) bool
    hard_label: Optional[torch.Tensor] = None,  # (B, T) long, optional aux CE
    kl_weight: float = 1.0,
    ce_weight: float = 0.2,
    class_weights: Optional[torch.Tensor] = None,  # (R,) to upweight the rare SYNERGISTIC class
) -> torch.Tensor:
    logp = F.log_softmax(logits, dim=-1)
    kl = F.kl_div(logp, soft_target, reduction="none").sum(-1)  # (B, T)
    mask = seg_mask.float()
    loss = (kl * mask).sum() / mask.sum().clamp_min(1.0) * kl_weight
    if hard_label is not None and ce_weight > 0:
        ce = F.cross_entropy(
            logits.reshape(-1, logits.size(-1)),
            hard_label.reshape(-1),
            weight=class_weights,
            reduction="none",
        ).reshape(hard_label.shape)
        loss = loss + (ce * mask).sum() / mask.sum().clamp_min(1.0) * ce_weight
    return loss
