# -*- coding: utf-8 -*-
"""Backbone wrapper around a frozen video-LLM.

Default backbone: **Qwen/Qwen3-VL-8B-Instruct** (2026). The wrapper is family-aware, so
Qwen2.5-VL and LLaVA-Video also work by changing `model_id` (see FAMILY detection).

Responsibilities:
  1. score_probe(): text-only / vision-only / joint passes for a segment probe -> margins
     (used to build self-supervised role labels).
  2. build_answer_inputs()/answer: QA over a *role-allocated* frame set (+ inline ASR text).
  3. pooled_segment_features(): cheap per-segment (visual, text) features for the router,
     both in the LLM embedding space (dim = llm_hidden) so the router is backbone-robust.

Version notes:
  - Qwen3-VL needs transformers >= 4.57 (or `pip install git+https://github.com/huggingface/transformers`).
    Input building uses the new `processor.apply_chat_template(..., tokenize=True, return_dict=True)`
    pattern and does NOT use qwen_vl_utils.
  - Qwen2.5-VL uses the older `process_vision_info` + `processor(text=, images=, videos=)` pattern
    (kept for backward compat; requires `qwen-vl-utils`).
  - Lines whose behavior depends on the exact model/transformers version are marked [VERIFY].
    The frame-budget path used in all main experiments does NOT touch positional encodings.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import torch
from PIL import Image

from .pid_labels import ProbeMargins, SegmentProbe
from .roles import Role, RoleBudget, allocate_frames


@dataclass
class BackboneConfig:
    model_id: str = "Qwen/Qwen3-VL-8B-Instruct"   # 2026 default; 4B for fast iteration, 30B-A3B(MoE)/32B for strength
    dtype: str = "bfloat16"
    device_map: str = "auto"
    attn_impl: str = "sdpa"                 # "flash_attention_2" if flash-attn built
    fps_sample: float = 1.0
    trust_remote_code: bool = True


def _detect_family(model_id: str) -> str:
    m = model_id.lower()
    if "qwen3-vl" in m or "qwen3vl" in m:
        return "qwen3vl"
    if "qwen2.5-vl" in m or "qwen2_5_vl" in m or "qwen2-vl" in m:
        return "qwen2_5_vl"
    if "llava" in m:
        return "llava_video"
    return "qwen3vl"  # sensible default


def _load_model(model_id: str, family: str, dtype, cfg: BackboneConfig):
    kw = dict(torch_dtype=dtype, device_map=cfg.device_map,
              attn_implementation=cfg.attn_impl, trust_remote_code=cfg.trust_remote_code)
    if family == "qwen3vl":
        try:
            from transformers import Qwen3VLForConditionalGeneration as M      # dense
        except ImportError:
            from transformers import Qwen3VLMoeForConditionalGeneration as M   # MoE (30B-A3B / 235B-A22B)
        # MoE ids need the Moe class:
        if "a3b" in model_id.lower() or "a22b" in model_id.lower():
            from transformers import Qwen3VLMoeForConditionalGeneration as M
        return M.from_pretrained(model_id, **kw)
    if family == "qwen2_5_vl":
        from transformers import Qwen2_5_VLForConditionalGeneration as M
        return M.from_pretrained(model_id, **kw)
    if family == "llava_video":
        from transformers import LlavaOnevisionForConditionalGeneration as M   # [VERIFY] class for your LLaVA-Video ckpt
        return M.from_pretrained(model_id, **kw)
    from transformers import AutoModelForImageTextToText as M
    return M.from_pretrained(model_id, **kw)


class RoleCompressBackbone:
    def __init__(self, cfg: BackboneConfig, lora_adapter_path: Optional[str] = None):
        self.cfg = cfg
        self.family = _detect_family(cfg.model_id)
        from transformers import AutoProcessor
        dtype = getattr(torch, cfg.dtype)
        self.processor = AutoProcessor.from_pretrained(cfg.model_id, trust_remote_code=cfg.trust_remote_code)
        self.model = _load_model(cfg.model_id, self.family, dtype, cfg)
        self.model.eval()
        for p in self.model.parameters():
            p.requires_grad_(False)
        if lora_adapter_path:
            from peft import PeftModel
            self.model = PeftModel.from_pretrained(self.model, lora_adapter_path)
            self.model.eval()
        self.device = next(self.model.parameters()).device
        self.tokenizer = self.processor.tokenizer

    # ---------------------------------------------------------- dims
    @property
    def llm_hidden(self) -> int:
        cfg = self.model.config
        tc = getattr(cfg, "text_config", None)
        return int(getattr(tc, "hidden_size", None) or getattr(cfg, "hidden_size"))

    def feature_dims(self) -> Tuple[int, int]:
        """(d_visual, d_text) for the router. Both live in LLM embedding space -> equal to llm_hidden."""
        h = self.llm_hidden
        return h, h

    # ------------------------------------------------------ input building
    @staticmethod
    def _to_pil(frames: Sequence[np.ndarray]) -> List[Image.Image]:
        return [Image.fromarray(f) if not isinstance(f, Image.Image) else f for f in frames]

    def _video_content(self, frames: Sequence[np.ndarray]) -> dict:
        # both Qwen3-VL and Qwen2.5-VL accept a list of PIL frames under the "video" key
        return {"type": "video", "video": self._to_pil(frames)}

    def _build_inputs(self, messages):
        """Return a processor inputs dict on device. Branches on backbone family."""
        if self.family == "qwen3vl":
            inputs = self.processor.apply_chat_template(
                messages, tokenize=True, add_generation_prompt=True,
                return_dict=True, return_tensors="pt",
            )
            inputs.pop("token_type_ids", None)  # per Qwen3-VL docs
            return inputs.to(self.device)
        if self.family == "qwen2_5_vl":
            text = self.processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
            from qwen_vl_utils import process_vision_info
            image_inputs, video_inputs = process_vision_info(messages)
            inputs = self.processor(text=[text], images=image_inputs, videos=video_inputs,
                                    padding=True, return_tensors="pt")
            return inputs.to(self.device)
        # generic (llava-video etc): try the new unified path
        inputs = self.processor.apply_chat_template(
            messages, tokenize=True, add_generation_prompt=True, return_dict=True, return_tensors="pt")
        return inputs.to(self.device)

    # ------------------------------------------------------------ scoring
    @torch.no_grad()
    def score_probe(self, probe: SegmentProbe, frames: Sequence[np.ndarray]) -> ProbeMargins:
        m_text = self._score_pass(probe, frames=None, use_text=True)
        m_vis = self._score_pass(probe, frames=frames, use_text=False)
        m_joint = self._score_pass(probe, frames=frames, use_text=True)
        return ProbeMargins(m_text=m_text, m_vision=m_vis, m_joint=m_joint)

    @torch.no_grad()
    def _score_pass(self, probe: SegmentProbe, frames: Optional[Sequence[np.ndarray]], use_text: bool) -> float:
        content = []
        if frames is not None and len(frames) > 0:
            content.append(self._video_content(frames))
        ctx = probe.asr_text if use_text else ""
        content.append({"type": "text", "text": _probe_prompt(probe.question, ctx, probe.choices)})
        messages = [{"role": "user", "content": content}]
        if probe.choices:
            return self._mcq_margin(messages, probe)
        return self._open_loglik(messages, probe.gold)

    @torch.no_grad()
    def _mcq_margin(self, messages, probe: SegmentProbe) -> float:
        inputs = self._build_inputs(messages)
        out = self.model(**inputs)
        next_logits = out.logits[0, -1]
        letters = [chr(ord("A") + i) for i in range(len(probe.choices))]
        ids = [self._letter_id(l) for l in letters]
        scores = torch.stack([next_logits[i] for i in ids])
        gold_idx = letters.index(probe.gold) if probe.gold in letters else 0
        logp = torch.log_softmax(scores, dim=-1)
        gold = logp[gold_idx]
        others = torch.cat([logp[:gold_idx], logp[gold_idx + 1:]])
        return float((gold - others.max()).item() if others.numel() else gold.item())

    def _letter_id(self, letter: str) -> int:
        return self.tokenizer.encode(letter, add_special_tokens=False)[0]

    @torch.no_grad()
    def _open_loglik(self, messages, gold_text: str) -> float:
        prompt_inputs = self._build_inputs(messages)
        gold_ids = self.tokenizer(gold_text, add_special_tokens=False, return_tensors="pt").input_ids.to(self.device)
        full_ids = torch.cat([prompt_inputs["input_ids"], gold_ids], dim=1)
        model_kwargs = {k: v for k, v in prompt_inputs.items() if k not in ("input_ids", "attention_mask")}
        out = self.model(input_ids=full_ids, attention_mask=torch.ones_like(full_ids), **model_kwargs)
        logits = out.logits[0, :-1]
        targets = full_ids[0, 1:]
        start = prompt_inputs["input_ids"].shape[1] - 1
        logp = torch.log_softmax(logits[start:], dim=-1)
        tok_lp = logp[torch.arange(targets[start:].numel(), device=self.device), targets[start:]]
        return float(tok_lp.mean().item())

    # ------------------------------------------------ router features (LLM space)
    @torch.no_grad()
    def pooled_segment_features(self, frames_per_seg: Sequence[Sequence[np.ndarray]], asr_per_seg: Sequence[str]):
        """(vis (T,H), txt (T,H), scal (T,4)) with H = llm_hidden. Backbone-frozen -> cache to disk."""
        H = self.llm_hidden
        vis_feats, txt_feats, scal_feats = [], [], []
        for frames, asr in zip(frames_per_seg, asr_per_seg):
            vis_feats.append(self._encode_frames_mean(frames, H))
            txt_feats.append(self._encode_text_mean(asr, H))
            var = float(np.stack(frames).astype(np.float32).var()) if len(frames) else 0.0
            scal_feats.append(torch.tensor([
                1.0,
                1.0 if asr.strip() else 0.0,
                min(1.0, var / 5000.0),
                min(1.0, len(asr) / 200.0),
            ], device=self.device))
        return torch.stack(vis_feats), torch.stack(txt_feats), torch.stack(scal_feats)

    @torch.no_grad()
    def _encode_frames_mean(self, frames: Sequence[np.ndarray], H: int) -> torch.Tensor:
        """Mean of the LLM input embeddings over the visual tokens of the segment.
        Robust across backbones (no vision-tower internals). [VERIFY] visual-token id lookup."""
        if not frames:
            return torch.zeros(H, device=self.device)
        content = [self._video_content(frames), {"type": "text", "text": "."}]
        inputs = self._build_inputs([{"role": "user", "content": content}])
        emb = self.model.get_input_embeddings()(inputs["input_ids"])[0]  # (L, H)
        vis_id = self._visual_token_id()
        if vis_id is not None:
            mask = (inputs["input_ids"][0] == vis_id)
            if mask.any():
                return emb[mask].mean(0).float()
        return emb.mean(0).float()  # fallback: pool whole (visual-dominated) sequence

    @torch.no_grad()
    def _encode_text_mean(self, text: str, H: int) -> torch.Tensor:
        if not text.strip():
            return torch.zeros(H, device=self.device)
        ids = self.tokenizer(text, add_special_tokens=False, return_tensors="pt").input_ids.to(self.device)
        return self.model.get_input_embeddings()(ids)[0].mean(0).float()

    def _visual_token_id(self) -> Optional[int]:
        cfg = self.model.config
        for attr in ("video_token_id", "image_token_id", "video_token_index", "image_token_index"):
            v = getattr(cfg, attr, None)
            if isinstance(v, int):
                return v
        for tok in ("<|video_pad|>", "<|image_pad|>", "<video>", "<image>"):
            tid = self.tokenizer.convert_tokens_to_ids(tok)
            if isinstance(tid, int) and tid >= 0:
                return tid
        return None

    # --------------------------------------------------------- QA forward
    def build_answer_inputs(
        self,
        question: str,
        choices: Optional[List[str]],
        seg_frames: Sequence[Sequence[np.ndarray]],
        seg_roles: Sequence[Role],
        seg_asr: Sequence[str],
        budget: RoleBudget,
    ):
        """Assemble the role-allocated multimodal prompt. Returns (inputs, visual_token_count, messages)."""
        seg_counts = [len(f) for f in seg_frames]
        keep_local = allocate_frames(seg_roles, seg_counts, budget)
        kept_frames: List[np.ndarray] = []
        transcript_lines: List[str] = []
        for i, (frames, role, asr, local_idx) in enumerate(zip(seg_frames, seg_roles, seg_asr, keep_local)):
            for li in local_idx:
                kept_frames.append(frames[li])
            if asr.strip():
                tag = "[speech]" if role in (Role.REDUNDANT, Role.UNIQUE_TEXT) else "[audio]"
                transcript_lines.append(f"{tag} seg{i}: {asr.strip()}")
        transcript = "\n".join(transcript_lines)

        content = []
        if kept_frames:
            content.append(self._video_content(kept_frames))
        content.append({"type": "text", "text": _qa_prompt(question, transcript, choices)})
        messages = [{"role": "user", "content": content}]
        inputs = self._build_inputs(messages)
        return inputs, self._visual_token_count(inputs), messages

    def _visual_token_count(self, inputs) -> int:
        grid = inputs.get("video_grid_thw", None)
        if grid is not None:
            merge = getattr(getattr(self.model.config, "vision_config", self.model.config), "spatial_merge_size", 2)
            return int(sum((t * h * w) // (merge * merge) for t, h, w in grid.tolist()))
        vis_id = self._visual_token_id()
        if vis_id is not None and "input_ids" in inputs:
            return int((inputs["input_ids"][0] == vis_id).sum().item())
        return 0

    def answer_logits(self, inputs):
        return self.model(**inputs).logits

    @torch.no_grad()
    def generate_answer(self, inputs, max_new_tokens: int = 64) -> str:
        gen = self.model.generate(**inputs, max_new_tokens=max_new_tokens, do_sample=False)
        trimmed = gen[:, inputs["input_ids"].shape[1]:]
        return self.processor.batch_decode(trimmed, skip_special_tokens=True)[0].strip()


# --------------------------------------------------------------------- prompts
def _probe_prompt(question: str, ctx: str, choices: Optional[List[str]]) -> str:
    parts = []
    if ctx.strip():
        parts.append(f"Transcript: {ctx.strip()}")
    parts.append(f"Question: {question.strip()}")
    if choices:
        parts.append("\n".join(f"{chr(ord('A') + i)}. {c}" for i, c in enumerate(choices)))
        parts.append("Answer with the letter only.")
    else:
        parts.append("Answer concisely.")
    return "\n".join(parts)


def _qa_prompt(question: str, transcript: str, choices: Optional[List[str]]) -> str:
    parts = []
    if transcript.strip():
        parts.append("Speech/audio transcript (with segment tags):")
        parts.append(transcript.strip())
    parts.append(f"Question: {question.strip()}")
    if choices:
        parts.append("\n".join(f"{chr(ord('A') + i)}. {c}" for i, c in enumerate(choices)))
        parts.append("Answer with the letter only.")
    else:
        parts.append("Answer concisely based on the video and transcript.")
    return "\n".join(parts)
