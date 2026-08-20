# -*- coding: utf-8 -*-
"""Backbone wrapper around a frozen video-LLM (default Qwen2.5-VL-7B-Instruct).

Responsibilities:
  1. score_probe(): run text-only / vision-only / joint passes for a segment probe and
     return margins -> used to build self-supervised role labels (pid_labels).
  2. answer(): run QA over a *role-allocated* set of frames (+ inline ASR text), with an
     optional LoRA adapter. This is the RoleCompress inference/training forward.
  3. pooled_segment_features(): cheap per-segment (visual, text) features for the router.

IMPORTANT — model-version-dependent bits are marked [VERIFY]. They are correct for
transformers>=4.49 Qwen2.5-VL; if you bump versions, re-check the flagged lines
(processor kwargs, the visual-token id, and get_rope_index). The frame-budget path (used
in all main experiments) does NOT touch positional encodings and is robust; the
token-merge path (optional) does and is flagged.

Everything here runs the backbone in eval() with torch.no_grad() except the LoRA-adapted
answer() during training.
"""
from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import torch

from .pid_labels import ProbeMargins, SegmentProbe
from .roles import Role, RoleBudget, allocate_frames


@dataclass
class BackboneConfig:
    model_id: str = "Qwen/Qwen2.5-VL-7B-Instruct"
    dtype: str = "bfloat16"
    device_map: str = "auto"
    attn_impl: str = "flash_attention_2"   # or "sdpa"
    max_pixels: int = 360 * 420            # per-frame token cap (keeps memory sane on long video)
    fps_sample: float = 1.0                # raw frames/sec before role allocation
    trust_remote_code: bool = True


class RoleCompressBackbone:
    def __init__(self, cfg: BackboneConfig, lora_adapter_path: Optional[str] = None):
        self.cfg = cfg
        from transformers import AutoProcessor
        # [VERIFY] class name for the chosen backbone
        from transformers import Qwen2_5_VLForConditionalGeneration as _Model

        dtype = getattr(torch, cfg.dtype)
        self.processor = AutoProcessor.from_pretrained(cfg.model_id, trust_remote_code=cfg.trust_remote_code)
        self.model = _Model.from_pretrained(
            cfg.model_id, torch_dtype=dtype, device_map=cfg.device_map,
            attn_implementation=cfg.attn_impl, trust_remote_code=cfg.trust_remote_code,
        )
        self.model.eval()
        for p in self.model.parameters():
            p.requires_grad_(False)

        if lora_adapter_path:
            from peft import PeftModel
            self.model = PeftModel.from_pretrained(self.model, lora_adapter_path)
            self.model.eval()

        self.device = next(self.model.parameters()).device
        self.tokenizer = self.processor.tokenizer

    # ------------------------------------------------------------------ scoring
    @torch.no_grad()
    def score_probe(self, probe: SegmentProbe, frames: Sequence[np.ndarray]) -> ProbeMargins:
        """Return margins for text-only / vision-only / joint answering of a segment probe.

        `frames`: list of HxWx3 uint8 arrays for THIS segment (already sampled).
        Margin definition:
          - MCQ  : logit-margin = logprob(gold letter) - max logprob(other letters).
          - open : length-normalized logprob(gold answer).
        """
        m_text = self._score_pass(probe, frames=None, use_text=True)
        m_vis = self._score_pass(probe, frames=frames, use_text=False)
        m_joint = self._score_pass(probe, frames=frames, use_text=True)
        return ProbeMargins(m_text=m_text, m_vision=m_vis, m_joint=m_joint)

    @torch.no_grad()
    def _score_pass(self, probe: SegmentProbe, frames: Optional[Sequence[np.ndarray]], use_text: bool) -> float:
        content = []
        if frames is not None and len(frames) > 0:
            content.append({"type": "video", "video": list(frames)})  # [VERIFY] processor accepts list of np frames
        ctx = probe.asr_text if use_text else ""
        prompt = _probe_prompt(probe.question, ctx, probe.choices)
        content.append({"type": "text", "text": prompt})
        messages = [{"role": "user", "content": content}]
        if probe.choices:  # MCQ: score letter logits at first decoded position
            return self._mcq_margin(messages, probe)
        return self._open_loglik(messages, probe.gold)

    def _build_inputs(self, messages, gold_text: Optional[str] = None):
        text = self.processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        # [VERIFY] qwen_vl_utils.process_vision_info extracts image/video inputs from messages
        from qwen_vl_utils import process_vision_info
        image_inputs, video_inputs = process_vision_info(messages)
        inputs = self.processor(
            text=[text], images=image_inputs, videos=video_inputs,
            padding=True, return_tensors="pt",
        )
        return inputs.to(self.device)

    @torch.no_grad()
    def _mcq_margin(self, messages, probe: SegmentProbe) -> float:
        inputs = self._build_inputs(messages)
        out = self.model(**inputs)
        next_logits = out.logits[0, -1]  # logits for the first answer token
        letters = [chr(ord("A") + i) for i in range(len(probe.choices))]
        # token id of each option letter (leading space variant handled)
        ids = [self._letter_id(l) for l in letters]
        scores = torch.stack([next_logits[i] for i in ids])
        gold_idx = letters.index(probe.gold) if probe.gold in letters else 0
        logp = torch.log_softmax(scores, dim=-1)
        gold = logp[gold_idx]
        others = torch.cat([logp[:gold_idx], logp[gold_idx + 1:]])
        margin = (gold - others.max()).item() if others.numel() else gold.item()
        return float(margin)

    def _letter_id(self, letter: str) -> int:
        # prefer the tokenization used right after the assistant header
        cand = self.tokenizer.encode(letter, add_special_tokens=False)
        return cand[0]

    @torch.no_grad()
    def _open_loglik(self, messages, gold_text: str) -> float:
        # teacher-forced length-normalized logprob of the gold answer
        prompt_inputs = self._build_inputs(messages)
        gold_ids = self.tokenizer(gold_text, add_special_tokens=False, return_tensors="pt").input_ids.to(self.device)
        full_ids = torch.cat([prompt_inputs["input_ids"], gold_ids], dim=1)
        attn = torch.ones_like(full_ids)
        # visual inputs must be carried along for the joint/vision passes
        model_kwargs = {k: v for k, v in prompt_inputs.items() if k not in ("input_ids", "attention_mask")}
        out = self.model(input_ids=full_ids, attention_mask=attn, **model_kwargs)
        logits = out.logits[0, :-1]
        targets = full_ids[0, 1:]
        start = prompt_inputs["input_ids"].shape[1] - 1
        sel_logits = logits[start:]
        sel_targets = targets[start:]
        logp = torch.log_softmax(sel_logits, dim=-1)
        tok_lp = logp[torch.arange(sel_targets.numel(), device=self.device), sel_targets]
        return float(tok_lp.mean().item())

    # ------------------------------------------------------ router features
    @torch.no_grad()
    def pooled_segment_features(self, frames_per_seg: Sequence[Sequence[np.ndarray]], asr_per_seg: Sequence[str]):
        """Return (vis (T,d_visual), txt (T,d_text), scal (T,4)) cheap features for the router.

        vis: mean vision-encoder feature over a couple frames of the segment.
        txt: mean LLM input-embedding of the segment's ASR text.
        scal: [seg_seconds_norm, has_speech, visual_var, asr_len_norm].
        Cache these to disk (they are backbone-frozen) so router training is fast.
        """
        vis_feats, txt_feats, scal_feats = [], [], []
        for frames, asr in zip(frames_per_seg, asr_per_seg):
            vis_feats.append(self._encode_frames_mean(frames))
            txt_feats.append(self._encode_text_mean(asr))
            var = float(np.stack(frames).astype(np.float32).var()) if len(frames) else 0.0
            scal = torch.tensor([
                1.0,                                   # seg_seconds_norm (filled by caller if known)
                1.0 if asr.strip() else 0.0,           # has_speech
                min(1.0, var / 5000.0),                # visual_var (rough)
                min(1.0, len(asr) / 200.0),            # asr_len_norm
            ], device=self.device)
            scal_feats.append(scal)
        return torch.stack(vis_feats), torch.stack(txt_feats), torch.stack(scal_feats)

    @torch.no_grad()
    def _encode_frames_mean(self, frames: Sequence[np.ndarray]) -> torch.Tensor:
        if not frames:
            return torch.zeros(self.model.config.vision_config.hidden_size, device=self.device)
        # [VERIFY] use the visual tower directly; here we approximate with the processor+visual merge.
        content = [{"type": "video", "video": list(frames)}, {"type": "text", "text": "."}]
        inputs = self._build_inputs([{"role": "user", "content": content}])
        # Pull the vision features via the model's visual module if exposed:
        try:
            pv = inputs["pixel_values_videos"]; grid = inputs["video_grid_thw"]
            feats = self.model.visual(pv, grid_thw=grid)  # (Nvis, d) [VERIFY name]
            return feats.mean(0).float()
        except Exception:
            # fallback: mean of the input embeddings at visual positions
            emb = self.model.get_input_embeddings()(inputs["input_ids"])
            return emb[0].mean(0).float()

    @torch.no_grad()
    def _encode_text_mean(self, text: str) -> torch.Tensor:
        if not text.strip():
            return torch.zeros(self.model.config.hidden_size, device=self.device)
        ids = self.tokenizer(text, add_special_tokens=False, return_tensors="pt").input_ids.to(self.device)
        emb = self.model.get_input_embeddings()(ids)
        return emb[0].mean(0).float()

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
        """Assemble the role-allocated multimodal prompt.

        - Frames: per-segment kept frames according to role (frame-budget path).
        - Text: ASR of REDUNDANT/UNIQUE_TEXT segments is injected inline (with timestamps)
          so their content is preserved as language; synergy/unique-visual ASR is also kept
          (text is cheap) but their visual frames are the point.
        Returns a `processor(...)` inputs dict on device, plus the effective visual token count.
        """
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
            content.append({"type": "video", "video": kept_frames})
        prompt = _qa_prompt(question, transcript, choices)
        content.append({"type": "text", "text": prompt})
        messages = [{"role": "user", "content": content}]
        inputs = self._build_inputs(messages)
        vtok = self._visual_token_count(inputs)
        return inputs, vtok, messages

    def _visual_token_count(self, inputs) -> int:
        grid = inputs.get("video_grid_thw", None)
        if grid is None:
            return 0
        # tokens after the 2x2 spatial merge Qwen2.5-VL applies: prod(t,h,w)/merge^2
        merge = getattr(self.model.config.vision_config, "spatial_merge_size", 2)
        total = 0
        for t, h, w in grid.tolist():
            total += (t * h * w) // (merge * merge)
        return int(total)

    def answer_logits(self, inputs):
        """Forward for QA (LoRA-adapted at train time). Returns logits."""
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
        opts = "\n".join(f"{chr(ord('A') + i)}. {c}" for i, c in enumerate(choices))
        parts.append(opts)
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
