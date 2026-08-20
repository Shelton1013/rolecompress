# -*- coding: utf-8 -*-
"""ASR transcript extraction + alignment of speech to video segments.

Uses faster-whisper. Output: list of {start, end, text}. Then align_to_segments()
buckets utterances into the video segments (by temporal overlap) to build the per-segment
ASR text used as the cheap dense modality.
"""
from __future__ import annotations

from typing import Dict, List, Sequence

_model = None


def _get_model(size: str = "large-v3", device: str = "auto", compute_type: str = "float16"):
    global _model
    if _model is None:
        from faster_whisper import WhisperModel
        if device == "auto":
            try:
                _model = WhisperModel(size, device="cuda", compute_type=compute_type)
            except Exception:
                _model = WhisperModel(size, device="cpu", compute_type="int8")
        else:
            _model = WhisperModel(size, device=device, compute_type=compute_type)
    return _model


def transcribe(audio_or_video_path: str, language: str = None, size: str = "large-v3") -> List[Dict]:
    model = _get_model(size)
    segs, _info = model.transcribe(
        audio_or_video_path, language=language, vad_filter=True, beam_size=1,
        condition_on_previous_text=False, word_timestamps=False,
    )
    return [{"start": round(s.start, 2), "end": round(s.end, 2), "text": s.text.strip()} for s in segs]


def align_to_segments(utterances: Sequence[Dict], seg_spans: Sequence) -> List[str]:
    """seg_spans: objects with .start/.end (segment.Segment). Returns per-segment ASR text
    (utterances overlapping the segment, concatenated)."""
    out = []
    for seg in seg_spans:
        a, b = seg.start, seg.end
        txt = " ".join(u["text"] for u in utterances if u["end"] > a and u["start"] < b and u["text"])
        out.append(txt.strip())
    return out
