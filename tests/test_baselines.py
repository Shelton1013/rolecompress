# -*- coding: utf-8 -*-
"""GPU-free tests for baseline frame-keep logic and benchmark adapters."""
import numpy as np

from rolecompress.baselines import _global_topk_to_local, saliency_frame_keep, uniform_frame_keep
from rolecompress.benchmarks import videomme, egoschema, _strip_letter_prefix, _letter


def test_global_topk_budget():
    scores = [[0.1, 0.9], [0.5], [0.2, 0.3, 0.8]]
    keep = _global_topk_to_local(scores, keep_total=2)
    total = sum(len(x) for x in keep)
    assert total == 2
    # the two highest (0.9 in seg0, 0.8 in seg2) should be chosen
    assert 1 in keep[0] and 2 in keep[2]


def test_global_topk_min_one():
    assert sum(len(x) for x in _global_topk_to_local([[0.1], [0.2]], keep_total=0)) >= 1


def test_saliency_keep_budget():
    segs = [[np.random.randint(0, 255, (16, 16, 3), dtype=np.uint8) for _ in range(4)] for _ in range(3)]
    keep = saliency_frame_keep(segs, keep_total=5)
    assert sum(len(x) for x in keep) == 5


def test_uniform_keep():
    segs = [[np.zeros((8, 8, 3), np.uint8)] * 6 for _ in range(2)]
    keep = uniform_frame_keep(segs, per_seg=2)
    assert all(len(x) == 2 for x in keep)


def test_strip_letter_prefix():
    assert _strip_letter_prefix("A. a cat") == "a cat"
    assert _strip_letter_prefix("B) dog") == "dog"
    assert _strip_letter_prefix("no prefix") == "no prefix"


def test_letter():
    assert _letter(0) == "A" and _letter(3) == "D"


def test_videomme_adapter_letter_answer(tmp_path):
    (tmp_path / "vid1.mp4").write_bytes(b"x")
    row = {"duration": "long", "videoID": "vid1", "question": "q?",
           "options": ["A. one", "B. two", "C. three", "D. four"], "answer": "C"}
    u = videomme(row, str(tmp_path))
    assert u["answer"] == "C" and u["choices"][0] == "one" and u["path"].endswith("vid1.mp4")


def test_videomme_filters_short(tmp_path):
    (tmp_path / "v.mp4").write_bytes(b"x")
    row = {"duration": "short", "videoID": "v", "question": "q", "options": ["A. a", "B. b"], "answer": "A"}
    assert videomme(row, str(tmp_path)) is None


def test_egoschema_index_answer(tmp_path):
    (tmp_path / "u1.mp4").write_bytes(b"x")
    row = {"q_uid": "u1", "question": "q",
           "option 0": "zero", "option 1": "one", "option 2": "two", "option 3": "three", "option 4": "four",
           "answer": 2}
    u = egoschema(row, str(tmp_path))
    assert u is not None and u["answer"] == "C" and u["choices"][2] == "two"
