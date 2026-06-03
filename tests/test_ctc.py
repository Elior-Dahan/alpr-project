"""CTC encode -> decode round-trip test (requires keras + TF backend)."""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

keras = pytest.importorskip("keras")

from src.ocr import (  # noqa: E402
    NUM_CLASSES,
    TIME_STEPS,
    build_crnn,
    ctc_greedy_decode,
    encode_label,
)


def _one_hot_probs(label_indices: list[int], time_steps: int = TIME_STEPS) -> np.ndarray:
    """Lay out non-blank label indices across timesteps, blanks elsewhere.

    Produces a clean signal that a greedy CTC decoder must collapse back to
    the original string (one char per timestep, separated by blanks).
    """
    chars = [i for i in label_indices if i > 0]
    probs = np.zeros((1, time_steps, NUM_CLASSES), dtype=np.float32)
    probs[0, :, 0] = 1.0  # default everything to blank
    t = 0
    for ci in chars:
        if t >= time_steps:
            break
        probs[0, t, 0] = 0.0
        probs[0, t, ci] = 1.0
        t += 2  # leave a blank gap so repeated chars are not merged
    return probs


def test_encode_label_padding():
    enc = encode_label("KL45")
    assert len(enc) == 12  # MAX_LABEL_LEN
    assert enc[:4] != [0, 0, 0, 0]
    assert enc[4:] == [0] * 8


def test_ctc_round_trip():
    text = "KL45C4411"
    probs = _one_hot_probs(encode_label(text))
    decoded = ctc_greedy_decode(probs)
    assert decoded[0] == text


def test_crnn_output_shape():
    model = build_crnn()
    assert model.output_shape == (None, TIME_STEPS, NUM_CLASSES)
