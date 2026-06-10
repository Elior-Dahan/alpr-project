"""Tests for OpenCV preprocessing (no GPU/TF needed)."""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.data import OCR_HEIGHT, OCR_WIDTH
from src.preprocessing import four_point_transform, order_corners, preprocess_plate


def test_order_corners_from_shuffled():
    # Canonical TL, TR, BR, BL.
    canonical = np.array([[0, 0], [100, 0], [100, 50], [0, 50]], dtype=np.float32)
    shuffled = canonical[[2, 0, 3, 1]]  # arbitrary permutation
    ordered = order_corners(shuffled)
    assert np.allclose(ordered, canonical)


def test_four_point_transform_shape():
    img = np.zeros((120, 200, 3), np.uint8)
    corners = np.array([[10, 10], [180, 15], [185, 95], [12, 90]], dtype=np.float32)
    out = four_point_transform(img, corners, output_size=(64, 256))
    assert out.shape == (64, 256, 3)


def test_preprocess_plate_output_shape():
    img = np.random.randint(0, 255, (300, 400, 3), dtype=np.uint8)
    bbox = (50, 60, 200, 120)
    out = preprocess_plate(img, bbox)
    assert out.shape == (OCR_HEIGHT, OCR_WIDTH)
    assert out.dtype == np.uint8
