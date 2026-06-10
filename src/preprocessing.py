"""OpenCV preprocessing: crop -> rectify (4-point warp) -> grayscale -> CLAHE.

Bridges detection and OCR. The detector emits an axis-aligned box; this module
expands and crops it, attempts to recover the plate's true four corners for a
perspective rectification, then normalizes contrast so the OCR model sees a
consistent (OCR_HEIGHT, OCR_WIDTH) grayscale image.
"""

from __future__ import annotations

import cv2
import numpy as np

from src.data import OCR_HEIGHT, OCR_WIDTH

BBox = tuple[int, int, int, int]  # (xmin, ymin, xmax, ymax)


def order_corners(pts: np.ndarray) -> np.ndarray:
    """Order four points as [top-left, top-right, bottom-right, bottom-left].

    TL = min(x+y), BR = max(x+y), TR = max(x-y), BL = min(x-y).
    (Top-right has large x and small y, so x-y is largest there.)
    """
    pts = pts.reshape(4, 2).astype(np.float32)
    s = pts.sum(axis=1)
    d = pts[:, 0] - pts[:, 1]  # x - y
    return np.array(
        [
            pts[np.argmin(s)],  # top-left
            pts[np.argmax(d)],  # top-right
            pts[np.argmax(s)],  # bottom-right
            pts[np.argmin(d)],  # bottom-left
        ],
        dtype=np.float32,
    )


def find_plate_corners(crop_bgr: np.ndarray) -> np.ndarray | None:
    """Try to find the 4 corners of the plate within ``crop_bgr``.

    Canny edges -> dilate -> external contours -> the largest contour that
    covers >=20% of the crop -> ``approxPolyDP``. If that yields 4 points use
    them; otherwise fall back to ``minAreaRect``. Returns ordered corners
    (4, 2) float32, or ``None`` if nothing plate-like is found.
    """
    gray = cv2.cvtColor(crop_bgr, cv2.COLOR_BGR2GRAY)
    gray = cv2.GaussianBlur(gray, (5, 5), 0)
    edges = cv2.Canny(gray, 50, 150)
    edges = cv2.dilate(edges, np.ones((3, 3), np.uint8), iterations=1)

    contours, _ = cv2.findContours(
        edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
    )
    if not contours:
        return None

    crop_area = crop_bgr.shape[0] * crop_bgr.shape[1]
    largest = max(contours, key=cv2.contourArea)
    if cv2.contourArea(largest) < 0.20 * crop_area:
        return None

    peri = cv2.arcLength(largest, True)
    approx = cv2.approxPolyDP(largest, 0.02 * peri, True)
    if len(approx) == 4:
        return order_corners(approx)

    # Fall back to the minimum-area rotated rectangle.
    box = cv2.boxPoints(cv2.minAreaRect(largest))
    return order_corners(box)


def four_point_transform(
    image: np.ndarray, corners: np.ndarray, output_size: tuple[int, int] = (OCR_HEIGHT, OCR_WIDTH)
) -> np.ndarray:
    """Warp the quadrilateral ``corners`` to a frontal ``output_size`` (H, W)."""
    h, w = output_size
    dst = np.array([[0, 0], [w - 1, 0], [w - 1, h - 1], [0, h - 1]], dtype=np.float32)
    matrix = cv2.getPerspectiveTransform(order_corners(corners), dst)
    return cv2.warpPerspective(image, matrix, (w, h))


def preprocess_plate(
    full_image: np.ndarray,
    bbox: BBox,
    output_size: tuple[int, int] = (OCR_HEIGHT, OCR_WIDTH),
    margin: float = 0.10,
) -> np.ndarray:
    """Full preprocessing pipeline -> (H, W) uint8 grayscale.

    1. Expand ``bbox`` by ``margin`` and clamp to image bounds.
    2. Crop; attempt corner detection and perspective rectification.
       Fall back to a direct resize if no plate-like quad is found.
    3. Grayscale, then CLAHE (clipLimit=2.0, tile=(8, 8)).
    """
    h_img, w_img = full_image.shape[:2]
    xmin, ymin, xmax, ymax = bbox
    bw, bh = xmax - xmin, ymax - ymin
    ex, ey = int(bw * margin), int(bh * margin)
    x0 = max(0, xmin - ex)
    y0 = max(0, ymin - ey)
    x1 = min(w_img, xmax + ex)
    y1 = min(h_img, ymax + ey)
    crop = full_image[y0:y1, x0:x1]

    if crop.size == 0:
        crop = full_image[ymin:ymax, xmin:xmax]

    h, w = output_size
    corners = find_plate_corners(crop)
    if corners is not None:
        warped = four_point_transform(crop, corners, output_size)
    else:
        warped = cv2.resize(crop, (w, h), interpolation=cv2.INTER_CUBIC)

    gray = cv2.cvtColor(warped, cv2.COLOR_BGR2GRAY)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    return clahe.apply(gray)
