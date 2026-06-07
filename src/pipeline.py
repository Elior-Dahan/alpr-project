"""End-to-end ALPR inference: detect -> preprocess -> read.

    image -> detect_plate (box) -> preprocess_plate (64x256 gray)
          -> CRNN -> ctc_greedy_decode -> plate text
"""

from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np

from src.detection import detect_plate, load_detector
from src.ocr import ctc_greedy_decode, load_ocr_model
from src.preprocessing import preprocess_plate


class ALPRPipeline:
    """Chains the detector, OpenCV preprocessing, and the CRNN reader."""

    def __init__(self, detector_path: str | Path, ocr_weights_path: str | Path):
        self.detector = load_detector(detector_path)
        self.ocr_model = load_ocr_model(ocr_weights_path)

    def run(self, image_path: str | Path) -> list[dict]:
        """Run the full pipeline on one image.

        Returns a one-element list (single plate per image) with the bbox,
        decoded text, and the preprocessed crop.
        """
        img = cv2.imread(str(image_path))
        if img is None:
            raise FileNotFoundError(f"could not read image: {image_path}")

        bbox = detect_plate(img, self.detector)
        crop = preprocess_plate(img, bbox)  # (64, 256) uint8

        batch = (crop.astype(np.float32) / 255.0)[None, ..., None]  # (1,64,256,1)
        probs = self.ocr_model.predict(batch, verbose=0)
        plate_text = ctc_greedy_decode(probs)[0]

        return [{"bbox": bbox, "plate_text": plate_text, "crop": crop}]
