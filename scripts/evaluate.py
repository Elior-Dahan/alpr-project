#!/usr/bin/env python
"""Evaluate one stage of the ALPR system on the test split.

    --stage detection  -> mean IoU + accuracy@IoU>=0.5
    --stage ocr         -> exact-match accuracy + CER (on cropped plates)
    --stage pipeline    -> full-plate exact-match on PREDICTED boxes + per-state

Usage:
    python scripts/evaluate.py --stage {detection|ocr|pipeline}
"""

from __future__ import annotations

import argparse
import csv
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import Levenshtein  # noqa: E402


def _cer(pred: str, true: str) -> float:
    if not true:
        return 0.0 if not pred else 1.0
    return Levenshtein.distance(pred, true) / len(true)


# --- Detection ------------------------------------------------------------

def evaluate_detection(detection_dir: Path, model_path: Path) -> None:
    from src.detection import load_detector, make_dataset

    model = load_detector(model_path)
    test_ds = make_dataset(detection_dir / "test.csv", batch_size=16)

    ious, n = [], 0
    for images, boxes in test_ds:
        preds = model.predict(images, verbose=0)
        gt = boxes.numpy()
        for p, t in zip(preds, gt):
            x0 = max(p[0], t[0]); y0 = max(p[1], t[1])
            x1 = min(p[2], t[2]); y1 = min(p[3], t[3])
            inter = max(0.0, x1 - x0) * max(0.0, y1 - y0)
            area_p = max(0.0, p[2] - p[0]) * max(0.0, p[3] - p[1])
            area_t = max(0.0, t[2] - t[0]) * max(0.0, t[3] - t[1])
            union = area_p + area_t - inter
            ious.append(inter / (union + 1e-7))
            n += 1
    ious = np.asarray(ious)
    print(f"Detection — samples: {n}")
    print(f"  mean IoU:          {ious.mean():.4f}")
    print(f"  accuracy@IoU>=0.5: {(ious >= 0.5).mean():.4f}")


# --- OCR (on pre-cropped plates) -----------------------------------------

def evaluate_ocr(labels_csv: Path, model_path: Path) -> None:
    from src.ocr import ctc_greedy_decode, load_ocr_model, make_dataset, _read_labels_csv

    model = load_ocr_model(model_path)
    paths, _ = _read_labels_csv(labels_csv, "test")
    # Recover ground-truth strings directly from the CSV for scoring.
    truths = []
    base = labels_csv.parent
    with labels_csv.open() as f:
        for row in csv.DictReader(f):
            if row["split"] == "test":
                truths.append(row["plate_text"])

    test_ds = make_dataset(labels_csv, "test", batch_size=32)
    preds: list[str] = []
    for images, _labels in test_ds:
        probs = model.predict(images, verbose=0)
        preds.extend(ctc_greedy_decode(probs))

    exact = np.mean([p == t for p, t in zip(preds, truths)])
    cer = np.mean([_cer(p, t) for p, t in zip(preds, truths)])
    print(f"OCR — samples: {len(truths)}")
    print(f"  exact-match: {exact:.4f}")
    print(f"  CER:         {cer:.4f}")


# --- End-to-end pipeline (predicted boxes) -------------------------------

def evaluate_pipeline(
    detection_dir: Path, detector_path: Path, ocr_path: Path
) -> None:
    from src.pipeline import ALPRPipeline

    pipe = ALPRPipeline(detector_path, ocr_path)

    correct = 0
    total = 0
    per_state = defaultdict(lambda: [0, 0])  # state -> [correct, total]
    with (detection_dir / "test.csv").open() as f:
        for row in csv.DictReader(f):
            true = row["plate_text"]
            try:
                result = pipe.run(row["image_path"])[0]
            except FileNotFoundError:
                continue
            pred = result["plate_text"]
            total += 1
            state = true[:2]
            per_state[state][1] += 1
            if pred == true:
                correct += 1
                per_state[state][0] += 1

    print(f"Pipeline — samples: {total}")
    print(f"  full-plate exact-match: {correct / total:.4f}" if total else "  no samples")
    print("  per-state accuracy:")
    for state in sorted(per_state):
        c, t = per_state[state]
        print(f"    {state}: {c}/{t} = {c / t:.3f}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate an ALPR stage.")
    parser.add_argument("--stage", required=True, choices=["detection", "ocr", "pipeline"])
    parser.add_argument("--detection-dir", default="datasets/detection", type=Path)
    parser.add_argument("--ocr-labels", default="datasets/ocr/labels.csv", type=Path)
    parser.add_argument("--detector", default="models/detection/detector.keras", type=Path)
    parser.add_argument("--ocr-model", default="models/ocr/crnn_best.keras", type=Path)
    args = parser.parse_args()

    if args.stage == "detection":
        evaluate_detection(args.detection_dir, args.detector)
    elif args.stage == "ocr":
        evaluate_ocr(args.ocr_labels, args.ocr_model)
    else:
        evaluate_pipeline(args.detection_dir, args.detector, args.ocr_model)


if __name__ == "__main__":
    main()
