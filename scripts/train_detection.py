#!/usr/bin/env python
"""Train the MobileNetV2 plate detector.

Usage:
    python scripts/train_detection.py [--epochs 100] [--batch-size 16]
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.detection import train  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser(description="Train the plate detector.")
    parser.add_argument("--detection-dir", default="datasets/detection", type=Path)
    parser.add_argument("--model-out", default="models/detection/detector.keras", type=Path)
    parser.add_argument("--epochs", default=100, type=int)
    parser.add_argument("--batch-size", default=16, type=int)
    parser.add_argument("--lr", default=1e-3, type=float)
    args = parser.parse_args()

    train(
        detection_dir=args.detection_dir,
        model_out=args.model_out,
        epochs=args.epochs,
        batch_size=args.batch_size,
        lr=args.lr,
    )


if __name__ == "__main__":
    main()
