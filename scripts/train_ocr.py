#!/usr/bin/env python
"""Train the CRNN OCR model.

Usage:
    python scripts/train_ocr.py [--epochs 50] [--batch-size 32]
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.ocr import train  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser(description="Train the CRNN OCR model.")
    parser.add_argument("--labels-csv", default="datasets/ocr/labels.csv", type=Path)
    parser.add_argument("--model-out", default="models/ocr/crnn_best.keras", type=Path)
    parser.add_argument("--epochs", default=50, type=int)
    parser.add_argument("--batch-size", default=32, type=int)
    parser.add_argument("--lr", default=1e-3, type=float)
    args = parser.parse_args()

    train(
        labels_csv=args.labels_csv,
        model_out=args.model_out,
        epochs=args.epochs,
        batch_size=args.batch_size,
        lr=args.lr,
    )


if __name__ == "__main__":
    main()
