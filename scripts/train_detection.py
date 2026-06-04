#!/usr/bin/env python
"""Train the MobileNetV2 plate detector (two-phase: Huber warmup -> Huber+GIoU).

Usage:
    python scripts/train_detection.py [--warmup-epochs 8] [--finetune-epochs 8]
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.detection import (  # noqa: E402
    FINETUNE_EPOCHS,
    FINETUNE_LR,
    WARMUP_EPOCHS,
    WARMUP_LR,
    train,
)


def main() -> None:
    parser = argparse.ArgumentParser(description="Train the plate detector.")
    parser.add_argument("--detection-dir", default="datasets/detection", type=Path)
    parser.add_argument("--model-out", default="models/detection/detector.keras", type=Path)
    parser.add_argument("--batch-size", default=16, type=int)
    parser.add_argument("--warmup-epochs", default=WARMUP_EPOCHS, type=int)
    parser.add_argument("--warmup-lr", default=WARMUP_LR, type=float)
    parser.add_argument("--finetune-epochs", default=FINETUNE_EPOCHS, type=int)
    parser.add_argument("--finetune-lr", default=FINETUNE_LR, type=float)
    args = parser.parse_args()

    train(
        detection_dir=args.detection_dir,
        model_out=args.model_out,
        batch_size=args.batch_size,
        warmup_epochs=args.warmup_epochs,
        warmup_lr=args.warmup_lr,
        finetune_epochs=args.finetune_epochs,
        finetune_lr=args.finetune_lr,
    )


if __name__ == "__main__":
    main()
