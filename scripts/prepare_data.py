#!/usr/bin/env python
"""Run the full data pipeline and print real counts.

    parse_voc -> deduplicate -> plate_aware_split
              -> build_detection_manifest + build_ocr_dataset

Usage:
    python scripts/prepare_data.py [--data-root data] [--out datasets]
"""

from __future__ import annotations

import argparse
import sys
from collections import Counter
from pathlib import Path

# Allow running as a plain script (python scripts/prepare_data.py).
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.data import (  # noqa: E402
    build_detection_manifest,
    build_ocr_dataset,
    deduplicate,
    parse_voc,
    plate_aware_split,
)


def main() -> None:
    parser = argparse.ArgumentParser(description="Build ALPR datasets from VOC data.")
    parser.add_argument("--data-root", default="data", type=Path)
    parser.add_argument("--out", default="datasets", type=Path)
    parser.add_argument("--seed", default=42, type=int)
    args = parser.parse_args()

    print(f"Parsing VOC annotations under {args.data_root} ...")
    anns = parse_voc(args.data_root)
    print(f"  clean annotations: {len(anns)}")
    by_source = Counter(a.source for a in anns)
    for src, n in sorted(by_source.items()):
        print(f"    {src}: {n}")

    deduped = deduplicate(anns)
    print(f"  after video dedup: {len(deduped)}")
    by_source_dd = Counter(a.source for a in deduped)
    for src, n in sorted(by_source_dd.items()):
        print(f"    {src}: {n}")

    splits = plate_aware_split(deduped, seed=args.seed)
    print("  split sizes:")
    for split, items in splits.items():
        print(f"    {split}: {len(items)}")

    # Sanity check: no plate leaks across splits.
    plate_sets = {s: {a.plate_text for a in items} for s, items in splits.items()}
    leak = (
        (plate_sets["train"] & plate_sets["val"])
        | (plate_sets["train"] & plate_sets["test"])
        | (plate_sets["val"] & plate_sets["test"])
    )
    print(f"  plate leakage across splits: {len(leak)} (expected 0)")

    det_dir = args.out / "detection"
    print(f"Writing detection manifests to {det_dir} ...")
    build_detection_manifest(splits, det_dir)

    ocr_dir = args.out / "ocr"
    print(f"Building OCR crops in {ocr_dir} ...")
    labels_path = build_ocr_dataset(splits, ocr_dir)
    n_rows = sum(1 for _ in labels_path.open()) - 1
    print(f"  OCR samples written: {n_rows}  ({labels_path})")

    print("Done.")


if __name__ == "__main__":
    main()
