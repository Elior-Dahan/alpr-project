"""Data pipeline for the Indian ALPR system.

Parses Pascal VOC annotations, filters noise, deduplicates video frames,
performs a plate-aware train/val/test split, and builds the per-stage
datasets:

  * detection -> a CSV manifest of normalized boxes (no image copying)
  * ocr       -> cropped grayscale plates (OCR_HEIGHT x OCR_WIDTH) + labels.csv

All count estimates in the plan (~1,591 clean, video 654->137, ~880 train)
are confirmed at run time against the real data by ``scripts/prepare_data.py``.
"""

from __future__ import annotations

import csv
import random
import re
import xml.etree.ElementTree as ET
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path

import cv2

# --- Plate text validation ------------------------------------------------

# Standard Indian plate, e.g. KL45C4411, MH20BY4465, DL13S0155.
STANDARD_PLATE_RE = re.compile(r"^[A-Z]{2}\d{1,2}[A-Z]{1,3}\d{1,4}$")
# Extended: BH-series / commercial plates that end in a trailing letter,
# e.g. MH20TC830C.
EXTENDED_PLATE_RE = re.compile(r"^[A-Z]{2}\d{2}[A-Z]{2,3}\d{1,4}[A-Z]?$")

# Image extensions we accept (case-insensitive).
_IMAGE_EXTS = {".jpg", ".jpeg", ".png"}

# Crop geometry shared between OCR dataset building, preprocessing, and the CRNN
# input shape (everything downstream reads these — never hard-code the size).
# Larger than the original 64x256 to use more of the detail in the 384px detector
# crops; the width is the OCR sequence axis, so it is kept generous.
OCR_HEIGHT = 96
OCR_WIDTH = 320
MIN_CROP_SIDE = 10
MIN_BOX_SIDE = 5


@dataclass
class PlateAnnotation:
    """A single license-plate annotation parsed from a VOC XML file."""

    image_path: Path
    plate_text: str
    xmin: int
    ymin: int
    xmax: int
    ymax: int
    img_width: int
    img_height: int
    source: str  # top-level folder: google_images | video_images | State-wise_OLX

    @property
    def box_area(self) -> int:
        return max(0, self.xmax - self.xmin) * max(0, self.ymax - self.ymin)


def is_valid_plate(text: str) -> bool:
    """True if ``text`` looks like a real Indian plate (standard or extended)."""
    return bool(STANDARD_PLATE_RE.match(text) or EXTENDED_PLATE_RE.match(text))


def _source_of(xml_path: Path, data_root: Path) -> str:
    """Top-level collection folder under ``data_root`` containing ``xml_path``.

    ``rel.parts`` is ``(collection, ..., file.xml)``; the first part is the
    collection folder. If the XML sits directly in ``data_root`` (no
    subfolder), fall back to the immediate parent's name.
    """
    rel = xml_path.relative_to(data_root)
    if len(rel.parts) > 1:
        return rel.parts[0]
    return xml_path.parent.name


def _resolve_image(xml_path: Path) -> Path | None:
    """Find the image paired with an XML file.

    The dataset is consistent: the XML stem (before ``.xml``) is the image
    filename verbatim (e.g. ``foo.jpg.xml`` -> ``foo.jpg``, ``KL10.xml`` ->
    ``KL10.jpg``). We trust the ``<filename>`` field first, then fall back to
    matching the stem against any sibling image.
    """
    # Try the sibling whose name equals the XML stem.
    stem_candidate = xml_path.with_suffix("")  # strips ".xml"
    if stem_candidate.suffix.lower() in _IMAGE_EXTS and stem_candidate.exists():
        return stem_candidate

    # Fall back: same base name with a known image extension.
    base = xml_path.name[: -len(".xml")]
    for ext in _IMAGE_EXTS:
        cand = xml_path.with_name(base + ext)
        if cand.exists():
            return cand
        cand = xml_path.with_name(Path(base).stem + ext)
        if cand.exists():
            return cand
    return None


def parse_single_xml(xml_path: Path, data_root: Path) -> PlateAnnotation | None:
    """Parse one VOC XML into a :class:`PlateAnnotation`, or ``None`` if invalid.

    Returns ``None`` when: the object/bbox is missing, the box is degenerate
    (side < ``MIN_BOX_SIDE``), the paired image is absent, or the plate text
    fails both validation patterns.
    """
    try:
        root = ET.parse(xml_path).getroot()
    except ET.ParseError:
        return None

    obj = root.find("object")
    if obj is None:
        return None

    name_el = obj.find("name")
    box_el = obj.find("bndbox")
    if name_el is None or box_el is None or not name_el.text:
        return None

    plate_text = name_el.text.strip().upper()
    if not is_valid_plate(plate_text):
        return None

    try:
        xmin = int(float(box_el.findtext("xmin")))
        ymin = int(float(box_el.findtext("ymin")))
        xmax = int(float(box_el.findtext("xmax")))
        ymax = int(float(box_el.findtext("ymax")))
    except (TypeError, ValueError):
        return None

    if (xmax - xmin) < MIN_BOX_SIDE or (ymax - ymin) < MIN_BOX_SIDE:
        return None

    image_path = _resolve_image(xml_path)
    if image_path is None:
        return None

    # Prefer the declared size; fall back to reading the image header.
    size_el = root.find("size")
    img_w = img_h = None
    if size_el is not None:
        try:
            img_w = int(float(size_el.findtext("width")))
            img_h = int(float(size_el.findtext("height")))
        except (TypeError, ValueError):
            img_w = img_h = None
    if not img_w or not img_h:
        img = cv2.imread(str(image_path))
        if img is None:
            return None
        img_h, img_w = img.shape[:2]

    # Clamp the box to the image bounds.
    xmin = max(0, min(xmin, img_w - 1))
    xmax = max(0, min(xmax, img_w))
    ymin = max(0, min(ymin, img_h - 1))
    ymax = max(0, min(ymax, img_h))
    if (xmax - xmin) < MIN_BOX_SIDE or (ymax - ymin) < MIN_BOX_SIDE:
        return None

    return PlateAnnotation(
        image_path=image_path,
        plate_text=plate_text,
        xmin=xmin,
        ymin=ymin,
        xmax=xmax,
        ymax=ymax,
        img_width=img_w,
        img_height=img_h,
        source=_source_of(xml_path, data_root),
    )


def parse_voc(data_root: str | Path) -> list[PlateAnnotation]:
    """Walk ``data_root`` and return all valid annotations.

    Skips Windows ``*:Zone.Identifier`` sidecar files.
    """
    data_root = Path(data_root)
    annotations: list[PlateAnnotation] = []
    for xml_path in sorted(data_root.rglob("*.xml")):
        # rglob does not match the ADS sidecars on its own, but guard anyway.
        if "Zone.Identifier" in xml_path.name:
            continue
        ann = parse_single_xml(xml_path, data_root)
        if ann is not None:
            annotations.append(ann)
    return annotations


def deduplicate(annotations: list[PlateAnnotation]) -> list[PlateAnnotation]:
    """Collapse repeated ``video_images`` frames to one per plate.

    For ``source == "video_images"`` keep only the annotation with the largest
    bounding-box area (sharpest / closest frame). Other sources pass through
    unchanged.
    """
    best_video: dict[str, PlateAnnotation] = {}
    passthrough: list[PlateAnnotation] = []
    for ann in annotations:
        if ann.source == "video_images":
            cur = best_video.get(ann.plate_text)
            if cur is None or ann.box_area > cur.box_area:
                best_video[ann.plate_text] = ann
        else:
            passthrough.append(ann)
    return passthrough + list(best_video.values())


def plate_aware_split(
    annotations: list[PlateAnnotation],
    train_frac: float = 0.80,
    val_frac: float = 0.10,
    seed: int = 42,
) -> dict[str, list[PlateAnnotation]]:
    """Split annotations 80/10/10 grouping whole plates together.

    Grouping by ``plate_text`` guarantees the same physical plate never
    appears in more than one split, preventing leakage in evaluation.
    """
    groups: dict[str, list[PlateAnnotation]] = defaultdict(list)
    for ann in annotations:
        groups[ann.plate_text].append(ann)

    plate_keys = list(groups.keys())
    random.Random(seed).shuffle(plate_keys)

    n = len(plate_keys)
    n_train = int(n * train_frac)
    n_val = int(n * val_frac)

    split_keys = {
        "train": plate_keys[:n_train],
        "val": plate_keys[n_train : n_train + n_val],
        "test": plate_keys[n_train + n_val :],
    }
    return {
        split: [ann for key in keys for ann in groups[key]]
        for split, keys in split_keys.items()
    }


def build_detection_manifest(
    splits: dict[str, list[PlateAnnotation]], out_dir: str | Path
) -> dict[str, Path]:
    """Write ``datasets/detection/<split>.csv`` with normalized boxes.

    Columns: ``image_path, xmin_n, ymin_n, xmax_n, ymax_n, plate_text``
    (coords each in [0, 1]). ``plate_text`` lets end-to-end evaluation join
    predicted boxes back to ground truth. No images are copied; the tf.data
    loader reads originals and resizes.
    """
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    written: dict[str, Path] = {}
    for split, anns in splits.items():
        csv_path = out_dir / f"{split}.csv"
        with csv_path.open("w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(
                ["image_path", "xmin_n", "ymin_n", "xmax_n", "ymax_n", "plate_text"]
            )
            for a in anns:
                writer.writerow(
                    [
                        str(a.image_path),
                        f"{a.xmin / a.img_width:.6f}",
                        f"{a.ymin / a.img_height:.6f}",
                        f"{a.xmax / a.img_width:.6f}",
                        f"{a.ymax / a.img_height:.6f}",
                        a.plate_text,
                    ]
                )
        written[split] = csv_path
    return written


def build_ocr_dataset(
    splits: dict[str, list[PlateAnnotation]], out_dir: str | Path
) -> Path:
    """Build the OCR crops into ``datasets/ocr/<split>/`` + labels.csv.

    Crops go through the SAME preprocessing as the live pipeline
    (:func:`src.preprocessing.preprocess_plate` — expand+crop, optional
    perspective warp, grayscale, CLAHE) so the OCR trains on exactly the input
    distribution it is served at inference time (no train/serve skew).

    Returns the path to ``labels.csv`` (columns: split, filename, plate_text).
    ``filename`` is relative to ``out_dir``.
    """
    # Local import: preprocessing imports from this module, so importing it at
    # module top level would be circular. By call time `data` is fully loaded.
    from src.preprocessing import preprocess_plate

    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    labels_path = out_dir / "labels.csv"

    with labels_path.open("w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["split", "filename", "plate_text"])
        for split, anns in splits.items():
            split_dir = out_dir / split
            split_dir.mkdir(parents=True, exist_ok=True)
            for idx, ann in enumerate(anns):
                img = cv2.imread(str(ann.image_path))
                if img is None:
                    continue
                # (H, W) uint8 grayscale — identical to inference preprocessing.
                gray = preprocess_plate(img, (ann.xmin, ann.ymin, ann.xmax, ann.ymax))
                # Unique filename: plate text + running index avoids collisions
                # when the same plate appears across sources.
                fname = f"{ann.plate_text}_{idx:05d}.png"
                rel = Path(split) / fname
                cv2.imwrite(str(out_dir / rel), gray)
                writer.writerow([split, str(rel), ann.plate_text])
    return labels_path
