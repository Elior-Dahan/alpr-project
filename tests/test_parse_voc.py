"""Tests for VOC parsing and plate-text validation (no GPU/TF needed)."""

from __future__ import annotations

import sys
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.data import is_valid_plate, parse_single_xml


def test_valid_plate_standard():
    assert is_valid_plate("KL45C4411")
    assert is_valid_plate("MH20BY4465")
    assert is_valid_plate("DL13S0155")


def test_valid_plate_extended_bh_commercial():
    # Trailing letter (commercial / BH-series style).
    assert is_valid_plate("MH20TC830C")


def test_invalid_plate_noise():
    assert not is_valid_plate("TERRANO")
    assert not is_valid_plate("CRETA")
    assert not is_valid_plate("W0BNP300")  # leading digit in state code
    assert not is_valid_plate("")


def _write_xml(path: Path, filename: str, name: str, box=(10, 10, 90, 40), size=(100, 60)):
    w, h = size
    xmin, ymin, xmax, ymax = box
    path.write_text(
        f"""<annotation>
  <filename>{filename}</filename>
  <size><width>{w}</width><height>{h}</height><depth>3</depth></size>
  <object>
    <name>{name}</name>
    <bndbox><xmin>{xmin}</xmin><ymin>{ymin}</ymin>
            <xmax>{xmax}</xmax><ymax>{ymax}</ymax></bndbox>
  </object>
</annotation>"""
    )


def test_parse_single_xml_valid(tmp_path):
    img_path = tmp_path / "KL10.jpg"
    cv2.imwrite(str(img_path), np.zeros((60, 100, 3), np.uint8))
    xml_path = tmp_path / "KL10.xml"
    _write_xml(xml_path, "KL10.jpg", "KL45C4411")

    ann = parse_single_xml(xml_path, tmp_path)
    assert ann is not None
    assert ann.plate_text == "KL45C4411"
    assert (ann.xmin, ann.ymin, ann.xmax, ann.ymax) == (10, 10, 90, 40)
    assert ann.source == tmp_path.name or ann.source == "."


def test_parse_single_xml_rejects_noise(tmp_path):
    img_path = tmp_path / "bad.jpg"
    cv2.imwrite(str(img_path), np.zeros((60, 100, 3), np.uint8))
    xml_path = tmp_path / "bad.xml"
    _write_xml(xml_path, "bad.jpg", "TERRANO")

    assert parse_single_xml(xml_path, tmp_path) is None


def test_parse_single_xml_rejects_degenerate_box(tmp_path):
    img_path = tmp_path / "tiny.jpg"
    cv2.imwrite(str(img_path), np.zeros((60, 100, 3), np.uint8))
    xml_path = tmp_path / "tiny.xml"
    _write_xml(xml_path, "tiny.jpg", "KL45C4411", box=(10, 10, 12, 12))

    assert parse_single_xml(xml_path, tmp_path) is None


def test_parse_single_xml_missing_image(tmp_path):
    xml_path = tmp_path / "ghost.xml"
    _write_xml(xml_path, "ghost.jpg", "KL45C4411")
    assert parse_single_xml(xml_path, tmp_path) is None
