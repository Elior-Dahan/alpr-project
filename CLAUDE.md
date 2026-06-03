# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

Indian Automatic License Plate Recognition (ALPR/ANPR) system. The goal is to detect and read license plates from car images sourced from Indian classified ads (OLX) and Google Images. Indian plates use the format `<state-code><district-number><series><number>` (e.g., `KL45C4411`, `MH20TC830C`).

## Environment

The pipeline targets **Python 3.12** (TensorFlow has no 3.14 wheel). The committed
`alpr-env/` was created with 3.14 and must be recreated:

```bash
rm -rf alpr-env
python3.12 -m venv alpr-env
source alpr-env/bin/activate
pip install -r requirements.txt
```

The whole stack is **TensorFlow/Keras 3 + OpenCV** — no PyTorch/YOLO. Detection is
done as bbox regression with a MobileNetV2 backbone, not an object detector.

## Architecture

Three modular stages, orchestrated end-to-end by `src/pipeline.py::ALPRPipeline`:

1. **Detection** (`src/detection.py`) — MobileNetV2 → Dense(4, sigmoid) regresses one
   normalized `[xmin,ymin,xmax,ymax]` box. Loss = Huber + (1−GIoU); metric = mean IoU.
   One plate per image, so a single box suffices.
2. **Preprocessing** (`src/preprocessing.py`) — `preprocess_plate()`: expand+crop the box,
   recover plate corners (Canny→contours→approxPolyDP), 4-point perspective warp (falls back
   to plain resize), grayscale, CLAHE → `(64, 256)` uint8.
3. **OCR** (`src/ocr.py`) — CRNN: 4-block CNN (width axis becomes `T=32` time steps) →
   2× BiLSTM → Dense logits `(B, 32, 37)`. Trained with built-in `keras.losses.CTC`;
   decoded with `keras.ops.ctc_decode`. **The Dense output is linear logits, not softmax**
   (CTC loss/decode apply softmax internally). Charset = 36 chars + blank at index 0.

`src/data.py` is the shared data layer: `parse_voc` (filters noise via the two plate regexes,
skips Zone.Identifier and degenerate boxes), `deduplicate` (video frames → one per plate, largest
box), `plate_aware_split` (groups by plate text so no plate leaks across train/val/test),
`build_detection_manifest` (CSV of normalized boxes + plate_text), `build_ocr_dataset` (cropped
grayscale plates + `labels.csv`).

## Commands

```bash
python scripts/prepare_data.py                 # parse → dedup → split → build datasets (prints counts)
python scripts/train_detection.py              # train detector → models/detection/detector.keras
python scripts/train_ocr.py                    # train CRNN → models/ocr/crnn_best.keras
python scripts/evaluate.py --stage detection   # mean IoU + acc@IoU>=0.5
python scripts/evaluate.py --stage ocr         # exact-match + CER on cropped plates
python scripts/evaluate.py --stage pipeline    # end-to-end exact-match on predicted boxes + per-state
pytest tests/                                  # offline unit tests (parse/preprocess/ctc)
```

Generated `datasets/` and `models/` are git-ignored.

## Data Structure

~1,700 annotated images across three collections, all using **Pascal VOC XML** format:

```
data/
  google_images/          # Car images from Google search
  video_images/           # Car images extracted from video
  State-wise_OLX/         # OLX classified ads, organized by Indian state code
    KL/  MH/  HR/  DL/  TN/  KA/  ... (35 states/UTs)
```

Each image has a paired `.xml` annotation file with the same base name. Annotation structure:

```xml
<annotation>
  <filename>KL10.jpg</filename>
  <size><width>272</width><height>363</height><depth>3</depth></size>
  <object>
    <name>KL45C4411</name>          <!-- license plate text -->
    <bndbox>
      <xmin>58</xmin><ymin>201</ymin><xmax>130</xmax><ymax>230</ymax>
    </bndbox>
  </object>
</annotation>
```

The `<name>` field holds the ground-truth plate text; `<bndbox>` is the plate region in the image. Some images have `Zone.Identifier` sidecar files (Windows metadata) — ignore these.

## State Code Reference

Indian state/UT codes used as folder names and plate prefixes: AN, AP, AR, AS, BR, CG, CH, DL, DN, GA, GJ, HP, HR, JH, JK, KA, KL, LA, MH, ML, MN, MP, MN, MZ, NL, OD, PB, PY, RJ, SK, TN, TR, TS, UK, UP, WB.
