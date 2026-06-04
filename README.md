# Indian ALPR — Automatic License Plate Recognition

An end-to-end pipeline that detects and reads Indian vehicle license plates from
car photos. Built entirely on **TensorFlow / Keras 3 + OpenCV** (no PyTorch/YOLO).

Indian plates follow the format `<state-code><district-number><series><number>`
— e.g. `KL45C4411`, `MH20TC830C`.

---

## How it works

Three modular stages, chained end-to-end by `src/pipeline.py`:

```
 car image
    │
    ▼
┌─────────────────────┐   1. Detection  (src/detection.py)
│  MobileNetV2 +      │   Regresses one normalized box [xmin,ymin,xmax,ymax].
│  bbox-regression    │   Loss = Huber + (1 − GIoU); metric = mean IoU.
└─────────────────────┘
    │  box
    ▼
┌─────────────────────┐   2. Preprocessing  (src/preprocessing.py)
│  crop → 4-pt warp   │   Expand & crop the box, recover plate corners
│  → grayscale → CLAHE│   (Canny→contours→approxPolyDP), perspective-rectify,
└─────────────────────┘   grayscale, contrast-normalize → 64×256.
    │  64×256 gray plate
    ▼
┌─────────────────────┐   3. OCR  (src/ocr.py)
│  CRNN: CNN → BiLSTM │   4-block CNN (width axis → T=32 steps) → 2× BiLSTM →
│  → CTC              │   Dense logits (B,32,37). keras.losses.CTC training,
└─────────────────────┘   keras.ops.ctc_decode greedy decoding.
    │
    ▼
  "KL45C4411"
```

The character set is 36 classes (`A–Z`, `0–9`) plus a CTC blank at index 0.

---

## Dataset

~1,700 annotated images in **Pascal VOC XML** format across three collections,
under `data/` (git-ignored):

| Collection        | Description                              |
|-------------------|------------------------------------------|
| `google_images/`  | Car images from Google search            |
| `video_images/`   | Frames extracted from video (many dupes) |
| `State-wise_OLX/` | OLX classified ads, foldered by state    |

Each image has a paired `.xml` with the plate text (`<name>`) and bounding box
(`<bndbox>`). The data pipeline filters annotation noise, deduplicates repeated
video frames (one sharpest frame per plate), and splits **by plate identity** so
no plate leaks across train/val/test.

Validated counts from the real data:

| Stage              | Count                      |
|--------------------|----------------------------|
| Clean annotations  | 1592                       |
| After video dedup  | 1092 (video 633 → 133)     |
| Train / Val / Test | 881 / 103 / 108            |
| Plate leakage      | 0                          |

---

## Setup

Requires **Python 3.12** (TensorFlow has no 3.14 wheel). Using
[`uv`](https://github.com/astral-sh/uv):

```bash
uv python install 3.12
uv venv --python 3.12 alpr-env
uv pip install --python alpr-env/bin/python -r requirements.txt
source alpr-env/bin/activate
```

Or with a standard `python3.12`:

```bash
python3.12 -m venv alpr-env
source alpr-env/bin/activate
pip install -r requirements.txt
```

---

## Usage

### Notebook (report-style walkthrough)

`alpr_pipeline.ipynb` runs the entire pipeline end-to-end with rich explanations,
EDA, training curves, and sample-prediction visualizations — a self-contained
alternative to the CLI below. It trains with small default epochs (or loads saved
models if present), so it executes on CPU in minutes:

```bash
pip install -r requirements-dev.txt   # adds jupyter/nbconvert
jupyter notebook alpr_pipeline.ipynb
```

### CLI scripts

```bash
# 1. Build datasets from the raw VOC data (prints real counts + leakage check)
python scripts/prepare_data.py

# 2. Train the detector  → models/detection/detector.keras
python scripts/train_detection.py        # --epochs 100 --batch-size 16

# 3. Train the OCR model → models/ocr/crnn_best.keras (+ char_map.json)
python scripts/train_ocr.py              # --epochs 50 --batch-size 32

# 4. Evaluate any stage
python scripts/evaluate.py --stage detection   # mean IoU + acc@IoU≥0.5
python scripts/evaluate.py --stage ocr         # exact-match + CER on crops
python scripts/evaluate.py --stage pipeline    # end-to-end exact-match + per-state
```

`prepare_data.py` reads originals and writes:
- `datasets/detection/{train,val,test}.csv` — normalized boxes + plate text
- `datasets/ocr/{train,val,test}/*.png` + `datasets/ocr/labels.csv`

### Inference in code

```python
from src.pipeline import ALPRPipeline

pipe = ALPRPipeline(
    detector_path="models/detection/detector.keras",
    ocr_weights_path="models/ocr/crnn_best.keras",
)
result = pipe.run("path/to/car.jpg")[0]
print(result["plate_text"], result["bbox"])
```

---

## Project layout

```
requirements.txt
src/
  data.py           parse_voc, deduplicate, plate_aware_split,
                    build_detection_manifest, build_ocr_dataset
  detection.py      MobileNetV2 detector: build / train / detect_plate
  preprocessing.py  preprocess_plate (crop, warp, CLAHE)
  ocr.py            build_crnn, CTC loss/decode, train
  pipeline.py       ALPRPipeline (end-to-end inference)
scripts/
  prepare_data.py   train_detection.py  train_ocr.py  evaluate.py
tests/              parse_voc · preprocess · ctc round-trip
data/               raw VOC images (git-ignored)
datasets/  models/  generated artifacts (git-ignored)
```

---

## Testing

Offline unit tests (no GPU; `test_ctc` needs Keras installed):

```bash
pytest tests/
```

Covers plate-text validation & VOC parsing, perspective-warp corner ordering and
output shape, and a CTC encode → decode round-trip.

---

## Targets

| Metric                          | Target   |
|---------------------------------|----------|
| Detection mean IoU / acc@0.5    | > ~0.85  |
| OCR exact-match / CER           | > 80% / < 5% |
| End-to-end full-plate accuracy  | reported per state |

---

## Notes

- Detection is framed as **single-box regression** (one plate per image), not a
  general object detector — simpler and sufficient for this dataset.
- The OCR head outputs **linear logits, not softmax**: both `keras.losses.CTC`
  and `keras.ops.ctc_decode` apply softmax internally.
- Training needs a GPU for reasonable speed; the pipeline runs CPU-only for
  inference and tests.
