# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

Indian Automatic License Plate Recognition (ALPR/ANPR) system. The goal is to detect and read license plates from car images sourced from Indian classified ads (OLX) and Google Images. Indian plates use the format `<state-code><district-number><series><number>` (e.g., `KL45C4411`, `MH20TC830C`).

## Environment

Python 3.14 virtual environment at `alpr-env/`. Activate with:

```bash
source alpr-env/bin/activate
```

No ML packages are installed yet — install as needed (e.g., `pip install opencv-python torch ultralytics`).

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
