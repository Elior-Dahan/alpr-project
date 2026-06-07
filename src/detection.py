"""Stage 1 — license-plate detection as bounding-box regression (Keras).

A MobileNetV2 backbone with a small regression head predicts a single
normalized box ``[xmin, ymin, xmax, ymax]`` in [0, 1]. Trained with a
Huber + (1 - GIoU) loss and a mean-IoU metric. One plate per image, so a
single box is sufficient.

Training runs in three phases (Huber warmup -> Huber + GIoU -> backbone
fine-tune); see :func:`train` for the rationale.
"""

from __future__ import annotations

from pathlib import Path

import cv2
import keras
import numpy as np
import tensorflow as tf
from keras import layers
from keras.applications.mobilenet_v2 import preprocess_input

INPUT_SIZE = 384  # plates are tiny in full frames; higher res aids localization


# --- Model ----------------------------------------------------------------

def build_detector(input_size: int = INPUT_SIZE) -> keras.Model:
    """MobileNetV2 backbone -> GAP -> Dense(256) -> Dropout -> Dense(4, sigmoid)."""
    backbone = keras.applications.MobileNetV2(
        weights="imagenet",
        include_top=False,
        input_shape=(input_size, input_size, 3),
    )
    inputs = keras.Input(shape=(input_size, input_size, 3))
    x = backbone(inputs, training=False)
    x = layers.GlobalAveragePooling2D()(x)
    x = layers.Dense(256, activation="relu")(x)
    x = layers.Dropout(0.3)(x)
    outputs = layers.Dense(4, activation="sigmoid", name="box")(x)
    return keras.Model(inputs, outputs, name="plate_detector")


# --- Loss & metric --------------------------------------------------------

def _box_iou_components(y_true, y_pred):
    """Return (intersection, union, enclosing-area) for batched xyxy boxes.

    Shared by both ``giou_loss`` and the ``MeanIoU`` metric. Boxes are
    ``[xmin, ymin, xmax, ymax]``; the intersection is the overlap rectangle
    (clamped to >= 0), and ``enclose`` is the area of the smallest axis-aligned
    box containing both — the extra term GIoU needs to penalise non-overlapping
    predictions (plain IoU is 0 and gives no gradient when boxes are disjoint).
    """
    # Intersection rectangle: max of the mins, min of the maxes.
    x0 = tf.maximum(y_true[:, 0], y_pred[:, 0])
    y0 = tf.maximum(y_true[:, 1], y_pred[:, 1])
    x1 = tf.minimum(y_true[:, 2], y_pred[:, 2])
    y1 = tf.minimum(y_true[:, 3], y_pred[:, 3])
    inter = tf.maximum(0.0, x1 - x0) * tf.maximum(0.0, y1 - y0)

    area_t = tf.maximum(0.0, y_true[:, 2] - y_true[:, 0]) * tf.maximum(
        0.0, y_true[:, 3] - y_true[:, 1]
    )
    area_p = tf.maximum(0.0, y_pred[:, 2] - y_pred[:, 0]) * tf.maximum(
        0.0, y_pred[:, 3] - y_pred[:, 1]
    )
    union = area_t + area_p - inter

    # Smallest enclosing box.
    cx0 = tf.minimum(y_true[:, 0], y_pred[:, 0])
    cy0 = tf.minimum(y_true[:, 1], y_pred[:, 1])
    cx1 = tf.maximum(y_true[:, 2], y_pred[:, 2])
    cy1 = tf.maximum(y_true[:, 3], y_pred[:, 3])
    enclose = tf.maximum(0.0, cx1 - cx0) * tf.maximum(0.0, cy1 - cy0)
    return inter, union, enclose


def giou_loss(y_true, y_pred):
    """1 - GIoU, averaged over the batch (lower is better, range [0, 2]).

    GIoU = IoU - (area_enclosing - area_union) / area_enclosing. The second term
    shrinks as the prediction moves toward the target even while they don't yet
    overlap, so the loss keeps a useful gradient where IoU alone would be flat.
    """
    eps = 1e-7
    inter, union, enclose = _box_iou_components(y_true, y_pred)
    iou = inter / (union + eps)
    giou = iou - (enclose - union) / (enclose + eps)
    return tf.reduce_mean(1.0 - giou)


def detector_loss(y_true, y_pred):
    """Combined box loss: Huber on the 4 coordinates + (1 - GIoU).

    Huber gives a stable per-coordinate gradient; GIoU adds an overlap-aware
    term so the box is optimised as a whole rectangle, not four independent
    numbers.
    """
    huber = keras.losses.Huber()(y_true, y_pred)
    return huber + giou_loss(y_true, y_pred)


class MeanIoU(keras.metrics.Metric):
    """Streaming mean IoU over predicted vs. ground-truth boxes (normalized xyxy).

    Accumulates the IoU sum and sample count across batches so ``result()``
    reports the epoch-wide average rather than a single batch's value.
    """

    def __init__(self, name: str = "mean_iou", **kwargs):
        super().__init__(name=name, **kwargs)
        self.total = self.add_weight(name="total", initializer="zeros")
        self.count = self.add_weight(name="count", initializer="zeros")

    def update_state(self, y_true, y_pred, sample_weight=None):
        inter, union, _ = _box_iou_components(y_true, y_pred)
        iou = inter / (union + 1e-7)
        self.total.assign_add(tf.reduce_sum(iou))
        self.count.assign_add(tf.cast(tf.shape(iou)[0], tf.float32))

    def result(self):
        return self.total / (self.count + 1e-7)

    def reset_state(self):
        self.total.assign(0.0)
        self.count.assign(0.0)


# --- Data pipeline --------------------------------------------------------

def _load_manifest(csv_path: str | Path):
    """Read a detection manifest CSV into (paths, boxes) arrays."""
    import csv as _csv

    paths, boxes = [], []
    with Path(csv_path).open() as f:
        reader = _csv.DictReader(f)
        for row in reader:
            paths.append(row["image_path"])
            boxes.append(
                [
                    float(row["xmin_n"]),
                    float(row["ymin_n"]),
                    float(row["xmax_n"]),
                    float(row["ymax_n"]),
                ]
            )
    return paths, np.asarray(boxes, dtype=np.float32)


def _decode_and_resize(path, box, augment):
    """Load and resize one image to the network input, returning (image, box).

    The target ``box`` is already normalized to [0, 1] in the manifest, so it
    needs no adjustment when the image is resized. Augmentation is photometric
    only (brightness/contrast) — no geometric flips, which would invalidate the
    box and orient plates unnaturally.
    """
    raw = tf.io.read_file(path)
    img = tf.io.decode_image(raw, channels=3, expand_animations=False)
    img = tf.image.resize(img, (INPUT_SIZE, INPUT_SIZE))
    img = tf.cast(img, tf.float32)
    if augment:
        # Photometric only — these leave the target box valid (no geometry).
        # Done in [0, 1] because random_hue/saturation convert via HSV.
        img = img / 255.0
        img = tf.image.random_brightness(img, 0.1)
        img = tf.image.random_contrast(img, 0.85, 1.15)
        img = tf.image.random_hue(img, 0.05)
        img = tf.image.random_saturation(img, 0.85, 1.15)
        img = tf.clip_by_value(img, 0.0, 1.0) * 255.0
    # MobileNetV2 expects inputs scaled to [-1, 1], not [0, 255] or [0, 1].
    img = preprocess_input(img)
    return img, box


def make_dataset(
    csv_path: str | Path, batch_size: int = 16, augment: bool = False, shuffle: bool = False
) -> tf.data.Dataset:
    """Build a tf.data pipeline from a detection manifest CSV."""
    paths, boxes = _load_manifest(csv_path)
    ds = tf.data.Dataset.from_tensor_slices((paths, boxes))
    if shuffle:
        ds = ds.shuffle(len(paths), reshuffle_each_iteration=True)
    ds = ds.map(
        lambda p, b: _decode_and_resize(p, b, augment),
        num_parallel_calls=tf.data.AUTOTUNE,
    )
    return ds.batch(batch_size).prefetch(tf.data.AUTOTUNE)


# --- Training -------------------------------------------------------------

# Three-phase schedule. Adding GIoU from a cold start produces noisy gradients
# while boxes don't yet overlap, which can saturate the output sigmoid into an
# inverted box (xmax < xmin) where GIoU's clamps *and* Huber's saturated sigmoid
# both kill the gradient — the model collapses (mean_iou stuck at 0). So we warm
# up with Huber alone until boxes overlap (Phase 1), refine with Huber + GIoU
# (Phase 2), then unfreeze the top of the backbone (Phase 3). Each epoch count
# is a max budget — EarlyStopping on val mean IoU cuts each phase short. Tune
# these knobs freely.
WARMUP_EPOCHS = 10
WARMUP_LR = 1e-4
FINETUNE_EPOCHS = 120
FINETUNE_LR = 3e-5
BACKBONE_FT_EPOCHS = 50
BACKBONE_FT_LR = 1e-5
WEIGHT_DECAY = 1e-4  # AdamW decoupled weight decay (all phases)
EARLY_STOPPING_PATIENCE = 15
BACKBONE_FT_LAYERS = 10  # top-N backbone layers unfrozen in Phase 3


def train(
    detection_dir: str | Path = "datasets/detection",
    model_out: str | Path = "models/detection/detector.keras",
    batch_size: int = 16,
    warmup_epochs: int = WARMUP_EPOCHS,
    warmup_lr: float = WARMUP_LR,
    finetune_epochs: int = FINETUNE_EPOCHS,
    finetune_lr: float = FINETUNE_LR,
    backbone_ft_epochs: int = BACKBONE_FT_EPOCHS,
    backbone_ft_lr: float = BACKBONE_FT_LR,
    weight_decay: float = WEIGHT_DECAY,
) -> keras.Model:
    """Train the detector in three phases; return the global-best model.

    Phase 1 warms up with Huber only (stable box regression); Phase 2 re-compiles
    the *same* model with the full Huber + GIoU loss to refine localisation;
    Phase 3 unfreezes the top ``BACKBONE_FT_LAYERS`` backbone layers (BatchNorm
    kept in inference mode) and fine-tunes at a very low LR. See the module-level
    note for why the warmup is necessary.

    A single ``ModelCheckpoint`` is shared across all three phases, so it tracks
    the global-best ``val_mean_iou`` rather than resetting each phase; the saved
    checkpoint is reloaded at the end so the returned model is that best — not the
    weights Phase 3 happened to finish on. AdamW adds mild weight decay throughout.

    Raises ``RuntimeError`` if the warmup fails to produce a non-zero mean IoU
    within the first epoch — a sign the data/manifest is wrong, so we stop rather
    than proceed into the GIoU phase.
    """
    detection_dir = Path(detection_dir)
    model_out = Path(model_out)
    model_out.parent.mkdir(parents=True, exist_ok=True)

    # Photometric augmentation on (box-safe) to fight the train/val IoU gap.
    train_ds = make_dataset(
        detection_dir / "train.csv", batch_size, augment=True, shuffle=True
    )
    val_ds = make_dataset(detection_dir / "val.csv", batch_size)

    model = build_detector()

    # One checkpoint shared across all phases, so it tracks the global-best
    # val_mean_iou rather than resetting each phase.
    ckpt = keras.callbacks.ModelCheckpoint(
        str(model_out), save_best_only=True, monitor="val_mean_iou", mode="max"
    )

    def early() -> keras.callbacks.EarlyStopping:
        return keras.callbacks.EarlyStopping(
            monitor="val_mean_iou", mode="max",
            patience=EARLY_STOPPING_PATIENCE, restore_best_weights=True,
        )

    def opt(lr: float) -> keras.optimizers.Optimizer:
        return keras.optimizers.AdamW(learning_rate=lr, weight_decay=weight_decay)

    # Phase 1 — Huber warmup.
    print(f"[detector] Phase 1: Huber warmup — {warmup_epochs} epochs @ lr={warmup_lr}")
    model.compile(optimizer=opt(warmup_lr), loss=keras.losses.Huber(), metrics=[MeanIoU()])
    hist = model.fit(
        train_ds, validation_data=val_ds, epochs=warmup_epochs, callbacks=[ckpt],
    )

    # Gate: stable regression should yield a non-zero IoU within one epoch.
    first_iou = hist.history["mean_iou"][0]
    if not first_iou > 0:
        raise RuntimeError(
            f"Warmup failed: mean_iou={first_iou} after epoch 1 (expected > 0). "
            "Stopping before the GIoU phase — investigate the data/manifest."
        )

    # Phase 2 — Huber + GIoU fine-tune.
    print(
        f"[detector] Phase 2: Huber+GIoU fine-tune — up to {finetune_epochs} epochs @ lr={finetune_lr}"
    )
    model.compile(optimizer=opt(finetune_lr), loss=detector_loss, metrics=[MeanIoU()])
    model.fit(
        train_ds, validation_data=val_ds,
        epochs=finetune_epochs, callbacks=[ckpt, early()],
    )

    # Phase 3 — unfreeze the top backbone layers (BatchNorm stays in inference
    # mode) and fine-tune at a very low LR to push past the frozen ceiling.
    backbone = next(l for l in model.layers if isinstance(l, keras.Model))
    backbone.trainable = True
    for layer in backbone.layers[:-BACKBONE_FT_LAYERS]:
        layer.trainable = False
    for layer in backbone.layers:
        if isinstance(layer, keras.layers.BatchNormalization):
            layer.trainable = False
    n_trainable = sum(l.trainable for l in backbone.layers)
    print(
        f"[detector] Phase 3: backbone fine-tune — up to {backbone_ft_epochs} epochs @ "
        f"lr={backbone_ft_lr} ({n_trainable}/{len(backbone.layers)} backbone layers trainable)"
    )
    model.compile(optimizer=opt(backbone_ft_lr), loss=detector_loss, metrics=[MeanIoU()])
    model.fit(
        train_ds, validation_data=val_ds,
        epochs=backbone_ft_epochs, callbacks=[ckpt, early()],
    )

    # The shared checkpoint holds the global-best across phases. Reload it so the
    # returned model is that best, not the in-memory Phase 3 weights.
    return load_detector(model_out)


# --- Inference ------------------------------------------------------------

def detect_plate(image_bgr: np.ndarray, model: keras.Model) -> tuple[int, int, int, int]:
    """Predict a single plate box in original-image pixel coordinates."""
    h, w = image_bgr.shape[:2]
    rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
    resized = cv2.resize(rgb, (INPUT_SIZE, INPUT_SIZE)).astype(np.float32)
    batch = preprocess_input(resized[None, ...])
    xmin_n, ymin_n, xmax_n, ymax_n = model.predict(batch, verbose=0)[0]
    xmin = int(round(xmin_n * w))
    ymin = int(round(ymin_n * h))
    xmax = int(round(xmax_n * w))
    ymax = int(round(ymax_n * h))
    # Guard against inverted predictions.
    xmin, xmax = sorted((max(0, xmin), min(w, xmax)))
    ymin, ymax = sorted((max(0, ymin), min(h, ymax)))
    return xmin, ymin, xmax, ymax


def load_detector(model_path: str | Path) -> keras.Model:
    """Load a saved detector with its custom loss/metric objects."""
    return keras.models.load_model(
        model_path,
        custom_objects={
            "detector_loss": detector_loss,
            "giou_loss": giou_loss,
            "MeanIoU": MeanIoU,
        },
    )
