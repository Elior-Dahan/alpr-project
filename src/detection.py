"""Stage 1 — license-plate detection as bounding-box regression (Keras).

A MobileNetV2 backbone with a small regression head predicts a single
normalized box ``[xmin, ymin, xmax, ymax]`` in [0, 1]. Trained with a
Huber + (1 - GIoU) loss and a mean-IoU metric. One plate per image, so a
single box is sufficient.
"""

from __future__ import annotations

from pathlib import Path

import cv2
import keras
import numpy as np
import tensorflow as tf
from keras import layers
from keras.applications.mobilenet_v2 import preprocess_input

INPUT_SIZE = 224


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
    """Return (intersection, union, enclosing-area) for batched xyxy boxes."""
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
    """1 - GIoU, averaged over the batch."""
    eps = 1e-7
    inter, union, enclose = _box_iou_components(y_true, y_pred)
    iou = inter / (union + eps)
    giou = iou - (enclose - union) / (enclose + eps)
    return tf.reduce_mean(1.0 - giou)


def detector_loss(y_true, y_pred):
    """Huber on the 4 coords + (1 - GIoU)."""
    huber = keras.losses.Huber()(y_true, y_pred)
    return huber + giou_loss(y_true, y_pred)


class MeanIoU(keras.metrics.Metric):
    """Mean IoU over predicted vs. ground-truth boxes (normalized xyxy)."""

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
    raw = tf.io.read_file(path)
    img = tf.io.decode_image(raw, channels=3, expand_animations=False)
    img = tf.image.resize(img, (INPUT_SIZE, INPUT_SIZE))
    img = tf.cast(img, tf.float32)
    if augment:
        img = tf.image.random_brightness(img, 0.1)
        img = tf.image.random_contrast(img, 0.9, 1.1)
        img = tf.clip_by_value(img, 0.0, 255.0)
    img = preprocess_input(img)  # scales to [-1, 1]
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

def train(
    detection_dir: str | Path = "datasets/detection",
    model_out: str | Path = "models/detection/detector.keras",
    epochs: int = 100,
    batch_size: int = 16,
    lr: float = 1e-3,
) -> keras.Model:
    """Fine-tune the detector; save the best checkpoint by val mean IoU."""
    detection_dir = Path(detection_dir)
    model_out = Path(model_out)
    model_out.parent.mkdir(parents=True, exist_ok=True)

    train_ds = make_dataset(
        detection_dir / "train.csv", batch_size, augment=True, shuffle=True
    )
    val_ds = make_dataset(detection_dir / "val.csv", batch_size)

    model = build_detector()
    model.compile(
        optimizer=keras.optimizers.Adam(lr),
        loss=detector_loss,
        metrics=[MeanIoU()],
    )
    callbacks = [
        keras.callbacks.ModelCheckpoint(
            str(model_out),
            save_best_only=True,
            monitor="val_mean_iou",
            mode="max",
        ),
        keras.callbacks.ReduceLROnPlateau(
            monitor="val_mean_iou", mode="max", factor=0.5, patience=5
        ),
        keras.callbacks.EarlyStopping(
            monitor="val_mean_iou", mode="max", patience=15, restore_best_weights=True
        ),
    ]
    model.fit(train_ds, validation_data=val_ds, epochs=epochs, callbacks=callbacks)
    return model


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
