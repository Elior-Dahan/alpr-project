"""Stage 3 — plate text recognition (CRNN + CTC, Keras 3).

A lightweight 4-block CNN reduces a (64, 256, 1) grayscale plate to a feature
map whose width axis (T=32) is the sequence axis, which two BiLSTM layers and a
Dense layer turn into per-timestep character *logits* (not softmax —
``keras.losses.CTC`` and ``keras.ops.ctc_decode`` apply softmax internally).
Trained with the built-in ``keras.losses.CTC`` (Keras 3) and decoded greedily
via ``keras.ops.ctc_decode``.
"""

from __future__ import annotations

from pathlib import Path

import keras
import numpy as np
import tensorflow as tf
from keras import layers

from src.data import OCR_HEIGHT, OCR_WIDTH

# --- Character set --------------------------------------------------------

CHARS = "ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789"
BLANK_INDEX = 0  # CTC blank / padding / mask
NUM_CLASSES = len(CHARS) + 1  # 37: blank + 36 chars
# Real characters occupy indices 1..36.
CHAR_TO_IDX = {c: i + 1 for i, c in enumerate(CHARS)}
IDX_TO_CHAR = {i + 1: c for i, c in enumerate(CHARS)}

MAX_LABEL_LEN = 12  # longest Indian plate (e.g. MH20TC830C) fits comfortably
TIME_STEPS = 32  # CNN reduces width 256 -> 32 (>= 2 * max label length)


def encode_label(text: str) -> list[int]:
    """Map a plate string to integer indices (1..36), padded with 0 to MAX_LABEL_LEN.

    Index 0 is the CTC blank and doubles as the padding/mask value, so a padded
    label like ``[11, 12, 4, 5, 0, 0, ...]`` ("KL45") is unambiguous: the loss
    treats the trailing zeros as "no character here" rather than real classes.
    """
    idxs = [CHAR_TO_IDX[c] for c in text if c in CHAR_TO_IDX]
    idxs = idxs[:MAX_LABEL_LEN]
    return idxs + [BLANK_INDEX] * (MAX_LABEL_LEN - len(idxs))


# --- Model ----------------------------------------------------------------

def build_crnn() -> keras.Model:
    """CRNN: 4-block CNN -> 2x BiLSTM -> Dense logits. Output (B, TIME_STEPS, NUM_CLASSES).

    The CNN turns the image into a horizontal sequence of feature vectors (one
    per output column), which the BiLSTMs read left-to-right and right-to-left
    before the Dense layer scores each timestep over the 37 classes. CTC then
    aligns this length-32 sequence to the variable-length plate text.
    """
    inputs = keras.Input(shape=(OCR_HEIGHT, OCR_WIDTH, 1), name="image")

    x = inputs
    # Each block halves H; only the first three also halve W. The asymmetric
    # last pool keeps the width (time) axis long enough for CTC:
    #   H: 64 -> 32 -> 16 -> 8 -> 4    (/16)
    #   W: 256 -> 128 -> 64 -> 32 -> 32 (/8, so TIME_STEPS = 32)
    pool_sizes = [(2, 2), (2, 2), (2, 2), (2, 1)]
    filters = [64, 128, 256, 256]
    for f, pool in zip(filters, pool_sizes):
        x = layers.Conv2D(f, 3, padding="same", use_bias=False)(x)
        x = layers.BatchNormalization()(x)
        x = layers.Activation("relu")(x)
        x = layers.MaxPooling2D(pool)(x)

    # x: (B, H'=4, W'=32, C=256). Move width to the front so it becomes the time
    # axis, then fold the remaining height and channels into one feature vector
    # per timestep: (B, 32, 4*256=1024).
    x = layers.Permute((2, 1, 3))(x)  # (B, 32, 4, 256)
    x = layers.Reshape((TIME_STEPS, -1))(x)  # (B, 32, 1024)

    x = layers.Dropout(0.25)(x)
    x = layers.Bidirectional(layers.LSTM(256, return_sequences=True))(x)
    x = layers.Dropout(0.25)(x)
    x = layers.Bidirectional(layers.LSTM(256, return_sequences=True))(x)
    x = layers.Dropout(0.25)(x)
    # Linear logits: keras.losses.CTC and keras.ops.ctc_decode both expect
    # logits and apply softmax internally. (Greedy decode is argmax-invariant.)
    outputs = layers.Dense(NUM_CLASSES, name="logits")(x)

    return keras.Model(inputs, outputs, name="crnn")


# --- CTC decoding ---------------------------------------------------------

def ctc_greedy_decode(probs: np.ndarray) -> list[str]:
    """Greedy-decode model logits (B, T, NUM_CLASSES) into plate strings.

    ``keras.ops.ctc_decode`` merges repeats and strips the blank (mask_index=0);
    padding positions come back as -1. We map the remaining indices to chars.
    """
    probs = np.asarray(probs)
    batch = probs.shape[0]
    seq_len = np.full((batch,), probs.shape[1], dtype="int32")
    decoded, _ = keras.ops.ctc_decode(
        probs, sequence_lengths=seq_len, strategy="greedy", mask_index=BLANK_INDEX
    )
    seq = np.asarray(decoded[0])  # (B, T)
    results = []
    for row in seq:
        results.append("".join(IDX_TO_CHAR[int(i)] for i in row if int(i) > 0))
    return results


# --- Data pipeline --------------------------------------------------------

def _read_labels_csv(labels_csv: str | Path, split: str):
    import csv as _csv

    base = Path(labels_csv).parent
    paths, labels = [], []
    with Path(labels_csv).open() as f:
        for row in _csv.DictReader(f):
            if row["split"] != split:
                continue
            paths.append(str(base / row["filename"]))
            labels.append(encode_label(row["plate_text"]))
    return paths, np.asarray(labels, dtype=np.int32)


def _decode_gray(path, label, augment):
    raw = tf.io.read_file(path)
    img = tf.io.decode_png(raw, channels=1)
    img = tf.image.resize(img, (OCR_HEIGHT, OCR_WIDTH))
    img = tf.cast(img, tf.float32) / 255.0
    if augment:
        img = tf.image.random_brightness(img, 0.1)
        img = tf.image.random_contrast(img, 0.9, 1.1)
        img = tf.clip_by_value(img, 0.0, 1.0)
    return img, label


def make_dataset(
    labels_csv: str | Path,
    split: str,
    batch_size: int = 32,
    augment: bool = False,
    shuffle: bool = False,
) -> tf.data.Dataset:
    """tf.data pipeline yielding ((B, 64, 256, 1) images, (B, MAX_LABEL_LEN) labels)."""
    paths, labels = _read_labels_csv(labels_csv, split)
    ds = tf.data.Dataset.from_tensor_slices((paths, labels))
    if shuffle:
        ds = ds.shuffle(len(paths), reshuffle_each_iteration=True)
    ds = ds.map(
        lambda p, y: _decode_gray(p, y, augment),
        num_parallel_calls=tf.data.AUTOTUNE,
    )
    return ds.batch(batch_size).prefetch(tf.data.AUTOTUNE)


# --- Training -------------------------------------------------------------

class ValExactMatch(keras.callbacks.Callback):
    """Log true decoded exact-match on the val set as ``val_exact_match``.

    CTC ``val_loss`` is a poor proxy for readability: it bottoms out early while
    the net still predicts a near-constant string (prior collapse), and only
    later learns to read. Selecting on ``val_loss`` therefore restores the
    collapsed weights (exact-match ~0). This callback decodes the val set each
    epoch and injects the real exact-match so checkpoint / early-stopping can
    select on it instead.
    """

    def __init__(self, val_ds: tf.data.Dataset):
        super().__init__()
        self.val_ds = val_ds

    def on_epoch_end(self, epoch, logs=None):
        logs = logs if logs is not None else {}
        preds, truths = [], []
        # Pair preds and truths in the same loop (per batch) — no ordering bug.
        for images, labels in self.val_ds:
            preds.extend(ctc_greedy_decode(self.model.predict(images, verbose=0)))
            # Ground truth = inverse of encode_label: indices 1..36 -> chars,
            # dropping the blank/pad index 0.
            for row in labels.numpy():
                truths.append("".join(IDX_TO_CHAR[int(i)] for i in row if int(i) > 0))
        em = float(np.mean([p == t for p, t in zip(preds, truths)]))
        logs["val_exact_match"] = em
        print(f" — val_exact_match: {em:.4f}")


def train(
    labels_csv: str | Path = "datasets/ocr/labels.csv",
    model_out: str | Path = "models/ocr/crnn_best.keras",
    epochs: int = 40,
    batch_size: int = 32,
    lr: float = 1e-3,
) -> keras.Model:
    """Train the CRNN with CTC loss; checkpoint/early-stop on val exact-match.

    Monitors ``val_exact_match`` (via :class:`ValExactMatch`) rather than
    ``val_loss`` — see that callback for why. ``restore_best_weights`` returns the
    highest-exact-match epoch's weights.
    """
    model_out = Path(model_out)
    model_out.parent.mkdir(parents=True, exist_ok=True)

    train_ds = make_dataset(labels_csv, "train", batch_size, augment=True, shuffle=True)
    val_ds = make_dataset(labels_csv, "val", batch_size)

    model = build_crnn()
    model.compile(optimizer=keras.optimizers.Adam(lr), loss=keras.losses.CTC())
    # ValExactMatch must run first so it injects 'val_exact_match' into logs
    # before ModelCheckpoint / EarlyStopping read it the same epoch.
    callbacks = [
        ValExactMatch(val_ds),
        keras.callbacks.ModelCheckpoint(
            str(model_out), save_best_only=True,
            monitor="val_exact_match", mode="max",
        ),
        keras.callbacks.EarlyStopping(
            monitor="val_exact_match", mode="max",
            patience=20, restore_best_weights=True,
        ),
    ]
    model.fit(train_ds, validation_data=val_ds, epochs=epochs, callbacks=callbacks)
    return model


def load_ocr_model(model_path: str | Path) -> keras.Model:
    """Load a saved CRNN (CTC loss is a built-in, no custom objects needed)."""
    return keras.models.load_model(model_path)
