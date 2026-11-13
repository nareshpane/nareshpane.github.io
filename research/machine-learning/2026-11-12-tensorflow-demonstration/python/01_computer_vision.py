"""Synthetic urban vegetation segmentation with a small U-Net and color baseline.

Independent program: python 01_computer_vision.py --quick --output ../outputs/vision
All scenes are synthetic; there is no satellite data or claim of real-world validity.
NumPy-only functions can be imported without TensorFlow. Training needs requirements.txt.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import platform
import struct
import time
import zlib

os.environ.setdefault("CUDA_VISIBLE_DEVICES", "-1")
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")
import numpy as np


# SECTION 1 — A reproducible, editable data-generating process.
def generate_scene(seed: int, size: int = 64) -> tuple[np.ndarray, np.ndarray]:
    """Return RGB float32[H,W,3] and vegetation float32[H,W,1].

    Semantic masks are painted with the same geometry as objects, BEFORE shadows,
    illumination, and sensor noise. Green roofs are explicitly non-vegetation.
    Dry/shadowed plants remain vegetation. Later objects occlude earlier objects.
    """
    rng = np.random.default_rng(seed)
    yy, xx = np.mgrid[:size, :size]
    image = np.empty((size, size, 3), dtype=np.float64)
    image[:] = rng.uniform([.60, .57, .48], [.78, .73, .64])
    mask = np.zeros((size, size), dtype=bool)

    def paint(region, color, vegetation=False):
        image[region] = color
        mask[region] = vegetation

    # Broad lawn blocks surrounded by mineral pavement.
    for _ in range(int(rng.integers(1, 4))):
        x0, y0 = rng.integers(0, size - 10, 2)
        width, height = rng.integers(9, 25, 2)
        region = (xx >= x0) & (xx < x0 + width) & (yy >= y0) & (yy < y0 + height)
        paint(region, rng.uniform([.17, .32, .10], [.42, .66, .27]), True)

    # Cross streets with variable widths and markings; sidewalks precede asphalt.
    cx, cy = rng.integers(size // 4, 3 * size // 4, 2)
    half = int(rng.integers(3, 6))
    paint((abs(xx - cx) <= half + 1) | (abs(yy - cy) <= half + 1), [.68, .69, .66])
    road = (abs(xx - cx) <= half) | (abs(yy - cy) <= half)
    paint(road, rng.uniform(.19, .32, 3))
    paint(((xx == cx) & (yy % 9 < 4)) | ((yy == cy) & (xx % 9 < 4)), [.83, .79, .58])

    # Roofs include a deliberately confusing green-painted material.
    for _ in range(int(rng.integers(4, 10))):
        x0, y0 = rng.integers(1, size - 9, 2)
        width, height = rng.integers(5, 13, 2)
        region = (xx >= x0) & (xx < x0 + width) & (yy >= y0) & (yy < y0 + height)
        shadow = (xx >= x0 + 2) & (xx < x0 + width + 3) & (yy >= y0 + 2) & (yy < y0 + height + 3)
        image[shadow] *= .63
        color = (rng.uniform([.14, .33, .19], [.30, .55, .33]) if rng.random() < .25
                 else rng.uniform([.39, .33, .29], [.83, .70, .61]))
        paint(region, color)
        image[region & ((xx == x0) | (yy == y0))] = np.minimum(color + .11, 1)

    # Occasional water adds another blue-green confounder.
    if rng.random() < .4:
        wx, wy = rng.integers(7, size - 7, 2)
        water = ((xx - wx) / 7) ** 2 + ((yy - wy) / 4) ** 2 <= 1
        paint(water, rng.uniform([.10, .30, .31], [.25, .48, .53]))

    # Tree crowns are clustered but independently sampled; some are dry/brown.
    for _ in range(int(rng.integers(5, 23))):
        tx, ty = rng.integers(2, size - 2, 2)
        radius = float(rng.uniform(2, 5))
        canopy = (xx - tx) ** 2 + (yy - ty) ** 2 <= radius ** 2
        image[(xx - tx - 2) ** 2 + (yy - ty - 2) ** 2 <= radius ** 2] *= .60
        dry = rng.random() < .16
        color = (rng.uniform([.34, .31, .12], [.55, .48, .28]) if dry
                 else rng.uniform([.08, .26, .06], [.30, .58, .24]))
        paint(canopy, color, True)
        image[canopy] *= (1 + .08 * (xx[canopy] - tx) / radius)[:, None]

    # Multiplicative illumination and additive sensor noise never alter labels.
    lighting = rng.uniform(.72, 1.13) * (1 + rng.uniform(-.18, .18) * xx / size)
    image *= lighting[..., None]
    image += rng.normal(0, .022, image.shape)
    return np.clip(image, 0, 1).astype(np.float32), mask[..., None].astype(np.float32)


def make_dataset(n: int, seed: int, size: int = 64):
    """Independent child seeds, with exact scene duplication checked explicitly."""
    seeds = np.random.SeedSequence(seed).generate_state(n, dtype=np.uint64)
    pairs = [generate_scene(int(s), size) for s in seeds]
    images = np.stack([p[0] for p in pairs])
    masks = np.stack([p[1] for p in pairs])
    fingerprints = [hashlib.sha256(im.tobytes() + m.tobytes()).hexdigest() for im, m in pairs]
    assert len(set(fingerprints)) == n
    assert images.shape == (n, size, size, 3) and masks.shape == (n, size, size, 1)
    assert np.isfinite(images).all() and set(np.unique(masks)) <= {0., 1.}
    return images, masks


def color_score(images: np.ndarray) -> np.ndarray:
    """Fixed excess-green score, NOT a trained probability or neural prediction."""
    score = 2 * images[..., 1] - images[..., 0] - images[..., 2]
    return np.clip((score + .05) / .55, 0, 1)[..., None]


def segmentation_metrics(truth, scores, threshold: float = .5) -> dict:
    """Micro aggregation across every pixel; empty union/Dice is defined as one."""
    y, p = np.asarray(truth) >= .5, np.asarray(scores) >= threshold
    tp, fp = int(np.sum(y & p)), int(np.sum(~y & p))
    fn, tn = int(np.sum(y & ~p)), int(np.sum(~y & ~p))
    return {"tp": tp, "fp": fp, "fn": fn, "tn": tn,
            "accuracy": (tp + tn) / y.size,
            "precision": tp / (tp + fp) if tp + fp else 0.,
            "recall": tp / (tp + fn) if tp + fn else 0.,
            "iou": tp / (tp + fp + fn) if tp + fp + fn else 1.,
            "dice": 2 * tp / (2 * tp + fp + fn) if 2 * tp + fp + fn else 1.}


# SECTION 2 — A compact encoder–decoder, with same-resolution skip connections.
def build_model(tf, size=64):
    layers = tf.keras.layers

    def block(x, channels):
        x = layers.Conv2D(channels, 3, padding="same", activation="relu")(x)
        return layers.Conv2D(channels, 3, padding="same", activation="relu")(x)

    inputs = layers.Input((size, size, 3))
    skip1 = block(inputs, 16)                 # [N,64,64,16]
    skip2 = block(layers.MaxPool2D()(skip1), 32)  # [N,32,32,32]
    middle = block(layers.MaxPool2D()(skip2), 64) # [N,16,16,64]
    x = block(layers.Concatenate()([layers.UpSampling2D()(middle), skip2]), 32)
    x = block(layers.Concatenate()([layers.UpSampling2D()(x), skip1]), 16)
    outputs = layers.Conv2D(1, 1, activation="sigmoid")(x)
    return tf.keras.Model(inputs, outputs, name="compact_vegetation_unet")


def composite_loss(tf, y_true, y_pred):
    """BCE + 0.5 soft Dice loss, each image equally weighted, epsilon=1e-6."""
    # Probability clipping is confined to logarithms; Dice uses raw sigmoid p.
    clipped = tf.clip_by_value(y_pred, 1e-7, 1 - 1e-7)
    bce = -tf.reduce_mean(y_true * tf.math.log(clipped) +
                          (1 - y_true) * tf.math.log1p(-clipped), axis=(1, 2, 3))
    intersection = tf.reduce_sum(y_true * y_pred, axis=(1, 2, 3))
    mass = tf.reduce_sum(y_true + y_pred, axis=(1, 2, 3))
    dice = (2 * intersection + 1e-6) / (mass + 1e-6)
    return tf.reduce_mean(bce + .5 * (1 - dice))


def segmentation_loss(y_true, y_pred):
    """Named serializable loss; avoid a lambda closing over the TensorFlow module."""
    import tensorflow as tf
    return composite_loss(tf, y_true, y_pred)


# SECTION 3 — Small dependency-free scientific output (standard PNG).
def write_rgb_png(path: Path, image: np.ndarray):
    """Write RGB uint8 pixels using only Python's standard library."""
    image = np.asarray(image, dtype=np.uint8)
    height, width, channels = image.shape
    assert channels == 3

    def chunk(kind, data):
        return struct.pack(">I", len(data)) + kind + data + struct.pack(">I", zlib.crc32(kind + data) & 0xffffffff)

    rows = b"".join(b"\x00" + image[j].tobytes() for j in range(height))
    path.write_bytes(b"\x89PNG\r\n\x1a\n" +
                     chunk(b"IHDR", struct.pack(">IIBBBBB", width, height, 8, 2, 0, 0, 0)) +
                     chunk(b"IDAT", zlib.compress(rows, 9)) + chunk(b"IEND", b""))


def write_examples(path, images, masks, scores):
    """Five columns: RGB, truth, thresholded fit, overlay, errors; red FP/blue FN."""
    rows = []
    for image, truth, score in zip(images, masks, scores):
        y, p = truth[..., 0] > .5, score[..., 0] >= .5
        rgb = np.rint(image * 255).astype(np.uint8)
        truth_rgb, predicted_rgb = np.repeat(y[..., None], 3, -1) * 255, np.repeat(p[..., None], 3, -1) * 255
        overlay = rgb.copy().astype(float)
        overlay[p] = .55 * overlay[p] + .45 * np.array([58, 190, 87])
        errors = np.full_like(rgb, 242)
        errors[~y & p] = [190, 62, 52]
        errors[y & ~p] = [54, 104, 178]
        rows.append(np.concatenate([rgb, truth_rgb, predicted_rgb, overlay.astype(np.uint8), errors], axis=1))
    write_rgb_png(path, np.repeat(np.repeat(np.concatenate(rows, axis=0), 3, 0), 3, 1))


# SECTION 4 — Training, validation, fair baseline selection, and held-out test.
def run(args):
    import tensorflow as tf

    tf.keras.utils.set_random_seed(args.seed)
    tf.config.experimental.enable_op_determinism()
    counts = (80, 20, 24) if args.quick else (1000, 200, 200)
    split_seeds = np.random.SeedSequence(args.seed).generate_state(3)
    splits = [make_dataset(n, int(seed)) for n, seed in zip(counts, split_seeds)]
    (x_train, y_train), (x_val, y_val), (x_test, y_test) = splits
    # Exact duplicate detection across all splits, not merely within each split.
    hashes = [hashlib.sha256(im.tobytes()).hexdigest() for x, _ in splits for im in x]
    assert len(set(hashes)) == sum(counts)
    val_scores = color_score(x_val)
    thresholds = np.linspace(.1, .9, 17)
    threshold = float(max(thresholds, key=lambda q: segmentation_metrics(y_val, val_scores, q)["iou"]))

    def dataset(x, y, training=False):
        ds = tf.data.Dataset.from_tensor_slices((x, y))
        if training:
            ds = ds.shuffle(len(x), seed=args.seed, reshuffle_each_iteration=True)
        return ds.batch(8).prefetch(tf.data.AUTOTUNE)

    model = build_model(tf)
    loss_fn = tf.keras.utils.register_keras_serializable(package="Monograph")(segmentation_loss)
    model.compile(optimizer=tf.keras.optimizers.Adam(1e-3), loss=loss_fn, jit_compile=False)
    start = time.perf_counter()
    history = model.fit(dataset(x_train, y_train, True), validation_data=dataset(x_val, y_val),
                        epochs=2 if args.quick else 35, verbose=2,
                        callbacks=[tf.keras.callbacks.EarlyStopping(monitor="val_loss", patience=5,
                                                                    restore_best_weights=True),
                                   tf.keras.callbacks.TerminateOnNaN()])
    probabilities = model.predict(x_test, batch_size=8, verbose=0)
    assert probabilities.shape == y_test.shape and np.isfinite(probabilities).all()
    args.output.mkdir(parents=True, exist_ok=True)
    model_path = args.output / "vegetation.keras"
    model.save(model_path)
    restored = tf.keras.models.load_model(model_path, compile=False)
    np.testing.assert_allclose(restored.predict(x_test[:2], verbose=0), probabilities[:2], atol=1e-6)

    # Show failure cases as well as representative cases, sorted by per-image IoU.
    per_image = [segmentation_metrics(y, p) for y, p in zip(y_test, probabilities)]
    order = np.argsort([m["iou"] for m in per_image])[:4]
    write_examples(args.output / "worst_segmentation_examples.png", x_test[order], y_test[order], probabilities[order])
    output = {"provenance": "actual TensorFlow training on synthetic scenes", "seed": args.seed,
              "environment": {"python": platform.python_version(), "numpy": np.__version__, "tensorflow": tf.__version__},
              "configuration": {"counts": counts, "epochs_max": 2 if args.quick else 35, "threshold": .5,
                                "baseline_threshold_selected_on_validation": threshold, "loss": "BCE + 0.5 soft-Dice"},
              "neural_metrics": segmentation_metrics(y_test, probabilities),
              "baseline_metrics": segmentation_metrics(y_test, color_score(x_test), threshold),
              "history": history.history, "seconds": time.perf_counter() - start,
              "worst_examples": [{"test_index": int(i), **per_image[i]} for i in order],
              "model_parameters": model.count_params(), "reload_verified": True}
    (args.output / "results.json").write_text(json.dumps(output, indent=2), encoding="utf-8")
    np.savez_compressed(args.output / "held_out_predictions.npz", images=x_test[:8], truth=y_test[:8], probabilities=probabilities[:8])
    print(json.dumps(output, indent=2))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--quick", action="store_true", help="80/20/24 scenes, at most two epochs; a smoke test, not convergence")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output", type=Path, default=Path(__file__).resolve().parents[1] / "outputs/vision")
    args = parser.parse_args()
    try:
        run(args)
    except ModuleNotFoundError as exc:
        raise SystemExit("Install requirements.txt in an isolated environment first.") from exc
