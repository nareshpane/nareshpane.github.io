"""Executable numerical checks: tensors, convolution, softmax, and differentiation.

Run: python 00_tensorflow_foundations.py --output ../outputs/foundations
Requires only the packages in requirements.txt. CPU execution is the default.
"""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

os.environ.setdefault("CUDA_VISIBLE_DEVICES", "-1")
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")
import numpy as np


def run(seed: int = 42) -> dict:
    import tensorflow as tf

    tf.keras.utils.set_random_seed(seed)
    tf.config.experimental.enable_op_determinism()
    # SECTION 1 — Shapes and broadcasting preserve the number of elements.
    a = tf.reshape(tf.range(12, dtype=tf.float32), (3, 4))
    b = a + tf.constant([10., 20., 30., 40.])
    assert tuple(b.shape) == (3, 4)
    np.testing.assert_array_equal(b[0], [10., 21., 32., 43.])
    images = tf.zeros((16, 64, 64, 3))
    assert int(tf.rank(images)) == 4

    # SECTION 2 — A dense forward pass; TF stores a kernel as input×output.
    x = tf.constant([[1., 2.]])
    mathematical_weights = tf.constant([[1., -1.], [2., .5]])
    weights = tf.transpose(mathematical_weights) # Column-vector W becomes Keras kernel W^T.
    bias = tf.constant([0., 1.])
    z = x @ weights + bias
    activation = tf.nn.relu(z)
    output = activation @ tf.constant([[.5], [.25]])
    np.testing.assert_allclose(z, [[-1., 4.]])
    np.testing.assert_allclose(activation, [[0., 4.]])
    np.testing.assert_allclose(output, [[1.]])

    # SECTION 3 — One manually checkable gradient update.
    # yhat=w*x+b=2; L=(yhat-y)^2/2=.5 for x=2,y=3,w=1,b=0.
    w, intercept = tf.Variable(1.), tf.Variable(0.)
    with tf.GradientTape() as tape:
        predicted = 2. * w + intercept
        loss = .5 * tf.square(predicted - 3.)
    dw, db = tape.gradient(loss, [w, intercept])
    np.testing.assert_allclose([loss, dw, db], [.5, -2., -1.])
    w.assign_sub(.1 * dw)
    intercept.assign_sub(.1 * db)
    np.testing.assert_allclose([w, intercept], [1.2, .1], rtol=1e-6)
    new_loss = .5 * tf.square(2. * w + intercept - 3.)
    np.testing.assert_allclose(new_loss, .125, rtol=1e-6)

    # SECTION 4 — Neural libraries use cross-correlation, without kernel reversal.
    patch = tf.reshape(tf.range(1, 10, dtype=tf.float32), (1, 3, 3, 1))
    kernel = tf.reshape(tf.constant([1., 0., 0., -1.]), (2, 2, 1, 1))
    filtered = tf.nn.conv2d(patch, kernel, strides=1, padding="VALID")
    np.testing.assert_allclose(filtered.numpy().ravel(), [-4.] * 4)

    # The 3×3 numerical diagram uses a distinct, fully visible edge kernel.
    diagram_kernel = tf.reshape(tf.constant([1., 0., -1.] * 3), (3, 3, 1, 1))
    diagram_output = tf.nn.conv2d(patch, diagram_kernel, strides=1, padding="VALID")
    np.testing.assert_allclose(diagram_output.numpy().ravel(), [-6.])

    # SECTION 5 — Stable softmax and sparse cross-entropy agree with arithmetic.
    logits = tf.constant([[2., 1., 0., -1.]])
    probabilities = tf.nn.softmax(logits)
    cross_entropy = tf.nn.sparse_softmax_cross_entropy_with_logits(
        labels=[0], logits=logits
    )
    shifted = np.array([0., -1., -2., -3.])
    expected = np.exp(shifted) / np.exp(shifted).sum()
    np.testing.assert_allclose(probabilities.numpy()[0], expected, rtol=1e-6)
    np.testing.assert_allclose(cross_entropy, [-np.log(expected[0])], rtol=1e-6)

    return {
        "seed": seed, "tensorflow": tf.__version__, "numpy": np.__version__,
        "image_batch_shape": list(images.shape), "dense_z": z.numpy().tolist(),
        "dense_output": output.numpy().tolist(),
        "gradient": [float(dw), float(db)], "updated_parameters": [float(w), float(intercept)],
        "updated_loss": float(new_loss), "cross_correlation": filtered.numpy().tolist(),
        "softmax": probabilities.numpy().tolist(), "cross_entropy": float(cross_entropy[0]),
        "checks": "all numerical assertions passed",
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output", type=Path, default=Path(__file__).resolve().parents[1] / "outputs/foundations")
    args = parser.parse_args()
    try:
        result = run(args.seed)
    except ModuleNotFoundError as exc:
        raise SystemExit("Install requirements.txt in an isolated environment first.") from exc
    args.output.mkdir(parents=True, exist_ok=True)
    (args.output / "checks.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps(result, indent=2))
