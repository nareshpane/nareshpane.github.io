"""Numerically compare handwritten NumPy gradients with TensorFlow autodiff.

Default (NumPy only):
    python 05_tensorflow_comparison.py
After installing requirements.txt, also check TensorFlow parity:
    python 05_tensorflow_comparison.py --tensorflow

This complete standalone program also checks the monograph's quadratic optimizer,
two-hidden-layer forward pass, and exact ReLU representation of the XOR corners.
It is a correctness comparison, not a performance benchmark. It does not train
any of the four application models or claim that TensorFlow is necessary here.
"""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import platform

os.environ.setdefault("CUDA_VISIBLE_DEVICES", "-1")
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")
import numpy as np


# SECTION 1 — A small model for which handwritten derivatives are reasonable.
def numpy_regression(steps: int = 100, learning_rate: float = .1) -> dict:
    """Fit y=wx+b to five exact observations using analytical gradients.

    Loss = mean((wx+b-y)^2)/2. Parameters, observations, and reductions are
    float64. Histories include the initial state, then every updated state;
    gradient_history[k] differentiates loss_history[k], before update k+1.
    """
    if steps < 1 or not np.isfinite(learning_rate) or learning_rate <= 0:
        raise ValueError("Require steps >= 1 and a finite positive learning rate")
    x = np.array([-2., -1., 0., 1., 2.], dtype=np.float64)
    y = 3. * x + 1.
    w, b = np.float64(0.), np.float64(0.)
    eta = np.float64(learning_rate)
    parameter_history = np.empty((steps + 1, 2), dtype=np.float64)
    gradient_history = np.empty((steps, 2), dtype=np.float64)
    loss_history = np.empty(steps + 1, dtype=np.float64)
    parameter_history[0] = [w, b]
    loss_history[0] = .5 * np.mean((w * x + b - y) ** 2)

    for step in range(steps):
        predictions = w * x + b
        residuals = predictions - y
        dw = np.mean(residuals * x)
        db = np.mean(residuals)
        gradient_history[step] = [dw, db]
        # Both gradients use the same pre-update parameters.
        w = w - eta * dw
        b = b - eta * db
        parameter_history[step + 1] = [w, b]
        loss_history[step + 1] = .5 * np.mean((w * x + b - y) ** 2)

    return {"final_parameters": parameter_history[-1].copy(),
            "parameter_history": parameter_history,
            "gradient_history": gradient_history,
            "loss_history": loss_history}


# SECTION 2 — The same model, observations, precision, and update convention.
def tensorflow_regression(steps: int = 100, learning_rate: float = .1) -> dict:
    """Use GradientTape to obtain the same two derivatives automatically.

    TensorFlow is imported only when this function is called. Eager execution
    keeps this tiny comparison transparent. No timing or speed claim is made.
    """
    import tensorflow as tf

    if steps < 1 or not np.isfinite(learning_rate) or learning_rate <= 0:
        raise ValueError("Require steps >= 1 and a finite positive learning rate")
    x = tf.constant([-2., -1., 0., 1., 2.], dtype=tf.float64)
    y = 3. * x + 1.
    w = tf.Variable(0., dtype=tf.float64)
    b = tf.Variable(0., dtype=tf.float64)
    eta = tf.constant(learning_rate, dtype=tf.float64)
    parameter_history = np.empty((steps + 1, 2), dtype=np.float64)
    gradient_history = np.empty((steps, 2), dtype=np.float64)
    loss_history = np.empty(steps + 1, dtype=np.float64)
    parameter_history[0] = [w.numpy(), b.numpy()]
    loss_history[0] = (.5 * tf.reduce_mean(tf.square(w * x + b - y))).numpy()

    for step in range(steps):
        with tf.GradientTape() as tape:
            predictions = w * x + b
            residuals = predictions - y
            loss = .5 * tf.reduce_mean(tf.square(residuals))
        dw, db = tape.gradient(loss, [w, b])
        tf.debugging.assert_all_finite(dw, "Nonfinite slope gradient")
        tf.debugging.assert_all_finite(db, "Nonfinite intercept gradient")
        gradient_history[step] = [dw.numpy(), db.numpy()]
        # Both taped gradients were computed before either assign_sub operation.
        w.assign_sub(eta * dw)
        b.assign_sub(eta * db)
        parameter_history[step + 1] = [w.numpy(), b.numpy()]
        loss_history[step + 1] = (.5 * tf.reduce_mean(tf.square(w * x + b - y))).numpy()

    return {"final_parameters": parameter_history[-1].copy(),
            "parameter_history": parameter_history,
            "gradient_history": gradient_history,
            "loss_history": loss_history}


# SECTION 3 — Exact trajectories for the quadratic learning-rate illustration.
def quadratic_trajectory(learning_rate: float, steps: int = 25,
                         theta_initial: float = 3., theta_star: float = 0.,
                         curvature: float = 1.) -> dict:
    """L(theta)=a(theta-theta_star)^2/2, with gradient a(theta-theta_star).

    The error multiplier is 1-eta*a. For a>0, convergence from every initial
    point requires 0<eta<2/a. This scalar-quadratic condition is not a general
    guarantee for nonlinear neural-network optimization.
    """
    if steps < 1 or curvature <= 0 or not np.isfinite(learning_rate):
        raise ValueError("Require steps >= 1, positive curvature, finite rate")
    theta = np.empty(steps + 1, dtype=np.float64)
    theta[0] = theta_initial
    for step in range(steps):
        theta[step + 1] = theta[step] - learning_rate * curvature * (theta[step] - theta_star)
    loss = .5 * curvature * (theta - theta_star) ** 2
    return {"learning_rate": float(learning_rate), "error_multiplier": 1 - learning_rate * curvature,
            "theta": theta, "loss": loss}


# SECTION 4 — Two nonlinear hidden layers, with explicit column-vector notation.
def two_hidden_forward() -> dict:
    """Verify every intermediate value in the article's two-hidden-layer example."""
    x = np.array([1., 2.], dtype=np.float64)
    w1 = np.array([[1., -1.], [2., .5]], dtype=np.float64)
    b1 = np.array([0., 1.], dtype=np.float64)
    z1 = w1 @ x + b1
    a1 = np.maximum(z1, 0.)
    w2 = np.array([[1., .25], [0., .5]], dtype=np.float64)
    b2 = np.array([0., -1.], dtype=np.float64)
    z2 = w2 @ a1 + b2
    a2 = np.maximum(z2, 0.)
    head = np.array([.5, .5], dtype=np.float64)
    output = head @ a2
    return {"x": x, "w1": w1, "b1": b1, "z1": z1, "a1": a1,
            "w2": w2, "b2": b2, "z2": z2, "a2": a2, "head": head,
            "output": float(output)}


def xor_relu_example() -> dict:
    """XOR on the four binary corners from two ReLU features.

    h1=max(x1-x2,0), h2=max(x2-x1,0); h1+h2=|x1-x2|.
    The output is exactly XOR for binary inputs, not a calibrated probability
    for arbitrary real inputs. One affine function cannot fit all four targets.
    """
    points = np.array([[0., 0.], [0., 1.], [1., 0.], [1., 1.]], dtype=np.float64)
    targets = np.array([0., 1., 1., 0.], dtype=np.float64)
    h1 = np.maximum(points[:, 0] - points[:, 1], 0.)
    h2 = np.maximum(points[:, 1] - points[:, 0], 0.)
    return {"inputs": points, "targets": targets, "hidden_features": np.column_stack([h1, h2]),
            "predictions": h1 + h2}


# SECTION 5 — Assertions independently connect formulas to executed calculations.
def validate_numpy_demonstrations() -> dict:
    regression = numpy_regression()
    parameters = regression["parameter_history"]
    losses = regression["loss_history"]
    np.testing.assert_allclose(regression["gradient_history"][0], [-6., -1.], rtol=0, atol=1e-15)
    np.testing.assert_allclose(parameters[1], [.6, .1], rtol=0, atol=1e-15)
    np.testing.assert_allclose(losses[:2], [9.5, 6.165], rtol=0, atol=2e-15)
    # Symmetric x has mean(x)=0 and mean(x^2)=2. Therefore the parameter
    # errors contract independently by 1-2*eta=.8 and 1-eta=.9.
    step_indices = np.arange(101, dtype=np.float64)
    closed_form = np.column_stack([3. * (1. - .8 ** step_indices), 1. - .9 ** step_indices])
    np.testing.assert_allclose(parameters, closed_form, rtol=1e-12, atol=2e-15)
    expected_losses = (closed_form[:, 0] - 3.) ** 2 + .5 * (closed_form[:, 1] - 1.) ** 2
    np.testing.assert_allclose(losses, expected_losses, rtol=1e-10, atol=2e-15)
    assert np.isfinite(losses).all() and np.all(np.diff(losses) <= 0.)
    assert losses[-1] < 4e-10

    trajectories = {"slow": quadratic_trajectory(.1), "stable": quadratic_trajectory(.8),
                    "oscillatory": quadratic_trajectory(1.5), "divergent": quadratic_trajectory(2.2)}
    for trajectory in trajectories.values():
        expected_theta = 3. * trajectory["error_multiplier"] ** np.arange(26)
        np.testing.assert_allclose(trajectory["theta"], expected_theta, rtol=1e-13, atol=1e-15)
        assert np.isfinite(trajectory["theta"]).all()
    for name in ["slow", "stable", "oscillatory"]:
        assert np.all(np.diff(trajectories[name]["loss"]) < 0.)
    assert np.all(trajectories["oscillatory"]["theta"][1:] * trajectories["oscillatory"]["theta"][:-1] < 0.)
    assert np.all(np.diff(trajectories["divergent"]["loss"]) > 0.)

    forward = two_hidden_forward()
    for key, expected in [("z1", [-1., 4.]), ("a1", [0., 4.]),
                          ("z2", [1., 1.]), ("a2", [1., 1.])]:
        np.testing.assert_array_equal(forward[key], expected)
    assert forward["output"] == 1.
    xor = xor_relu_example()
    np.testing.assert_array_equal(xor["predictions"], xor["targets"])
    return {"checks": ["NumPy regression gradients and first update agree with hand arithmetic",
                       "all 101 regression states and losses agree with independent closed forms",
                       "25-step slow, stable, oscillatory, and divergent quadratic trajectories agree with exact multiplier",
                       "two hidden ReLU layers and linear head match every stated intermediate value",
                       "two ReLU difference features represent all four XOR binary corners exactly"],
            "numpy_regression": regression, "quadratic_trajectories": trajectories,
            "two_hidden_forward": forward, "xor": xor}


def json_ready(value):
    """Convert NumPy results without changing the computed values."""
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, dict):
        return {key: json_ready(item) for key, item in value.items()}
    if isinstance(value, (tuple, list)):
        return [json_ready(item) for item in value]
    return value


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--tensorflow", action="store_true",
                        help="also execute TensorFlow and check all gradients, states, and losses")
    parser.add_argument("--report", type=Path,
                        default=Path(__file__).resolve().parents[1] / "data/comparison-validation.json")
    args = parser.parse_args()
    result = validate_numpy_demonstrations()
    result["environment"] = {"python": platform.python_version(), "numpy": np.__version__,
                             "dtype": "float64", "device_default": "CPU"}
    result["configuration"] = {"x": [-2., -1., 0., 1., 2.], "y": [-5., -2., 1., 4., 7.],
                               "initial_parameters": [0., 0.], "learning_rate": .1, "steps": 100,
                               "loss": "0.5 * mean((w*x+b-y)**2)"}
    result["tensorflow"] = {"executed": False, "status": "not requested; run --tensorflow in the documented environment"}
    result["performance_benchmark"] = "not performed; this file tests numerical correctness only"
    if args.tensorflow:
        try:
            tf_result = tensorflow_regression()
        except ModuleNotFoundError as exc:
            raise SystemExit("TensorFlow is unavailable. Install requirements.txt in an isolated environment, then rerun --tensorflow.") from exc
        import tensorflow as tf
        errors = {}
        for key in ["gradient_history", "parameter_history", "loss_history", "final_parameters"]:
            np.testing.assert_allclose(tf_result[key], result["numpy_regression"][key], rtol=1e-11, atol=1e-12)
            errors[key] = float(np.max(np.abs(tf_result[key] - result["numpy_regression"][key])))
        result["tensorflow"] = {"executed": True, "version": tf.__version__,
                                "status": "all gradients, states, losses, and final parameters match within rtol=1e-11, atol=1e-12",
                                "maximum_absolute_differences": errors}
        result["checks"].append("TensorFlow float64 autodiff agrees with NumPy analytical gradients through all 100 updates")
    result = json_ready(result)
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(json.dumps(result, indent=2, allow_nan=False), encoding="utf-8")
    print(json.dumps({"report": str(args.report.resolve()), "environment": result["environment"],
                      "checks": result["checks"], "final_parameters": result["numpy_regression"]["final_parameters"],
                      "final_loss": result["numpy_regression"]["loss_history"][-1],
                      "tensorflow": result["tensorflow"], "performance_benchmark": result["performance_benchmark"]}, indent=2))


if __name__ == "__main__":
    main()
