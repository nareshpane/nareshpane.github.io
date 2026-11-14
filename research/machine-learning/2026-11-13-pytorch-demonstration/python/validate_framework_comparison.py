"""Execute the actual Chapter 7 snippets in separate framework interpreters.

No packages are installed and no scientific experiment artifacts are rebuilt.
Only data/framework-comparison-validation.json is written. These checks establish
example correctness, not training quality or a performance comparison.

Example:
    python validate_framework_comparison.py --torch-python PATH_TO_PYTHON
The TensorFlow interpreter defaults to this script's interpreter. PyTorch defaults
to the existing authoring environment in TEMP, when present, or this interpreter.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
import math
import os
from pathlib import Path
import subprocess
import sys
import tempfile

ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / "python"
REPORT = ROOT / "data" / "framework-comparison-validation.json"
SENTINEL = "FRAMEWORK_COMPARISON_JSON:"

# Each framework gets a fresh process. runpy executes the displayed canonical
# files, rather than a separately maintained copy of their example computations.
CHILD = r'''
import contextlib
import importlib.util
import io
import json
import math
from pathlib import Path
import platform
import runpy
import sys

framework, source_string = sys.argv[1:]
source = Path(source_string)
package = "torch" if framework == "pytorch" else "tensorflow"
if importlib.util.find_spec(package) is None:
    print("FRAMEWORK_COMPARISON_JSON:" + json.dumps({
        "status": "skipped", "reason": package + " is not installed",
        "python": platform.python_version(), "interpreter": sys.executable}))
    raise SystemExit(0)

import numpy as np
capture = io.StringIO()
with contextlib.redirect_stdout(capture):
    if framework == "pytorch":
        import torch
        torch.set_num_threads(1)
        gradient_namespace = runpy.run_path(str(source / "comparison_pytorch_gradient.py"))
        derivative = float(gradient_namespace["w"].grad)
        namespace = runpy.run_path(str(source / "comparison_pytorch_model.py"))
        model = namespace["model"]
        model.eval()
        with torch.inference_mode():
            default_output = model(torch.zeros((2, 10)))
        shape = list(default_output.shape)
        finite = bool(torch.isfinite(default_output).all())
        count = sum(parameter.numel() for parameter in model.parameters())
        version = torch.__version__
    else:
        import tensorflow as tf
        tf.config.threading.set_intra_op_parallelism_threads(1)
        tf.config.threading.set_inter_op_parallelism_threads(1)
        with tf.device("/CPU:0"):
            gradient_namespace = runpy.run_path(str(source / "comparison_tensorflow_gradient.py"))
            derivative = float(gradient_namespace["gradient"].numpy())
            namespace = runpy.run_path(str(source / "comparison_tensorflow_model.py"))
            model = namespace["model"]
            default_output = model(tf.zeros((2, 10)), training=False)
        shape = list(default_output.shape)
        finite = bool(np.isfinite(default_output.numpy()).all())
        count = model.count_params()
        version = tf.__version__
        assert tf.keras.backend.backend() == "tensorflow"

    assert derivative == 7.0, derivative
    assert float(capture.getvalue().strip()) == 7.0, capture.getvalue()
    assert shape == [2, 1], shape
    assert count == 385, count
    assert finite

    # Only this independent verification replaces the initially random weights.
    # Keras Dense stores input-by-output kernels; torch.nn.Linear stores their
    # transpose. Both use float32 for this matched-weight forward comparison.
    w1 = (np.arange(320, dtype=np.float32).reshape(32, 10) - 160) / 100
    b1 = np.linspace(-.2, .2, 32, dtype=np.float32)
    w2 = np.linspace(-.3, .3, 32, dtype=np.float32).reshape(1, 32)
    b2 = np.array([.125], dtype=np.float32)
    inputs = np.linspace(-1, 1, 20, dtype=np.float32).reshape(2, 10)
    reference = np.maximum(inputs @ w1.T + b1, 0) @ w2.T + b2
    if framework == "pytorch":
        with torch.no_grad():
            model[0].weight.copy_(torch.from_numpy(w1))
            model[0].bias.copy_(torch.from_numpy(b1))
            model[2].weight.copy_(torch.from_numpy(w2))
            model[2].bias.copy_(torch.from_numpy(b2))
        with torch.inference_mode():
            matched = model(torch.from_numpy(inputs)).numpy()
    else:
        model.set_weights([w1.T, b1, w2.T, b2])
        with tf.device("/CPU:0"):
            matched = model(tf.convert_to_tensor(inputs), training=False).numpy()
    assert np.isfinite(matched).all()
    np.testing.assert_allclose(matched, reference, rtol=1e-5, atol=1e-6)

result = {
    "status": "passed", "framework": framework, "version": version,
    "python": platform.python_version(), "numpy": np.__version__,
    "interpreter": sys.executable, "device": "cpu", "gradient_dtype": "float64",
    "gradient": derivative, "printed_gradient": capture.getvalue().strip(),
    "forward_shape": shape, "parameter_count": int(count),
    "default_forward_finite": finite, "matched_forward_dtype": "float32",
    "matched_forward": matched.tolist(), "numpy_reference": reference.tolist(),
    "matched_numpy_max_abs_error": float(np.max(np.abs(matched - reference))),
    "checks": ["actual canonical derivative snippet prints 7.0",
               "actual canonical model snippet returns shape (2, 1)",
               "model has 385 parameters and finite initial outputs",
               "controlled weights and inputs agree with independent NumPy forward"]
}
print("FRAMEWORK_COMPARISON_JSON:" + json.dumps(result, allow_nan=False))
'''

KERAS_CHILD = r'''
import importlib.util
import json
from pathlib import Path
import runpy
import sys

if importlib.util.find_spec("keras") is None:
    result = {"status": "skipped", "reason": "Keras is not installed in the PyTorch interpreter"}
else:
    namespace = runpy.run_path(str(Path(sys.argv[1]) / "comparison_keras_backend.py"))
    keras = namespace["keras"]
    assert keras.backend.backend() == "torch"
    result = {"status": "passed", "version": keras.__version__, "backend": keras.backend.backend(),
              "checks": ["actual canonical snippet selects torch before importing standalone keras"]}
print("FRAMEWORK_COMPARISON_JSON:" + json.dumps(result, allow_nan=False))
'''


def execute(interpreter: Path, code: str, arguments: list[str]) -> dict:
    if not interpreter.is_file():
        return {"status": "skipped", "reason": "Interpreter path does not exist", "interpreter": str(interpreter)}
    child_environment = os.environ.copy()
    child_environment.update(PYTHONIOENCODING="utf-8", PYTHONDONTWRITEBYTECODE="1",
                             CUDA_VISIBLE_DEVICES="-1", TF_CPP_MIN_LOG_LEVEL="2",
                             MPLCONFIGDIR=str(Path(tempfile.gettempdir()) / "codex-pytorch-comparison-mpl"))
    if arguments and arguments[0] == "tensorflow":
        child_environment["KERAS_BACKEND"] = "tensorflow"
    try:
        completed = subprocess.run([str(interpreter), "-B", "-c", code, *arguments],
                                   capture_output=True, text=True, encoding="utf-8",
                                   errors="replace", env=child_environment, timeout=180)
    except (OSError, subprocess.TimeoutExpired) as error:
        return {"status": "failed", "interpreter": str(interpreter), "error": str(error)}
    records = [line[len(SENTINEL):] for line in completed.stdout.splitlines() if line.startswith(SENTINEL)]
    if completed.returncode or len(records) != 1:
        return {"status": "failed", "interpreter": str(interpreter), "returncode": completed.returncode,
                "stdout": completed.stdout[-8000:], "stderr": completed.stderr[-8000:]}
    result = json.loads(records[0])
    result["stderr"] = completed.stderr.strip()
    return result


def main() -> None:
    authoring_torch = Path(tempfile.gettempdir()) / "codex-pytorch-20261113" / "Scripts" / "python.exe"
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--torch-python", type=Path,
                        default=authoring_torch if authoring_torch.is_file() else Path(sys.executable))
    parser.add_argument("--tensorflow-python", type=Path, default=Path(sys.executable))
    args = parser.parse_args()

    frameworks = {"pytorch": execute(args.torch_python, CHILD, ["pytorch", str(SOURCE)]),
                  "tensorflow": execute(args.tensorflow_python, CHILD, ["tensorflow", str(SOURCE)])}
    if frameworks["pytorch"]["status"] == "passed":
        keras = execute(args.torch_python, KERAS_CHILD, [str(SOURCE)])
    else:
        keras = {"status": "skipped", "reason": "PyTorch example execution is unavailable"}

    cross_framework = {"status": "skipped", "reason": "Both frameworks must execute before comparing outputs"}
    if all(result["status"] == "passed" for result in frameworks.values()):
        first = [value for row in frameworks["pytorch"]["matched_forward"] for value in row]
        second = [value for row in frameworks["tensorflow"]["matched_forward"] for value in row]
        passed = all(math.isclose(a, b, rel_tol=1e-5, abs_tol=1e-6) for a, b in zip(first, second))
        cross_framework = {"status": "passed" if passed else "failed", "dtype": "float32",
                           "rtol": 1e-5, "atol": 1e-6,
                           "max_abs_difference": max(abs(a-b) for a, b in zip(first, second)),
                           "protocol": "Same fixed input and replacement weights; Keras kernels transpose torch Linear weights"}

    failed = any(result["status"] == "failed" for result in [*frameworks.values(), keras, cross_framework])
    complete = all(result["status"] == "passed" for result in frameworks.values())
    files = sorted(SOURCE.glob("comparison_*.py"))
    report = {"status": "failed" if failed else "passed" if complete else "partial",
              "checked_at_utc": datetime.now(timezone.utc).isoformat(),
              "purpose": "Canonical Chapter 7 examples: correctness only, not a performance benchmark",
              "canonical_sources_sha256": {path.name: hashlib.sha256(path.read_bytes()).hexdigest() for path in files},
              "frameworks": frameworks, "matched_weight_parity": cross_framework, "keras_torch_backend": keras,
              "limitations": ["No model training or performance timing is performed",
                              "Identical architecture alone does not imply identical random initialization",
                              "Matched weights are applied only inside this verification, after executing original snippets"]}
    REPORT.parent.mkdir(parents=True, exist_ok=True)
    REPORT.write_text(json.dumps(report, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    summary = {"report": str(REPORT), "status": report["status"],
               "frameworks": {name: {key: result.get(key) for key in ["status", "version", "gradient", "forward_shape", "parameter_count"]}
                              for name, result in frameworks.items()},
               "matched_weight_parity": cross_framework, "keras_torch_backend": keras}
    print(json.dumps(summary, indent=2, allow_nan=False))
    if failed:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
