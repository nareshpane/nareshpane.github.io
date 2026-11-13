# TensorFlow: From Tensors to Intelligent Systems

Local static research monograph, created 12 November 2026. Open the sibling HTML through GitHub Pages or a local HTTP server. No build framework, external API, personal data, or browser machine-learning runtime is required.

## Evidence status

The authoring environment is **Python 3.13.1 / NumPy 2.4.0**, native Windows CPU. TensorFlow is **not installed**. Source syntax, generators, baseline fits, numerical invariants, exported data, and static-page checks were executed. Neural model imports, forward passes, optimization, and save/reload round trips remain unexecuted. Do not quote the included baselines as neural results.

Reproduction target (deliberate stable pins, not “latest”): **TensorFlow 2.20.0 / Keras 3.10.0 / NumPy 2.2.6**, Python 3.12 or 3.13. Compatibility is based on official release metadata; this exact training environment has not been run here. See [references.md](references.md) for release-specific evidence and APIs.

## Windows / PowerShell

Download the whole asset folder and open PowerShell in `python/`:

```powershell
py -0p
py -3.13 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
python -c "import tensorflow as tf; print(tf.__version__); print(tf.reduce_sum(tf.ones((2, 3)))); print(tf.config.list_physical_devices())"
python 00_tensorflow_foundations.py
python validate_experiments.py --tensorflow
python 01_computer_vision.py --quick --output outputs/vision-quick
python 02_natural_language_processing.py --quick --output outputs/nlp-quick
python 03_time_series_forecasting.py --quick --output outputs/forecast-quick
python 04_scientific_computing.py --quick --output outputs/heat-quick
```

If activation is restricted, use `.\.venv\Scripts\python.exe` directly. Do not change global execution policy. Native Windows CPU is the default; modern standard TensorFlow NVIDIA GPU execution uses Linux/WSL2. Native Windows CUDA support ended after TF 2.10. All scripts default `CUDA_VISIBLE_DEVICES=-1`; an explicitly pre-set environment variable can override this for an appropriately configured GPU environment.

After successful quick runs, omit `--quick` for fuller configurations. A fixed epoch/step count is not a convergence certificate. Inspect output metrics, independent validation, and failure examples; repeat seeds for substantive studies. Never add `.venv/`, training caches, or large output/model archives to the repository.

## Programs and defaults

Each of the six numbered files is independently executable and contains its own imports, generators, evaluation, and CLI. Copy Complete Python Program embeds its entire exact canonical text; no missing common helper is required.

| File | Full configuration | Quick configuration |
| --- | --- | --- |
| `00_tensorflow_foundations.py` | Shapes, dense pass, chain rule, one update, two cross-correlations, softmax assertions | No training |
| `01_computer_vision.py` | 1000/200/200 independent 64×64 scenes; 16/32/64-channel compact U-Net | 80/20/24 scenes, short fit |
| `02_natural_language_processing.py` | 240/60/60 messages **per class**, 4 classes; held-out template families, NB plus masked 24-D embeddings | 40/12/12 per class |
| `03_time_series_forecasting.py` | 2190 days; 56-day history, 6 features, 7 outputs; LSTM32; chronological 60/15/10/15 splits | 1460 days, 2 epochs |
| `04_scientific_computing.py` | Float64 tanh3×32, 1024 residual points, 4000 Adam steps | Width24, 256 residual points, 200 steps |
| `05_tensorflow_comparison.py` | NumPy regression, exact quadratic trajectories, two-hidden-layer forward pass, XOR checks | Optional `--tensorflow` verifies regression parity; no timing benchmark |

Seed is 42 by default (`--seed` supported). Every trained program saves a `.keras` model, verifies prediction parity after reload, and exports real metrics/history plus diagnostic artifacts. Exact filenames and settings are documented inside each source. Modern Keras saving and loss serialization are used; actual runtime validation awaits TensorFlow installation.

## Rebuild and validate assets

Only NumPy is needed for the numerical generator and baseline checks:

```powershell
python -B generate_demo_assets.py
python -B validate_experiments.py
python -B 05_tensorflow_comparison.py
python -B build_monograph.py
python -B validate_page.py
```

Run these from `python/`, or pass their paths from elsewhere. The asset generator resolves its default output relative to its own location. It intentionally imports numerical functions from the canonical application files, which lazily import TensorFlow only for neural execution. `-B` avoids bytecode cache directories. The article includes 75 numbered subsections across six chapters, including four application-specific TensorFlow contribution conclusions. The HTML is maintained directly; retained `CODE` and `FULL` comment markers allow `build_monograph.py` to update embedded panels deterministically. Tokenizer-based highlighting preserves the exact code text, including newlines and special characters. `validate_page.py` checks text parity against the sources and parses every embedded program/fragment.

Serve the repository from its root using a local-only server, for example:

```powershell
python -m http.server 8765 --bind 127.0.0.1
```

Then visit `http://127.0.0.1:8765/research/machine-learning/2026-11-12-tensorflow-demonstration.html`. Direct `file:` viewing preserves prose/source/figures but may block JSON loading and ES modules. Stop the server with Ctrl+C. No Python server is required on GitHub Pages.

## Browser data and provenance

| Asset | Meaning |
| --- | --- |
| `data/vision-demo.json` | Three independent procedural scenes, integer RGB export, true masks, fixed excess-green scores, micro confusion counts; **not neural probabilities** |
| `data/nlp-demo.json` | Training-only conventional vocabulary, exact fitted multinomial NB probe probabilities/log weights, real held-out metrics, difficult requests; separate from future Keras vocabulary |
| `data/forecasting-demo.json` | Executed six-year generator, additive components, chronological baseline forecasts and residual intervals; metrics over **323 daily origins**, gallery every seventh origin |
| `data/scientific-demo.json` | Exact and stable FTCS field on 41×21 space–time grid, independent errors, final-time refinement table, guarded unstable trace; no PINN values |
| `data/generation-manifest.json` | Actual environment, seed, provenance and baseline metric summaries |
| `data/python-validation.json` | Source/numerical tests and explicit unexecuted TensorFlow status |
| `data/comparison-validation.json` | Executed NumPy checks and explicit unexecuted TensorFlow parity status; not a benchmark |
| `data/page-validation.json` | Static anchors/assets/headings/code-parity checks |

Forecast origin `o` always means **first unobserved index**: features `[o-L:o]`, targets `[o:o+H]`. Last value is `Y[o-1]`; seasonal-naïve offset `h=0..6` is `Y[o+h-7]`. Target membership stays within each chronological partition. Input scaling uses training days only. Calibration and test origins are daily and overlap; temporal dependence invalidates automatic exchangeability-based coverage claims. Gallery thinning affects display only. Day zero is Monday 6 January 2020.

The browser text interface tokenizes any text using lowercasing, ASCII punctuation deletion, whitespace splitting, OOV1, padding0, and length48. It shows fitted probabilities **only for exact exported presets**. Edited text becomes tokenization-only. The exported NB log weights make a faithful conventional extension possible, but the interface does not impersonate arbitrary neural inference.

Heat coordinates, temperatures, and κ are dimensionless. The stable browser FTCS implementation recomputes when κ/resolution changes, enforces r≤0.4, and interpolates time. The independent instability visualization calculates amplification of a specified high-frequency mode and stops before overflow. No previously trained neural field is reused at a new diffusivity.

## Design, accessibility and licenses

CSS, five authored ES modules, original editable SVG scenes and mathematical diagrams, executed numerical diagnostic images, local MathJax, and compact JSON comprise the page. Navigation is independently scrollable and collapsible, with active chapter/subsection and progress state. All 26 canonical Python panels and the PowerShell panel use native disclosures, gray syntax-highlighted bodies, code icons, explicit Expand Code / Collapse Code labels, copy buttons, and optional full-height viewing. Six full Python programs and two complete short regression examples are distinguished from function excerpts. All controls have labels/focus states, charts have descriptions, and animations start paused, stop on leaving view, and have finite cycles. Copy uses Clipboard API with a selection-based fallback.

MathJax is the existing repository's local 3.x TeX-to-SVG renderer, copied into this dedicated folder for self-containment. Its [Apache 2.0 license](js/mathjax-LICENSE.txt) is retained. No font/CDN/network request is needed to read this page once its relative assets are available. TensorFlow/Keras are Apache 2.0 and NumPy BSD-3-Clause; their packages are not vendored here. All diagrams, exposition, synthetic generators, and observations were developed for this monograph; no external photograph or confidential dataset is used.

See [qa-report.md](qa-report.md) for actual browser verification, limitations, and the final file inventory. No changes were staged, committed, pushed, or published by this task.

## Revised foundations and authoring card

Section 1.3 compares custom numerical work with TensorFlow integration without asserting a speedup. Controlled scalar-graph, two-hidden-layer, and quadratic-optimization walkthroughs compute their displayed states. The four-line authoring card reproduces the user-supplied attribution values (model and reasoning) and updates Created to November 12, 2026 and Harness to Codex on Windows; it is presentation metadata, not a detected runtime report.

The source builder replaces only immediately adjacent generated panels and checks that article section IDs survive embedding. Its `SVG` markers embed the two animated diagrams from their canonical SVG assets. Rebuilding is idempotent.
