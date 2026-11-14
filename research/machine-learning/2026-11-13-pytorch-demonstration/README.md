# PyTorch: from computational graphs to scientific inference and intelligent decisions

Created November 13, 2026. A buildless seven-chapter mathematical monograph, four original synthetic scientific studies, independent classical implementations, genuine CPU PyTorch fits, interactive browser laboratories, and a concise final PyTorch–TensorFlow comparison.

Main page: `../2026-11-13-pytorch-demonstration.html`.

## Open locally

From PowerShell:

```powershell
Set-Location -LiteralPath 'D:\GitHub\nareshpane.github.io'
python -m http.server 8765 --bind 127.0.0.1
```

Open http://127.0.0.1:8765/research/machine-learning/2026-11-13-pytorch-demonstration.html . Serve the repository root, not just this asset directory: one existing shared MathJax renderer lives elsewhere in the repository. HTTP is also necessary for JSON fetches and ES modules. GitHub Pages uses the same relative paths. No build framework, API key, Python web backend, or external inference service is required.

The article, native source disclosures/downloads, and mathematical text alternatives remain readable without JavaScript. Browser laboratories require JavaScript. Python models do not execute in the browser.

## Preliminary reference inspection and scope

Before creating this project, the complete TensorFlow reference HTML and all 54 associated files were inspected, including source, data/validation records, SVG structures, existing screenshots, and the published page. Its narrow independently scrollable navigation and source-generated disclosures informed engineering choices. Its prose, applications, scientific examples, experiment outputs, and dedicated assets were not reused. The reference and `research.html` were not edited. Existing unrelated Git work was preserved; nothing was staged, committed, or pushed.

The only shared runtime is the existing tracked file:

`research/miscellaneous/harmonized-system/details-of-machine-learning-models/js/vendor/tex-svg.js`

SHA256: `d4295dc33744836935c1399feece5159577b34c5c8ffb9f1c6324cd82e03a882`.

Its adjacent `MATHJAX-LICENSE.txt` remains in place. This is outside the TensorFlow folder and avoids another 2 MB library copy. Copy this shared renderer/license separately if extracting the page out of the repository. Text math remains the fallback if it is missing.

## Organization

- `css/styles.css`: responsive academic design, fit-content blue/peach equations, native gray source components, mobile drawer, print and reduced-motion styling.
- `js/navigation.js`: static enhanced six-chapter TOC, independent scroll, active subsection, progress, expand/collapse, focus-managed mobile drawer.
- `js/code-controls.js`: exact-source copy with fallback and independent full-height expansion.
- `js/utils.js`: SVG charting and finite playback with offscreen/hidden-page pause and reduced-motion handling.
- `js/foundations.js`: software responsibilities, real small tensor matrices, derivative-step sweep, scalar graph, neuron arithmetic.
- `js/demonstrations.js`: survival and historical graph laboratories, plus initialization of the coordinated river/city components.
- `js/river-lab.js`, `js/allocation-lab.js`: compact laboratory views and controls; the corresponding `*-model.js` modules expose the original numerical calculations for independent verification.
- `js/comparison.js`: actual CPU timing/quality/replication dashboard and compilation table.
- `images/`: thirteen original editable SVGs, illustration documentation, and selected browser QA screenshots.
- `python/`: canonical scientific programs, shared utilities, page content/compiler, exports, replication, benchmarks, validation, dependencies, and PowerShell reproduction.
- `outputs/seed-42/`: substantive canonical results, synthetic NPZ datasets, and eight Matplotlib SVG diagnostics.
- `outputs/replications.json`: compact substantive seed-42/43/44 evaluation records.
- `outputs/smoke/`: executed quick-pipeline results and diagnostic figures; explicitly not convergence evidence.
- `data/`: thinned browser views, raw benchmark/compilation data, structure/source counts, provenance hashes, numerical/static/browser validation records.
- `references.md`: checked primary sources and attribution.
- `qa-report.md`: actual executed tests, observations, limitations, and Git scope.

## Chapters

The structure is generated from `python/monograph_content.py`; `data/structure.json` is the authoritative machine-readable subsection list. It currently contains **61 subsections**: 13 foundations, 10 survival, 10 graph, 10 river, 9 allocation, and 9 synthesis.

1. Understanding PyTorch: Tensors, Derivatives, and Computational Graphs.
2. Biostatistics: Survival Analysis and Competing Risks.
3. Financial Networks: Learning From Transaction Graphs.
4. Environmental Science: Differentiable River-Contamination Inference.
5. Operations Research: Risk-Aware Emergency Resource Allocation.
6. Comparative Evaluation, Reproducibility, and Synthesis.

The progression distinguishes software framework, mathematical algorithm, statistical model, numerical verification, and scientific evidence. Every application explicitly separates equivalent computations (Level A) from differing models/decision procedures (Level B).

## Executed environment and installation

Executed on Windows AMD64, CPython 3.13.1, NumPy 2.4.0, SciPy 1.18.1, scikit-learn 1.9.1, Matplotlib 3.10.8, and torch 2.15.0+cpu. The per-run JSON records detected processor/platform, numerical threadpools, torch threads, dtype, seed, and device. A temporary virtual environment with system site packages supplied the existing numerical stack and isolated torch/plotting additions; global Python was not modified.

For independent reproduction, create a fresh environment. Consult https://pytorch.org/get-started/locally/ for a compatible platform/device command. The tested CPU wheel was verified against the official CPU package index rather than assuming an arbitrary CUDA build. Example from the repository's machine-learning directory:

```powershell
$venv = Join-Path $env:TEMP 'pytorch-monograph-repro'
python -m venv $venv
$py = Join-Path $venv 'Scripts\python.exe'
& $py -m pip install -r '.\2026-11-13-pytorch-demonstration\python\requirements.txt'
& $py -m pip install torch==2.15.0+cpu --index-url https://download.pytorch.org/whl/cpu
& $py -c "import torch; print(torch.__version__); print(torch.cuda.is_available())"
```

CPU execution is the implemented reproducibility target. GPU adaptation is optional future work; the programs intentionally operate on CPU rather than providing an untested device switch. Identical seeds need not produce identical bits on another library release/device.

## Run experiments and regenerate the page

The complete `python/reproduce.ps1` automates the tested CPU setup and steps. The following commands assume the asset folder is the current directory and `$py` points to the isolated Python:

```powershell
$env:PYTHONDONTWRITEBYTECODE = '1'
& $py python/00_pytorch_foundations.py --output outputs/seed-42
& $py python/run_all.py --quick --output outputs/smoke
& $py python/run_all.py --output outputs/seed-42
& $py python/replicate_experiments.py --output outputs
& $py python/benchmark_experiments.py --output outputs/seed-42
& $py python/05_framework_comparisons.py --output outputs/seed-42
& $py python/validate_experiments.py
& $py python/build_monograph.py
& $py python/generate_demo_assets.py
& $py python/validate_page.py
node --check js/main.js
node --check js/navigation.js
node --check js/code-controls.js
node --check js/utils.js
node --check js/foundations.js
node --check js/demonstrations.js
node --check js/comparison.js
node js/validate_numerics.mjs
```

Each application accepts `--seed`, `--quick`, and `--output`. To run it separately:

```powershell
& $py python/01_survival_analysis.py --seed 42 --output outputs/seed-42
& $py python/02_financial_graph_learning.py --seed 42 --output outputs/seed-42
& $py python/03_river_inverse_problem.py --seed 42 --output outputs/seed-42
& $py python/04_resource_allocation.py --seed 42 --output outputs/seed-42
```

Keep `common.py` and the sibling files together. Downloading a displayed function excerpt alone does not supply its imports/data generator. The code components label excerpts explicitly. Source text is extracted from canonical files and checked byte-for-text parity after HTML decoding. `explore_api.py` is the sole substantial disclosure open by default. Its output is bounded and source/signature failures are caught.

Every source panel also links to `python/python-source.zip`, a deterministic bundle of all Python/PowerShell sources and requirements so sibling imports remain together. The full asset folder is still needed to rebuild the article and reuse recorded results/illustrations.

After editing canonical source or prose, rebuild the page, regenerate assets/hashes, then validate. The compiler is deterministic; it does not hand-maintain a second program inside HTML. Timing results naturally differ between reruns.

## Experimental specifications

### Survival

1,800 independent subjects, 12 time bins, two causes, two synthetic covariates plus intercept. Conditional probabilities come from a three-category softmax. Censoring is at **end of bin**; a tied event is observed. A 60/20/20 subject split separates fit, validation selection, and final test. Same-model NumPy/torch fits share float64 data, zero initialization, ridge, L-BFGS-B, and tolerances. The nonlinear 16-unit model is a separate capacity comparison, trained for 350 epochs with best validation NLL selection.

IPCW Brier uses known generator G(t)=P(C≥t). Calibration groups use complete latent outcomes privileged to this synthetic audit; they are not a real-data censoring solution. Three independent seeds are descriptive replications, not a precise confidence interval. No real clinical causal or treatment claim is made.

### Financial graph

48 accounts, four institutions, 36 days × 120 events. A latent seven-account circular regime mixes with background transfers; independent thinning gives noisy labels, and amounts overlap. The historical graph uses exactly the previous three days. Scaling uses training days only; thresholds use validation only. Engineered logistic features already include neighborhood statistics. Native torch message passing uses repeated-edge incoming mean aggregation and an edge MLP. The 160-epoch neural run selects validation BCE. New-account evaluation is a new generator/seed and shorter early-prevalence process, distinct from late chronological drift.

### River

Dimensionless [0,1] spatial domain, T=1.2, 61 points, v=.35, D=.005, Gaussian source width .04, zero initial and Dirichlet endpoint values. Explicit upwind/centered stepping enforces **A+2B≤.75**. Both frameworks use exactly the same discretization and L-BFGS-B bounds/initialization. NumPy derivatives use three-point differencing; torch differentiates through all steps. A manufactured forced exact solution independently checks refinement at 31/61/121 points. Source recovery is a same-model/same-grid control; it does not assess physical misspecification. Interleaved held-out sensor rows share one trajectory. Additional fits vary sensor count/noise with three noise replicates each.

### Allocation

Eight districts, three stations, capacities 12/10/11. Scenario demand totals 18…31 keep the complete-network experiment feasible. 300 training and 160 independent test scenarios. A row-softmax policy and eta jointly optimize mean cost + .7 CVaR90 + .25 equal-district cost variance + 1.5 scaled squared capacity excess. Analytical and autograd gradients are checked on the identical objective. SciPy L-BFGS and torch Adam (1,200 steps) are different optimization procedures; training-time differences are not isolated framework speedups.

Integer dispatch uses deterministic score-ranked greedy realization, not probability rounding or nearest-point projection. Exact scenario-specific cost transportation is a separate objective; its integral LP structure makes MILP optional mathematically. Nearest preferences have station-index fallback; cost-ranked greedy is the stronger simple comparator. Browser shortages explicitly leave demand unserved with a disclosed 15-unit proxy penalty, without violating station capacity.

## Benchmark/evidence policy

Equivalent kernels use resident identical inputs, float64 CPU, one numerical thread, three warm-up calls, 31 repetitions, and median/IQR/raw samples. Initialization, plotting, disk, training, and compilation are excluded from steady kernel calls. River finite-difference versus reverse sensitivities changes derivative algorithms and is labeled. Full-model times are separate single-run observations. AOT eager capture works; Inductor compatibility failed on this host (see raw reason), so no optimizing compiled speedup is claimed.

Native CPU tensor allocator peak memory, GPU speed/memory, end-to-end production throughput, real observations, broad uncertainty coverage, exact discrete multiobjective optimality, and external deployment are **not measured**. We do not substitute Python-only tracemalloc for native tensor memory. Each laboratory identifies whether it uses analytical browser arithmetic, genuine fitted outputs, or a JavaScript simulator/optimizer.

## Data/provenance schema

Canonical JSON is human-readable and finite (`allow_nan=False`). Synthetic datasets use compressed NPZ with numeric arrays and no pickle objects. Browser JSON preserves actual profiles/embeddings/metrics; river display fields and allocation iterates are explicitly thinned. Benchmark JSON retains raw milliseconds and call counts; compilation JSON retains first-call/steady measurements or a failure type/reason. `replications.json` stores full evaluation metrics without redundant checkpoints. Manifest SHA256 hashes cover canonical source, browser data, original SVG assets, and deterministic structure records; the manifest and mutable validation reports are excluded from their own hash set.

See the QA report for tested browser widths, actual interactions, and remaining verification limits.

## Mathematical typesetting and Chapter 7 — current presentation

The latest pass audited all **39 existing numbered equations**. Every one now has genuine mathematical fallback content; explanatory sentences appear outside equation panels. **38 TeX sources** received alignment, notation, or formula revisions, including six specific clarifications: the gradient displacement identity, column-vector VJP, distinct baseline/learned edge feature dimensions, explicit inverse-loss sensor/time indices, a separate stencil-matrix symbol, and the allocation objective's optimized threshold dependency. The scientific programs and recorded results were not changed.

There are now **51 sequential numbered equations**, stable formula anchors, and six semantic HTML mathematical figures. The Jacobian figure uses actual bracketed matrices, column input/output vectors, forward blue and reverse peach lanes, and the explicit transpose relationship to the row convention. The other teaching figures use the same math renderer rather than improvised SVG text. Their earlier SVG source files are retained as historical geometry references. The live neuron calculation now renders mathematical expressions synchronized with its controls.

One TeX source drives both the existing **MathJax 3 SVG renderer** and a precompiled **native MathML fallback**. `python/math_typesetting.py` supplies consistent inline/display components; `js/render_math.mjs` compiles the cache with undefined-command/error suppression disabled. Mathematical alphabet styles are normalized for MathML Core compatibility. The page remains mathematical if site scripts or MathJax fail. The only added runtime library asset is the matching 4 KB `boldsymbol` extension, with its Apache 2.0 license in `js/mathjax-extensions/`; a second renderer is not shipped.

Chapter 7 has six subsections: shared mathematics, an API table, the same derivative, neural-network construction/training, Keras 3, and practical choice including Windows GPU constraints. The four paired examples and tiny Keras example are open on load; Chapters 1–6 retain the existing API-tour-only default exception. All five examples were executed: TensorFlow 2.21.0 and PyTorch 2.15.0+cpu both returned derivative **7.0**, model output shape **(2, 1)**, and **385 parameters**. A separate matched-weight float32 check agreed exactly on two fixed inputs; the independently initialized examples do not claim identical predictions. Keras 3.15.1 selected the PyTorch backend successfully. These are correctness checks, not performance benchmarks.

Normal rebuilding uses the shipped cache and requires no Node package installation:

```powershell
python python/build_monograph.py
python python/generate_demo_assets.py
python python/validate_page.py
python python/validate_typesetting.py
```

The numerical validators require the documented scientific environment. To edit TeX or introduce a new formula, collect sources and regenerate the cache with a temporary MathJax installation matching the shared renderer:

```powershell
$mathBuild = Join-Path $env:TEMP 'pytorch-math-build'
npm install --prefix $mathBuild --no-audit --no-fund --ignore-scripts mathjax-full@3.2.2
$env:MONOGRAPH_COLLECT_MATH = '1'
python python/build_monograph.py
Remove-Item Env:\MONOGRAPH_COLLECT_MATH
node js/render_math.mjs $mathBuild
python python/build_monograph.py
python python/validate_typesetting.py
python python/generate_demo_assets.py
python python/validate_page.py
```

Collection mode writes only mathematical sources, not a placeholder article. The normal compiler rejects a formula missing from the cache. The complete asset folder is required for rebuilding; the Python source ZIP preserves sibling imports but does not contain datasets, MathML cache, or figures.

Use `python/validate_framework_comparison.py --torch-python <interpreter> --tensorflow-python <interpreter>` when both dependencies are available in separate environments. It runs the canonical files, checks shapes/derivatives/parameter counts, and tests an explicitly matched weight set. No global package installation is required. See [qa-report.md](qa-report.md), [typesetting audit](data/typesetting-audit.json), and [exact changed-file paths](data/typesetting-changed-files.json).

## Earlier visual refinement — November 13, 2026

The existing monograph was refined in place. All six chapters, 61 subsections, 26 canonical source panels, experimental programs, and recorded scientific results remain. The page now has 39 numbered equations and six additional original mathematical diagrams. The body and reading surface are white (`#FFFFFF`); contained peach worked examples, blue algorithms, green conceptual material, and restrained lavender assumptions use a common panel design. Source disclosures retain their neutral gray treatment.

Section 4.7 is a coordinated desktop 2 × 2: river/source transport, three sensor trajectories with observations, the spatial field at the selected time, and the **recorded intensity-profiled source objective**. Physical sliders, sensor selection, and time controls sit beside their relevant views. All four share playback and source state. The fixed objective profile remains labeled with its original velocity/dispersion; changing the live physics displays a reference-only notice. Recorded fit playback retains full-precision Python parameters and distinguishes recorded loss from a freshly profiled objective. The river schematic redraws at the available width to keep mobile text readable.

Section 5.7 groups policy/capacity, risk/fairness, and demand controls. Its smaller city view shows station load/capacity numerically and with bars; a neighboring statistics panel reports demand, service, shortages, capacity, serving cost, shortage penalty, total cost, utilization, district penalized cost SD, and violations. The latter SD includes shortage costs and is distinct from the smooth training fairness variance. Separate compact charts show genuine recorded/calculated optimization history and realized costs over the 160 original held-out demands. Added D8 stress demand applies only to the selected realization. The unchanged shortage penalty is **15 proxy units per unserved unit**. SciPy exposes only its stored final objective; direct rules do not invent iteration paths. Risk/fairness edits explicitly select the 100-step JavaScript optimizer on 48 stored training scenarios.

At 1440 × 900 the river lab is about **827 px** high, versus **2,155 px** before refinement; the city lab is about **845 px**, versus **1,483 px**. The river's complete controls, four views, readouts, and note fit together in that viewport. The city fits nearly entirely; its full-section QA image uses a 1000 px viewport height. Small screens stack views and confine wide city diagrams, mathematical diagrams, and tables to local scroll regions. Scientific figures are centered and bounded; the five-column river sensitivity table uses about **561 px**, with centered numeric headings and values and unchanged precision.

New worked calculations explain JVP/VJP orientation, competing-risk probability budgets and censoring, synchronous duplicate-edge aggregation, an actual saved river sensor value and its source-intensity derivative, fractional CVaR and risk tradeoffs, and soft overload versus feasible integer decisions. Analytical examples are explicitly separated from empirical results. `python/refinement_content.py` adds these explanations through the existing compiler; `python/refinement_examples.py` calculates their values. The six new SVGs illustrate these mechanisms, using no external images.

No model was refitted for this pass. `data/refinement-baseline.json` records the pre-edit hashes; `data/refinement-validation.json` confirms all **32 protected scientific artifacts** and **56 outside-scope files** are byte-identical. This includes the existing `research.html` state and the TensorFlow reference. The original numerical validator still passes. New checks verify the teaching cases against independent torch/NumPy calculations, compare browser integer flows and costs to Python on **640 policy/scenario combinations**, and verify **7,680 stress cases**. Five-width browser results and nine selected screenshots are linked in [qa-report.md](qa-report.md).

To verify this presentation pass against the **retained checkout and recorded artifacts**, from this asset directory with the documented Python environment activated:

```powershell
python python/validate_refinement.py
node js/validate_refinement.mjs
node js/validate_numerics.mjs
Get-ChildItem -LiteralPath js -Filter '*.js' | ForEach-Object { node --check $_.FullName }
python python/build_monograph.py
python python/generate_demo_assets.py
python python/validate_page.py
```

The preservation audit requires the baseline files at their original repository paths. Deliberately rerunning empirical fits or benchmarks changes their hashes; the unchanged scientific reproduction workflow above remains available separately. No staging, commit, push, publication, or out-of-scope edits were performed.
