# Current revision verification — 12 November 2026

This record covers the comprehensive revision and the subsequent refinement request, including the four-line authoring card. It supersedes the counts, design description, screenshots, and file inventory in the historical creation record below. Revision scope is exclusively the sibling monograph HTML and this asset folder. No staging, commits, pushes, or publication.

## Implemented changes

- Removed the old breadcrumb/date bar, made the page white, retained the narrow independent contents sidebar, and replaced the inconsistent paragraph-width restriction with one shared reading width. The title occupies a full row and fits on one line at the tested 1440 px desktop width; the explanatory artwork follows below.
- Added the requested top-right card at 290 × 139 CSS pixels with 12 px text, 25 px line spacing, cream fill, green border, rust labels, and dark blue values. Created is November 12, 2026; Harness is Codex on Windows. Model and Reasoning retain the user-supplied screenshot values, GPT Sol-6.1 and High; these are requested attribution text, not detected runtime telemetry.
- Refined the original overview and application scenes, added four coordinated illustrated application cards, and retained the unboxed evidence note and every original chapter.
- All 26 canonical Python disclosures plus the PowerShell disclosure use the shared light-gray component. Each has a line-art code symbol, native details/summary semantics, explicit Expand Code / Collapse Code text, a rotating chevron, and a toolbar outside the summary. Syntax highlighting, copying, downloads, independent panel states, full-height expansion, and visible focus remain available. Labels work without JavaScript through native open-state CSS. PowerShell is labeled correctly.
- Equation boxes fit their mathematical content, retain numbers, and keep all explanatory paragraphs outside. Green supports foundational relationships; peach marks gradient updates, optimization derivations, and the FTCS update; blue supports probability, statistical expressions, and differential operators. Inline expressions and aligned derivations were visually checked.
- Added Section 1.3, “What if TensorFlow did not exist?”, with equivalent complete short NumPy and GradientTape regression programs, manual derivative calculations, numerical alternatives, software responsibilities, and qualified performance discussion. Existing Chapter 1 anchors remain stable; numbering now ends at 1.13.
- Rebuilt the two-lane image-pipeline figure with vertically aligned stages and four benefit callouts: integrated data pipelines, optimized tensor computation, automatic differentiation, and less custom training infrastructure. It explicitly allows optimized custom libraries and other autodiff frameworks. The caption distinguishes the conceptual file-decoding pipeline from this experiment's in-memory arrays. Development efficiency, pipeline throughput, runtime, and statistical accuracy are separate concepts; no speed multiplier or neural accuracy improvement is claimed.
- Rebuilt the scalar graph as forward computation, reverse sensitivities, and before/after update rows. Observed x/y and trainable w/b are separate entry nodes. Solid forward arrows and dashed coral derivative flow accompany local derivatives. Seven controlled steps highlight the active operation and synchronize exact values. The final comparison records the particular 75% loss decrease, with no general descent guarantee.
- Added exact controlled two-hidden-layer and quadratic-optimization demonstrations, with explicit matrices, XOR expressiveness, learning-rate regimes, shape conventions, and derivations.
- Added four distinct concluding subsections: 2.13 segmentation responsibilities and a comparison table; 3.13 conventional tokenization versus integrated embeddings; 4.14 baseline indexing versus an LSTM's direct seven-horizon output; 5.15 FTCS versus nested neural derivatives and parameter optimization. Six extra canonical excerpts support these comparisons. Each separates implementation convenience, model capacity, potential computation gains, observed baseline evidence, and remaining scientific responsibilities.
- Added or clarified numerical broadcasting, BCE, mask-aware pooling, forecast availability, conformal ranking, zero-field loss, and tanh derivative examples. Fixed ambiguous “one-parameter predictor” wording to identify both weight and bias.
- Hardened source embedding to replace only immediately adjacent generated panels; fresh markers cannot swallow intervening prose. Section IDs are checked before writing. Animated SVGs are embedded from canonical files, and external SVG dimensions reserve layout space.

## Executed validation

Environment: Python 3.13.1, NumPy 2.4.0, Windows CPU. TensorFlow is absent.

- Existing experiment validator: all six reported source/numerical/asset groups passed, covering generators, splits, vocabulary, leakage boundaries, metrics, exact heat derivatives, FTCS stability and convergence, JSON, and canonical numerical export parity.
- New comparison program: all five groups passed. They check analytical regression gradients and the first update, all 101 states against closed forms, 25-step quadratic trajectories in four regimes, every intermediate two-hidden-layer value, and all four XOR corners.
- Independent JavaScript assertions checked the scalar graph, hidden-layer values, and 100 quadratic updates against closed-form recurrences. New tanh first-time and second-space derivative formulas agreed with independent finite differences at two nontrivial points.
- Syntax checks passed for all five authored JavaScript modules. Every Python source and every embedded canonical example parsed.
- Final static page validation: 26 canonical Python panels, six complete programs, 111 headings, 198 explicit anchors, 20 external images, approximately 23,076 prose words, and zero errors. All 75 subsection boundaries and expected chapter counts are retained. All 27 code disclosures contain the four shared header components.
- Rebuilding a second time produced byte-identical HTML. Every embedded Python panel's text is exactly equal to its canonical source or defined complete-example extraction.

## Browser verification actually performed

Tested through a temporary loopback HTTP server in the Codex in-app Chromium browser. The temporary viewport override was reset and the owned server stopped after QA.

- Inspected the opening/card, four application openings, both revised mathematical figures, the network and optimization demonstrations, all four new concluding sections, and advanced loss/probability/uncertainty/physics sections. Inspected collapsed and expanded gray panels on desktop and mobile.
- Checked widths 1920, 1600, 1440, 1280, 1024, 768, and 390 px. No whole-document horizontal overflow and no overflowing disclosure header at any width. Paragraph widths were respectively 1360, 1289, 1129, 969, 737, 705, and 341 px. The short update equation remained approximately 204–258 px wide. Local scrolling for long source, formulas, tables, and diagrams is intentional.
- The authoring card measured exactly 290 × 139 CSS pixels and fit the 390 px mobile view. The desktop title remained a single line.
- All 27 panels opened and collapsed independently and exposed the correct action label. All 27 copy actions yielded exact source text, including six full programs and the PowerShell block. The first asynchronous clipboard read was stale; a separate completed copy/read verified exact equality. Enter and Space toggling, mobile copying, and full-height expansion were exercised. Clipboard fallback was not force-triggered.
- TOC Expand all opened six chapter groups and Collapse all closed all six. Each new concluding anchor aligned near 24 px on desktop and highlighted the corresponding entry. Mobile navigation closed the drawer, aligned the selected subsection near 68 px, and highlighted it correctly.
- Graph Next advanced through all seven stages; final values were w=1.2, b=0.1, a=2.5, loss=0.125 and end controls disabled. Play completed the seven-step sequence and stopped; pause, restart, and offscreen stopping were checked.
- Network stepping reached z1=[-1,4], a1=[0,4], z2=[1,1], a2=[1,1], output=1. Quadratic UI checks: eta 1.5 gives theta=-1.5 and loss=1.125 after one step; eta 2.2 gives theta=4.32 and loss=9.3312 after two steps with an off-plot warning; eta 1 gives theta=0 and loss=0 after one step.
- Existing image threshold/error controls reproduced TP128/FP3/FN110/TN3855 at threshold 0.8 on the first scene. Editing the text input showed tokenization-only status. Forecast controls updated origin/history/target alignment and displayed errors. Heat controls reproduced exact u=0.372708 at t=0.5, kappa=0.2, x=0.5, with 21-point FTCS error 0.00106251; requested unstable ratio 0.65 produced mode multiplier -1.595993.
- All 20 article images loaded. No MathJax error elements were present, including after opening the advanced quadratic derivation. All 29 equation panels contained zero explanatory paragraphs. Final console inspection returned no warnings or errors.

Proof: [desktop opening and authoring card](images/qa-desktop.jpg), [mobile expanded code panel](images/qa-mobile.jpg).

## Limits

TensorFlow imports, neural training, GradientTape parity, neural metrics, save/reload execution, and speed benchmarks were not run. The supplied optional TensorFlow checks remain available. Existing executed conventional data have not been relabeled as neural results. No Firefox/Safari, screen-reader, native-device, GPU, OS reduced-motion toggle, forced clipboard fallback, or exhaustive external-link HTTP audit is claimed. Reduced-motion and no-JavaScript fallbacks were source-reviewed. No deployment occurred.

## Exact revision file inventory

Modified relative to the pre-revision snapshot (20 files):

- [2026-11-12-tensorflow-demonstration.html](../2026-11-12-tensorflow-demonstration.html)
- [2026-11-12-tensorflow-demonstration/README.md](README.md)
- [2026-11-12-tensorflow-demonstration/css/styles.css](css/styles.css)
- [2026-11-12-tensorflow-demonstration/data/page-validation.json](data/page-validation.json)
- [2026-11-12-tensorflow-demonstration/images/computational-graph.svg](images/computational-graph.svg)
- [2026-11-12-tensorflow-demonstration/images/computer-vision-scene.svg](images/computer-vision-scene.svg)
- [2026-11-12-tensorflow-demonstration/images/nlp-university.svg](images/nlp-university.svg)
- [2026-11-12-tensorflow-demonstration/images/overview.svg](images/overview.svg)
- [2026-11-12-tensorflow-demonstration/images/qa-desktop.jpg](images/qa-desktop.jpg)
- [2026-11-12-tensorflow-demonstration/images/qa-mobile.jpg](images/qa-mobile.jpg)
- [2026-11-12-tensorflow-demonstration/images/scientific-laboratory.svg](images/scientific-laboratory.svg)
- [2026-11-12-tensorflow-demonstration/images/seasonal-city.svg](images/seasonal-city.svg)
- [2026-11-12-tensorflow-demonstration/js/code-controls.js](js/code-controls.js)
- [2026-11-12-tensorflow-demonstration/js/demonstrations.js](js/demonstrations.js)
- [2026-11-12-tensorflow-demonstration/js/main.js](js/main.js)
- [2026-11-12-tensorflow-demonstration/js/navigation.js](js/navigation.js)
- [2026-11-12-tensorflow-demonstration/python/build_monograph.py](python/build_monograph.py)
- [2026-11-12-tensorflow-demonstration/python/validate_page.py](python/validate_page.py)
- [2026-11-12-tensorflow-demonstration/qa-report.md](qa-report.md)
- [2026-11-12-tensorflow-demonstration/references.md](references.md)

Added (9 files):

- [2026-11-12-tensorflow-demonstration/data/comparison-validation.json](data/comparison-validation.json)
- [2026-11-12-tensorflow-demonstration/images/application-card-forecast.svg](images/application-card-forecast.svg)
- [2026-11-12-tensorflow-demonstration/images/application-card-language.svg](images/application-card-language.svg)
- [2026-11-12-tensorflow-demonstration/images/application-card-physics.svg](images/application-card-physics.svg)
- [2026-11-12-tensorflow-demonstration/images/application-card-vision.svg](images/application-card-vision.svg)
- [2026-11-12-tensorflow-demonstration/images/manual-vs-tensorflow-pipeline.svg](images/manual-vs-tensorflow-pipeline.svg)
- [2026-11-12-tensorflow-demonstration/images/network-composition.svg](images/network-composition.svg)
- [2026-11-12-tensorflow-demonstration/js/foundations.js](js/foundations.js)
- [2026-11-12-tensorflow-demonstration/python/05_tensorflow_comparison.py](python/05_tensorflow_comparison.py)

No files were removed. SHA256 comparison confirms the five original numbered Python programs, requirements, baseline generator/validator, experimental JSON, numerical diagnostic assets, and vendored MathJax are unchanged. The pre-revision and final `research.html` hash is `239ccbc4fba1286d9a181cb47cbbcdccbbaddbc6a11073f1578cc6e5033afaf0`. Unrelated tracked and untracked work was preserved.

---

# Historical creation record

The following is the original creation-stage record. Its original panel counts, warm-background design description, and index insertion refer to that earlier stage; the current revision above governs the delivered page.

# Verification record

Date: 12 November 2026. Scope: the new monograph, its dedicated asset folder, and one insertion in the Machine Learning subsection of the existing `research.html`. No staging, commits, pushes, or deployment.

## Executed Python and static checks

- Python **3.13.1**, NumPy **2.4.0**, native Windows CPU. The target training environment is separately pinned in `python/requirements.txt`.
- `python -B python/generate_demo_assets.py`: completed; deterministic procedural scenes, fitted count-based naïve Bayes, conventional forecast baselines and chronological residual intervals, exact heat fields and stable FTCS. Actual data provenance is in `data/generation-manifest.json`.
- `python -B python/validate_experiments.py`: all seven validation groups passed. All Python sources parse and compile without creating bytecode caches; no TensorFlow import is needed for these numerical tests.
- Checks include scene geometry/mask alignment and dimensions; independent scenes; normalized-text deduplication and separated template families; training-only vocabulary; NB probability normalization; hand-calculated confusion metrics; additive positive demand; training-only scalers; legal window boundaries and future-sentinel isolation; weekly baseline indexing; corrected conformal order statistics; independent analytical heat derivatives and conditions; stable FTCS; rejected unstable settings, guarded unstable trace, and mesh convergence; finite JSON and export parity.
- `python -B python/build_monograph.py`: embeds **17** canonical Python panels, including **five** complete standalone programs. Python tokenization provides restrained highlighting without changing copied text.
- `python -B python/validate_page.py`: zero errors for unique anchors, heading hierarchy, relative assets, local links, alternatives, SVG view boxes, finite JSON, embedded Python syntax, and exact text parity with canonical sources. Approximate prose count includes equation alternatives/captions and excludes displayed code: about **17,000** words.
- `node --check` passed for all four authored JavaScript modules. The local MathJax renderer was reused from the repository; its Apache license is included.

## Browser checks actually performed

Served from a temporary loopback HTTP server rooted at the existing repository; tested in the Codex in-app Chromium browser. The server was stopped after verification. Screenshots are durable local QA artifacts, not a hosted deployment.

- Rendered page title, subtitle, warm palette, opening figure, four problem cards, six chapters, tables, mathematical panels, illustrations, numerical figures, code panels, and laboratories inspected in representative views across the article.
- **24** MathJax expressions rendered, with **zero** `mjx-merror` elements. Final browser console inspection reported no warnings or errors. All 16 article images were successfully loaded after visiting their sections; no laboratory error statuses remained.
- Document width checked at **1440, 1280, 1024, 768, and 390 px**: no whole-document horizontal overflow. Internal code/table/equation scrolling is intentional. Temporary viewport override reset afterward.
- Desktop navigation: expand all, collapse all, independent chapter links, active chapter/subsection, independent TOC scrolling, progress, and preservation of deliberately collapsed groups verified. Fixed an initial combined anchor-offset issue so anchor targets align with highlighting.
- Added image dimensions from SVG view boxes and loaded the small SVG group eagerly after observing lazy-image layout shifts during anchor jumps. PNGs retain reserved dimensions and lazy loading.
- Mobile drawer: open, navigate to a subsection, close automatically, and accurate active subsection verified at 390 px. Code disclosure, Copy Code, and full-height expansion verified on mobile without page overflow.
- All five complete Python program copy actions produced text exactly equal to their source panels (foundation copy rechecked after its asynchronous clipboard operation completed). A nested-tape fragment also copied exactly, including indentation, operators, and Unicode comments. Static checks independently compare every code panel against its source file. Clipboard API worked; the legacy fallback was source-reviewed but not force-triggered.
- Tensor shape control: batch 32 produced 393,216 entries and 1,572,864 float32 bytes. Gradient control at η=0.2 produced w=1.4, b=0.2, prediction=3, loss=0. Convolution mean control at final position produced 19, and restart returned the first-position mean 7.
- Image controls: channel view, threshold, error view, overlay, and opacity state verified. Threshold 0.8 on the first scene changed counts to TP128/FP3/FN110/TN3855, with the corresponding metrics recomputed.
- Text presets display actual stored conventional probabilities; editing arbitrary text switches to tokenization-only with OOV/padding encoding. No fabricated arbitrary neural probabilities appear.
- Forecast component toggles recompute the displayed sum/ACF. Moving the gallery origin and selecting a shorter displayed horizon updates history/target indices and actual baseline errors. Baseline/interval controls, aggregate horizon MAE, residual ACF, and coverage table render.
- Heat time, κ, spatial resolution, and stability controls verified. At t=0.5, κ=0.2, x=0.5, exact u=0.372708; 21-point FTCS ratio0.4 and maximum profile error0.00106251. Requested unstable ratio0.65 gives |G|1.595993 for the specified error mode. Animation advanced to t=1 and stopped. Restart and offscreen/visibility stopping behavior are implemented; all animations begin paused.
- Existing Research page inspected with its Machine Learning disclosure open: exactly one TensorFlow entry. Following that link opened the new page with the correct title and relative URL.

Screenshots: [desktop opening](images/qa-desktop.jpg), [mobile image inspector](images/qa-mobile.jpg).

## Actual numerical results, with their population

| Calculation | Executed result | Interpretation |
| --- | --- | --- |
| Color baseline | IoU0.422249, accuracy0.921387 | Micro counts over three illustrative scenes, **not a neural test score** |
| Count-based NB | Accuracy0.758333, macro-F10.746269 | 240 template-held-out synthetic test messages |
| Seasonal-naïve forecast | MAE55.164236, RMSE71.152207 MWh/day | All323 daily final-test origins, seven horizons |
| Seasonal residual interval | Coverage0.830164 | Empirical final-test average; no dependent-time-series guarantee |
| Stable FTCS | Global relative L²0.000305225 | Exact reference comparison over41×21 space–time grid |

The 47 forecasting gallery paths are thinned only for display. Calibration uses213 daily origins and rank172 for α0.2. Final-time-only FTCS refinement errors have a different denominator from the full-field norm and are labeled separately.

## Limits and unexecuted checks

**TensorFlow is not installed.** The foundations TensorFlow script, four neural quick/full runs, TensorFlow forward/gradient/optimizer tests, trained-model saving/reloading, convergence, neural metrics, and fitted neural visualizations were **not executed**. The optional `validate_experiments.py --tensorflow` branch is supplied for that environment. Static mathematical/source review corrected a likely Keras lambda-serialization failure, but source review is not runtime evidence.

No Firefox, Safari, native Edge, screen-reader, device-hardware, or real GPU test is claimed. Reduced-motion and no-JavaScript behavior were implemented and source-reviewed; an OS preference toggle and full offline/file-protocol test were not run. With no JavaScript, text alternatives are the default equation display and source downloads/native disclosures remain available. File-protocol browsers may block ES modules/JSON: serve over HTTP or use GitHub Pages. External reference links were checked against primary web sources; exhaustive HTTP status checks of every live external URL were not performed.

## Scope preservation and complete file inventory

`research.html` had existing user modifications. Its pre-task SHA256 was `5ef6d26dfea38981ba6ce44cc288bcd579fc53c403450706cb70d01ffe1a07e1`. The index integration inserted exactly one byte sequence after the existing Machine Learning list opening; removing that inserted sequence reproduces the original bytes. Post-insertion hash: `239ccbc4fba1286d9a181cb47cbbcdccbbaddbc6a11073f1578cc6e5033afaf0`. Other pre-existing tracked and untracked work was preserved.

New main file: `research/machine-learning/2026-11-12-tensorflow-demonstration.html`.

All remaining new files are under `research/machine-learning/2026-11-12-tensorflow-demonstration/`:

- `README.md`, `references.md`, `qa-report.md`.
- `css/styles.css`.
- `js/main.js`, `js/navigation.js`, `js/code-controls.js`, `js/demonstrations.js`, `js/mathjax-tex-svg.js`, `js/mathjax-LICENSE.txt`.
- `python/requirements.txt`, `python/00_tensorflow_foundations.py`, `python/01_computer_vision.py`, `python/02_natural_language_processing.py`, `python/03_time_series_forecasting.py`, `python/04_scientific_computing.py`, `python/generate_demo_assets.py`, `python/validate_experiments.py`, `python/build_monograph.py`, `python/validate_page.py`.
- `data/vision-demo.json`, `data/nlp-demo.json`, `data/forecasting-demo.json`, `data/scientific-demo.json`, `data/generation-manifest.json`, `data/python-validation.json`, `data/page-validation.json`.
- `images/overview.svg`, `images/tensor-concepts.svg`, `images/computational-graph.svg`, `images/backpropagation.svg`, `images/computer-vision-scene.svg`, `images/unet-architecture.svg`, `images/convolution.svg`, `images/nlp-university.svg`, `images/seasonal-city.svg`, `images/forecasting-window.svg`, `images/scientific-laboratory.svg`, `images/heat-stencil.svg`.
- `images/vision-baseline-diagnostics.png`, `images/nlp-baseline-confusion.svg`, `images/heat-exact.png`, `images/heat-ftcs-error.png`, `images/qa-desktop.jpg`, `images/qa-mobile.jpg`.

Modified existing file: `research.html` (single new Machine Learning entry only, preserving pre-existing changes).
