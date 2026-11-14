# QA record — November 13, 2026

This report records observed execution, not intended tests. Numerical/static/browser records are in `data/`. Scope is exclusively the PyTorch sibling HTML and this asset folder. No staging, commit, push, index edit, or TensorFlow reference modification.

## Latest pass: mathematical typesetting and Chapter 7

The current article has **7 chapters, 67 subsections, 51 numbered equations, 220 rendered mathematical fragments, and 31 canonical source panels**. All 39 earlier numbered equations were audited; the mathematical fallback path, notation, six teaching figures, and final framework-comparison chapter are complete. The original scientific artifacts remain byte-identical.

See the [full completion report](typesetting-report.md), [equation-by-equation audit](data/typesetting-audit.json), [browser evidence](data/typesetting-browser-validation.json), and [exact modified paths](data/typesetting-changed-files.json). These current counts and screenshots supersede presentation counts in the historical passes below.

## Earlier visual refinement — November 13, 2026

The existing page and complete asset inventory were inspected before modification: six chapters / 61 subsections, navigation, equations, tables, code disclosures, every JavaScript dependency, Python programs, numerical artifacts, responsive rules, and existing QA images. The new attachment contained the written request only; no separate new screenshot files were present. The actual existing page and its earlier QA images supplied the visual baseline. The previous build record below is retained as historical evidence; the inventory and layout counts in this section supersede its original presentation counts.

### Presentation and explanatory changes

- White body and main reading surface (`rgb(255, 255, 255)` verified at every requested width). Selective coordinated peach, blue, spring-green, and lavender panels; common labels, small mathematical symbols, borders, and spacing. Gray source disclosures remain distinct.
- River 4.7: a compact desktop 2 × 2 retains the four scientific roles—transport schematic, spatial profile, sensor trajectories/observations, and the stored intensity-profiled source objective. Local controls, consistent station colors, shared time/source state, and one playback group replace the long vertical layout. The responsive schematic maintains readable mobile labels. Axis margins were widened after inspecting objective tick labels.
- The original source-profile curve is **not** recomputed when live velocity/dispersion changes; its notice makes the fixed reference explicit. Recorded inverse iterates use full-precision parameters. The fit dot is saved fitting loss, not a newly profiled objective or confidence interval. Time is explicitly fixed at 1.2 during fit replay.
- City 5.7: grouped policy/capacity, risk/fairness, and demand controls; a smaller centered street/flow diagram; numerical station loads, utilization bars, and full/unavailable states; adjacent realized-service statistics; compact history and cost-distribution charts. Serving cost and 15-unit-per-unserved shortage penalty remain separately visible. The realized penalized district SD is labeled separately from the smooth objective's fairness variance.
- City history is actual saved PyTorch history or the calculated 100-step browser sequence. SciPy exposes its final point only; direct rules expose their decision mechanism without an invented history. Distribution curves evaluate the selected policy on all 160 original held-out demands at the selected capacity; extra D8 stress demand is explicitly limited to the selected scenario. λ/μ changes select the disclosed JS optimizer on 48 stored train scenarios.
- Standalone figures and charts are centered with bounded widths. Tables use content width; numeric headings and cells are centered and tabular. The sensitivity table retains five columns and original decimal precision at about **560.85 px**, versus the previous approximately 1117 px full-width treatment. Descriptive text remains left aligned; long tables have local scrolling.
- Six new editable vector figures explain JVP/VJP direction, competing-risk mass accounting, synchronous repeated-edge aggregation, forward/inverse computation, fractional CVaR, and soft versus feasible allocation. All were visually inspected in the integrated browser. New worked examples and intermediate reductions connect these diagrams to the unchanged scientific programs; analytical examples are not presented as measured findings.
- Final article retains **6 chapters, 61 subsections, and 26 canonical source panels**, with **39 numbered equations and 21 article image elements**. Only `explore_api.py` is open by default. Equation numbering is assigned in document order after inserting explanations. Final exact inventory is recorded in `data/page-validation.json`.

### Scientific preservation and numerical checks

Before editing, `data/refinement-baseline.json` captured **32 scientific artifact hashes** and **56 outside-scope hashes**, plus all 61 subsection IDs and baseline geometry. All 88 files remain byte-identical. The 32 include all retained JSON/NPZ experiment outputs, the six numbered scientific programs, and their protected shared execution/validation programs. The 56 cover the TensorFlow reference and the pre-existing `research.html` state. No scientific model was refitted or benchmark rerun during refinement.

The aggregate fingerprint of the 21 original JSON/NPZ output files remains `563617f007ae99ead786a8474add20966a0714b2ebcac670aecadf1b9102dd6d` (SHA256 over sorted `relative/path:sha256(bytes)` lines). Existing scientific assumptions, fitted parameters, held-out results, and benchmark evidence are retained.

Executed after refinement:

- Original `validate_experiments.py`: **all six numerical groups pass**. The existing `validate_numerics.mjs` still checks every river field value against Python, maximum discrepancy **1.7763568394002505e−15**, and nine stable physics combinations.
- New `validate_refinement.py`: **six groups pass**. Checks preservation/anchors/source-panel count; Jacobian/JVP/VJP via torch.func; competing-risk incidence and event/censor probabilities; native index_add with duplicate edges; unchanged river solver linearity and autograd sensor/loss derivatives; allocation feasibility, fractional CVaR, and risk-ranking arithmetic. It exports exact canonical Python flow fixtures.
- New `validate_refinement.mjs`: **640 exact flow/cost comparisons** against canonical Python (four stored policies × 160 test demands); **7,680** integer-flow, demand-conservation, capacity, serving-cost, and shortage-penalty stress checks; 100 finite normalized browser iterates; and an independent first-iterate objective reduction. Fractional-tail CVaR reproduces 11/3.
- Python parsing, all authored JS/MJS syntax checks, deterministic HTML rebuild, all local assets/anchors/unique IDs, SVG XML and accessibility metadata, all 26 source-panel text comparisons, exports/provenance hashes, and deterministic source ZIP parity pass. The new mathematical examples are exported in `data/worked-examples.json`; numerical fixture/report files are audit artifacts, not browser-loaded empirical data.

### Observed responsive geometry

All tests used the actual local page in the Codex in-app Chromium browser, a no-store loopback server, and temporary viewport overrides. The scrollbar accounts for the 15 px client-width difference.

| Viewport width | Client / document scroll width | River lab height | City lab height | River columns |
| --- | --- | --- | --- | --- |
| 390 | 375 / 375 | 1641 px | 1865 px | 1 |
| 768 | 753 / 753 | 1438 px | 1158 px | 1 |
| 1024 | 1009 / 1009 | 1410 px | 1140 px | 1 |
| 1440 | 1425 / 1425 | **827 px** | **845 px** | **2** |
| 1920 | 1905 / 1905 | 827 px | 828 px | 2 |

At 1440 px the river lab decreased from **2154.57 to 826.84 px (61.6%)** and the city from **1482.75 to 844.61 px (43.0%)**. The complete river lab, including controls/readouts/note, fits in the 900 px viewport below its subsection title. The city is nearly complete at 900 px; its 1000 px screenshot captures the entire laboratory and surrounding context. Narrow layouts deliberately stack panels. The mobile city diagram scrolls within a 302 px container over a 540 px drawing; the page itself remains 375 px wide. Station text bounding boxes fit inside their pills. No source disclosure header overflowed at any tested width.

### Observed interaction and rendering checks

- All **39 MathJax expressions rendered**, zero math errors, zero laboratory errors, all four applications and benchmark data initialized. Final console inspection: **zero errors or warnings**.
- River reaches all **11 saved fit iterates**. Final source is **(0.2589787265303483, 0.8045371286461034)** and live fitting loss **0.33414256116024144**, matching the stored fit. Extreme v=.7 / D=.015 gives **A+2B=.7499999999999999**. Sensor selection, physical controls, time Restart=0, Play/Pause, fixed-reference notices, and mode transitions work.
- City policy selector exercises PyTorch, SciPy, nearest, greedy, and JS. λ/μ edits select browser step 99; Restart/Next/Previous return through the real sequence. A selected stress case at scenario 160 with two half-capacity stations and +16 D8 units gives **46 demand, 11 served, 35 unserved, 525 shortage penalty, zero capacity violations**. Station labels show 6/6 full, 5/5 full, and unavailable. Browser Play advances; navigating offscreen clears its play state and stops progression.
- TOC Collapse all closes six groups; Expand all opens six. Mobile drawer marks main inert, wraps Tab/Shift+Tab, restores focus to Contents on Escape, and closes after anchor navigation. The stacked mobile river and local city scrolling were inspected.
- Source copying of the complete foundation program matches all **3987 characters**; full-height toggle and keyboard disclosure work. Clipboard content was restored. All 26 canonical source texts also pass independent static equality checks. Dashboard switching and numeric alignment in its dynamically built replication table work.
- All six new SVGs, worked derivative panel, probability panel, centered sensitivity table, both laboratories, opening, and comparison chart were visually inspected. Earlier original-build interaction evidence remains below; this pass does not claim an additional forced clipboard-denial or OS preference test.

Machine-readable evidence: [Python refinement validation](data/refinement-validation.json), [JavaScript refinement validation](data/refinement-js-validation.json), [browser refinement validation](data/refinement-browser-validation.json), and [page inventory](data/page-validation.json).

### Selected final screenshots

| Evidence | Screenshot |
| --- | --- |
| White introduction and application cards | [Opening](images/qa-refinement-opening.jpg) |
| Blue, peach, green and lavender scientific panels | [Probability budget](images/qa-refinement-probabilities.jpg) |
| Complete new mathematical illustration | [Derivative directions](images/qa-refinement-derivatives.jpg) |
| Rendered intermediate mathematical derivation | [Worked Jacobian products](images/qa-refinement-mathematics.jpg) |
| Compact numeric table and centered empirical figure | [Sensitivity table](images/qa-refinement-table.jpg) |
| Complete 2 × 2 river laboratory at 1440 × 900 | [River](images/qa-refinement-river.jpg) |
| Complete city laboratory at 1440 × 1000 | [City](images/qa-refinement-city.jpg) |
| Actual CPU comparison chart and compact replication table | [Comparison](images/qa-refinement-comparison.jpg) |
| Narrow-screen navigation and stacked laboratory | [Mobile river](images/qa-refinement-mobile.jpg) |

### Limits and scope

No unresolved numerical or functional defect was observed in these checks. No Firefox/Safari, physical-phone, screen-reader, forced clipboard-denial, or browser JavaScript-disabled audit is claimed. Reduced-motion and hidden-page handling remain implemented and source-reviewed; offscreen stopping was actually exercised. Local horizontal scrolling is intentional for large mathematical/city diagrams, long equations, source, and wide tables. No native compilation, GPU, or real-world performance evidence was added. Temporary browser tabs and viewport overrides were cleaned up. Changes remain confined to the authorized PyTorch HTML and dedicated asset folder; no staging, commit, push, or publication.

---

## Original build record (before refinement)

## Executed scientific work

- Installed the verified torch 2.15.0+cpu wheel and Matplotlib 3.10.8 in a temporary isolated environment. Global Python was not changed. CPython 3.13.1, NumPy 2.4.0, SciPy 1.18.1, scikit-learn 1.9.1; Windows AMD64; float64 CPU, one torch thread. Benchmark/runner contexts also limit BLAS/OpenMP threadpools.
- Ran the foundation checks and all four application quick runs, retaining explicitly marked smoke artifacts.
- Ran all four substantive experiments at seeds 42, 43, and 44; seed-42 canonical arrays/figures and compact independent-seed evaluations are retained. Survival uses 1,800 subjects/350 neural epochs; graph uses 48 accounts, 4,320 events/160 epochs; river includes matched inverse fits and 12 sensor/noise sensitivity fits per substantive seed; allocation uses 300 train/160 test scenarios and 1,200 Adam steps plus conventional optimization/transportation solves.
- Ran repeated equivalent-work CPU benchmarks: three warm-up calls, 31 repetitions per implementation, identical resident inputs/dtype/thread limits, median/IQR/raw samples. No concurrent model-training process ran during the benchmark.
- Ran compilation compatibility outside the restrictive subprocess sandbox after an initial sandbox/temp-cleanup failure. Inductor failed its compatibility probe; AOT eager executed. Its overhead and steady-state results are recorded separately. AOT eager is not labeled an optimizing Inductor backend.
- Numerical validation passed six substantial groups: exact scalar/tensor/torch.func/state checks; censor/event likelihood, independent finite gradients and probability/censor weights; graph future-data sentinel and aggregation edge cases; combined physical stability/boundaries/forward parity/gradcheck/manufactured refinement; allocation numerical gradients/fractional CVaR/integer feasibility/exact-cost comparison/infeasible rejection; finite artifacts and substantive replication status.

## Scientific qualifications

Survival IPCW uses known synthetic censor survival; calibration uses privileged complete latent outcomes. Graph labels are independently thinned injected-regime labels, and new-account testing does not have the late chronological prevalence shift. River source generation/fitting shares a grid/model and held-out rows share a trajectory. Allocation feasibility applies after explicit realization; its soft capacity penalty is not a hard constraint. Its exact solver minimizes cost, while learned/static policies use a different composite objective. Training times with different optimizers cannot isolate framework speed.

Native tensor CPU peak memory, GPU benchmarks, large-scale device performance, real external datasets, broad uncertainty coverage, actual emergency travel time, and operational deployment are unmeasured. The evidence system and browser laboratories disclose these limits.

## Selected substantive results (seed 42)

| Result | Actual value / interpretation |
| --- | --- |
| Exact scalar loss/update | 4.5 → 2.53125; gradient (−6,−3); new (w,b)=(1.3,0.15) |
| Survival analytical/autograd maximum gradient error | 4.9960e−16 |
| Final-test survival NLL | Linear 2.695411; neural 2.695993; no neural improvement in this realization |
| Chronological graph average precision | Logistic 0.556941; message passing 0.572467 |
| New-account graph average precision | Logistic 0.562027; message passing 0.553619; the ordering reverses |
| River source truth / recovered | (0.260000,0.800000) / (0.258979,0.804537) |
| River source gradient relative discrepancy | 8.4706e−10 |
| JavaScript/Python entire river-field maximum discrepancy | 1.7764e−15 |
| Feasible allocation mean proxy cost | Greedy 3.263823; torch policy 3.264692; SciPy policy 3.294820; exact cost comparator 3.245211 |
| Realized allocation capacity violations | Zero for every generated held-out scenario |

The three-seed evaluations are retained in `outputs/replications.json` and displayed by domain in the dashboard. The same-model survival final objectives agree; graph aggregation agrees; river NumPy and torch fields agree; the 25-coordinate allocation analytical/autograd gradient agrees independently. No result is marketed as universal superiority.

Conventional median divided by torch median is 1.4818× for survival loss/gradient, 2.3012× for incoming-neighbor aggregation, 0.1959× for river sensitivities, and 0.2577× for allocation loss/gradient on the specific recorded CPU inputs. Above one favors torch for that kernel. River derivative algorithms differ; full training timings with different models/optimizers are not framework-speedup measurements. Raw samples and IQRs are retained.

Inductor failed because `cl` was not found (`InvalidCxxCompiler`). AOT eager executed and its first-call and repeated timings are retained. It is not an optimizing Inductor result. No attempt was made to install a C++ toolchain or infer a successful native compile.

## Static/source and numerical integrity

- Final article: **6 chapters, 61 subsections (13/10/10/10/9/9), 33 numbered equations, 176 IDs, 26 canonical source panels, 15 article image elements**. Approximately 18,731 words including navigation/labels and excluding code/script/style; this is an inventory measure, not a prose-only word count.
- All Python sources parse; PowerShell parser reports zero errors; all seven authored JS modules pass syntax checking. The bounded API exploration file was actually executed, including caught unavailable Tensor signatures/source handling.
- All 26 embedded source components match canonical full files or AST-defined excerpts exactly after HTML decoding. The only default-open substantial source is `explore_api.py`. Python and PowerShell are highlighted without altering copy text. A deterministic ZIP keeps shared imports and setup alongside the numbered sources.
- All local assets and anchors resolve, all IDs are unique, image dimensions/alt text are present, SVG XML/viewBoxes/title descriptions validate, generated structure matches the document, JSON is finite, NPZ loads without pickle, browser exports match canonical results, and manifest hashes match the final source/data/assets.
- Rebuilding produces byte-identical HTML. Structure, subsection counts, and measured tables come from one content specification and real results rather than separately maintained counts.
- Independent Node assertions compare every value of the JavaScript river field with the executed Python field and check nine velocity/dispersion control combinations for combined stability, positivity, finite output, and boundary values.
- Allocation diagnostic labels were shortened and clarified after visual inspection, and the SVGs were regenerated directly from retained results without changing numerical metrics. A final allocation quick run verified the refactored diagnostics and null soft-overload metric for the exact solver.

## Actual browser and interaction checks

Served the repository root through temporary loopback HTTP servers. Final tests used a no-store preview server to prevent stale module caches. Tested the actual page in the Codex in-app Chromium browser; temporary viewport overrides were reset and agent-created tabs closed after QA.

| Requested viewport | Document client width | Document scroll width | Whole-page overflow |
| --- | --- | --- | --- |
| 390 | 375 | 375 | None |
| 768 | 753 | 753 | None |
| 1024 | 1009 | 1009 | None |
| 1440 | 1425 | 1425 | None |
| 1920 | 1905 | 1905 | None |

The difference between viewport and client width is the vertical scrollbar. No source disclosure header overflowed at any width. Curve plots redraw to a 320-unit mobile viewBox with readable axes instead of shrinking desktop labels. Large graph/stencil diagrams, long equations, tables, and source have explicit local scroll areas.

- Inspected hero, metadata, all four illustrated cards and their actual destinations, fit-content equations, tensor workbench, scalar graph, neural unit calculation, API namespace diagram/code, survival curves/timelines/calibration, transaction message stages, river transport/history/sensor curves, city flows/shortage states, benchmark dashboard, all empirical figure sections, and collapsed/expanded gray source. All 15 image elements loaded successfully after visiting their sections.
- All **33 MathJax expressions rendered**, zero `mjx-merror` elements, all four application laboratories plus benchmark data initialized, zero `.lab-error` elements. Final console inspection found **zero errors or warnings** after the corrected-page QA marker.
- All **26 actual copy actions** matched displayed canonical source. The first clipboard read was initially stale and was rechecked in a separate completed copy/read call. Updated allocation source and PowerShell source were rechecked after their final changes. Mobile PowerShell copying was exercised with Enter. Enter/Space disclosures and bounded/full-height expansion were exercised. Clipboard fallback is implemented/source-reviewed but was not force-triggered.
- Expand all opened six TOC groups; collapse all closed six. Chapter/subsection highlights, independent contents scrolling, automatic active-entry repositioning, reading progress, and desktop/mobile anchor offsets (approximately 24/65 px) were checked. All four application cards reached their corresponding chapters.
- Mobile drawer opens/closes, sets main/closed-sidebar inert state, wraps Tab and Shift+Tab between visible controls, returns focus to Contents on Escape, and closes after subsection navigation. Closed chapter descendants are explicitly excluded from focus wrapping.
- Scalar walkthrough reaches prediction **2.7500**, loss **2.53125**, gradients **−6.0000/−3.0000**; Previous/Restart work. Play advances, Pause stops, and leaving the viewport resets the play state and stops the interval.
- Tensor default matrix product is `[[22,28],[49,64]]`. Mismatched inner dimensions show a real invalid-state message; singleton addition produces `[[2,3,4],[6,7,8]]`; six float32 entries report **24 bytes** raw storage. Reshape and permutation are explicitly separate operations.
- Fixed network at x=(0.5,1) yields tanh output **0.1847**; selecting the third unit with ReLU yields z=h=**1.0500**, output **0.3300**. Connection labels were repositioned after observing overlap and the corrected diagram was visually checked.
- Survival profile/model/cause/horizon controls update actual stored probabilities. At bin 12 for the lower conventional profile, S/F1/F2 display **0.2937/0.2452/0.4611**, summing to **1.00000000**. Cause selection changes curve visibility; calibration retains its explicit bin-8 target.
- Transaction controls reach 120/120 candidates. Final candidate **13→38**, arrival **29.987**, stored score **0.0035** agrees with canonical probability 0.0034993217. Account selection reports complete incident historical edges and actual feature/aggregate/embedding values. Threshold/stage controls work.
- River extreme controls v=.7,D=.015 produce combined A+2B=**0.7500**, with automatic stable time steps. All 11 stored inverse iterates are reachable; final browser objective **0.3341** agrees with Python 0.3341425612 and recovered pair rounds to **(0.259,0.805)**. Browser simulation versus precomputed inference is explicit.
- Two stations, half capacity, and added hotspot demand yield **32 unserved units**, **100.0% utilization**, and **zero capacity violations**. Risk changes switch to the explicitly labeled JavaScript optimizer. Flows and station load labels reflect actual integer assignments.
- Dashboard switches among all domains; it shows actual median/IQR, ratio direction, repetitions, numerical errors, unavailable memory, genuine quality metrics, independent-seed rows, hardware/library context, and compilation status. No unsupported GPU or native-compile result appears.

Selected screenshots only: [desktop opening](images/qa-desktop.jpg) and [mobile probability laboratory](images/qa-mobile.jpg). No animation frame collection was retained.

## Issues found and resolved during QA

Corrected a derivative-lab output ID that interrupted later initialization; HTML-escaped TeX so a less-than sign could not become an HTML tag; removed double anchor offset; fixed focus wrapping through closed disclosure descendants; made closed mobile navigation inert; added no-JS mobile contents styling; redrew mobile curve axes; corrected an irrelevant soft-overload metric on exact transportation output; clarified that the exact solver has richer optimization rather than extra demand information; and improved overlapping network labels/long diagnostic axis text. Final assertions and browser checks above describe the corrected state.

## Verification limits

No Firefox/Safari/physical-phone/screen-reader audit, OS reduced-motion preference toggle, forced clipboard-denial fallback, or full browser JavaScript-disabled test is claimed. Reduced motion, hidden-page stopping, native details/downloads, static TOC, mathematical text alternatives, and no-JS mobile contents are implemented and source-reviewed; offscreen stopping is actually tested. `file:` browser policies may block modules/JSON; HTTP/GitHub Pages is the supported interactive route. Exhaustive live status checks of every external link were not performed; relevant primary documentation and bibliographic records were consulted, with the institutional preprint parsing limitation documented in references.md.

## Git and scope

`git status --short` reports the two requested items as untracked (`??`): the HTML and dedicated folder. Nothing was staged, committed, pushed, or published. The only pre-existing tracked modification remains `research.html`, already modified before this task; this task did not edit it. The TensorFlow page/asset folder have no diff. Existing unrelated untracked work remains present. No virtual environment, Python cache directory, node_modules, large library copy, or temporary compiler cache was placed inside the deliverable.

Open exactly:

`D:\GitHub\nareshpane.github.io\research\machine-learning\2026-11-13-pytorch-demonstration.html`

Serve `D:\GitHub\nareshpane.github.io` over HTTP and use `/research/machine-learning/2026-11-13-pytorch-demonstration.html`; the README and complete PowerShell reproduction file contain the commands.
