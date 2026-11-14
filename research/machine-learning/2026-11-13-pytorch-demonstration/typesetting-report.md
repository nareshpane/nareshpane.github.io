# Mathematical typesetting and Chapter 7 — completion record

Local project: `D:/GitHub/nareshpane.github.io/research/machine-learning/2026-11-13-pytorch-demonstration.html`.

This pass extends the existing monograph in place. All 61 previous subsection anchors, four application laboratories, canonical scientific programs, recorded arrays/results, compact river/city layouts, and reproducibility workflow remain. The final article has **7 chapters, 67 subsections, 51 numbered equations, 31 source panels, six semantic HTML mathematical figures, and fifteen image elements**. The six former mathematical SVGs remain as historical geometry references; their instructional content is now presented with actual typeset formulas.

## Audit coverage and corrections

Before editing, the complete PyTorch article/rendering system, all 39 numbered display sources, plain-formula SVGs, relevant dynamic mathematical readouts, and their numerical implementations were inspected. The complete local TensorFlow reference and its assets were read; the published reference was verified byte-identical. The new attachment contained the written request only, so the actual page and existing QA images provided the visual baseline.

**39 existing equation elements were audited; all 39 required fallback/typographic treatment.** Each already contained genuine TeX, but its alternative presentation was ordinary explanatory prose. A renderer startup problem could therefore expose sentence-like colored boxes. The valid delimiter escaping was retained. All 39 now have mathematical fallback content and external interpretation; **38 TeX sources** also received notation, alignment, or formula revisions. Six numbered-display clarifications are recorded below. They do not change experimental algorithms or results.

| Previous number | Final number | Clarification |
| --- | --- | --- |
| 7 | 8 | Gradient sensitivity and parameter displacement: use the update identity, allowing a zero gradient. |
| 10 | 11 | Column VJP and output seed; the previous row product was numerically valid and is explicitly related by transpose. |
| 23 | 25 | Separate the 27-dimensional baseline features from the learned 26-dimensional edge input. |
| 30 | 35 | Separate the three sensor indices from the fitting-time index set; retain 36 fitting pairs. |
| 33 | 38 | Use P for the interior stencil matrix, distinct from scalar Courant coefficient A. |
| 39 | 45 | Keep the optimized threshold explicit in the joint allocation objective. |

Examples of sentence replacements include the verbal forward-step description replaced by the actual upwind/centered finite-difference scheme, the no-event-logit sentence replaced by reference logits and the categorical normalization formula, and the neural-network description replaced by affine/activation equations with explicit dimensions. Equations, explanations, and assumptions now have separate roles. A short qualifier such as the implemented stability margin can remain inside mathematics; explanatory sentences do not.

The complete [before/after inventory](data/typesetting-audit.json) maps every old equation to its final formula, linked anchor, and number. Twelve additional displays supply broadcasting, probability conservation, boundary conditions, transport coefficients/restrictions, explicit shortage accounting, the threshold derivative, row probabilities, benchmark definitions, and the final chapter's shared learning/derivative/model mathematics. Numbering is sequential 1–51 in document order, with stable content-derived anchors and clickable numbers; diagrams do not hardcode page equation numbers.

## Typography and figures

The existing MathJax 3 TeX/SVG engine remains the single JavaScript mathematical renderer. Proper fractions, operators, real-number spaces, summations/products, gradients, Greek letters, bracketed matrices, and column vectors share one source and font system. Short panels use natural width; longer expressions use aligned rows and local keyboard-accessible overflow regions. The white background and selective blue, peach, green, and lavender panels remain.

A build-time cache compiles exactly the same TeX to native MathML. Mathematical alphabet variants are normalized for MathML Core, including blackboard real-number symbols and bold vector alphabets. Thus renderer failure does not replace an equation with prose. A script-free copy was actually inspected at 1440 and 390 pixels: all 51 mathematical fallbacks were present, with no page overflow. This is a targeted native-rendering test, not a browser-wide JavaScript-disable test of every laboratory/navigation behavior.

The small matching `boldsymbol` extension and its Apache 2.0 license are included locally; no second renderer or another full MathJax bundle is shipped. The precompiler disables error/undefined-command suppression and rejects invalid TeX. Collection mode does not overwrite the article with temporary placeholders.

All six reconstructed mathematical figures were visually inspected. The Jacobian visual retains blue forward and peach reverse lanes, gives input/output spaces, aligns actual matrices and vectors, and distinguishes the column products Jv and J-transpose-u. Its caption states the equivalent row convention. Numerical values are rounded for display while computations retain full precision. The other figures show probability budgets, synchronous duplicate-edge means, forward/inverse sensitivity flow, fractional CVaR, and soft versus integer allocation. The live neuron calculation uses genuine synchronized mathematics; selecting its third ReLU unit gives preactivation/activation 1.05 and output 0.33.

## Final Chapter 7

1. **Two frameworks, one mathematical foundation** — model, mean squared loss, gradient, update, and a compact training diagram with optimizer/parameter feedback.
2. **PyTorch and TensorFlow at a glance** — one bounded comparison table, with modern eager/graph/compiler distinctions.
3. **The same derivative in two implementations** — TensorFlow on the left, PyTorch on the right; both short float64 examples visible by default with copy/download controls.
4. **Neural-network construction and training** — paired 10–32–1 affine/ReLU/affine definitions, parameter/shape checks, and explicit initialization qualifications.
5. **Keras 3 and the modern ecosystem** — backend selection before import, actual tiny example, portability qualifications.
6. **Choosing a framework in practice** — research/custom mathematics, training, infrastructure, deployment, hardware/OS/team fit, native Windows versus WSL2 CUDA guidance, and a balanced concluding panel.

The previous 26 source panels retain their original default policy: only the API exploration is open among Chapters 1–6. The five new short examples are open on load. Paired cards have equal desktop widths, natural full code height, consistent highlighting, and stack on narrow screens. There are no Run controls. All five actual copy actions matched canonical source, as did a mobile Enter-triggered copy; the clipboard was restored.

Official PyTorch, TensorFlow, and Keras sources were checked; precise API/installation qualifications and links are in [references.md](references.md). The chapter distinguishes tracing from XLA, training configuration from compiler transformation, and model architecture from matched numerical initialization. It makes no framework speed/accuracy ranking from the earlier classical-versus-PyTorch experiments.

## Examples actually executed

| Actual check | PyTorch 2.15.0+cpu | TensorFlow 2.21.0 |
| --- | --- | --- |
| Float64 polynomial derivative / printed result | 7.0 | 7.0 |
| Forward output for a 2 × 10 batch | 2 × 1 | 2 × 1 |
| Trainable parameters | 385 | 385 |
| Finite initial outputs | Passed | Passed |

A separate controlled-weight float32 test copies the same parameters using the required Keras/PyTorch kernel transpose. The two fixed input outputs agree across frameworks with maximum absolute difference **0.0**; comparison with the independent NumPy calculation is within **9.5367431640625e-07**. This verifies the tested mapping, not universal floating-point identity or equality of independent initializations. The exact standalone Keras backend file also executed successfully with **Keras 3.15.1 and backend torch**.

The [execution artifact](data/framework-comparison-validation.json) records canonical source hashes, actual interpreter/library versions, shapes, counts, outputs, and qualifications. Existing TensorFlow and isolated PyTorch installations were used. No global Python packages were installed for this pass. The temporary MathJax build dependency is outside the deliverable.

## Mathematical and scientific validation

- Original `validate_experiments.py`: all **six scientific validation groups pass**.
- New `validate_typesetting.py`: validates all original equation lineage, exact TeX/MathML correspondence, mathematical alphabet compatibility, anchors/numbering, external explanations, and 88 protected hashes. Independent matrix stepping matches the unchanged river solver at several actual time indices; the explicit threshold objective and tail-count derivative match the canonical allocation implementation.
- Strict cache compilation passes **187 unique expressions**. All **220 mathematical fragments**, including **51 numbered equations**, render in the actual browser. No math errors, hidden mathematical SVGs, or visible raw delimiters were found.
- JavaScript river/Python whole-field parity remains **1.7763568394002505e-15**, with nine stable parameter combinations.
- Canonical Python/browser dispatch parity passes **640** policy/scenario comparisons; **7680** integer/capacity/conservation/shortage-cost stress cases pass.
- Static validation checks all local assets/anchors, unique IDs, source equality for all 31 panels, original default states, seven-chapter structure, and recursive manifest hashes including the extension/license. Final checks also cover Python and JS/MJS syntax, deterministic regeneration, and canonical source ZIP parity.
- All **32 protected scientific files** and **56 outside-scope files** remain byte-identical. This includes original JSON/NPZ outputs, scientific programs, the TensorFlow reference, and the pre-existing `research.html` state. No fit, benchmark, dataset, or trained parameter was changed.

The original output aggregate fingerprint remains `563617f007ae99ead786a8474add20966a0714b2ebcac670aecadf1b9102dd6d`. The preservation baseline and individual hashes are in [typesetting-baseline.json](data/typesetting-baseline.json).

## Actual browser validation

| Viewport width | Client / scroll width | Numbered equations rendered | All math fragments rendered | Math errors |
| --- | --- | --- | --- | --- |
| 390 | 375 / 375 | 51 / 51 | 220 / 220 | 0 |
| 768 | 753 / 753 | 51 / 51 | 220 / 220 | 0 |
| 1024 | 1009 / 1009 | 51 / 51 | 220 / 220 | 0 |
| 1440 | 1425 / 1425 | 51 / 51 | 220 / 220 | 0 |
| 1920 | 1905 / 1905 | 51 / 51 | 220 / 220 | 0 |

All five widths have white body backgrounds, no whole-page horizontal overflow, no source-header overflow, and no hidden mathematical SVGs. The short Chapter 7 polynomial fits the 390 px layout without equation scrolling. A larger Jacobian definition was actually panned with ArrowRight in its focused region: 283 px client width, 350 px content, 40 px horizontal movement; the page remained 375 px wide.

The compact laboratories are retained: about **829 px river / 849 px city** at 1440 px, compared with approximately 827 / 845 px after the previous visual pass. The minor difference does not change their coordinated layout. All four application labs and benchmark data initialize. River replay reaches the exact saved source **(0.2589787265303483, 0.8045371286461034)** and loss **0.33414256116024144**. The city shortage case still gives **46 demand, 11 served, 35 unserved, 525 penalty, zero capacity violations**.

TOC collapse/expand covers all seven chapters. Mobile navigation closes after selection, marks main inert while open, wraps Tab, and returns focus on Escape. The final subsection is now correctly active at the page bottom, with progress 100%. Final corrected-page console inspection found **zero errors and warnings**. Temporary viewport overrides, browser tabs, native test copy, and preview server were cleaned up.

## Selected final screenshots

| Requested evidence | Artifact |
| --- | --- |
| Revised equation panel | [Neural-network equations](images/qa-typesetting-equation.jpg) |
| Jacobian forward/reverse figure | [Actual matrices and column products](images/qa-typesetting-jacobian.jpg) |
| Competing-risk derivation | [Probability equations](images/qa-typesetting-probabilities.jpg) |
| River numerical scheme | [Coefficients, update, restrictions](images/qa-typesetting-river.jpg) |
| Chapter 7 comparison table | [Compact API table](images/qa-typesetting-comparison-table.jpg) |
| Side-by-side derivative examples | [Expanded gradient cards](images/qa-typesetting-gradients.jpg) |
| Side-by-side model examples | [Expanded model cards](images/qa-typesetting-models.jpg) |
| Practical conclusion | [Windows note and summary](images/qa-typesetting-conclusion.jpg) |
| Explicit allocation objective | [Joint threshold objective](images/qa-typesetting-allocation.jpg) |
| Training cycle | [Optimizer and parameter feedback](images/qa-typesetting-training.jpg) |
| Native fallback | [Script-free mathematical figure](images/qa-typesetting-native.jpg) |
| Mobile layout | [Readable equation and stacked code](images/qa-typesetting-mobile.jpg) |

## Issues resolved and limits

Actual QA found and resolved a missing bold-symbol component, a CSS display override that disrupted native inline MathML, tiny native vertical scrollbars, an adjacent-string TeX command boundary, an overly broad feedback-SVG selector affecting math on narrow screens, and the final-subsection navigation highlight. Strict compilation and actual browser checks describe the corrected state.

No unresolved numerical or functional defect was observed in the tested cases. Firefox/Safari, physical devices, screen readers, OS reduced-motion preference changes, a browser-wide JavaScript-disabled workflow, forced clipboard denial, GPU execution, and cross-framework performance benchmarks were not tested. Native mathematical fonts can differ by platform; exact pixel parity is not claimed. Existing reduced-motion/hidden/offscreen playback safeguards remain. Formal scientific assumptions and operational limitations remain those of the retained experiments.

## Exact file paths and scope

The complete source/document/generated-data/screenshot path list is in [typesetting-changed-files.json](data/typesetting-changed-files.json), with absolute Windows paths. The only page changed is the requested PyTorch HTML; all other changes are inside its dedicated asset folder. Some exports and validation records were regenerated byte-identically and are distinguished in the file list. No TensorFlow project, `research.html`, other research project, or unrelated Git state was edited. Nothing was staged, committed, pushed, or published.
