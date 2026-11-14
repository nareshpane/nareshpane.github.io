# Original editable illustrations

All thirteen SVG sources were drawn specifically for the PyTorch monograph. They use native vector geometry, system fonts, accessible titles/descriptions, and no external bitmap or TensorFlow-project dependencies.

The current article renders the six mathematical teaching figures through `python/math_diagrams.py` as semantic HTML with genuine MathJax/MathML formulas. Their original SVG files below are retained as historical geometry references, rather than loaded as the current mathematical diagrams. This avoids ordinary SVG-text approximations of matrices, vectors, and subscripts. The seven contextual SVGs and eight empirical diagnostic images remain in use. There are six HTML mathematical figures plus fifteen article image elements.

| File | View box | Purpose |
| --- | --- | --- |
| overview.svg | 1100 × 450 | A differentiable graph connects four distinct scientific questions. |
| application-survival.svg | 400 × 225 | A synthetic clinic, competing-event curves, and individual outcome timeline. |
| application-networks.svg | 400 × 225 | Institutions, accounts, and directed time-indexed transaction relationships. |
| application-river.svg | 400 × 225 | A contamination source, transported plume, monitoring stations, and field researcher. |
| application-allocation.svg | 400 × 225 | Synthetic city districts, emergency stations, demand nodes, flows, and capacity markers. |
| conventional-vs-pytorch.svg | 1200 × 418 | Two aligned objective-computation workflows; derivative maintenance is distinct from validation. |
| namespaces.svg | 1000 × 500 | Eight keyboard-accessible links to public PyTorch documentation. |
| ad-products.svg | 960 × 560 | The same numerical Jacobian propagates an input tangent or an output cotangent. |
| competing-risk-budget.svg | 960 × 560 | Conditional probability budgets, cumulative incidence, and event/censor likelihoods. |
| message-aggregation.svg | 960 × 530 | Synchronous incoming means with repeated edges distinguish equal-feature nodes. |
| inverse-workflow.svg | 960 × 480 | Source parameters, the stable discrete solver, sensor residuals, and reverse sensitivities. |
| cvar-tail.svg | 960 × 540 | Fractional upper-tail mass and the equivalent auxiliary-threshold calculation. |
| soft-vs-feasible.svg | 960 × 570 | Soft overload and a separate feasible integer realization in a two-district example. |

The four card scenes also appear as editable groups in the overview. Arrows communicate dependencies, transport, transactions, or assignment according to context; these are explanatory illustrations, not experimental observations or measured outputs. The survival identity is a standard probability identity. No performance result is asserted by the comparison diagram.

The six refinement diagrams are analytical or conceptual constructions, not additional model fits or measured outcomes. Their calculations are checked by `python/validate_refinement.py`. They retain a readable minimum width inside a keyboard-accessible local scrolling region on small screens.

Source checks cover XML parsing, view boxes, titles/descriptions, unique IDs, self-contained vector geometry, and official-domain documentation links. All six new illustrations were visually inspected inside the actual Chromium page. Selected `qa-refinement-*.jpg` screenshots cover the opening, colored panels, derivatives, mathematics, a compact numerical table, the complete river/city laboratories, comparison evidence, and the mobile river layout. The two earlier `qa-desktop.jpg` / `qa-mobile.jpg` files document the original build. See `../qa-report.md` for observed results and verification limits.

The latest `qa-typesetting-*.jpg` files document the equation overhaul, genuine Jacobian matrices/products, probability derivation, river scheme, explicit allocation objective, Chapter 7 table and expanded code pairs, training cycle, conclusion, native mathematical fallback, and mobile layout. These supersede earlier screenshots for current mathematical typography.
