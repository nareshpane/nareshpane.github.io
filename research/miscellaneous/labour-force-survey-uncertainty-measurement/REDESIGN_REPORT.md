# LFS presentation redesign — October 13, 2026

The article remains a draft. No files were staged, committed, pushed, or added to the research index.

## Files

Existing files revised:

- `../labour-force-survey-uncertainty-measurement.html`: film markup, navigation, opening cards, requested removals, source-level mathematical line breaks.
- `js/enhancements.js`: retired the old film renderer; retained numerical APIs and the donor, imputation and replicate laboratories.
- `css/enhancements.css`: removed retired film styling and the rule making every inline MathJax expression scrollable.
- `js/app.js`: replaced repeated person emojis with heterogeneous dwelling-frame illustrations and information marks; retained statistical calculations and controls.

New runtime assets:

- `js/documentary.js`: deterministic 108-second documentary and two static explanatory illustrations.
- `js/illustrations.js`: original SVG buildings, landscapes, people, monitors and six opening pictograms.
- `css/documentary.css`: film, mobile scene presentation, compact introductory cards, navigation and mathematics.

New supporting scripts: `scripts/redesign_journey.py`, `scripts/refine_documentary_math.py`, `scripts/verify_documentary.js`, and `scripts/verify_redesign_preservation.py`. The first two retain the presentation transformation; the latter two check browser behaviour and preservation. The browser test uses installed Chromium and Node's built-in CDP/WebSocket facilities, without installing dependencies.

## Film and illustrations

A persistent SVG scene graph is constructed once. A single requestAnimationFrame timeline controls camera position, easing, object placement, opacity, panel handover, questions, records, imputation transfers and publication. Replay and seeking restore state instead of inserting scenes. IntersectionObserver suspends playback offscreen; document visibility and reduced-motion changes reconcile the same clock.

Canada → Ontario urban blocks → British Columbia suburbs → Saskatchewan rural dwellings → PEI one-stage exception → Yukon quarterly exception → Ontario strata and neighbourhood → dwelling selection → six provincial panels → varied households → interviewing → processing → analysis → dissemination → “How reliable are these estimates?”

Real provincial geometry comes from the existing local map data. The conceptual neighbourhood is anchored at Lambert-map point (350, 388), verified inside Ontario's path. Boundaries of domains, strata and PSUs in the neighbourhood story are illustrative. The camera holds each geographic focus, zooms toward the neighbourhood, and follows response and estimate records through adjoining rooms. Scene durations vary. Quintic interpolation, coordinated fades, and moving records replace periodic SVG replacement and unrelated province selection.

Original artwork includes high-rise and medium-rise apartments, townhouses, duplexes, detached houses, rural fields, roads, sidewalks, parks and trees; individuals, couples, children, multigenerational and shared households; interviewer/computer, telephone and electronic-form symbols; distinct desks, analysts, monitors, intervals and diagnostics; and generic media channels. Colour identifies dwelling rotation panels. One outgoing panel and one incoming panel move while five panels remain fixed. Imputation transfers keep an explicit distinction between estimated values and unknown truth.

On mobile, rotation panels use three columns, household groups and media use two columns, imputation sources stack below the records, and analysts occupy two desk rows. The static diagrams show real Canadian outlines and the synthetic dwelling-probability → linked people → weighted provincial-total relationship. An amber window in the heterogeneous population frame identifies a selected dwelling, rather than selecting an entire apartment building.

## Article presentation and mathematics

The six original titles remain, with centred custom SVG pictograms replacing descriptions. Their desktop arrangement remains three columns by two rows. Navigation has six compact columns and two desktop rows; smaller screens wrap without horizontal scrolling. Active-section highlighting is retained through viewport observation, and sticky navigation is disabled where it would consume excessive height. The full contents listing and the specified introductory lines are removed. Methodology references, historical/current distinctions and reproducibility information remain in relevant sections.

Long inline expressions are promoted to display mathematics. RHC group sizes, conditional probabilities and variance receive separate blocks. Multiline layouts also cover PSU/SSU probabilities, monthly-change covariance, deterministic and donor imputation, total-variance decompositions, ratio linearization, MSE, bootstrap rescaling, adjusted CVs and worked calculations. MathJax's accessible mathematical output remains enabled. No scrollbar is hidden over a clipped formula.

No numerical or substantive formula corrections were required. Equivalent expressions were reformatted, and auxiliary notation (`v_Δ` for a variance and `B_i` for the joint logical condition) makes narrow-screen derivations readable. The conditional RHC estimator and variance, HT arguments, covariance terms, and historical replicate scaling retain their assumptions. Technical distinctions were checked against the [2025 Guide](https://www150.statcan.gc.ca/n1/pub/71-543-g/71-543-g2025001-eng.htm) and [2017 sample-design](https://www150.statcan.gc.ca/n1/pub/71-526-x/2017001/chapt2-eng.htm) and [variance-estimation](https://www150.statcan.gc.ca/n1/pub/71-526-x/2017001/chapt7-eng.htm) chapters; the historical RHC machinery is not presented as reconfirmed current production methodology.

## Validation

- Browser audit at 1440, 768, 390 and 320 pixels; all collapsible mathematics expanded; 150 rendered MathJax expressions, no rendering errors, no formula overflow or equation scrollbars, no full-page horizontal overflow, and valid anchors and unique IDs.
- Autoplay on entering the viewport, suspension offscreen, pause/resume, Previous/Next during playback, replay reset, timeline seeking and automatic completion at 108 seconds.
- Fifty rapid replay/next/previous cycles with stable SVG node count; 433 quarter-second deterministic timeline samples with finite transformations; Ontario focus point inside the actual province outline.
- Reduced-motion still-scene operation; original laboratory and provincial controls; donor transfer/reset; weighted replication and missing-data experiment; static empirical tables with JavaScript disabled; direct local-file opening.
- Numerical suite checks seeded simulations, sampling uniqueness, MSE, intervals, covariance and bootstrap. Empirical validation checks eleven geographies, count identities, denominators, changes, SEs and worked examples.
- Twenty-three data/source/statistics files checked against pre-edit hashes; all 23 article sections and eight statistical forms preserved; twenty local HTML asset references checked.

Screenshots and machine-readable results are in `figures/redesign-*.png`, `figures/redesign-browser-results.json`, and `figures/redesign-preservation-results.json`. Browser startup initially failed because the system drive was full; final checks use an isolated temporary profile on the workspace drive and remove that profile afterwards.

The page and its asset directory were already untracked at task start. Ordinary `git diff --stat` therefore has no output. A `git diff --no-index` comparison with the saved pre-edit HTML provides the page-change review without staging anything.
