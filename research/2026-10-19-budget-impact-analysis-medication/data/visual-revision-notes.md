# Illustration revision — October 20, 2026

Updated the existing working page inside the dedicated project folder. The requested sibling `research/2026-10-19-budget-impact-analysis-medication.html` did not exist; no duplicate primary page, redirect, index change, or move was introduced.

## Scope

- All twelve opening scenes now use a coordinated original cast: research analyst; varied residents; two different dispensing environments; hospital nurse, physician, paramedic and laboratory scientist; programmer; linkage analyst; health economist; and four policy professionals.
- Retained maps, relational tables, legible SQL, diagnostic-code highlights, moving records, medication packages, calendars, switching groups, stock counts and financial charts. Raster charts inside illustrations are decorative; actual labels and analytical charts remain separate and model-driven.
- Expanded six compact SVG patient silhouettes with distinct hair, build, clothing, stance and accessories; used them in the population, treatment and cohort grids. Individual symbols remain schematic and do not assign demographics or clinical severity to model records.
- Updated clinical assessment, eligibility decision tree, Special Authority, authorization pathway, both prevalence approaches, DAD/MSP/NACRS/PLIS source artwork, pharmacy payment illustration and selected limitations. Preserved the existing Hurley schematics, dose equation and useful analytical figures.
- Added a short general-service illustration caption, a local data-URI favicon, and minimum-width containment to prevent grid children from expanding the document on narrow screens. Complex analytical figures retain their existing horizontal scrolling.
- Added a restrained scene fade; pause, replay, speed, scrubbing, scene navigation and the 94-second timeline remain intact. Reduced-motion mode disables the fade.

## Assets and provenance

`figures/illustrations/professional-cast.png` is one original generated atlas containing twelve distinct compositions, not a gallery added to the page. Its actual dimensions are 1448 × 1086; each tile is 362 × 362. The prompt requested a larger size; the generated output was retained at its native resolution. The asset, central tile references, accessible descriptions, complete reference audit and generation prompt are retained locally. See `figures/illustrations/README.md` and `reference-inventory.json`.

All twenty supplied screenshots were accessible and visually inspected. Their republication rights were unverified, so none was embedded or copied into the repository. They informed subject selection and general visual direction only. No watermark or attribution was removed. The new work is original AI-generated illustration; it does not establish rights to the supplied references.

## Validation

- Numerical regression: 2,769 checks pass. Baseline three-year incremental impact remains CAD 128,236.64828571305.
- Model and original CDA-AMC PDF retain their original SHA-256 values.
- Presentation tests: 43 checks pass, including simulated speed, pause/resume, replay, scrubbing, scene jumps and the final hold.
- Revision tests: 95 checks pass, including all twelve scenes, 94-second duration, clinical teaching arithmetic and live forecast propagation.
- Actual Chrome browser regression passes at desktop 1440 × 1000, tablet 820 × 1180 and mobile 390 × 844. Verified model edits/reset, strict cohort selection, dosing controls, repeatable Monte Carlo, real CSV download, replay/pause, scrub, keyboard focus, no document overflow, no external requests and zero console errors.
- Captured all twelve scenes and twelve affected subsections at all three viewport sizes. Inspected desktop scene contact sheet, mobile scene sheet and affected subsection screenshots. Corrected neighboring-tile bleed, SQL/table spacing and narrow-screen grid containment during review. Screenshots and contact sheets are in `figures/qa/illustrations/`; raw browser results are in `data/illustration-browser-review.json` and `data/browser-validation.json`.
- Browser checks use the existing local development Playwright installation; the website remains dependency-free. Touch hardware and a screen-reader session were not tested.

No analytical assumptions, citations, prices, equations, engine logic, source PDF, unrelated research projects, or research-index entries were edited. No staging, commit or push was performed.
