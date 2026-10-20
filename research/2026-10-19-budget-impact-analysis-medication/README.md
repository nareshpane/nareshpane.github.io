# Interactive BC secukinumab budget impact analysis

Research draft created October 19, 2026. Dependency-free static HTML/CSS/ES modules; no build or network dependency. Original supplied PDF retained unchanged. No index update, commit or push.

Revised October 20, 2026. See [revision notes](data/revision-notes.md) for the change manifest, official-source checks and validation scope. The calculation engine is byte-identical to the original implementation.

Targeted refinements completed October 20, 2026: compact coloured policy symbols, smaller Hurley cards, a redesigned prevalence overlap, expanded terminology, national-versus-provincial context and a compact metadata panel. See [targeted revision report](data/targeted-refinements.md) and [focused source diff](data/targeted-refinements.patch).

## Location decision

The request contained conflicting instructions: an exact HTML path inside this dedicated folder, but also text saying the primary page must be directly in `research/`. The implementation uses the **exact requested path**, inside the permitted writable folder, without creating a duplicate or redirect:

`research/2026-10-19-budget-impact-analysis-medication/2026-10-19-budget-impact-analysis-medication.html`

## Run

From `D:\GitHub\nareshpane.github.io`:

```powershell
python -m http.server 8000
```

Open:

`http://localhost:8000/research/2026-10-19-budget-impact-analysis-medication/2026-10-19-budget-impact-analysis-medication.html`

Do not open as `file://`: browsers restrict ES module loading under that scheme.

## Files

- `2026-10-19-budget-impact-analysis-medication.html`: all twelve teaching sections, citations, navigation, controls and explanatory content.
- `css/style.css`: peach/cream/green responsive design, chart overflow, focus and reduced-motion styles.
- `js/model.js`: pure population, deterministic synthetic records, case definition, dose-event costing, treatment transitions, payer split, sensitivity decomposition and seeded Monte Carlo calculations.
- `js/assumptions.js`: source registry and structured editable defaults with evidence labels, dates, units and descriptions.
- `js/art.js`: retained shared illustration entry point, now re-exporting the richer vector vocabulary.
- `js/illustrations.js`: editable patient silhouettes, analytical vectors and shared scene compositions.
- `js/budget-view.js`: shared CAD 3.2 million illustration and live-budget view selection.
- `figures/bc-centred.svg`, `scripts/recenter-bc-map.py`: existing BC geometry recentered on a 126°W central meridian for the standalone BC scene.
- `js/editorial-art.js`, `figures/illustrations/`: centrally referenced original artwork atlas, accessible role descriptions, complete reference inventory and generation prompt.
- `scripts/review-illustrations.mjs`, `scripts/illustration-contact-sheets.mjs`: actual browser scene/subsection captures and screenshot contact sheets.
- `js/storyboard.js`, `js/video.js`: 13 illustrated scenes, 51-second timeline at 1× and synchronized playback controls.
- `js/terminology.js`: accessible full-name expansions for compact SVG labels; a closed terminology reference also appears near methodology.
- `scripts/review-targeted-refinements.mjs`: desktop/mobile before-and-after measurements and captures of the targeted changes.
- `js/education.js`: prevalence/clinical teaching aids, decision tree, reimbursement storyboard, cost equation, payer examples and live limitation scenarios.
- `css/revision.css`: compact metadata, video and scene-navigation layout, new illustrated panels and responsive refinements.
- `figures/canada-provinces.svg`, `scripts/extract-canada-map.py`: 13 unaltered Natural Earth province/territory geometries from the existing local reference.
- `scripts/review-storyboard.mjs`, `figures/qa/`: development-only SVG raster previews and inspected contact sheet; not browser screenshots.
- `scripts/test-revision.mjs`, `data/revision-validation.json`: engine-identity, formatting, new teaching arithmetic, storyboard and layout-order assertions.
- `js/app.js`: controls, chart rendering, CSV export, accessible tooltips and timeline playback.
- `figures/bc-outline.svg`: Natural Earth v5.1.2 public-domain BC geometry reused from the local design-reference map.
- `scripts/extract-bc-map.py`: map extraction from the existing repository reference, needed only to regenerate the geometry.
- `scripts/test-model.mjs`: automated numerical tests and seeded randomized invariant checks; generates `data/numerical-validation.json`.
- `scripts/test-site.py`: local asset, DOM-ID, module dependency and PDF hash audit; generates `data/static-validation.json`.
- `scripts/test-presentation.mjs`: lightweight DOM-stub initialization and control-handler smoke tests; generates `data/presentation-validation.json`. This is not browser rendering validation.
- `scripts/test-browser.mjs`: executable desktop/tablet/mobile browser checks and screenshot capture; development-only Playwright dependency, never used by the website.
- `data/evidence-notes.md`: source findings, page numbers, historical/current distinctions and model conventions.
- `data/numerical-validation.json`, `data/presentation-validation.json`, `data/static-validation.json`, `data/http-validation.json`: actual executed validation reports.
- `data/browser-validation.json`, `data/browser-validation-status.json`, `data/illustration-browser-review.json`: actual passing browser regression and responsive screenshot review results.

## Numerical tests

From this dedicated folder, using Node 24+ and Python 3:

```powershell
node scripts/test-model.mjs
node scripts/test-presentation.mjs
node scripts/test-revision.mjs
python scripts/test-site.py
node --check js/app.js
node --check js/art.js
```

With the HTTP server running, `python scripts/test-site.py --http` also verifies the page and all thirteen required page/module/graphic requests are served with HTTP 200 and appropriate MIME types.

Core validation: DAD/MSP/NACRS counts 170/1240/250; inclusion–exclusion union **1,350**; broad/repeated/strict case counts **1,350 / 917 / 270**; all default eligibility filters retain **379** synthetic people. Synthetic severity/response flags require clinical validation in any actual study.

Historical top-down estimate: **121.7396480468** expected publicly funded biologic users. Fixed N=120, zero growth: secukinumab stocks **36/48/60**, new switches **36/12/12**. Initial default includes 10% utilization growth: active stock **120/132/145.2**, secukinumab stock **36/52.8/72.6**.

Default q4w first-year and maintenance costs: **$28,242.88 / $22,947.34**. Q2w: **$49,425.04 / $45,894.68**. Adalimumab: **$25,448.58 / $24,506.04**. Exact [0,52) convention intentionally differs from the review's annualized tables.

Live model baseline public budget impact: **$128,236.64828571305** over three years. Fixed no-growth demonstration: **$93,279.60**. Negative annual impacts or totals are supported and labelled illustrative savings. All values depend on historical prices and hypothetical assumptions; they are not official forecasts.

The default policy illustration uses CAD 0.8 / 1.0 / 1.4 million annual increments, totalling CAD 3.2 million. A synchronized budget-view selector switches the animation and annual/cumulative charts to the live model. Editing BIA assumptions selects the live view automatically. The calculation audit, sensitivity analysis and CSV export always retain model-derived values. See [context and animation revision](data/context-animation-revision.md).

The revised publication displays aggregate values in **CAD $000** with zero decimals: three-year reference **9,758**, new scenario **9,886**, incremental **128**. This is presentation rounding only; the underlying model and CSV remain dollar-level and unrounded. Individual patient and unit prices remain dollars. New payer selections affect the separate patient example only.

## Browser QA: passed on October 20, 2026

The final revision was tested in actual Chrome at desktop (1440 × 1000), tablet (820 × 1180) and mobile (390 × 844) sizes using an approved browser launch outside the sandbox. All browser regressions pass, with zero console errors and no document overflow. This supersedes the earlier sandbox launch limitation. All thirteen scenes and twelve affected subsections were captured at each size. See [visual revision notes](data/visual-revision-notes.md), [artwork inventory](figures/illustrations/README.md), and [scene contact sheet](figures/qa/illustrations/desktop-contact.png). Touch hardware and a screen-reader session were not tested.

With an existing Playwright installation on an unrestricted machine, keep the HTTP server running and run:

```powershell
# Optional: specify where an existing Playwright package is installed
$env:PLAYWRIGHT_MODULE = 'C:/path/to/node_modules/playwright'
# Optional: use an installed Chrome rather than a Playwright browser
$env:BIA_BROWSER_PATH = 'C:/Program Files/Google/Chrome/Application/chrome.exe'
node scripts/test-browser.mjs
```

The script blocks external requests, captures desktop/tablet/mobile top and budget screenshots in `figures/qa/`, tests model-control propagation, reset, cohorts, dosing, Monte Carlo repeatability, playback/scrubbing, CSV downloads, page overflow and basic keyboard access, and records console errors. It writes a passing browser report only after completing all checks. Screenshot inspection, actual browser zoom and touch usability still require human review.

## Interpretation and limitations

BC listed secukinumab for HS on August 13, 2025. This is a retrospective educational counterfactual. Current BC eligibility is summarized from the shared HS Special Authority form; consult it for actual applications. Historical national sponsor/CDA-AMC results are separate from the BC model and are never allocated by population share.

Reference market is 100% adalimumab biosimilars in an already publicly covered biologic cohort. Other current-market biologics, outcomes, adverse-event costs, background therapy, non-drug burden, net rebates, real benefit adjudication and real BC clinical eligibility observations are not estimated. Do not treat antibiotic users as automatically eligible switchers. Advanced return-to-adalimumab reloading, year-boundary exits and one-fill-per-administration fees are explicit simplifications.

All original human illustrations, facilities and flows are schematic. Venn and treatment-flow areas are not proportional; counts reconcile. Population bars use a labelled logarithmic count scale. Small-screen complex charts scroll horizontally to preserve text size.

Payer example is an independent illustrative secukinumab course: eligible charges only, no other family claims, annual resets, private insurance applied once to residual costs. The aggregate public cost weight is not a universal PharmaCare reimbursement percentage. PharmaNet non-public amounts are not automatically final patient out-of-pocket burden.

The Monte Carlo parameters are assumed distributions, not empirical uncertainty. Sorted shares alter marginal distributions; independence assumptions and omitted correlations limit interpretation. The waterfall is an exact sequential, order-dependent model decomposition, not a causal allocation or an additive attribution of the national review difference.

Independent research and educational illustration. Numerical scenarios are not official BC Ministry of Health forecasts.
