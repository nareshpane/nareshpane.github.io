# CPI monograph refinement report — October 30, 2026

This is the prior refinement record. See [FINAL_POLISH_REPORT.md](FINAL_POLISH_REPORT.md) for the current presentation, five displayed illustrations, household modes, formatting conventions and asset cleanup.

The existing 18-chapter page has been refined in place. The headline observations, processed official CPI dataset, original replication results and 30 original input files remain unchanged. No Git staging, commit, push or deployment was performed. `research.html` was not edited.

## Implementation files

| File | Change |
|---|---|
| `../2026-10-30-canada-consumer-price-index.html` | Rebuilt main page, preserving eighteen chapters. |
| `app.js` | Shared miniature dataset; coordinated aggregation trace; accessible clickable animation stages; balanced synthesis; immediate long-page navigation. |
| `refinements.css` | New visual layer for hero, cards, panels, hierarchy, figures, diagrams and responsive layouts. The original `styles.css` remains intact. |
| `refinements.js` | New set explorer, sampling frame, quality adjustment, weighted geometric mean, calculation tracing and geographical comparisons. |
| `scripts/build_page.py` | Calls the targeted refinement functions before writing HTML. |
| `scripts/refinement_content.py` | New scholarly additions and compact hero replacement. |
| `scripts/prepare_refinements.py` | Copies supplied images unchanged and extracts additional weights from the existing official ZIP. |
| `scripts/fetch_sources.py` | Adds five primary-source reference pages to the existing retrieval workflow. |
| `scripts/validate_refinements.py` | Adds geographical HTML-release comparisons, image integrity and baseline preservation checks. |
| `scripts/validate_math.cjs` | Exercises actual page calculation functions with numerical invariants and endpoint cases. |
| `data/processed/refinement-data.json`, `.js` | Additional official weights and image provenance; separate from unchanged `cpi.json`. |
| `images/*.png`, `images/README.md` | Nine supplied illustrations and their provenance explanation. |
| `references/` | Five additional official HTML references: Chapters 5–6, geographical release, CPI portal and 2026 basket release. |
| `data/metadata/retrievals.json`, `manifest.json` | Refreshed source retrieval records. |
| `README.md`, `validation/` | Reproduction notes, numerical results, browser evidence and screenshots. |

## Presentation and content

- Compact four-row metadata banner with the requested historical model label. Removed masthead, old research-record panel, decorative basket/quotation and hero chapter-count label.
- Two title lines at 1280, 1440, 1600 and 1920 pixels. The headline cards retain 170.0, +3.1%, +0.1% and +0.4%; values are reduced from 42 to 36 pixels, labels enlarged from 11 to 14, and notes from 10 to 12 at normal desktop size. Equal card heights, two-column tablet and single-column phone layouts are verified.
- Consistent spacing, peach compact equation panels, blue sampling panels, green empirical details and amber limitations. Long equations, tables and narrow-screen plots use contained scrolling. Mobile plot labels retain readable size and an explicit scrolling instruction.
- The classification opens at its root, shows all eight major components and leaves deeper branches collapsed. Independent native disclosure controls reveal the authentic SDMX hierarchy. Selected-group details identify codes, vectors, indices, rates and both 2025-price and April 2026-price weights.
- The set explorer separates authentic product containment, the goods/services partition and overlapping special aggregates. Energy crosses shelter, transport and recreation; exclusion measures are not presented as disjoint branches.
- One milk/rental-service/gasoline universe connects Chapters 4, 6 and 8 and the final synthesis. A milk-price change updates elementary means, repriced expenditures, the original aggregation diagram and animation, the detailed trace and synthesis. Exceptional explicit elementary weighting is separately identified.
- Sampling now includes the documented geography → outlets → product-offers framework, source-specific exceptions and a selectable 60-offer, two-month artificial population. Empty samples remain unavailable. No official sampling confidence interval is invented.
- Added interactive assumed laptop quality adjustment, explicit weighted geometric means, aggregate-household weighting, donor-relative missing-price explanation, seasonal-product treatment and vintage-specific geographical strata.
- Geography includes schematic west-to-east tiles with northern capital cities separately identified, a fixed 0–7% colour scale, a ten-province ranking, separate city comparison, 2–4 historical trajectories and matching BC/Alberta food, shelter and transport comparisons. Official geographical expenditure shares remain distinct from the synthetic three-region aggregation experiment. Missing component series remain unavailable.
- Five-step official arithmetic exposes observations, substitution, unrounded calculation and published rounding for annual, monthly, adjusted monthly and Ontario comparisons.
- The synthesis uses ten uniform icon cards in a desktop 5 × 2 directional sequence and a smaller-screen two-column sequence. All six animations share stage selection, one timeline, play/pause and restart.

## Supplied illustrations

All nine originals were copied byte for byte, with natural aspect ratios and CSS background blending. The two trolley scenes illustrate shopping and collection; online selection and mobile clothing illustrate offers and replacements; the wallet accompanies expenditure shares; food accompanies contributions; neighbourhood buildings accompany regional components; towers accompany household purchasing power; the skyline accompanies geographic comparisons. The skyline is explicitly conceptual and does not identify a Canadian city. Exact source filenames, dimensions and hashes are in `data/processed/refinement-data.json`.

## Validation

| Check | Result |
|---|---|
| Original release/source comparisons | **242 pass:** 143 exact matches; 99 within published rounding tolerance; zero failures. |
| Additional geographical HTML-release checks | **70 pass:** fourteen geographies × three index observations and two rates. These cross-check another format of the same release, rather than adding independent retail-price estimates. |
| Calculation-engine invariants | **17 pass**, including shared price endpoints, equal-weight geometric identity, fixed-quantity aggregation, linked levels, quality adjustment and synthetic regional aggregation. |
| Original data integrity | **30 original file hashes match**; `cpi.json` and `replication.json` match their pre-refinement hashes. |
| Illustration integrity | **9 source/copy hashes match.** |
| Static dependencies | Eighteen chapters, 36 unique local links, 60 font files; no missing dependencies, duplicate IDs or broken internal anchors. |
| External page references | **17 URLs return HTTP 200** after redirects, with certificate verification enabled. |
| Responsive browser audit | All eighteen chapters measured at **1920, 1600, 1440, 1280, 1024, 768 and 390 pixels**; no document or chapter overflow. |
| Rendered content | 27 equation boxes, 21 populated SVG chart areas, six players and nine PNG illustrations. Fresh final Chromium console: no warnings or errors; no KaTeX errors. |

Actual browser controls were exercised for basket quantities, weights, formula/history/seasonality selectors, package size, quality adjustment, contributions, weighted trimming, geographic selection, purchasing-power conversion, audit filters and sampling. The new tests also cover independent tree expansion, set membership, shared-price propagation, empty/reset samples, the geographic selection guard, unavailable city components, normalized regional weights at zero/100% endpoints, and all official walkthrough variants. Player advancement, pause, restart, direct stage selection and timeline endpoints were checked.

Evidence: `validation/layout-audit.json`, `functional-audit.json`, `regression-controls.json`, `math-checks.json`, `refinement-checks.json`, `asset-check.json`, `reference-links.json`, chapter screenshots and detail screenshots. The earlier `browser-qa.md` is retained as the initial implementation record.

## Methodological and testing boundaries

The eight-component reconstruction remains Level C: **169.92945615087487 versus 170.0**, a difference of **−0.07054384912513 index points**. Detailed offer construction, full internal compatible weights, imputations and quality treatments are unavailable; the page does not claim a complete independent official CPI reconstruction. Seasonal and core results are retrieved/source-checked rather than re-estimated. Artificial sampling, assumed quality values and regional scenarios remain explicitly synthetic.

The geographical display is a labelled schematic, not a boundary map. Methodological strata counts are tied to the 2023 reference-paper vintage. Browser QA used local HTTP and the in-app Chromium engine; direct `file://`, other browser engines and a deployed GitHub Pages site were not tested. The original unavailable reference-paper index URL is replaced in page navigation by working chapter references and the retained complete PDF.
