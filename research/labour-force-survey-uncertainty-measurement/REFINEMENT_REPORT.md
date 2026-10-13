# Targeted LFS visual refinements

## Latest small presentation pass

- The existing two-row desktop navigation is now a warm rounded panel with a sage border and modest padding. Labels, anchors, active states and responsive wrapping are unchanged.
- Toronto and Vancouver have taller varied-height towers, denser apartment blocks, crossings, lamps and restrained traffic details. Kelowna has lower housing, lawns, trees and a cul-de-sac. Edmonton's foreground residential areas are lower and greener. Northern Ontario and rural Saskatchewan have smaller, fewer homes and more open landscape. Camera logic, geography, scene timing and the 133-second duration are unchanged.
- Removed the bootstrap pseudocode and collapsed Python code reproduction, which duplicated the existing formulas and demonstration. Retained the coordinated-replicate explanation, all derivations, access qualifications, source references and a compact numerical-check link.
- Province summary cards retain their contents and grid layout, with a 28-pixel gap below Table 3 and subtle sage borders.
- Laboratory A initializes its existing 500-sample SRS default; Laboratory F initializes its existing 1,000-replicate bootstrap default. Both use the same seeded routines as manual runs and resets. Subsequent parameter edits still require the existing Run button. Both distributions fit phone widths with fewer axis labels; resizing redraws the cached estimates without resimulating. The bootstrap SVG viewBox now includes its bottom axis label.
- The article's six-month rotation component uses five neutral continuing cards and one lightly tinted entering card, with a sage border and bold status. Interview numbers and rotation logic are unchanged; phones show two rows of three cards.
- Modified in this pass: the primary HTML, `js/app.js`, `js/regional-scenes.js`, `css/visual-refinements.css`, `scripts/verify_visual_refinements.js` and this report. Added `scripts/refine_presentation.py` and `figures/presentation-*.png` screenshots; refreshed the existing browser results and scene screenshots. No new published asset type or dependency.
- Local Chrome checks pass at 1440, 768, 390 and 320 pixels, including initial charts before any submit, deterministic reset, nav borders/anchors, table/card separation, rotation handover/play/pause/previous/reset, autoplay, playback controls, reduced motion and expanded mathematics. No page or equation overflow, missing anchors or JavaScript errors. Numerical statistics and exact-enumeration bootstrap checks pass. All 152 existing source equations and 27 protected source/data/statistical files remain unchanged.
- Version-control status remains the two untracked LFS paths shown below. `git diff --stat` remains empty. Nothing staged, committed or pushed.

The following records describe the earlier, larger visual refinement pass.

The existing SVG scene graph, deterministic central timeline, viewport-aware playback and bottom controls are retained. Duration increases from 108 to **133 seconds**, with 19 navigable stages. There are no new dependencies or independent animation timers.

## Animation

- Canada remains the geographic anchor. Seven residential environments zoom from and return to their mapped locations: Toronto, northern Ontario, Vancouver, Kelowna, Regina, rural Saskatchewan and Edmonton. PEI and Yukon retain their distinct design explanations.
- Toronto has towers, multiple apartment dwellings and a conceptual neighbourhood boundary; northern Ontario contrasts forests, winter ground and widely separated homes. Vancouver adds waterfront and mountains; Kelowna adds an interior lake, hills and mixed housing. Regina and Edmonton retain substantial urban scale, while rural Saskatchewan shows fields, farmhouses and dispersed roads.
- Camera interpolation and opacity use the existing timeline. Geographic points are derived from the map projection and tested inside their corresponding provincial outlines.
- The sampling scene names Primary Sampling Unit (PSU) and Secondary Sampling Unit (SSU), with a province/stratum/PSU/dwelling/eligible-person hierarchy.
- Rotation identifies the outgoing panel's sixth interview, the replacement's first interview and the other five retained panels. The month indicator advances and the figure distinguishes dwelling rotation from changing household composition.
- The winter landscape is illustrative. Coverage qualifications distinguish target-population exclusions from sparsely populated remote frame exclusions and weighting controls, following the [2025 LFS Guide, section 4](https://www150.statcan.gc.ca/n1/pub/71-543-g/71-543-g2025001-eng.htm#a4). Existing historical 2017/current 2025 qualifications remain.

## Static explanations

- The weighting figure now follows one dwelling → three residents → three base weights of 45 → weighted contribution 135. Thin arrows replace the black wedge. Its three display formulas and caption distinguish representation from household size and explain final weight adjustments.
- Dwelling, Household, and Respondent and observation have sage panels and distinct pictures. Their original definitions are preserved immediately below the panels.
- Employed, Unemployed and Outside the labour force have respectful illustrations and separate green, amber and blue-green backgrounds. Their substantive explanations are unchanged.
- Seven visible tabs use the original scenario labels and unchanged classification rules. Selection supports clicks, arrow keys, Home/End, roving focus and an associated tab panel. The old select is hidden while preserving the original calculation interface.
- The estimand illustration separates target population, covered frame, strata, selected geographic PSUs, dwelling SSUs and eligible people, with an explicit coverage distinction.
- The synthetic design map has five stages: mixed geography, stratification, PSU formation, PSU selection and dwelling selection. Its caption explicitly disclaims replication of the production algorithm.
- Three compact mathematical aids illustrate clustering/design effects, PSU-level replicate weight configurations, and sampling variability versus systematic shifts. Separate phone layouts preserve readable labels. Quantitative chart colours are unchanged.

## Files

Existing files modified:

- `research/labour-force-survey-uncertainty-measurement.html`
- `js/documentary.js`
- `js/app.js`

New assets within this project's folder:

- `js/regional-scenes.js`: geographic environments and deterministic regional state.
- `js/visual-refinements.js`: reusable vector illustrations and accessible tabs.
- `css/visual-refinements.css`: scoped palettes and responsive layouts.
- `scripts/refine_visuals.py`, `scripts/polish_refinements.py`, `scripts/verify_visual_refinements.js`: implementation, preservation and browser checks.
- `figures/refinement-baseline.json`, `figures/refinement-browser-results.json` and `figures/refinement-*.png`: preservation snapshot, test results and desktop/tablet/mobile screenshots.
- This report.

No raster artwork, framework, shared site asset or unrelated research page was changed.

## Validation

- Chrome browser checks at 1440, 768, 390 and 320 CSS pixels: no page overflow, equation overflow, equation scrollbars, duplicate IDs, missing anchors or JavaScript errors. All mathematical details were expanded.
- Viewport autoplay/offscreen suspension, pause/resume, replay, previous/next, automatic completion, reduced-motion navigation and 50 rapid control cycles pass. Replay does not accumulate SVG nodes; 533 timeline samples have finite transformations.
- All seven classification cases and keyboard selection pass; the superseded dropdown is visually hidden.
- Existing laboratories, donor/reset, imputation, bootstrap, provincial controls, no-script tables and direct local-file opening pass.
- `verify_statistics.js` passes numerical implementation, seed, sampling, MSE, interval, covariance and bootstrap checks.
- `validate_analysis.py` passes all 11 geographies, identities, denominators, changes, standard errors, examples and local paths.
- Preservation audit confirms all **152 existing source equations**, **23 sections**, forms and **27 protected data/source/statistical files** unchanged. No mathematical correction or numerical result change was needed.
- Desktop and phone screenshots were inspected for the regional scenes, rotation, tabs, weighting, conceptual boxes, population hierarchy, synthetic design and mathematical aids.

## Version control

The LFS page and asset directory were already untracked at the start. Nothing was staged, committed or pushed.

`git status --short`:

```text
?? research/labour-force-survey-uncertainty-measurement.html
?? research/labour-force-survey-uncertainty-measurement/
```

`git diff --stat` has no output because these project files are untracked.
