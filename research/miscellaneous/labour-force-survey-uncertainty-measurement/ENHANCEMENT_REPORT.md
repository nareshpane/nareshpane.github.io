# LFS research page enhancement — October 13, 2026

Implemented in `D:/GitHub/nareshpane.github.io/research/labour-force-survey-uncertainty-measurement.html`. No staging, commit or push. No unrelated pages or shared assets edited.

## Files changed and created

Changed the existing HTML, `js/app.js` (conceptual person/dwelling pictograms only), `sources/guide-2025.txt` (PDF text extraction), and this project's implementation documentation. Created `css/enhancements.css`, `js/enhancements.js`, `js/map-data.js`, `animation-assets/canada-map.svg`, `sources/methodology-2017.txt`, this report, and local enhancement authoring/verification scripts under `scripts/`. Browser checks generate desktop/mobile screenshots and `figures/enhancement-browser-results.json`.

Geographic paths reuse the reference provincial trade page's Natural Earth v5.1.2 public-domain geometry, Lambert conformal conic projection with parallels 49°/77° and central meridian −96°. Local embedded path data make the animation work under both HTTP hosting and direct file opening. No trade data or trade-specific animation was copied. The reference page's actual explorer.css supplies the peach gradient, system/Segoe UI body fonts and Georgia headings. Metadata follows the Section 338 page's emoji/custom SVG convention.

## Existing material preserved

All 85 original DOM IDs remain. The sampling distributions, variance, CV, confidence/test, total-error, seasonal-adjustment, provincial comparison, September 2026 case-study, laboratory and worked-example sections were verified against the starting HTML and remain byte-for-byte unchanged. Existing pipeline, rotation illustration, all calculators and Laboratory F remain functional. The bootstrap section was expanded and its short Python example replaced with the new unequal-weight example. Quantitative points, bars and interval whiskers remain conventional statistical marks.

SHA-256 checks confirm that official-rows.csv, september-2026.json, validation-results.json, official-data.js and statistics.js are unchanged. The initial repository state already had the LFS HTML and entire project folder untracked.

## Animation

Seven stages, 12 seconds each, 84 seconds total: (1) population/geography and dwelling environments; (2) strata, geographic PSUs and dwelling SSUs; (3) households and provincial rotation, with separate territorial visits; (4) in-person/telephone computer-assisted and online collection, including proxy reporting; (5) encrypted records, editing, imputation, nonresponse and calibration; (6) collaborative analysis and quality review; (7) publication, generic media and public interpretation.

The main desktop map is 650–780 pixels high in its opening stages, with all 13 geographies selectable and a uniformly enlarged Atlantic inset. Selecting a geography expands a conceptual sampling scene. PEI bypasses a geographic PSU selection. Territories use a community-based explanation and eight quarterly-spaced interviews, not six monthly panels. All locations, people and records are illustrative.

Play, Pause, Replay, previous/next, a seven-stage selector, progress and geography controls work independently. Playback stops at completion. Selection pauses exploration; reduced motion uses manual still scenes. Native keyboard controls and Enter/Space map selection are supported. No scroll-event interception or external animation service.

## Mathematics added

Domain/stratum/PSU/SSU distinctions; frame construction versus selection; current cluster/SBgR description; per-draw PPS versus inclusion probabilities; conditional RHC estimator and variance; historical rounded-ISR cancellation; stratified two-stage probabilities and base weights; an eight-PSU worked example; PEI and clustering/design-effect implications.

Missing-data definitions, deterministic rules, carry-forward error, donor classes and variance, whole-record imputation, imputation versus weight adjustments, bias and total-variance decomposition, MCAR/MAR/MNAR, a donor transfer animation, and a seeded response/donor experiment.

Ordinary iid bootstrap benchmark; replicate totals and ratios; multinomial derivation of a rescaled cluster variance; a seeded unequal-weight simulation with original totals, replicate histogram/table, variance, SE, nominal normal interval and descriptive quantiles; collapsible Python example. The old equal-weight bootstrap calculator is retained.

## Official authority and methodological boundaries

Read local `sources/guide-2025.pdf` (49 pages) and `sources/methodology-2017.pdf` (109 pages), consulting relevant full sections; checked the official guide and Chapters 2 and 7 online. Citations appear beside the new material.

Primary authority: [2025 Guide](https://www150.statcan.gc.ca/n1/pub/71-543-g/71-543-g2025001-eng.htm), §4 pp. 19–23 (2021 Census-based redesign, April–September 2025 phase-in, four stratum types, approximately 100–400 dwellings, SBgR, PEI, rotation/territories and adjusted-CV allocation), §5 p. 24 (collection/proxy/security), §6 pp. 25–28 (imputation, coding, integrated composite calibration and processing changes), §7 pp. 29–34 (quality), §9 pp. 39–42 (dissemination and secure bootstrap-weight access), Appendix C p. 49.

Historical technical authority: [2017 Methodology](https://www150.statcan.gc.ca/n1/pub/71-526-x/71-526-x2017001-eng.htm), §§2.4–2.6 pp. 13–23 (2011 DA construction, historical 100–600-household range, PEI rationale, RHC and controlled rounding), Chapter 5 pp. 35–39 (hot-deck classes and longitudinal donors), Chapter 6 (weighting), Chapter 7 pp. 49–54 (January 2015 jackknife-to-bootstrap transition, Rao–Wu, 1,000 replicates, coordination and composite calibration).

The 2025 guide does not name the current first-stage RHC algorithm, spell out its new geographic merge/split rules, or confirm all historical rescaling/transition/single-PSU details. These remain explicitly historical or unconfirmed. It documents processing platform changes in 2019/2020 and imputation improvements in 2023. The simulations are educational, not production LFS methods. Percentile quantiles are not promoted as validated LFS confidence intervals. The small bootstrap example has no validated interval coverage claim. Existing MathJax remains an internet dependency; the new animation and simulations are local. Browser verification covers Chromium on Windows, not a full cross-browser/device audit.

## Verification

- Node syntax checks for added and modified JavaScript.
- Existing `verify_statistics.js` and `validate_analysis.py` pass, including 11 empirical geographies and official-data identities.
- `verify_enhancements.js` validates all worked inclusion probabilities and totals, deterministic seeds, multiplicities/denominators and missingness behaviour. Monte Carlo bootstrap variance 0.00016827 agrees with exact enumeration variance 0.00016801 (4,096 ordered replicate combinations).
- Chromium/CDP: real-time 84-second full playback to a stopped endpoint, pause/resume, Replay, previous/next, stage selection, all 13 geographic selections, PEI/territorial branching, keyboard selection, donor transfer/reset, both new simulations and existing interactive components.
- Desktop 1440px and mobile 390px: all seven scene layouts have no document overflow or SVG text beyond the viewport; screenshots reviewed. Atlantic leaders and panel label wrapping refined. Reduced-motion and direct-file opening pass.
- 150 MathJax containers rendered without errors; original anchors retained, local asset paths verified, no duplicate IDs or missing in-page links. No browser console exceptions or failed network requests in the test run. Static empirical tables remain available with JavaScript disabled.

## Requested Git output

`git status --short`:

```
?? research/labour-force-survey-uncertainty-measurement.html
?? research/labour-force-survey-uncertainty-measurement/
```

`git diff --stat` produced no output because both requested paths were already untracked. Nothing was staged to manufacture a tracked diff. A separate `git diff --no-index --stat` against the saved starting HTML reports the enhancement; this is a baseline comparison, not a Git-index diff.
