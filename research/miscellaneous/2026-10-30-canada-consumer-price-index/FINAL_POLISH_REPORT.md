# Final CPI visual and interaction refinements

Completed October 30, 2026. This is a targeted refinement of the existing monograph, preserving all 18 chapters, the source data, mathematical explanations, navigation and six animation players.

## Presentation and illustrations

- The four headline cards now have equal compact heights (about 132px on desktop), a shared sage border, four restrained basket/calendar SVG symbols, 34px values, 14px labels and 12px notes. The values remain 170.0, +3.1%, +0.1% and +0.4%.
- Five original PNGs are displayed with reserved aspect ratios: the trolley beside the opening basket question, food beside the goods/services classification, the wallet beside the normalized expenditure calculation, the alternate shopper beside the sampling population, and residential buildings beside regional component weighting. Four other supplied images remain available but are not displayed. No original raster was edited or enlarged beyond its native dimensions. Generic illustration captions were removed; provenance remains in the image documentation and source metadata.
- Shared percentage helpers apply one decimal to summaries and visual labels, two decimals to official source weights and meaningful small differences, and three or four decimals to detailed arithmetic. Formatting suppresses negative zero. Calculations continue to use full-precision inputs.
- Treemap rows and columns use the original fractional weights, replacing integer grid rounding. One-decimal percentages are centered; short labels keep narrow categories readable. Native tooltips and selected-tile text give the full name, original two-decimal weight, period and source. The table retains 16.83/16.93 for Food and 28.51/28.30 for Shelter, with larger type, aligned numerical columns, light row colors and an explanation of price updating versus quantity changes.
- Large panel backgrounds are fainter while text and controls retain contrast. The two hero links and their exclusive CSS are removed. Sidebar chapter text is 1.3px larger; its width, sticky layout, keyboard access and active-section tracking are preserved.

## Household and sampling experiments

The household experiment defaults to the Statistics Canada 2025 expenditure basket, price-updated to April 2026. Its published weights sum to 100.01%; the model normalizes these original weights without first rounding them. A separate fictional basket is available. Changing a category share redistributes the remaining share proportionally, including a defined fallback after a 100% endpoint. Live slider values, a visible 100.0% total, mode-aware reset and original/modified comparison panels make the state explicit.

The calculation is exact within its stated one-period fixed-share model: P = sum(w_j * (1 + g_j/100)); with only category k changing, 100(P - 1) = w_k * g_k. This is clearly distinguished from the official CPI contribution formula and from published inflation. For example, the normalized official Food share and a hypothetical +5% shock yield about +0.85 percentage points, shown as +0.8% at summary precision.

The 60-offer synthetic population now explains selection directly. Native buttons show included, deselected and filtered-out states, retain keyboard focus, and update the count automatically. Filters restrict calculation eligibility while retaining saved selections in the full grid. Both population and subset use equal-weight geometric means; neither is presented as a probability-sample estimate. A compact common-axis comparison updates alongside the detailed logarithmic calculation. An empty subset is unavailable rather than zero. Restore all offers clears filters and restores all 60 selections.

## Modified files and reproducibility

The main HTML, `app.js`, `refinements.js`, `styles.css` and `refinements.css` contain the page changes. `scripts/build_page.py` and `scripts/refinement_content.py` reproduce the markup; `scripts/prepare_refinements.py` preserves the updated image documentation when rerun. `scripts/validate_math.cjs` adds meaningful normalization, endpoint and display-format checks. The new `scripts/audit_assets.py` reports only explicitly identified generated artifacts and preserves deletion records; it does not itself delete files. README, image documentation and this report describe the final state. No source dataset or original illustration was changed.

## Validation

- **121 numerical checks passed**, covering the existing engine plus both basket modes, all eight categories at 0%, 50% and 100%, normalization, nonnegative finite shares, endpoint recovery, full-precision effects and formatting.
- **242 original official-release comparisons passed:** 143 exact and 99 within published rounding tolerance, with zero failures. The original dataset and replication JSON hashes are unchanged.
- **70 additional geographic source comparisons passed** against the same HTML release. These are source matches, not new independent microprice estimates.
- All **30 original input-file hashes** and **nine supplied PNG hashes** match; 32 local dependencies, 60 KaTeX font files, unique IDs and all 18 chapter anchors pass the asset checker.
- Browser evidence records 17 functional cases, including official/synthetic reset, category and share changes, negative shocks, keyboard toggling, preserved selection under filters, empty subsets and restoration. A filtered four-offer example independently gives index 105.52243003576045, matching the displayed 105.52 and detailed 105.5224.
- Chromium via local HTTP was checked at **1920, 1440, 1280, 1024, 768 and 390px**. No document overflow, card mismatch or treemap label clipping was found. Treemap center errors are under 0.01px; normalized area-share errors are under 0.002 percentage points from pixel rounding.
- All 18 navigation links were clicked and confirmed to update both the hash and active chapter. All five displayed PNGs load with preserved proportions. Final console capture contains no errors or warnings. The page retains 27 equation panels and six players.

Machine-readable evidence is in `validation/math-checks.json`, `replication.json`, `refinement-checks.json`, `asset-check.json`, `polish-functional.json`, `polish-layout.json`, `polish-navigation.json` and `polish-final-browser.json`. Five representative final screenshots remain in the validation folder.

## Asset cleanup and boundaries

The completed audit removed **63 files totaling 8,389,267 bytes**: superseded generated QA screenshots/contact sheets and one regenerable Python import cache. Before removal, each exact candidate was checked for literal and dynamic references, resolved inside this project, and verified against its recorded SHA-256. The [asset audit](ASSET_AUDIT.md) lists **every deleted filename and its reason**; `validation/asset-audit.json` also records size and hash. Original downloads, three PDFs, all nine original PNGs, scripts, manifests, source inspection evidence and numerical reports remain. Unknown files were retained.

This is local Chromium validation; other browser engines and `file://` operation were not verified. The eight-component approximation remains 169.92945615087487 versus the official 170.0; unpublished microdata still prevent complete independent CPI reconstruction. The household and selection experiments retain their stated illustrative assumptions.

No unrelated page was edited. Existing changes elsewhere in the repository, including `research.html`, were left alone. Nothing was staged, committed or pushed.
