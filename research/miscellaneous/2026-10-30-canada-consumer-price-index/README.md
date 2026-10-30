# Canada’s Consumer Price Index research monograph

Main page: `../2026-10-30-canada-consumer-price-index.html`.

Editorial creation date: October 30, 2026. Empirical cutoff: September 2026, released October 19, 2026. The page is a static, local-first HTML study. There is no server-side calculation and no runtime network API. HTML, CSS, JavaScript, SVG illustrations, data and KaTeX are retained here.

## Open locally

From the `research/miscellaneous` directory:

```powershell
python -m http.server 8765 --bind 127.0.0.1
```

Then open `http://127.0.0.1:8765/2026-10-30-canada-consumer-price-index.html`. All paths are relative and are suitable for GitHub Pages. Embedded JavaScript data wrappers avoid fetch restrictions associated with direct file opens; direct `file://` rendering was not verified because the preview browser permits only HTTP and HTTPS. HTTP serving was verified. No files have been staged, committed or pushed.

## Original inputs preserved

- `cpi-stat-canada-18100004-SDMX/18100004_1.xml`: SDMX 2.1 structure-specific data, 1,104,793 observations through August 2024.
- `18100004_2.xml` through `18100004_26.xml`: 25 monthly updates, September 2024 through September 2026, 2,005 observations each. These are monthly patches, not geography partitions.
- `18100004_Structure.xml`: dimension codebooks, product hierarchy, geography, units and status definitions.
- `cpi-reference-paper.pdf`: February 20, 2023 edition, 71 pages.
- `cpi-reference-paper-2019-archived.pdf`: February 27, 2019 edition, 83 pages.
- `cpi-september-report.pdf`: September 2026 release, 11 pages, including Tables 1–5.

Original filenames and bytes are unchanged. SHA-256 checksums, SDMX Prepared timestamps and coverage are in `data/metadata/input-inventory.json`. Original download dates are unknown and are not inferred from Prepared timestamps. The inventory also records the structure file. Extracted text and a rendered release-table page support reproducible inspection of the PDFs.

## Additional primary sources

The source fetcher retrieves only missing files and retains the original ZIP downloads:

- Statistics Canada Table 18-10-0007-01 (`data/18100007-eng.zip`), both price periods for expenditure weights.
- Statistics Canada Table 18-10-0006-01 (`data/18100006-eng.zip`), officially seasonally adjusted series.
- Rebekah Howlett, *An Analysis of the 2026 Consumer Price Index Basket Update, Based on 2025 Expenditures*, June 15, 2026, Catalogue 62F0014M2026005.
- Consumer Prices Division concordance table (document 2301_D66_V1), mapping older publications and CANSIM identifiers to current tables and views.
- September 2026 HTML release and Bank of Canada CPI page. The September core row in the Bank table is parsed by semantic HTML column headers and independently compared with release Table 4.
- The linked 2023 methodology Chapter 8. The originally supplied `2023001/index-eng.htm` returns HTTP 404; the preserved methodology PDF and linked chapter remain usable.
- KaTeX 0.16.11 with its MIT licence and referenced fonts, stored under `vendor/`.

`data/metadata/retrievals.json` records acquisition/check timestamps, URL, byte size, availability and hashes. A cached file is reused; its subsequent check timestamp is not a new download date. The manifest includes those acquisition records. The Bank page initially failed Python certificate validation; Windows curl retrieved it with certificate verification enabled. The script supports this fallback without disabling TLS checks. Web search was unavailable; official sources were retrieved directly.

## Rebuild

Python 3.10+ is required for preparation; `pypdf` is needed only for PDF extraction. All other Python operations use the standard library. KaTeX is already local. No npm installation or JavaScript build system is needed.

Run these commands from `research/miscellaneous`:

```powershell
python 2026-10-30-canada-consumer-price-index/scripts/inspect_sources.py
# Optional: requires network access; existing downloads are reused.
python 2026-10-30-canada-consumer-price-index/scripts/fetch_sources.py
python 2026-10-30-canada-consumer-price-index/scripts/prepare_data.py
python 2026-10-30-canada-consumer-price-index/scripts/validate_cpi_replication.py
python 2026-10-30-canada-consumer-price-index/scripts/build_page.py
python 2026-10-30-canada-consumer-price-index/scripts/check_assets.py
```

`inspect_sources.py` must only be used to regenerate a baseline when the raw inputs are intentionally unchanged or newly acquired. To verify an existing baseline, run `check_assets.py` before regenerating the inventory. Otherwise a changed source could be inadvertently accepted as a new baseline.

## Transformations and outputs

- The extractor uses `ElementTree.iterparse` and clears each completed series. It retains all national product series and regional all-items/eight-major-component series rather than duplicating the full raw dataset.
- It merges monthly patches by geography, product and month, rejecting conflicting observations. All 1,154,918 observations are scanned; 499 selected series are retained.
- `data/processed/cpi.json` is the structured dataset. `cpi-data.js` wraps the same data to support static/file loading without fetch. Both contain source vectors, dimension IDs, DGUIDs, original labels, units, available flags and reference-period information.
- `data/processed/series-map.csv` maps public vectors and labels. The raw SDMX UOM is preserved. Iqaluit’s all-items reference is explicitly recorded as December 2002=100 from release Table 2; it must not be inferred from the generic UOM code alone. Product-specific reference suffixes are retained.
- `release-tables.json` is independently parsed from release PDF text. National, regional and seasonally adjusted tables provide comparators; Table 5 provides contributor **growth rates**, not percentage-point amounts.
- `bank-core.json` contains the independently extracted Bank of Canada September core values and source.
- `validation/replication.json`, `.csv` and `.md` contain values, differences, tolerances, category, status and sources. The browser uses `replication-data.js`.
- `data/metadata/manifest.json` combines sources, hashes, coverage, known vintages and transformations.
- `scripts/build_page.py` contains the scholarly content and builds the main HTML. Edit content there and rebuild; `app.js` and `styles.css` are maintained directly.
- `validation/asset-check.json` checks links, original hashes, fonts and chapter structure. `validation/browser-qa.md` records browser testing and observed fixes.

## What was validated

There are 242 independently compared release/source observations and rates: 143 matched at numerical equality and 99 growth rates fell within the documented one-decimal rounding tolerance. The dashboard has 288 rows including 44 calculations without a separate release-table comparator, one Level C approximation and one Level D limitation. The 242 include three Bank-versus-release core **source matches**, not independent core construction.

Headline arithmetic:

| Comparison | Inputs | Calculated | Published |
|---|---|---:|---:|
| Annual unadjusted | 170.0 / 164.9 | 3.0928% | 3.1% |
| Monthly unadjusted | 170.0 / 169.8 | 0.1178% | 0.1% |
| Monthly adjusted | 170.0 / 169.4 | 0.3542% | 0.4% |

Major components, selected special aggregates and gasoline are checked against Table 1. All provinces and Whitehorse, Yellowknife and Iqaluit are checked against Table 2. Adjusted components and exclusions are compared with Table 3. Twenty product rates are checked against Table 5. Additional city and product arithmetic is reported without inventing comparator values.

Published rates have one decimal place. Rate comparison tolerance is ±0.05 **percentage points**, inclusive to a tiny floating-point margin; index/source-level matches require 1e-8 numerical equality. Separate intervals expose the effect of input indices rounded to 0.1. These are rounding intervals, never confidence intervals. A missing observation remains unavailable and fails a required comparison. No missing observations are replaced with zero.

## Approximation and limits

The eight-component aggregation uses normalized 2025 basket shares at April 2026 link-month prices, component ratios from April to September and the official April headline anchor of 168.0. The rounded published shares sum to 100.01%, so normalization is explicit.

Result: **169.92945615087487**, against official **170.0**, difference **−0.07054384912513 index points**. A cross-basket annual approximation uses the 2024 basket’s April 2025 link shares for the September 2025–April 2026 segment, then the new basket factor: **3.0610282066%**. This remains Level C even though its annual rate rounds to 3.1%.

Broad component relatives, separately linked public series, rounded indices and rounded weights are insufficient to reproduce the detailed internal aggregation. This result is anchored to an official headline observation and does not independently reconstruct the earlier history. A weighted average of September component **levels** is displayed only as an invalid-method comparison, never as an alternative CPI estimate.

- Level A: independently implemented arithmetic using public observations.
- Level B: independent extraction/source matching of public series.
- Level C: approximate public-data aggregation, with its numerical discrepancy.
- Level D: product-offer construction unavailable without unpublished prices, detailed compatible weights, imputations, quality treatments and model specifications.

Official seasonal adjustment is retrieved, not re-estimated. The 2023 paper describes X-12-ARIMA; a subsequent production-method change has not been independently verified. Core values match two official sources; component adjustments, weights, histories and factor estimation have not been reconstructed. The weighted-trim, seasonal-wave and sampling experiments are explicitly synthetic. No official confidence interval is invented.

## Initial page inventory

All 18 requested chapters are implemented. There are 15 SVG plot areas, an expenditure treemap and 100% stack, expandable published classification, selected-branch aggregation trace, geographic tile selector, 26 original miniature icon designs, and five staged animations with play/pause, restart and timeline controls. Experiments cover basket prices and quantities, normalization, weights and price shocks, package-size change, elementary formulas, Laspeyres/Paasche/Fisher/Lowe, basket linking, history selection, seasonality, contributions, weighted trimming, regional selection, purchasing-power conversion, audit filtering and synthetic sampling.

Reduced-motion preferences prevent smooth navigation transitions; all animations start paused and only advance on an explicit play action or timeline input. Narrow screens use contained formula/table scrolling, responsive line/bar plots and labelled contained scrolling for wider diagrams. Plot points provide source values via titles and focus where practical; data tables and text alternatives support inspection.

The main page has 23 rendered equation boxes. The 12 unique external references passed HTTP HEAD checks; the optional `scripts/check_reference_links.py` regenerates that network smoke test. Screenshots and manual browser findings are in `validation/`. No deployed GitHub Pages URL was tested because publishing was outside scope.

## Completed visual and methodological refinements

See [REFINEMENT_REPORT.md](REFINEMENT_REPORT.md) for the current implementation, file list, illustration placements, browser checks and boundaries. The inventory above describes the initial implementation. The refined page has 27 equation boxes, 21 SVG chart areas and six staged players. The final polish displays five contextual PNG illustrations while retaining all nine originals. The original processed official dataset and replication report are unchanged.

Additional reproduction steps, from this asset directory:

```powershell
python scripts/prepare_refinements.py
python scripts/build_page.py
python scripts/validate_cpi_replication.py
python scripts/validate_refinements.py
node scripts/validate_math.cjs
python scripts/check_assets.py
```

The image preparation script reuses retained copies when source attachments are absent. When originals are available it verifies byte equality. `refinement-data.json` records both official weight vectors and supplied-image provenance. `refinement_content.py` applies targeted additions to the original eighteen chapters; `refinements.css` and `refinements.js` provide the additional presentation and interactions.

The four-row banner’s model text is the historical display label expressly requested by the author. It is not generated runtime model metadata. All responsive checks and numerical test results are retained under `validation/`.

## Final presentation and interaction polish

See [FINAL_POLISH_REPORT.md](FINAL_POLISH_REPORT.md) for the latest changes and [ASSET_AUDIT.md](ASSET_AUDIT.md) for the exact cleanup list. Source observations and official two-decimal weights remain unchanged. Visual percentages use one decimal by default; detailed source tables keep two decimals, and mathematical calculations retain useful additional precision. Formatting helpers never change calculation inputs.

The household experiment defaults to normalized April 2026 link-month weights from the 2025 official basket. Price shocks and manual weight changes are hypothetical. Synthetic mode starts from a fictional distribution. Sampling filters limit inclusion without erasing saved offer choices; native buttons retain keyboard focus. The compact comparison uses index points with earlier prices equal to 100.

`python scripts/audit_assets.py` audits only known generated QA artifacts and retains the existing deletion record. It does not delete files. Final representative screenshots and machine-readable test reports remain in `validation/`.
