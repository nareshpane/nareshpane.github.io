# Page 2: annual product geography

## Canada and Provinces Exposure Explorer

The addition sits after the opening animation and following figure, before the
unchanged Product Explorer. `css/geography-explorer.css` and
`js/geography-explorer.js` extend the current cards and controls. One initialization
hook in `js/app.js` passes its already-loaded datasets to the new module; no new
data download or dependency is introduced. All fourteen geographic summaries are
computed once and cached. The searchable full list and Clear control are independent
of the product search and the product's exposure-ribbon geography selector.

The source is the existing Statistics Canada CIMT domestic-export snapshot,
`ODPFN018_202512N.csv` (source SHA-256 in `data/metadata.json`), reference year 2025,
destination US, integer CAD, customs-based domestic exports excluding re-exports.
Official catalogue: <https://open.canada.ca/data/en/dataset/2909a648-5753-4924-878a-b069392d9cde>.
The retained builder sums all twelve months and US states at HS6 by origin. Source
retrieval date was not recorded in the trade metadata; its generated snapshot is
dated October 6, 2026. This addition reuses that snapshot without re-downloading it.

For each geography and HS4, sum **all** its HS6 values for the denominator, and
only HS6 members of `data/section338-hs6.json` for the exposed numerator. The same
supplied September detailed-line reconstruction used by the existing ribbon is
applied as unique HS6 set membership. Parent HS4 totals are checks, not added trade.
Summary cards sum these sector numerators and denominators; overall intensity is
100 times their ratio. All positive-exposure sectors are sorted by exposed CAD,
descending (ties by code), without a top-N cut-off. Bar width is exposed CAD relative
to the largest sector for the selected geography. Blue colour interpolates linearly
from RGB(161,199,229) at 0% to RGB(14,65,126) at 100% **sector** intensity.

Canada is constructed from the thirteen source origins because the archive has
no national-origin record. All sums reconcile to the retained source audit. There
is no additional authoritative national total to compare within this archive.
Source-absent observations are already explicit zeroes in the validated dense
arrays; unavailable scope or invalid/missing values produce an unavailable state.
A zero export denominator yields unavailable intensity. No provincial allocation
is estimated. HS6 coverage is a potential upper bound; detailed classifications,
packaged-only restrictions and shipment exceptions are unresolved, and the supplied
policy extracts do not certify current implementation or tariff liabilities.

Run from the repository root:

```powershell
node research/harmonized-system/2-section-338-hs4-hs6-exposure-canada/scripts/verify_geography.js
```

This independently checks every geography and sector against HS6 arithmetic and
the existing exposure model, including Alberta, Ontario and British Columbia,
plus missing values, duplicated scope, ranking, formatting and unchanged opening
and Product Explorer content. The local server instructions below still apply.
Rendered checks also passed in installed headless Edge: all fourteen selections,
Clear, keyboard search, focused tooltips, full rankings, widths and colours,
anchor links, independent product/ribbon selections, introductory animation,
unique IDs and no page overflow at 320, 390, 768, 1024 and 1440 pixels.
The reproducible browser check starts and stops its own temporary local server:

```powershell
node research/harmonized-system/2-section-338-hs4-hs6-exposure-canada/scripts/verify_geography_browser.js
```

It needs Node 22+ and an installed Edge/Chrome (or `BROWSER_EXE`), installs nothing,
uses an isolated temporary browser profile, and leaves screenshots in the printed
temporary directory. The historical review notes below describe earlier additions.

Research draft at `../section-338-hs4-hs6-exposure-canada.html`. This reuses the existing
`2-section-338-hs4-hs6-exposure-canada/{css,js,data,scripts}` layout. No framework, package
installation, browser library or site build is needed. Raw sources stay outside Git.

From the **repository root** on Windows:

```powershell
python research/harmonized-system/2-section-338-hs4-hs6-exposure-canada/scripts/build_data.py
python research/harmonized-system/2-section-338-hs4-hs6-exposure-canada/scripts/verify_static.py
node research/harmonized-system/2-section-338-hs4-hs6-exposure-canada/scripts/verify_app.js
python -m http.server 8000
```

Open <http://localhost:8000/research/harmonized-system/section-338-hs4-hs6-exposure-canada.html>.
The previous public route, `../preferential-tariffs-canada.html`, is a small
compatibility redirect to the new page, with a normal fallback link.
`py -m http.server 8000` also works if the Windows Python launcher is installed.
Stop the server with Ctrl+C. Use HTTP rather than a `file://` URL for JSON fetches.
Node is needed only for the optional search-function checks, not for the page or builder.

## Sources and reproducibility

The saved `scripts/source-inspection.txt` and inspection scripts establish actual
schemas, row counts, encodings and file contents. The production builder uses
only Python's standard library, `pathlib`, and a configuration block at the top.
Paths can also be supplied through `--statcan-dir`, `--section338-dir`, and
`--cbsa-file`. It never writes into any raw-source directory.

The local archive is `CIMT-CICM_Dom_Exp_2025`. Programmatic header selection chooses
**ODPFN018_202512N.csv**, the HS6 dataset; 016 is HS8 and 020 is HS2. Do not combine
these overlapping aggregation levels. UTF-8 comma-delimited columns are:

| Column | Meaning / treatment |
| --- | --- |
| YearMonth/AnnéeMois | Filter 2025; sum all twelve months |
| HS6 | String identifier; preserve leading zeroes |
| Country/Pays | Filter US |
| Province | One of the thirteen origin abbreviations |
| State/État | Sum U.S. states into destination US |
| Value/Valeur | Integer Canadian-dollar value |
| Quantity/Quantité | Retained for complete-record checks, not visualized |
| Unit of Measure/Unité de Mesure | Retained for complete-record checks |

Lookups have no header: fixed-width Windows-1252 text. Use date-valid HS6 export
descriptions and geography entries. The supplied English country name is
“United States of America”. HS4 descriptions reuse CBSA explicit headings or
heading paths confirmed across **all** children. National export provisions
9802/9901 have no compatible heading description or Canadian import detail here.
2026 import tariff items are context; they are not a 2025 export classification
concordance or a single rate applying to every item within HS6.

## Output schema

All production JSON is minified. No monthly values are emitted.

- `metadata.json`: source SHA-256, headers, origin order, audit, description source
  labels and CBSA snapshot provenance.
- `search-index.json`: `[code, description, descriptionSourceIndex]` rows. HS4
  and HS6 identifiers remain strings. Descriptions occur once in this index.
- `exports-hs4-2025-us.json`, `exports-hs6-2025-us.json`: product-code keys and
  thirteen-element CAD arrays, in `metadata.origins` order. Absent observations
  are zero. Canada is the array sum; it is not a fourteenth origin.
- `section338-hs6.json`: unique matching universe, detailed-line amendments,
  partial/full ban lists, source hashes/URLs, matching counts and exposure audit.
- `tariffs-0.json` through `tariffs-9.json`: lazy first-digit bundles of Canadian
  tariff rows. Each bundle pools repeated strings. Rows are
  `[tariffItem, statisticalSuffix, descriptionIndex, unitIndex, mfnIndex,
  preferencesIndex]`. The last four indices refer to that bundle's `strings`.
  Keep blank cells and separate rates. Source URLs and warnings remain per HS6.

The core initial payload is approximately 0.85 MB uncompressed. Tariff bundles
load only for an HS6 selection and are cached during the session.

## Validation outcome

| Check | Independently generated result |
| --- | ---: |
| Raw observations, all destinations | 1,212,795 |
| Raw value, all destinations | CAD 720,769,312,708 |
| Filtered 2025 U.S. observations | 808,936 |
| Exact duplicates removed | 0 |
| Positive annual HS6 product-origin rows | 17,132 |
| Positive annual HS4 product-origin rows | 6,289 |
| Dense HS6 product-origin cells, including zeroes | 59,059 |
| Dense HS4 product-origin cells, including zeroes | 15,275 |
| Origins / months aggregated | 13 / 12 |
| HS4 / HS6 products | 1,175 / 4,543 |
| Canada → U.S. domestic exports | CAD 517,282,733,046 |
| Reconstructed Section 338 HS6 codes | 412 |
| Positive-export HS6 matches | 396 |
| Exports in matched HS6 codes | CAD 36,362,891,123 |
| Exposure share | 7.0295969303% |

The rounded reference targets reconcile: CAD 517.28B and CAD 36.36B. Differences
from the rounded targets are CAD 2,733,046 and CAD 2,891,123 respectively.
They do not feed any aggregation or classification choice. Every displayed HS4
and HS6 reconciles exactly to its thirteen origins, and each HS4 reconciles to
its HS6 children. Provincial totals reconcile before and after aggregation.
Future exact-record repeats cause the builder to stop for review rather than
silently deleting coincident observations.

## Section 338 scope decision

The two supplied CSVs are identical by bytes, SHA-256 and records: 554 lines,
373 July HS6 prefixes. Use `s338_products_browser.csv` as the canonical copy.
Local September alcohol/auto Annex II texts replace HTS8 parents and retain
specific HTS10 children; Annex I confirms additions/removals. Amend **detailed
lines first**, then exclude unrestricted ban lines and collapse to unique HS6.
Packaged-only bans do not remove an entire tariff line: other goods can remain
tariff-covered. Their lines are retained and explicitly flagged as partially banned.

Diagnostic results: July has 373 HS6 codes / 360 positive matches / CAD 33.184B;
after amendments, before bans: 420 / 404 / CAD 37.090B. Incorrectly excluding
all packaged-only lines gives 401 / 385 / CAD 35.587B. Preserving their residual
scope yields the validated 412 / 396 / CAD 36.363B.

The original reference described September 29 bans, but local ban annexes alone
do not establish those effective dates. This is labelled a supplied September
scope reconstruction, never a certified current legal schedule. Rates and legal
status are not inferred from July CSV fields. No local file establishes Yale
Budget Lab authorship; that attribution is not invented. Scope matches are
not duties paid, tariff revenue, export reductions or economic losses.

The canonical CSV also contains six lines not found verbatim by the code parser
in the corresponding July text extracts: 22042961, 22042981, 39211900, 39219050,
62104055 and 62105055. They remain part of the **supplied CSV** universe; these
local extracts cannot independently certify their July inclusion. This source
gap is another reason to keep the reconstruction labelled as supplied context.

## Review and testing

`verify_static.py` checks actual data, independent HS4-child reconciliations,
Canadian tariff prefix integrity, HTML IDs/assets/navigation, and HTTP responses
when the local server is running. `verify_app.js` runs the actual production
search and formatting functions against generated data without a browser or
installed packages. These checks do not substitute for rendered browser testing.

In a browser, review at 1440, 1024, 768 and 390 px; search 8414, 841490, 9403,
8537, pump, furniture, electrical, vacuum pump, electrical panel and 0601.10.
Repeat HS4/HS6 selections; use Arrow Up/Down, Enter, Escape and Tab; inspect both
measure toggles, linked origin highlights, heatmap drilldown, tariff disclosure
and optional policy context. Use the OS/browser reduced-motion preference and
check the console. CSS disables transitions/animations under reduced motion;
JS disables numeric tweening. Narrow bars keep full-sized labels above tracks,
and the heatmap/tariff tables scroll independently.

The working session had no connected browser surface. HTTP, data, syntax and
search-function checks were performed; rendered-width, console and actual
keyboard/hover tests remain for visual review. No Playwright was installed.
No commit, staging or push was performed.

## HS4 Section 338 exposure ribbon

The focused addition uses `js/exposure-ribbon.js` and the same loaded HS6 arrays
and policy membership as the existing explorer. `build_data.py`, the generated
JSON and Section 338 reconstruction are unchanged. Canada is the default; a
local selector switches this one ribbon among all thirteen origins.

For heading g and geography p:

```
total(p,g) = sum(exports(p,h) for all HS6 h within g)
exposed(p,g) = sum(exports(p,h) for matched HS6 h within g)
unexposed(p,g) = total(p,g) - exposed(p,g)
intensity(p,g) = exposed(p,g) / total(p,g) * 100
```

The denominator is **all exports within this heading for this geography**. It
is neither the province-wide export total nor Canadian HS4 exports when a
province is selected. Zero totals produce `n/a`, not a fabricated 0% intensity.
Exposure value is dollars; intensity is the percentage, so neither substitutes
for the other.

Positive HS6 children are sorted by value, ties by code. Retain the largest
until they cover at least 95% of heading value, subject to a maximum of twelve
individual segments. Group the remainder separately as other exposed and other
unexposed HS6. Every child contributes mathematically, including those grouped;
zero-value children have no width. The independently scrollable ranked list
retains all children, with an exposed-only filter that never changes the ribbon
or denominator. Hover, keyboard focus and tap link each row to its segment.
The original reduced-motion rule covers the addition.

Run the independent verification against the production calculation functions:

```powershell
node research/harmonized-system/2-section-338-hs4-hs6-exposure-canada/scripts/verify_exposure.js
```

All **16,450** heading/geography combinations passed exact total, numerator,
complement and grouping checks. Canada exposure also equals the sum across all
thirteen origins. The existing search, static data and HTTP asset checks passed.

| Canada HS4 | Total CAD | Exposed CAD | Unexposed CAD | Intensity shown | HS6 children / matched |
| --- | ---: | ---: | ---: | ---: | ---: |
| 8414 | 865,827,277 | 387,382,560 | 478,444,717 | 44.7% | 10 / 1 |
| 8537 | 2,762,167,821 | 2,520,378,489 | 241,789,332 | 91.2% | 2 / 1 |
| 9403 | 3,429,096,138 | 2,584,043,774 | 845,052,364 | 75.4% | 12 / 10 |
| 1001 | 941,711,426 | 0 | 941,711,426 | 0.0% | 4 / 0 |
| 1210 | 287,022 | 287,022 | 0 | 100.0% | 1 / 1 |

There are 8,986 heading/geography combinations with no exports. The largest
heading by child count is 0303 with 26 children; its Canada ribbon uses eleven
visible segments. Tests cover missing-description fallback and reject HS4 totals
that disagree with children. No browser surface was connected for rendered
review. Inspect the ribbon's responsive layout and native keyboard/hover behavior
at the existing localhost URL before committing.
