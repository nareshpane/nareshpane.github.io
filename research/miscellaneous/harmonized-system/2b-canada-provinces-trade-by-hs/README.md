# Canada and Provincial Domestic Exports by HS4 and HS6, 2025

Draft page: ../canada-and-provinces-trade-by-hs.html.
This is a dependency-free static GitHub Pages page. The original Section 338 page/assets and main research.html are unchanged. Its new collection entry immediately follows Section 338.

## Local preview

From D:\GitHub\nareshpane.github.io:

~~~powershell
python -m http.server 8000 --bind 127.0.0.1 --directory .
~~~

Open <http://localhost:8000/research/harmonized-system/canada-and-provinces-trade-by-hs.html>.

The included start_preview.ps1 starts the same server in a hidden process, with its PID and logs in ignored .qa/. To stop it, read .qa/server.pid and run Stop-Process -Id followed by that PID. A foreground server stops with Ctrl+C. Do not start two servers on the same port.

Use HTTP for local testing: file:// can block JSON fetching and external SVG references. GitHub Pages needs only the included relative assets; it never runs Python.

## Source and inspection

External, read-only archive:

D:\Trade_Data_Scientist_Gov_Alberta\raw_data\statcan\CIMT-CICM_Dom_Exp_2025

| File | Inspected format | Use |
| --- | --- | --- |
| ODPFN018_202512N.csv | UTF-8 CSV; 1,212,795 monthly HS6 records | Primary observations |
| ODPFN016_202512N.csv | UTF-8 CSV; 1,265,782 HS8 records | Independent origin × HS6 check |
| ODPFN020_202512N.csv | UTF-8 CSV; 291,204 HS2 records | Independent month × origin × chapter check |
| ODPF_4_HS6XDesc.TXT | Windows-1252 fixed-width | HS6 export descriptions valid in 2025 |
| ODPF_5_HS2Desc.TXT, ODPF_8_ProvDesc.TXT | Same | Chapter and origin identifiers/names |
| ODPF_6_CtyDesc.TXT, ODPF_7_StateDesc.TXT | Same | Destinations and U.S. states |
| ODPF_2_HS8Desc.TXT, ODPF_9_UOMDesc.TXT | Same | Inspected ancillary metadata |
| English/French user notes | DOCX XML | Classification and validity-date documentation |

source-inspection.json retains every file's SHA-256, size, actual schema/dimension inventory, geographic totals, and accompanying notes extracted as text.

The actual HS6 header is:

~~~text
YearMonth/AnnéeMois,HS6,Country/Pays,Province,State/État,Value/Valeur,Quantity/Quantité,Unit of Measure/Unité de Mesure
~~~

YearMonth/AnnéeMois covers 202501–202512. There is no trade-flow column: the archive itself identifies **domestic exports**. Imports, total exports and re-exports are not interchangeable. The builder requires the supplied Dom_Exp_2025 archive identity and discovers product files by their headers.

Lookup start/end dates are at positions 11:17 and 18:24; English HS6 descriptions at 29:111; other English descriptions at 25:107. Origin numeric identifiers plus abbreviations occupy 0:11. Active validity intervals are checked before joining labels.

## Reproduce

Python 3.11+ standard library only:

~~~powershell
python research/harmonized-system/2b-canada-provinces-trade-by-hs/inspect_sources.py
python research/harmonized-system/2b-canada-provinces-trade-by-hs/prepare_trade_data.py
python research/harmonized-system/2b-canada-provinces-trade-by-hs/validate_static.py
~~~

For relocated input, pass --source "X:\...\CIMT-CICM_Dom_Exp_2025" to all three commands.

CSV processing streams rows. Duplicate checks retain one ordered month, and annual counters are sparse; full raw CSVs are not loaded into memory. Hashing streams too. Unexpected ordering, duplicates, repeated keys, suppression/missing monetary cells, incompatible codes or reconciliation differences stop the build. Assertions precede browser-data writes.

## Aggregation decisions

- Only January–December 2025 is selected. There are no annual summary rows or world-destination totals.
- Sum each integer-CAD HS6 observation once across destinations and U.S. states; do not layer HS2/HS8 observations into it or sum incompatible quantities.
- All 221 reported destination categories remain included. ZX (unknown/unspecified), ZZ (high seas), PC (Pacific Islands) and CA (Canada) each have $0 in this archive.
- The exact origin abbreviations are AB, BC, MB, NB, NL, NS, NT, NU, ON, PE, QC, SK and YT. Their active lookup has thirteen unique numeric identifiers, covering all ten provinces and three territories. No Canada or residual origin records exist.
- CANADA is therefore explicitly derived from these mutually exclusive, exhaustive origins. It reconciles at HS6, HS4 and grand-total levels. No independent national observation is available; reconciliation establishes internal consistency.
- HS6 codes stay six-character strings, including leading zeros; HS4/HS2 are their four-/two-digit prefixes.
- Zero records remain in audits; positive annual products enter rankings. Ties use ascending product code with consecutive ordinal ranks.
- Shares are stored to eight decimals. The page calculates widths/shares from exact integer values, so stored percentage rounding cannot change bar composition.
- Chapters 98/99 are Canadian special classification provisions. Values are observed domestic exports, not forecasts, tariff amounts, opportunity or vulnerability scores.

## Description provenance and limitation

HS6/HS2 labels use the official supplied StatCan lookups, restricted to validity dates overlapping 2025. They retain the source's abbreviated wording.

hs4-official-2025-extract.json retains rows and URLs from [Statistics Canada's 2025 Canadian Export Classification](https://www150.statcan.gc.ca/n1/pub/65-209-x/65-209-x2025001-eng.htm). Explicit headings take precedence. Unsplit .00 rows label a heading only when its active source lookup confirms one HS6 child, resolving export-specific Chapter 98/99 provisions.

Direct local downloads failed during TLS negotiation. Accessible official rows were retained through web-tool rendered-page/indexed-text extraction. **189 of 1,216 exported headings use the retained official CBSA T2026-2 heading-label supplement**, copied to hs4-cbsa-reference.json. Only explicit headings or heading paths confirmed across every child are accepted. These are descriptions only; no import values, rates or 2026 export observations enter the analysis. Both references use the HS 2022 heading hierarchy; a complete classification concordance is not asserted.

Each heading's source/URL appears in its detail panel, product-descriptions.json and validation-2025.json. No HS4/HS6 labels are missing.

Optional refresh when the official website is locally reachable:

~~~powershell
python research/harmonized-system/2b-canada-provinces-trade-by-hs/fetch_classification.py
python research/harmonized-system/2b-canada-provinces-trade-by-hs/prepare_trade_data.py
~~~

The downloader reconstructs the same exact-year extraction from official HTML and retains chapter hashes in classification-provenance.json. A failed download leaves the snapshot intact. Ordinary trade builds are offline.

## Output files and schema

| Files | Contents |
| --- | --- |
| trade-<ID>-2025.json (14 files) | Ranked HS4 totals with all positive HS6 children, one geography per file |
| geography-summary-2025.json | Fourteen summaries and HS4 comparison index |
| product-descriptions.json | HS2/HS4/HS6 labels and per-heading provenance |
| destinations-<ID>-2025.json (14 files) | HS4 → ranked individual destination codes and integer-CAD values, loaded per origin |
| destination-countries-2025.json | Official active CIMT names for 221 positive country/territory categories |
| animation-data-2025.json | Alberta's actual three largest headings/children and 2711's three largest destinations |
| validation-2025.json, source-inspection.json | Data checks, sources/hashes, schemas and notes |
| js/app.js, js/display.js, css/*.css | Static explorer, shared display formatting/palette and adapted template styles |
| js/origin-exposure-animation.js | Seven-sequence SVG timeline |
| animation-assets/animation-maps.svg, build_animation_maps.py | Dedicated Natural Earth v5.1.2 Canadian map and retained generator |
| animation-assets/world-map.svg, build_world_map.py | Dedicated Robinson-projected world land geometry and offline extraction script |
| validate_browser.js, ui-validation.json | Real Chromium checks and results |
| validate_static.py, static-validation.json | Serialized data, links and source-immutability checks |

Each trade file has geography, year, total, and headings. A heading record is:

~~~text
[HS4, integer CAD, share of geography (%),
  [[HS6, integer CAD, share of HS4 (%), share of geography (%)], ...]]
~~~

Ranks are array positions plus one. Comparisons map HS4 → geography → [integer CAD, rank, share of geography (%)]. Absent entries display “No positive recorded exports.”

Canada's trade file is 237 KB, descriptions 727 KB, summary/comparisons 271 KB (decimal). Separate geography files are cached/prefetched after Canada becomes usable. All uses a scrolling chart; tiny segments retain true widths and complete accessible table values.

## Destination rankings and display

The builder retains each monthly HS6 observation by origin × HS4 prefix × official country code before aggregating its exact integer value. U.S. state records are summed once. There is no world/all-country row to add. Each destination file contains geography, year and headings, where a heading maps to [[countryCode, integerCAD], ...]. Country ties use ascending official code; ranks are positions plus one.

The international ranking excludes ZX (unknown/unspecified), ZZ (high seas), PC (legacy Pacific Islands group) and CA (Canada), all $0 here. Ordinary separately reported territories remain under their official source names. **221 country/territory categories** have positive exports. Across Canada and its thirteen origins there are **125,952 positive origin–HS4–destination combinations**. Individual country sums reconcile to every HS4 total with **$0 difference**, without rescaling; Canada also equals the thirteen origins for every heading–country pair. Future nonzero excluded-category residuals are recorded in the audit and displayed alongside the chart.

All shares use the full origin-HS4 value, including countries outside the displayed Top N. Top 10 (default), Top 20 and All retain deterministic ranks. Omitted observations form a separate, unranked Other destinations total. Canada’s destination file is 563 KB; all destination files plus names are about 1.72 MB. They load only as needed and remain independent of the reference pages at runtime.

Clear in Section 02 removes the HS4 selection, HS6 details, comparison values and country chart. It preserves geography and HS4 Top N. Province changes keep the selected heading if it has positive exports there; otherwise the largest heading is selected. Province changes preserve a deliberately empty selection until another heading is chosen. The original geography-search clear control still resets to Canada and its largest heading. Request tokens prevent earlier destination loads from restoring stale bars.

Shared js/display.js formats **all user-facing shares to one decimal** (including 0.0%) without changing precision. HS6 ranks 1–3 use rgb(174,64,27), rgb(230,126,83), rgb(242,173,136); later ranks progress more gradually to visible peach. Widths and ranks retain exact source calculations. The metadata banner has the four requested fixed entries; the lighter body theme, blue instruction row and teal Clear button do not affect data.

## Animation

The first two Canadian origin/production scenes and existing ship, truck, rail, container artwork, cargo attachments and deterministic timeline are retained. Sequences 3–5 use the reference trade-sector page’s recognizable Natural Earth v5.1.2 50m world land geometry in Robinson projection. Routes connect Canada to the United States, United Kingdom, Japan, Brazil, India and South Africa. Road/rail freight remains in a separate schematic inland strip; overseas ships use coastal endpoints, with the Pacific route wrapping across the world-map seam. Routes illustrate global reach, not observed shipment volumes or product-specific transportation modes.

Seven sequence start times: 0, 6, 12, 19, 25, 32, 37 seconds; completion at 40 seconds at 1×. Sequence six selects Alberta and shows actual HS4 2709 ($119,844,663,416), 2711 ($11,762,290,066) and 3901 ($3,596,310,629). Sequence seven expands 2711 into its six HS6 children, then branches to its three largest recorded destinations: United States of America, Japan and Korea, South. The orange gradient uses HS6 rank within each heading. “Exposure” means relative export concentration/composition.

Autoplay suspends offscreen or in hidden tabs. Controls include pause/resume, replay, speed, position and a still illustration. Reduced motion displays the final frame. Replay resets cargo attachments, transforms, fades and wheels from the same timeline.

Recreate the dedicated world asset offline:

~~~powershell
python research/harmonized-system/2b-canada-provinces-trade-by-hs/animation-assets/build_world_map.py
~~~

This copies validated embedded land geometry from the unchanged local machine-learning trade-sector page; production uses only world-map.svg in this project.

## Validation

- **$720,769,312,708 CAD**, **14 geographies**, **1,216 HS4 headings**, **4,998 HS6 products**.
- Twelve months; 1,212,795 HS6 records; 354 zero-value source records.
- Zero exact duplicates or repeated observation keys; no missing monetary cells or explicit suppression markers.
- Exact HS4/HS6 sums, all Canada/origin sums and deterministic descending rankings.
- HS2 month × origin × chapter and HS8 origin × HS6 differences: **$0**.
- All individual-destination sums versus HS4, and national heading-country sums versus thirteen origins: **$0**.
- SHA-256 checks confirm the original 14 trade assets, summary/comparison index and product descriptions are byte-for-byte unchanged after extension.
- Real Chrome checks: every geography, search/clear and keyboard controls, all ranking limits, every Canada stack's children/proportions/gradient, HS4 selection, full HS6 details, focus/tap tooltips, comparison/absent values, resize, 768/390/320px layouts, all seven animation phases, replay/pause/speed/still mode, cargo attachments, and reduced motion.
- New checks cover fixed metadata, banner/Clear colours, one-decimal shares, first-three HS6 shades, destination synchronization, keyboard and actual mobile touch tooltips, Top 10/20/All on Alberta HS4 3901 (46 destinations), exact unranked Other totals, Clear during a pending response and empty downstream states.
- No browser JavaScript or asset-loading errors. All 1,216 headings rendered in roughly 0.2–0.3 seconds locally; timing varies by device.
- Desktop/mobile screenshots were visually inspected. Test-only artifacts stay in ignored .qa/.

With the preview server running, rerun browser checks using Node 22+ and installed Chrome or Edge:

~~~powershell
node research/harmonized-system/2b-canada-provinces-trade-by-hs/validate_browser.js
~~~

This uses Chromium's built-in DevTools protocol in an isolated headless profile; it adds no npm dependencies to the site. TRADE_TEST_BROWSER can select another installed Chromium executable.
