# Data and provenance

Retrieval dates, byte counts and SHA-256 hashes for unmodified input snapshots appear in [retrievals.json](metadata/retrievals.json). Agency publications, dataset pages and theory references appear in [source-manifest.json](metadata/source-manifest.json).

## Official inputs

| Folder | Files | Coverage used |
| --- | --- | --- |
| Canada | 18-10-0004-01 unadjusted CPI cube, including its original metadata | January 2000 to October 2026, national and selected subnational series |
| Canada | 18-10-0006-01 seasonally adjusted CPI cube | National all-items CPI, January 2000 to October 2026 |
| Canada | 18-10-0007-01 weights cube | 2023–2025 baskets calculated at their link-month prices |
| United States | Three public API time-series JSON responses | National CPI-U, January 2000 to October 2026 |
| United States | December relative importance tables and BLS annual variance workbooks | 2023–2025 |
| United States | Monthly response-rate workbook, response methodology HTML and archived October CPI release | January 2019–October 2026 response series and October 2026 release |

The original national API work used CU (CPI-U), U or S (unadjusted or seasonally adjusted), 0000 (U.S. city average), and the BLS item code. The official API returned the requested identifiers, and archived release tables provide spot checks. The geography acquisition also obtained the official flat-file series catalogue. See the raw archives and agency links for original labels and technical notes.

## Harmonized outputs

| Field | Meaning |
| --- | --- |
| country, agency | CA / Statistics Canada or US / BLS |
| category, label | Study key and source product classification |
| geography, series_id | Official geographic label and Statistics Canada vector or BLS series code |
| adjustment | NSA unadjusted or SA seasonally adjusted |
| reference_month, value | Calendar month in YYYY-MM and published index level |
| units | Index reference base as available |
| missing, status | Explicit missing indicator and source status, when reported |
| source, retrieved | Source file path and acquisition time |
| source_detail | JSON encoding of source details, including Canadian DGUID, coordinate, scalars and decimals or BLS footnotes |

The cross-country browser bundle uses keys such as CA_all and US_sa and a shared sorted month axis. Every observation is numeric or JSON null. Missing values have no numerical contribution to percentage changes and are never set to zero. Twelve-month changes use their two calendar endpoints. Chart paths break where a value is missing.

The growth rates CSV holds percentage changes and percentage-point gaps. The rolling CSV uses seasonally adjusted monthly rates, a 36-calendar-month window and pairwise valid correlation. Index levels and CSV results retain input or calculated numerical precision. The standard-error CSV holds BLS summaries by calendar year, geography, category and comparison interval. Standard errors are in percentage points.

The weight views show national major groups as the agencies report them. The partial-basket comparison uses food, shelter, clothing/apparel and transportation. Published shares are price-updated within this mapped partition and rescaled to its observed share. Cross-application normalizes to 100% of that partial basket. Omitted groups and approximation effects remain in the reconciliation residual.

## Simulation outputs

The population CSV describes the synthetic baseline outlet, product, price, response, quality and expenditure fields. The summary files describe the target, biases, variance, MSE, RMSE, mean absolute error, coverage, interval width, hypothetical data costs, timing and Monte Carlo uncertainty by scenario and design.

## Geographical extension

- `processed/geography/*_cpi.csv` contains unadjusted index levels and their calendar-month and calendar-year comparison levels. Changes equal 100 × (index / comparison index − 1). Rows run January 2018–October 2026 and use 2017 index levels for initial comparisons where available. Empty rate fields and the associated flags mark missing endpoints. Individual release dates are present for the checked June 2022 and October 2026 releases.
- Canada uses StatCan 18-10-0004-01 and the shared 18-10-0007 cube underlying tables 18-10-0007-01 and 18-10-0007-02. The weight CSVs distinguish all-items shares of Canada from product shares within local baskets. They retain both the 2025 and April 2026 price bases. Iqaluit has a published all-items index, with no component series or weight for the selected categories in the extracts.
- The BLS extracts cover national CPI-U, all four Census regions, all nine divisions, and seven selected local series geographies. All 126 area–category IDs were verified against `cu.series`. Five API batches supply subnational observations. When the API reached its daily threshold, South and West BLS flat files supplied the remaining series. Gzip snapshots include both original-response and compressed-file hashes in the retrieval manifest. Workbook Tables 4 and 7 report within-region product shares and each area's relative importance, respectively. They use 2024 spending and December 2025 prices.
- BLS Appendix 1 supplies the names, constituent counties, 2010 represented populations and publication cycles for the 75 PSUs in the 2018 design. The area aggregation appendix supplies the 32 basic-area crosswalk. The Census county file provides 2020 boundaries and the 2023 Gazetteer gives county representative coordinates where applicable. PSU markers summarize those locations. Canadian markers outside the territorial capitals are illustrative.
- `geography.json` and `geography-data.js` carry the same rates, weights and geometry for online and offline use. The geography weight and spatial labs calculate their scenarios on demand. The simulated geography uses the seed shown in its control panel.

