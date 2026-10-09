# Data provenance and honest outcome coverage

## Historical trade: 252 rows

Historical outcome statuses: 207 source-supported positive observations and 45 absent-flow grid zeros marked `data_status=imputed`, `trade_status=baci_grid_zero`. The numeric observed-outcome column retains these disclosed grid zeros for matching the original model convention. It never contains a 2025 scenario value.

Read-only supplied CEPII BACI HS2022 V202601 annual files:

- `BACI_HS22_Y2022_V202601.csv`
- `BACI_HS22_Y2023_V202601.csv`
- `BACI_HS22_Y2024_V202601.csv`
- `country_codes_V202601.csv`, `product_codes_HS22_V202601.csv`, `Readme.txt`

[CEPII BACI](https://www.cepii.fr/CEPII/en/bdd_modele/bdd_modele_item.asp?id=37). Local readme: version 202601, release 22 January 2026; v in thousand current USD. Sum every active HS6 child by exporter/destination/HS4/year after multiplying v by 1,000. No CAD values enter this teaching dataset. BACI is reconciled historical trade estimated from reporter data, not a forecast or a raw customs observation. `observed_baci_reconciled` marks source-supported positive trade. The original project's balanced-grid convention assigns zero to absent valid cells; `baci_grid_zero` explicitly marks this assumption. These zeros do not prove customs-reported zero shipments or distinguish reporting thresholds from physical absence.

Network: Canada, United States, Mexico, Germany, China, Japan, United Kingdom. Current-first ISO3 country-code records are used, preserving modern Germany rather than obsolete records. Unlike the original network, Germany replaces South Korea. Recompute all aggregates for the new network; do not reuse the original network's feature cache.

HS4 1001: **Wheat and meslin.**

HS4 8703: **Motor cars and other motor vehicles principally designed for the transport of persons (other than those of heading 87.02), including station wagons and racing cars.**

These headings exist in the BACI HS2022 metadata. Exact official heading labels use the repository's CBSA T2026-2 hierarchy, by code. [CBSA source](https://www.cbsa-asfc.gc.ca/trade-commerce/tariff-tarif/2026/html/tblmod-2-eng.html). This is a documented later wording vintage, not a change to trade classification or an introduced 2026 economic predictor. All trade remains within a single HS2022 release; no HS6 cross-revision concatenation or undocumented concordance is performed. The extract aggregates to headings, not Canadian HS8/HS10 codes.

## Economic and geographic covariates

[World Development Indicators](https://databank.worldbank.org/source/world-development-indicators), supplied `API_Download_DS2_EN*.csv`, updated 13 July 2026. Exact-year columns 2022, 2023, 2024:

- NY.GDP.MKTP.CD: GDP, current USD.
- SP.POP.TOTL: population, persons.
- NV.IND.MANF.ZS: manufacturing share, % GDP.
- IT.NET.USER.ZS: internet users, % population.

Missing manufacturing is retained as a blank in the dataset and flagged by field in `covariate_status`. Each fitted pipeline estimates its own training median; no future-year value substitutes for it. Classifier and positive-only regressor fit separate preprocessors. See saved medians, means and SDs in worked_examples.json and explicit training IDs in fit_records.json.

[CEPII GeoDist](https://www.cepii.fr/CEPII/en/bdd_modele/presentation.asp?id=6), supplied `dist_cepii.xls`: distw in km, contig and comlang_off. Geographic features are national, time invariant; distance is population weighted, not road or shipping route length. No invented locations or common-language coding.

External exporter-HS4 supply, importer-HS4 demand and world-HS4 demand sum all selected-heading BACI flows outside the internal seven-country network. Exclude every record with both endpoints in the network, including other headings' selected cells. The full source is scanned, so these sums are not constructed from the 336-row teaching table. Hashes and full annual scan counts are in data/source_manifest.json.

## Unavailable 2025 outcomes: 84 scenario rows

The supplied BACI directory has no 2025 annual file. The supplied StatCan 2025 data describe Canadian provincial domestic exports in CAD; they do not provide a symmetric seven-exporter national bilateral BACI panel. They are not substituted. `observed_exports_usd` is blank for every 2025 row. There is no observed 2025 holdout metric or observed 2025 case study.

For an explicitly artificial demonstration only, `illustrative_exports_usd` is generated from each matching 2024 value:

`scenario = trade_2024 * exp(sector_shock + epsilon)`

- wheat sector_shock = -0.08; cars = +0.06.
- epsilon drawn independently from N(0, 0.10²), NumPy default_rng(338).
- deterministic row order: year, alphabetic exporter, alphabetic destination, HS4.
- zero 2024 values stay zero; no artificial entry events are added.

This procedure is not fitted to any research model and not intended as a plausible trade forecast. It is recorded on every scenario row. All other 2025 inputs are carried 2024 forecast-origin covariates and labelled macro_year=2024. They are not represented as observed 2025 GDP/population/product flows. Scenario errors are stored separately from missing observed errors. The procedure inherently favors persistence and must not be used to conclude universal model superiority.

## Reproduction and vintages

Raw sources are read-only at TRADE_RAW_ROOT (default local archive location is in dataset_build.py). `dataset_build.py --raw` scans sources, validates classifications, re-extracts the 252 historical records and writes file hashes, then builds the 336-row table. The portable default uses the included data/source_observed_subset.csv after its SHA-256 is checked against the manifest. Thus the numerical analysis reproduces without the large archive; independently verifying raw-source fidelity requires the supplied archive. Source hashes include full-file SHA-256, not merely filenames or timestamps. No original model code is imported or run.

The revised BACI and WDI releases postdate forecast origins. All temporal targets/predictor years are ordered, but this is a retrospective experiment, not a reconstruction of data release vintages. Actual 2024 validation uses 2023 inputs. 2025 outcomes never fit or select anything. The frozen forecast CSV is hashed; metrics distinguish final training fits, internal chronological validation, unavailable observed holdout, and illustrative scenario comparison.

No paid API, website backend or dataset download is needed to reproduce from the compact extract. Raw data remain outside the public project. Extraction of this small teaching subset does not imply redistribution of the entire BACI or WDI databases.
