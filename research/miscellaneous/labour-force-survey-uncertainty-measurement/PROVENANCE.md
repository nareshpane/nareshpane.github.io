# Data provenance

Created October 13, 2026. This is an unpublished independent educational draft.

## Official sources

The `sources/manifest.json` records source URLs, UTC retrieval timestamps, file sizes and SHA-256 hashes. Official 2025 Guide and 2017 Methodology PDFs are stored locally, with HTML chapters 2, 5, 6, 7 and 8. The guide is authoritative for the 2025 redesign. Detailed 2017 algorithms describe the 2015 design; their complete continuity is not asserted. Guide sections 2, 4–7 and Appendix B were consulted. The official SDDS 3701 survey page and October 9, 2026 Daily release are archived.

`data/official-rows.csv` is a direct subset of the downloaded full Table 14-10-0287-01 CSV: reference months August and September 2026, Canada and ten provinces, Total - Gender, 15 years and over, Seasonally adjusted. Original columns, vectors, units, decimals and status codes are preserved. `data/september-2026.json` reorganizes these rows and subtracts August from September. Counts remain in thousands; rate SEs are in percentage points. `js/official-data.js` embeds the JSON for direct local-file use. It is generated, not a separate data source.

Published SEs of levels and monthly changes were located for employment, unemployment, unemployment rate, participation rate and employment rate in all eleven geographies. These are official estimated and rounded uncertainty inputs, not exact population variances. CVs, confidence intervals and Z statistics on the page are author calculations. No lookup CV was needed for these selected indicators. Full replicate weights, cross-province covariance, production degrees of freedom, September responding-household counts and provincial response rates are not supplied by this extract. No substitutes are fabricated.

All interval endpoints use a two-sided large-sample normal approximation, with matching published level/change SE. Change tests use SE(change) directly; no assumed correlation enters the official results. Rate differences subtract rounded published rates. Borderline tests are sensitive to rounding and are individually exploratory, without multiplicity adjustment. Statistical significance cannot establish causation or absence of nonsampling bias.

Guide Table 4.1 reports national target 63,596; the ten displayed provincial targets sum to 63,595. Both published numbers are retained, with the one-household discrepancy disclosed rather than reconciled arbitrarily. Allocations are design targets, not September response counts.

## Synthetic models

Sampling simulator: fixed N=12,000 labour-force members in two equal strata. Integer unemployed count round(Np) is split 25%/75% across strata. Unemployed cases are contiguous within each stratum, inducing high clustering. SRS samples persons without replacement; stratified SRS samples n/2 persons per stratum; cluster sampling selects n/20 of 600 whole clusters of 20. Equal weights estimate the population mean. This model deliberately separates designs and omits classification, calibration and random labour-force denominators.

Empirical moments divide by repetitions, so displayed variance and bias satisfy the empirical MSE decomposition exactly. Monte Carlo moments are not unbiased estimates of a production LFS variance.

Coverage experiment: 100 exactly normal draws with truth zero and known SE=1. Bootstrap: fixed unemployed cluster totals [[2,4,3,7],[6,10,8,12]], each cluster with L=50; sample three clusters with replacement within each stratum, rescale multiplicities by 4/3; replicate-mean centering and 1/B variance. No calibration or LFS-specific exceptions. Browser uses Mulberry32; Python uses NumPy's seeded generator, so draws differ.

CV curves use Bernoulli effective size inferred from selected p and SE and keep that size fixed across the curve. p means U/P here; it is not U/L. Seasonal example uses known additive components, not estimated X-13 components. User-entered confidence/design/covariance values are always synthetic.

## Reproduce

From repository root:

```text
python research/labour-force-survey-uncertainty-measurement/scripts/acquire_sources.py
python research/labour-force-survey-uncertainty-measurement/scripts/prepare_data.py
python research/labour-force-survey-uncertainty-measurement/scripts/render_static_tables.py
python research/labour-force-survey-uncertainty-measurement/scripts/refine_page.py
python research/labour-force-survey-uncertainty-measurement/scripts/validate_analysis.py
node research/labour-force-survey-uncertainty-measurement/scripts/verify_statistics.js
node research/labour-force-survey-uncertainty-measurement/scripts/verify_browser.js
```

Downloading again overwrites snapshots with a new retrieval vintage; archive the current manifest and extracts first if preserving this case study. The page never refreshes its data automatically. Official material is governed by the Statistics Canada Open Licence. MathJax requires network access; all interaction/data assets are local and no server backend is required.
