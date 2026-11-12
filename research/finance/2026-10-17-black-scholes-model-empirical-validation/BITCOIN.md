# Bitcoin options: executed empirical study

**Executed October 17, 2026.** Three manually supplied official academic CSVs were inspected and priced. No options data were re-downloaded. All principal comparisons use 178,236 identical held-out transactions. The page retains separate, labelled mathematical illustrations.

## Citation and covered publication

Hoang, L., & Baur, D. (2020). *Deribit’s Options Data*. The University of Western Australia. [Dataset DOI](https://doi.org/10.26182/5eabbcb4fbf28); [official files](https://research-repository.uwa.edu.au/en/datasets/deribits-options-data/). Original market source: Deribit. Related paper: [Hoang and Baur (2020), *Forecasting Bitcoin Volatility: Evidence from the Options Market*](https://doi.org/10.1002/fut.22144).

The official attachments are labelled CC BY; the accessible record does not identify a version. No version is invented. The three actual CSVs contain ordinary market fields and no extra restrictive notice or account/person identifier was found. Attributed aggregate results, independently derived figures and exactly twenty limited examples rely on the grant for material it covers. The 2020 deposit is distinguished from later exchange-platform terms; no retroactive prohibition is presumed. No full raw archive is published and no endorsement is implied. See [the specific rights record](BITCOIN_DATA_RIGHTS.md).

## Verified files and actual coverage

| Original file | Exact bytes | Rows | Columns |
|---|---:|---:|---:|
| instrument_name.csv | 493,948 | 6,724 | 5 |
| Deribit_w_IV.csv | 149,371,644 | 761,067 | 16 |
| Price_5m.csv | 160,944,532 | 2,697,313 | 7 |

These SHA-256 values fingerprint the received files; they are not independent provider-published checksums:

- `instrument_name.csv`: `a39733081781987aed135b619681064d8c62cc918e97679705ebe65fcc95ffb2`
- `Deribit_w_IV.csv`: `47af56f73630b54a025b554088c03771404af15bbe9091dddee7d1c644c903b2`
- `Price_5m.csv`: `694cba508d41f773af0c07d5090543436cf7cc09b45714345a097ff6a6a3572f`

Options: November 29, 2016 08:23:13.249 through February 11, 2020 05:12:08.854 UTC; 761,067 rows, 1,147 trading dates, 5,189 contracts, 432,102 calls and 328,965 puts. Catalog: 6,724 unique names (3,362 per side). The history contains 2,697,313 contiguous **one-minute** observations, despite its Price_5m filename, January 1, 2015 08:02 through February 17, 2020 11:14 under the declared UTC interpretation. The final intraday partial day is omitted from published daily-close/volatility figures.

## Original fields and independent observed premium

| Original field | Verified interpretation / analytical use |
|---|---|
| timestamp | Unix milliseconds; matches time_trade within 0–0.999 seconds. Primary UTC clock. |
| price | Recorded trade premium in BTC per unit BTC notional; primary observed target. |
| instrument_name | BTC-expiry-strike-C/P contract identifier. |
| index_price | BTC/USD index co-recorded in the trade record; valuation spot and premium conversion basis. Separate index update time is absent. |
| direction | Reported buy/sell direction; not a unique transaction or trader identifier. |
| amount | Trade quantity, not displayed bid/ask size. One zero-amount row excluded. |
| time_trade | Source text trade clock, subsecond-truncated relative to epoch. |
| strike; type | Reconciled against identifier on every row. |
| time_create; time_expire | Reported instrument creation and expiry, reconciled to catalog for sample expirations. All traded expirations are 08:00 UTC. Some unused 2016 catalog entries show 15:00 and are not overwritten. |
| date | Source trade date; checked against UTC epoch date. |
| irate | One supplied annual fractional T-bill proxy per source date; paper identifies Yahoo ^IRX. |
| price_USD | Equals price × index_price on all 761,067 rows to within $1e-7. This is a conversion of recorded BTC premiums, not an independently quoted USD market. |
| iv_Tbill | Vendor/researcher IV in annual percentage units; separate diagnostic only. Missing on 9,935 source rows, including 260 held-out rows retained in pricing. |
| tt | Source year fraction; independently checked against exact expiry minus epoch seconds / 31,536,000. |

The original paper identifies Deribit options trades as the source, consistent with independent price, direction and amount fields. **No observed price is reconstructed from IV.** There are no bid/ask columns or transaction IDs. Exact full-row duplicates are removed; distinct economic trades beyond identical rows cannot be conclusively deduplicated without IDs.

History fields: time_m, price_bitstamp, price_bitfinex, price_coinbase, price_itbit, price_kraken, price_avg. A dot is missing, never zero. The supplied price_avg equals the arithmetic mean of available named exchange prices on every row (max absolute difference < 5e-12). Missing constituent coverage is recorded in the public inventory. Composition changes and this average/index basis remain limitations.

## Inputs, timestamps and contract benchmark

European inverse BTC-linked options, USD strikes and BTC premiums/settlement are interpreted using the academic fields and [historical contract explanation](https://insights.deribit.com/education/what-is-an-options-contract/), [2020 expiry note](https://insights.deribit.com/exchange-updates/launch-of-btc-daily-options-on-deribit/) and [BTC payoff explanation](https://insights.deribit.com/education/cryptocurrency-call-options/). Modern settlement changes are not imposed on old trades. Historical fees, precise settlement index averaging and funding curves are not reconstructed. Calculations are per one BTC notional, not per gross transaction.

For an inverse call, terminal BTC payoff is max(S_T−K,0)/S_T. Current BTC premium × current recorded index converts to USD-equivalent premium. USD-numeraire spot BSM values the corresponding dollar intrinsic claim; converting its initial value to BTC divides by current spot, not terminal spot. This benchmark assumes frictionless spot conversion, constant volatility/rate and explicit q=0; q is not called a Bitcoin dividend. BTC lending, forward basis, collateral and fees are omitted. No independent historical forward is present, so Black–76 is not substituted.

d1 = [log(S/K) + (r−q+σ²/2)T]/(σ√T); d2=d1−σ√T. C=Se^(−qT)N(d1)−Ke^(−rT)N(d2); P=Ke^(−rT)N(−d2)−Se^(−qT)N(−d1). Existing tested pricing.py and bitcoin_pricing.py are reused.

History clock: naive time_m is interpreted as UTC. A deterministic month-stratified check against epoch-anchored trade indices supports UTC over tested −8/+8/+12 hour alternatives; this is inference, not original timezone metadata. The check is reproduced by bitcoin_uwa.py. For conservative matching, a minute labelled t becomes available at t+60 seconds. Join backward with a five-minute maximum. All eligible joins succeed; observed availability lag has median 30.1005 seconds and maximum 59.999 seconds. Core valuation spot remains the trade record’s index_price; the matched average is explicitly retained as a proxy.

H30/H60 use 30/60 consecutive completed UTC daily log returns, sample ddof=1, ×√365: 31/61 prior closes are required. The last supplied price within day d (23:59 minute) becomes usable at next UTC midnight. No forward filling, trade-day future close or current option IV enters an estimate. Median held-out H30=59.1366%, H60=63.4100%. Alternative completed-day boundaries at ±8 hours are sensitivity checks. Optional five-minute realized-volatility software is not the primary estimator and would require explicitly selecting an exact five-minute grid from the actual minute data.

Historical rates: source irate, annual fraction, grouped by source date and usable only at next UTC midnight. This is a conservative end-of-day availability assumption, not a release-vintage reconstruction. A direct historical FRED request failed with WinError 10061; no FRED file was fabricated or substituted. Rate inputs are explicitly not DGS3MO. Use r=2 log(1+y/2) as a semiannual-equivalent approximation, not an exact ^IRX discount-yield or zero-curve conversion. Evaluation continuous rates range 1.467602%–2.393619%; zero and ±50 bp sensitivities are computed. This proxy is not a BTC funding curve.

## Sample construction

| Stage | Retained | Excluded at stage |
|---|---:|---:|
| raw | 761,067 | 0 |
| valid_timestamp | 761,067 | 0 |
| unique_exact_source_rows | 746,943 | 14,124 |
| trade_id_unavailable_not_deduplicated | 746,943 | 0 |
| valid_symbol_and_spec | 746,943 | 0 |
| positive_executed_quantity | 746,942 | 1 |
| valid_prior_underlying | 746,942 | 0 |
| positive_observed_premium | 746,942 | 0 |
| positive_strike | 746,942 | 0 |
| maturity | 367,724 | 379,218 |
| moneyness | 250,453 | 117,271 |
| independent_volatility | 250,453 | 0 |
| known_rate | 250,453 | 0 |

Retained: 250,453 transactions, 2,635 contracts, 1,086 dates, December 31, 2016–February 11, 2020. Restrictions are actual 7–90 calendar days and 0.80≤K/S≤1.20. No pricing-residual outlier, IV-availability or invented liquidity/spread filter is used. The first private run retained the lone zero-amount source row; it is preserved only as a baseline audit. The **uwa-validated-20261017** run excludes it before fitting. All published results use that validated run.

## Chronological experiment and common sample

The cutoff January 1, 2019 UTC was chosen as a whole-year boundary before inspecting pricing errors. Training: 72,217 earlier eligible transactions, 1,186 contracts, 679 dates. Fit one constant annual σ=103.767725% using unweighted USD-premium MSE, bounded 1%–400%, then freeze it. This calibration is intentionally parsimonious and can transfer poorly across regimes. No volatility model or cutoff was changed to improve evaluation fit.

Evaluation: January 1, 2019 00:45:37.900–February 11, 2020 05:12:08.854 UTC; 178,236 transactions, 407 dates, 1,474 contracts, 94,286 calls and 83,950 puts. Training/evaluation transactions and dates do not overlap. Twenty-five contract identifiers span the chronological boundary; earlier trades of those contracts are allowed. This is an out-of-time observation holdout, not a completely unseen-contract experiment. H30, H60 and calibrated B use exactly the same rows.

| Model | n | MAE $ | Median absolute $ | RMSE $ | Bias $ | Mean absolute / spot (bp) |
|---|---:|---:|---:|---:|---:|---:|
| H30 | 178,236 | 78.7167 | 48.1633 | 120.1380 | -39.0390 | 99.2776 |
| H60 | 178,236 | 64.9389 | 38.1316 | 101.9822 | -30.8235 | 83.1877 |
| B | 178,236 | 201.6667 | 181.4624 | 240.2530 | 190.4265 | 284.3520 |

All dollar errors are per BTC notional; error=model−observed. BTC MAE is mean(|error|/spot). Additional metrics: median signed error, $5 and 1-spot-bp tolerance fractions, and mean premium-relative absolute error restricted to observed premiums ≥$5 (178,025 evaluation observations). No within-spread statistic is reported. Grouped tables cover side, UTC date, moneyness and maturity, with explicit counts.

| Paired comparison | Mean Δ $ | Median Δ $ | Improved transactions | Aggregate MAE reduction |
|---|---:|---:|---:|---:|
| H30 minus B absolute error | -122.9500 | -125.5331 | 15.87% | -156.19% |
| H60 minus B absolute error | -136.7278 | -142.0313 | 14.77% | -210.55% |

Positive Δ favors calibrated B. Here calibration worsens aggregate MAE; percentages are changes in aggregate means, not averages of individual percentage gains. H60 improves MAE over H30 by 17.5031% on this evaluation sample. This is no universal estimator ranking.

Supplementary all-eligible historical tests (not the main paired sample): H30 n=250,453, MAE $90.4632, RMSE $152.0213; H60 MAE $81.7316, RMSE $153.3136. All side/maturity/moneyness, time and sensitivity metrics are in the publication-safe bundle and private report.

## Inversion and uncertainty

Independent IV inversion succeeds on 177,977 held-out premiums; 259 fail bounds/bracket checks and remain in primary error metrics. Median IV=66.0296%. Across 177,975 joint valid vendor/computed values, mean absolute gap is about 0.00425 percentage points. Repricing after inversion is not independent model validation and IV is never Experiment A input.

Primary descriptive uncertainty: circular moving blocks of seven calendar days, 1,000 replicates, seed 20261017, preserving all trades within each selected day and trade-weighted errors. The fixed training sigma is not refitted. Whole-date IID clusters are a supplementary diagnostic. Seven-day blocks preserve some serial dependence, not arbitrary long-range regimes. No independent-trade p-values or universal forecasting inference is claimed.

| Model | Seven-day-block 95% MAE interval ($) |
|---|---|
| H30 | [64.6092, 94.5198] |
| H60 | [52.4317, 77.6658] |
| B | [177.6338, 228.5081] |

## Reproduction and private/public outputs

Run from the project root. No source download, installation or SPXW rerun is required. All persistent datasets remain under ignored project-root bitcoin_data/{raw,processed,private}. The root /bitcoin_data/ ignore rule is verified; ignore is not web-server access control, and this tree must not be deployed.

```powershell
$env:PYTHONDONTWRITEBYTECODE = '1'
git check-ignore -v bitcoin_data/raw/Deribit_w_IV.csv
# Choose a new run id to preserve existing analytical results:
python python/bitcoin_uwa.py --run-id replication-01
# The adapter creates inventories, matched inputs, prices and diagnostic uncertainty.
# Public export is explicit and requires the scoped publication review record:
python python/bitcoin_publish.py --run-id replication-01
# Recompute only diagnostics for an existing run if needed:
python python/bitcoin_diagnostics.py --run-id replication-01
python -m unittest discover -s tests
node tests/test_browser_math.cjs
node tests/test_dom.mjs `
  "$env:TEMP\bsm-research-20261017\linkedom.mjs"
```

The original three files remain unmodified under bitcoin_data/raw. bitcoin_uwa.py verifies immutable local ingest hashes and actual field relationships, emits complete inventories/clock/average audits and prepares the explicit uwa-manifest.json. Source row numbers preserve provenance. The generic bitcoin_reproduce.py handles chunked normalization, exact-row hashes, executed quantity, matching, pricing, filters and statistics; source/runtime flags do not create rights.

Processed run files: analytical-observations.csv.gz, historical-volatility.csv.gz and source-inventory.json. The root processed folder also contains source inventories, uwa_prior_rates.csv, uwa-manifest.json and audits. Private run files: evaluation-private.csv, bitcoin-results.json, private-report.html and six SVG figures. The source and public application use no external permanent data copies. Short-lived deterministic synthetic fixtures remain segregated in OS TEMP.

bitcoin_publish.py exports only attributed aggregates, daily low-frequency outputs and twenty explicitly identified examples to data/bitcoin-results.{json,js}; it does not copy the detailed evaluation table. The twenty examples are ten calls and ten puts at chronological evaluation quantiles. Display headings map to actual source columns in the chapter dictionary; model values and conversions are labelled derived. Sorting and side filtering affect only these examples.

Nine empirical SVG charts render from all selected evaluation aggregates: historical price, H30/H60, equality-line binned observed/model means, full residual histograms, discrete IV medians, moneyness, maturity, model error and call/put error. Figures expose model/side/maturity controls, counts, keyboard focus and tooltips. The error heatmap uses counted cells, no interpolation. For privacy/storage efficiency, scatter dots are conditional means within observed-premium quantile bins—not fabricated transactions. HTML is static GitHub Pages; no framework or backend was added.

## Interpretation and limitations

H60 fits this evaluation period better than H30 in both MAE/RMSE; frozen early-period sigma fits much worse. Larger dollar maturity errors do not establish a causal effect. Measured deviations are not allocated causally to jumps, noise, volatility changes or financing. Execution premiums differ from midpoints; no bid/ask or individual index-update clocks exist. Source timezone is inferred, exchange-average composition changes, daily/rate boundaries are assumptions, exact funding/fees/settlement replication is omitted, and trade/date dependence limits generalization. The Bitcoin and 2022 SPXW studies have different periods, price proxies and calibration designs. Their normalized errors are descriptive, not a controlled cross-asset ranking.

Original SPXW results and Chapters 01–06 are preserved. Numerical test success, fitted IV consistency and model price fit establish different things; none establishes trading profitability.

The reported counts describe the supplied academic files, not a verified census of every Deribit transaction. Earlier selection or preprocessing by the depositors cannot be reconstructed solely from these CSVs.

Software used: python 3.13.1, numpy 2.4.0, pandas 2.3.3, scipy 1.18.1. Existing NumPy/pandas/SciPy dependencies were reused; no package installation or web build system was added. Node checks and the existing optional external LinkeDOM harness are software verification, not browser rendering.
