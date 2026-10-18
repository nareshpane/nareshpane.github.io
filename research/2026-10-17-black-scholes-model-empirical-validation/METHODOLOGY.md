# Experimental protocol and interpretation

## Inputs and provenance

Actual source sample: 127 daily 34-column CSVs, July 1–December 30, 2022, from HistoricalData.net's public options sample. All 4,294,301 rows were read. Source is third-party historical market data, not an official Cboe quote distribution. Archive SHA-256 and manifest SHA-256 are in the aggregate bundle. Provider README, manifest, LICENSE.txt and validation implementation were inspected. Its `verify.py --complete` passed 127/127 files, including hashes, structure, documented invariants, calendar completeness and usable curves. This is not an independent exchange audit.

Independent references downloaded October 17, 2026:

- https://cdn.cboe.com/api/global/us_indices/daily_prices/SPX_History.csv
- https://fred.stlouisfed.org/graph/fredgraph.csv?id=DGS3MO&cosd=2022-07-01&coed=2022-12-30

SPX daily levels in the options files agree with Cboe on all 127 sample dates. SPXW differs on September 19 (−3.05), November 16 (+2.99), November 17 (−1.67), December 5 (−0.02), December 21 (−2.68), in index points, provider minus reference. All pricing spot inputs and historical returns use Cboe. The code checks the first root-level underlying observation each day; the provider's within-underlying consistency checks cover the repeated values. Substitution does not repair nonsynchronous option quotes. No quote_time is available in the 2022 sample.

## Unit of observation and selection

A row is a contract-day quote. A contract identifies root, expiration, option side and strike; its repeated daily rows are dependent observations. A date is a trading-date file, expiration is a calendar date, maturity is their calendar-day difference, strike is an exercise index level, and spot is a single harmonized SPX observation per date. Prices/errors are in index points (SPX multiplier $100); no per-contract-dollar conversion is used in the metrics.

Before model fitting, the code fixes: root SPXW, style E, settlement PM; 7–90 calendar days; K/S between 0.90 and 1.10 inclusive; finite positive bid and ask, ask strictly greater than bid (locked quotes excluded), both displayed sizes at least one, midpoint at least 5 points, spread/midpoint at most 10%. No volume or open-interest threshold is imposed because displayed quotes, not trades, are the proxy. Empty fields remain missing, never zero. Sequential counts and exclusions are displayed. Quality sensitivity caps of 2% and 5% are prespecified nested subsets of the 10% experiment, with baseline fits held fixed.

Duplicate contract-date keys and overlap of train/evaluation contract sets are asserted absent. Strike split is rounded K/5 modulo 2: even training, odd evaluation, invariant across dates. Training also requires 0.97 ≤ K/S ≤ 1.03. Both call and put at a given strike are assigned together. Odd strikes lacking valid calibration groups are excluded, and unused even strikes are not evaluation observations. This is contract-held-out same-date cross-sectional validation, NOT a temporal holdout or a forecasting test.

## Rates and carry

The manifest's historical Treasury annual-percent par curves have tenors 1/12, 0.25, 1, 2, 5, 10, 30 years. Missing tenors are dropped. Yields interpolate linearly; flat below/above the available ends. Annual decimal y becomes continuous r ≈ 2 log(1+y/2). This treats the par yield as a bond-equivalent rate proxy, not an exact discount factor or bootstrapped zero curve. With maturities below 90 days the first two nodes dominate. All 125 nonmissing historical DGS3MO nodes match the manifest 3-month node exactly; bond-market holidays October 10 and November 11 use October 7 and November 10 curves as recorded by the manifest. Same-day end-of-day rates are part of a contemporaneous pricing comparison, not information available at that morning's open.

For each date-expiration training group require at least 3 usable matched call/put strikes and at least 6 near-the-money training quotes. Derive each forward F_j = K_j + exp(rT)(C_j−P_j), then median F and q = r − log(F/S)/T. Reject q outside (−10%,15%). This pragmatic guard is broad; short-maturity q is sensitive to synchronization. Aggregate q quantiles and excluded group counts are exported. It is implied dividend/financing carry, not a pure independently observed dividend yield. Both A and B use it. Sensitivity: q ± 0.01 and a fixed 0.015 scenario (not an independently sourced ex-ante dividend estimate).

## Volatility and prices

Test A uses ddof=1 standard deviation of 20 daily log returns ending at the preceding trading date, annualized by sqrt(252). Require both 20 and 30-window values; earlier Cboe history supplies warm-up. Shift(1) prevents use of the quote date's closing return. Test A30 uses 30 returns. T is calendar days / 365, a separate model-clock convention. Physical mean return is not estimated or inserted into BSM.

Test B minimizes unweighted training midpoint squared error in one common volatility per date-expiration, bounded [0.03,1.5], SciPy bounded scalar optimization with xatol=1e-9. Training includes both sides; held-out quotes never supply their own σ or q. No vendor IVs enter either test. Only groups with evaluation rows are retained. BSM has constant rate/carry/volatility within each valuation, despite cross-group variation in calibrated parameters.

IV analysis separately inverts evaluation midpoint prices, with 60 iterations of vectorized bisection on [1e-6,5]. Reject missing target, T ≤ 0, prices at/below discounted intrinsic (1e-8 tolerance), prices at/above the theoretical upper bound, or outside the volatility bracket. It uses SciPy's normal CDF, independently implemented solver and the same r/q. Published medians group dates, sides and contracts; this is descriptive in-sample inversion, never an independent validation price. At least 30 valid solves per published cell.

## Metrics, uncertainty, publication

Residual e = theoretical minus midpoint. MAE = mean |e|; median absolute error = median |e|; RMSE = sqrt(mean e²); bias = mean e. Normalizations: mean(|e|/S)×10000 spot basis points and mean(|e|/(ask−bid)) spread units. No unstable midpoint percentage error. Within-spread uses inclusive comparisons against the recorded bid and ask. Contract-weighted metrics weight a contract-day equally, not each unique contract or date equally.

Moneyness bins: [0.90,0.95], (0.95,0.98], (0.98,1.02], (1.02,1.05], (1.05,1.10]. Maturity bins: 7–21,22–45,46–90 days. Date, side and joint strike/maturity breakdowns are exported. Histograms show −150 to +150 in 5-point bins and disclose omitted tails. Scatter/bar prices group observed midpoints into 25-point bins; only n≥30 bins are published. The original result bundle has no individual contract/date/price reconstruction table. The editorial revision separately publishes exactly 20 limited explanatory source rows with all 34 fields and clearly marked research quantities; see REVISION.md. No larger quote collection or raw chain is exported.

Cluster bootstrap: NumPy default_rng seed 20261017, 1,000 replicates, resample dates uniformly with replacement, carry all that date's observations, recalculate contract-weighted MAE from totals. Percentile 2.5/97.5 endpoints. The same RNG then generates 1,000 replicates for each side/maturity stratum in pandas sorted group order. These intervals allow intraday dependence but do not handle serial correlation or refit model parameters on a simulated data history. They condition on the actual fitted experiment.

Put-call parity audit uses held-out matched strikes and the training-derived carry. It is not the identity used to fit those same held-out contracts. Midpoint residuals under absent timestamps are not executable arbitrages. No option return forecast, realized trading profit or historical hedging performance is evaluated. Illustrative GBM and hedge animations are wholly synthetic and labeled separately.

## Limitations

Six months in one market cannot establish generality. Filtering favors larger and tighter quotes; quality sensitivity changes composition. Day-count is approximate, par yields are not zero rates, carry is calibrated, and raw quotes cannot be precisely synchronized. Source revisions can change reproducibility. Pooled smile cells can mix date regimes. More flexible calibration improves fit partly through information not available to historical-volatility-only forecasts. Discrete dividends, transaction costs, jumps and hedging implementation remain outside the model. The empirical pipeline does not identify the causal origin of a residual.

## Editorial supplement

The experiment and its original result bundles are unchanged. The optional `--supplement-output` branch runs after the held-out frame and strata are defined, exports the limited examples and new aggregates, and returns before the original result-writing path. See [REVISION.md](REVISION.md) for exact selection, field origins, source checks, aggregate definitions and license interpretation.


## Bitcoin empirical study — executed October 17, 2026

The three manually supplied [Hoang/Baur UWA CSVs](https://doi.org/10.26182/5eabbcb4fbf28) were inspected without re-downloading. Original BTC transaction premiums are separate from IV; all 761,067 source USD values reconcile to price × index_price. After 14,124 exact duplicate rows, one zero-amount record, maturity and moneyness restrictions, 250,453 transactions remain. H30/H60 historical pricing and a frozen earlier-period calibration share 178,236 later evaluation transactions across 407 dates.

Held-out MAE: H30 $78.7167, H60 $64.9389, calibrated $201.6667 per BTC notional. The fitted earlier σ transfers poorly; no favorable conclusion is forced. Chapter 07 includes nine additional empirical charts, twenty limited original examples with a data dictionary, complete filtering, seven-day-block uncertainty and careful SPXW comparisons. Figures distinguish binned observations, model values, inversion and illustrative mathematics.

All raw/processed/private files remain under ignored project-root bitcoin_data. No raw source or detailed analytical archive is public. The scoped academic CC BY basis, exact unversioned-license limitation, source hashes, minute-bar availability, inferred UTC history clock, prior-date Yahoo-identified T-bill proxy and all reproduction commands are in [BITCOIN.md](BITCOIN.md) and [BITCOIN_DATA_RIGHTS.md](BITCOIN_DATA_RIGHTS.md). FRED access failed; dataset rates are not relabelled DGS3MO. Source IV is never an independent historical-pricing input. The original SPXW methodology, results and Chapters 01–06 remain unchanged.
