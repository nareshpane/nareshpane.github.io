# Execution and QA report — October 17, 2026

## Actual empirical execution

The computation ran on genuine downloaded source files, not synthetic quotes. Archive download and extraction used `%TEMP%\bsm-research-20261017`, entirely outside this repository. curl could not use the configured proxy/Windows TLS path; Python urllib with a direct connection and normal TLS verification downloaded all three inputs successfully. No raw archive or bulk input is present in the project. The subsequent editorial revision adds exactly 20 limited explanatory source rows; see REVISION.md.

Provider `verify.py --complete`: exit 0, **127 / 127 files PASS**, complete calendar coverage and documented consistency. Input options archive SHA-256: `8517b069f4d41e44f950dc79a3cdab694359b0fae91664190011e39dd8e48b76`.

4,294,301 source rows → 1,635,844 European PM SPXW rows → 594,484 within maturity/moneyness → 560,650 quality-filtered rows → 241,365 reserved odd-strike observations → **238,836 evaluated quotes**, **17,543 unique contracts**, **127 dates**, **July 1–December 30, 2022**. 2,762 fitted date-expiration groups; 123,778 near-the-money training quote observations. Six date-expiration groups lacked enough pairs and 27 failed the carry guard. Full sequential counts are in the bundle and article.

Independent Cboe SPX comparison: all 127 dates agree for SPX; five SPXW discrepancies reproduce the provider's open issue. All spots harmonized to Cboe. FRED: all 125 available contemporaneous 3-month nodes agree. Neither exercise establishes synchronized option quotes.

| Metric, in index points unless stated | Test A | Test B |
|---|---:|---:|
| MAE | 11.6782 | 6.1593 |
| Median absolute error | 7.9762 | 4.7732 |
| RMSE | 16.1603 | 7.9157 |
| Bias (model − midpoint) | +1.7591 | −1.1251 |
| Within observed spread | 7.4968% | 5.9673% |
| Date-cluster 95% MAE interval | [10.5340, 12.8402] | [5.9301, 6.4012] |

1,000 bootstrap replicates, seed 20261017. Additional intervals stratify call/put × maturity. Matched held-out parity: 112,949 pairs; mean absolute residual 0.1200 points. Independent midpoint IV inversion rejects 249 of 238,836 targets. Pooled 22–45 day median IV drops from 26.56% in K/S 0.90–0.95 to 19.59% in 1.05–1.10. Pooled cross-date summaries do not establish a synchronized surface.

Market carry is used in both tests. Test B's better mean fit does not imply better within-spread coverage, forecasting performance or tradable profit. The article reports this explicitly.

## Automated checks

- Python unittest: 13 tests, including pricing reference/parity, deterministic and expiry limits, CDF behavior, invalid/missing inputs, analytical Greeks against finite differences, IV round trips, quality filters, metric calculations, aggregate partition reconciliation, interval/histogram consistency, publication thresholds, matching JSON/JS bundles, independent reconciliation, HTML IDs and every local link/asset.
- Node: syntax checks for both browser scripts and independent browser-pricing tests pass. Normal-CDF approximation is documented; test tolerance reflects its approximation error.
- DOM integration harness: **15 populated SVG charts**, both heatmaps, ten chapter links, mathematical and empirical controls, reset and browser checks exercised. Inputs changed actual calculated outputs. No NaN or Infinity chart coordinates. It uses LinkeDOM 0.18.12 standalone `worker.js`, downloaded to Windows TEMP, not a repository dependency. An initial assertion expected 18 charts; corrected to the actual 15 SVG charts (plus 2 HTML heatmaps). Flat-payoff and shared-gamma test cases were corrected to test changes where mathematics predicts a change; a DOM-only defaultValue polyfill accounts for LinkeDOM's form behavior.
- Pipeline assertions: duplicate contract-day keys absent; training and evaluation contracts disjoint.
- `git diff --check`, intended-file inventory, relative links and prohibited-folder/raw-data inspection performed before delivery.

## Rendering limitation

The browser connector returned no available browsers (`listBrowsers` = empty); an attempt to create the in-app browser reported it unavailable. Therefore **desktop/mobile visual rendering, formula-network loading and actual browser keyboard/animation playback were not visually verified**. The DOM harness is not a renderer and is not claimed as responsive visual QA. CSS includes responsive one-column layouts, scrollable formulas/tables/charts, focus outlines, large-enough chart canvases on narrow screens, and reduced-motion behavior. Animation starts paused and has explicit controls.

No test demonstrates authenticating every raw quote, live execution, historical trading profitability or causal attribution of pricing errors. No commit, staging or push was performed; unrelated pre-existing changes are preserved.

## Subsequent editorial revision

The report above describes the initial implementation. Current checks: 21 Python tests and 26 SVG chart DOM checks; all 20 new example rows verified against original external CSVs. Original result bundles pass preservation hashes. See [REVISION.md](REVISION.md) for the complete revision audit and unchanged browser-rendering limitation.


## Bitcoin access review — October 17, 2026

Completed cleanup was verified without repeating deletion. Chapters 01–06 and SPXW data pass their preservation hashes. The existing 21 Python tests, browser-math checks and 26-chart DOM suite passed before this review. Chapter 07 now names the Bitcoin investigation and documents the unresolved processing/publication gate, with no empirical observations or chart substitutes. See [BITCOIN_DATA_RIGHTS.md](BITCOIN_DATA_RIGHTS.md) for file-specific academic license evidence, the original paper’s availability statement, current Deribit terms and the historically important 2021 clause addition. No market file was requested or processed. A clarification draft is prepared, not sent. Raw/private paths named inside this repository are Git-ignored as a precaution, not treated as external storage or a license. No SPXW methodology or outputs changed.


## Bitcoin mathematical framework — October 17, 2026

See [BITCOIN.md](BITCOIN.md) for the six source modules, normalized interface (not a claimed UWA schema), independent daily and optional five-minute volatility, explicit UTC/rate availability, currency/settlement benchmark, chronological calibration and paired/cluster statistics. Chapter 07 contains eight subsections and three live mathematical charts; all teaching values are illustrative. Source and publication rights remain unresolved as documented in [BITCOIN_DATA_RIGHTS.md](BITCOIN_DATA_RIGHTS.md). No actual attachment was downloaded or processed, no dataset coverage or observed-premium field has been verified, and no market-derived result or observation table is published. The complete workflow was exercised only with explicitly labelled synthetic CSV fixtures. Original mathematical/SPXW chapters and result bundles are preserved.

### Final Bitcoin software checks

62 Python tests passed (21 original plus 41 Bitcoin checks). Independent browser-pricing checks and the DOM harness passed with 29 populated SVG charts. The harness exercises all Bitcoin inputs, unit selection, dedicated/global resets, a Python-referenced illustrative call, chapter deep links and preserved SPXW controls. All JavaScript syntax checks and HTML IDs/local assets passed. Chapters 01?06 exactly match their pre-extension hashes; original SPXW result-bundle hashes pass. The chapter fragment matches its embedded section. No raw/private market directory or market payload was created. Browser inventory was empty; desktop/mobile rendering, live MathJax/network loading and actual keyboard rendering were not visually verified. DOM simulation is not a browser layout test.

Rights and access gates remain unresolved: no actual CSV/RDS request, download, schema inventory or empirical market calculation was performed. Download behavior was exercised with mock HTTP responses; workflow results came only from explicitly synthetic fixtures outside the repository. Initial DOM reference extraction was corrected to read the price element rather than concatenated price/BTC text; no pricing calculation changed. No staging, commit or push.

## October 17, 2026 ? Official acquisition and project-local storage

The university small-file URL returned HTTP 403 through the browsing tool. A click of the official attachment provided no source file either. No retry loop, access-control bypass, larger-file request or empirical calculation followed. The user was given precise manual-save instructions for project-root bitcoin_data/raw. Exact source checksums, schemas, coverage and premium fields remain unknown. The local private acquisition receipt records the failed request; no raw file exists.

The rights review now explicitly relies on the academic deposit?s CC BY label for covered reuse, without presuming later platform terms retroactively invalidate it. Exact license version/file notices remain unverified. Added the requested compact attribution panel and bottom references, updated Chapter 08 and methods, and migrated defaults/storage validation to ignored bitcoin_data/raw, processed and private. Synthetic temporary fixtures remain explicitly segregated. Compressed processed datasets and inventories are tested; no real-data output or publication claim was made.

66 Python tests, JavaScript syntax and pricing checks, and the 29-chart DOM integration suite passed. New checks cover local storage, role/path boundaries, manual HTML rejection, UTF-8/BOM, attribution and processed-output reconciliation. Chapters 01?06 match their exact pre-Bitcoin hashes; original SPXW result preservation tests passed. Embedded fragment consistency, internal links and duplicate IDs pass. Browser inventory was empty, so desktop/mobile rendering and live MathJax loading were not visually verified. Gold cleanup was not repeated and no gold component was recreated. Unrelated working-tree changes remain untouched; no staging, commit or push.

## October 17, 2026 — Actual Bitcoin empirical validation

Three user-supplied official UWA CSVs were inspected locally without re-downloading: 761,067 options rows, 6,724 catalog instruments and 2,697,313 price-history minutes. SHA-256/headers/units/expiry/source relationships are recorded in BITCOIN.md and processed inventories. All source USD premiums equal supplied BTC price × recorded index; IV is separate and never an independent historical-pricing input. Source fields include no bid/ask or transaction ID. History is one-minute, not five-minute, and the arithmetic mean of available exchange columns. Raw files remain unchanged and Git-ignored.

Validated filtering retains 250,453 transactions after 14,124 exact duplicate rows, one zero-amount record, 7–90 day maturity and K/S 0.80–1.20. Earlier calibration trains on 72,217 trades; all three configurations evaluate the identical 178,236 later trades across 407 dates and 1,474 contracts. Cutoff January 1, 2019 was selected before errors. Frozen σ=103.767725%. MAE H30/H60/B = $78.7167/$64.9389/$201.6667 per BTC notional. Unfavorable calibration transfer is retained, not tuned away. Source IV missing/invalid inversion targets remain in primary errors.

Clock/availability safeguards: epoch verifies UTC trade text; history UTC is inferred via underlying alignment and disclosed. Minute proxy prices become available only after their full minute; daily closes become usable next midnight. Co-recorded index remains valuation spot. Rates are prior-date academic irate (^IRX per the paper), with an approximate continuous transformation; the independent FRED request failed WinError10061 and was not relabelled successful. The last partial source day is omitted from daily public plots. Seven-calendar-day moving-block uncertainty uses 1,000 replicates and seed 20261017; date-cluster intervals are supplementary and fixed fitted-sigma limitations are explicit.

The initial private run retained a zero-amount source row and remains an audit baseline only. Final published outputs use uwa-validated-20261017. During diagnostics QA, shifted-day estimates were explicitly re-aligned to globally sorted evaluation timestamps before pricing. Conservative completed-minute metadata/proxy joins were corrected without changing the primary recorded spot, historical volatilities, rate, maturity, prices or common evaluation set. CSV-reloaded categorical bins are reconstructed for private plotting; no source value was replaced.

Chapter 07 retains its eight subsections and original mathematical demonstrations, and adds nine actual empirical SVG charts, counted error cells, comparative tables and twenty verified, attributed examples. Academic CC BY labels are the scoped basis for covered outputs; version remains unspecified, and no later exchange-term retroactivity or endorsement is claimed. No raw archive or detailed processed dataset is embedded. Chapter 08 and methods explain reproducibility and ignored local data paths.

78 Python tests passed, including actual file hashes, counts/filters, original-source example matches, independent erf pricing, prior-window NumPy volatility, minute/rate timing, chronological separation, metric/group/paired reconciliation, vendor-IV missingness, bootstrap reproducibility and SPXW preservation. Browser-pricing, JavaScript syntax and the expanded 38-chart DOM suite passed, including selectors, sorting, reset, deep links and finite coordinates. Browser inventory was empty; desktop/mobile visual rendering and live MathJax loading were not verified. Chapters 01–06 match exact preservation hashes and original SPXW bundles remain byte-identical. Only this project changed; no staging, commit or push.
