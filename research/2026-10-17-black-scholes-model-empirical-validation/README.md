# Pricing uncertainty — Black–Scholes–Merton

Research draft created October 17, 2026. Main page is one directory above this folder:
`../2026-10-17-black-scholes-model-empirical-validation.html`.

The page uses plain HTML, CSS, JavaScript and SVG; no build or backend. MathJax 3.2.2 loads from jsDelivr. The empirical bundle is a script rather than a fetched resource, so the page also works through `file://` (formula typesetting needs internet).

## Files

- `css/article.css`: responsive peach/ivory visual system.
- `js/pricing.js`: browser BSM, Greeks, seeded illustrative GBM.
- `js/article.js`: navigation, accessible controls and SVG figures.
- `data/results.json` and `results.js`: identical publication-safe statistical outputs, the latter wrapped for direct local opening.
- `python/pricing.py`: independent vectorized pricing and bisection IV solver.
- `python/analyze.py`: actual-data pipeline, fixed contract split, calibration, clustered uncertainty and aggregates.
- `python/download_inputs.py`: downloads and extracts only to an external path.
- `tests/`: numerical, aggregate and browser-DOM checks.
- `METHODOLOGY.md`, `QA.md`: decisions, limitations and actual checks.

## Open locally (PowerShell, project root)

```powershell
Start-Process '..\2026-10-17-black-scholes-model-empirical-validation.html'
```

No raw inputs are required to view the computed article. The revision includes exactly 20 limited explanatory source rows; see REVISION.md for licensing scope and the supplement-only reproduction command. For an optional local HTTP server:

```powershell
python -m http.server 8001 --bind 127.0.0.1 --directory ../..
# Then open http://127.0.0.1:8001/research/2026-10-17-black-scholes-model-empirical-validation.html
```

## Reproduce

Execution used Python 3.13.1, NumPy 2.4.0, pandas 2.3.3, SciPy 1.18.1 and Node 24.19.0. Use an existing Python installation; no virtual environment, dependency directory or processing workspace belongs inside this project. If needed, install Python packages into an existing external environment:

```powershell
python -m pip install numpy==2.4.0 pandas==2.3.3 scipy==1.18.1
$env:PYTHONDONTWRITEBYTECODE = '1'
$raw = Join-Path $env:TEMP 'bsm-research-20261017'
python python/download_inputs.py --directory $raw
# If the configured system proxy is unavailable, --direct bypasses only that proxy.
# TLS verification remains enabled.
Get-Content "$raw\source\README.md"
Get-Content "$raw\source\LICENSE.txt"
Get-Content "$raw\source\verify.py"
python "$raw\source\verify.py" --complete "$raw\source\day_by_date"
python -m unittest discover -s tests -v
python python/analyze.py `
  --source "$raw\source" --spx "$raw\spx.csv" --fred "$raw\fred.csv" `
  --archive "$raw\options_sample_2022H2.zip" `
  --output data
node --check js/pricing.js
node --check js/article.js
node tests/test_browser_math.cjs
```

Run tests again after regenerating outputs to check aggregate consistency. The supplied provider verifier takes a few minutes. The pipeline does not need the stock sample and ignores all vendor IV and Greek columns. It reads the original options files, independent Cboe close history and historical FRED observations. All intermediate frames stay in memory; no raw quote extracts are written.

The archive is about 239 MB and extracts to substantially more. All raw sources and logs used during execution live under Windows TEMP. The downloader preserves existing inputs; remove or rename an external input yourself to refresh it. Provider revisions or Cboe/FRED revisions can change results; input archive and manifest hashes are in `data/results.json`.

## Publication policy

HistoricalData.net is credited visibly in the article. Its supplied sample license (2026-09-06) and online terms (2026-09-14) were read. Free samples may be evaluated; statistics, charts and model outputs may be published, but raw files and reconstructable substitutes may not. Exactly 20 explanatory contract-day rows now accompany the aggregate figures. No larger quote-level panel, contract identifier collection, raw rate curve, SPX close series, archive, source manifest or supplied vendor script is redistributed here. See REVISION.md for the selection, all-field source verification and interpretation of the limited-example permission. Price comparisons pool 25-point midpoint bins with at least 30 observations; IV statistics pool broad moneyness/maturity groups. These aggregations intentionally omit contract/date mappings.

This draft has not been added to `research.html`. No staging, commit, push or unrelated-page edit is part of this work.

## October 17 editorial revision

See [REVISION.md](REVISION.md) for chapter changes, the 20-observation extract, new figures, exact preservation hashes and supplement-only reproduction. Added assets: `css/revision.css`, `js/revision.js`, `data/illustrative-observations.js`, `data/supplement.js`, `data/revision-provenance.json`, `python/revision_outputs.py`, `python/revise_article.py` and `tests/test_revision.py`. Original pricing results remain unchanged.


## Bitcoin empirical study — executed October 17, 2026

The three manually supplied [Hoang/Baur UWA CSVs](https://doi.org/10.26182/5eabbcb4fbf28) were inspected without re-downloading. Original BTC transaction premiums are separate from IV; all 761,067 source USD values reconcile to price × index_price. After 14,124 exact duplicate rows, one zero-amount record, maturity and moneyness restrictions, 250,453 transactions remain. H30/H60 historical pricing and a frozen earlier-period calibration share 178,236 later evaluation transactions across 407 dates.

Held-out MAE: H30 $78.7167, H60 $64.9389, calibrated $201.6667 per BTC notional. The fitted earlier σ transfers poorly; no favorable conclusion is forced. Chapter 07 includes nine additional empirical charts, twenty limited original examples with a data dictionary, complete filtering, seven-day-block uncertainty and careful SPXW comparisons. Figures distinguish binned observations, model values, inversion and illustrative mathematics.

All raw/processed/private files remain under ignored project-root bitcoin_data. No raw source or detailed analytical archive is public. The scoped academic CC BY basis, exact unversioned-license limitation, source hashes, minute-bar availability, inferred UTC history clock, prior-date Yahoo-identified T-bill proxy and all reproduction commands are in [BITCOIN.md](BITCOIN.md) and [BITCOIN_DATA_RIGHTS.md](BITCOIN_DATA_RIGHTS.md). FRED access failed; dataset rates are not relabelled DGS3MO. Source IV is never an independent historical-pricing input. The original SPXW methodology, results and Chapters 01–06 remain unchanged.
