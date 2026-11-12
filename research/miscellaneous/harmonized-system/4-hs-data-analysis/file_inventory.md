# Files created and modified

Modified: `research/harmonized-system/machine-learning-trade-sector-prediction.html`.

All new published/analysis files are in the existing `4-hs-data-analysis/` folder. Raw and reference datasets were not copied or modified. Local processed caches and browser snapshots are git-ignored.

## New files

- `.gitignore`
- `browser_layout_checks.json`
- `build_model_comparison.py`
- `check_page_layout.py`
- `css\comparison.css`
- `data_audit.json`
- `data_audit.md`
- `destination_predictions.csv`
- `figures\model_01_actual_vs_predicted.svg`
- `figures\model_01_errors.svg`
- `figures\model_01_scatter.svg`
- `figures\model_02_actual_vs_predicted.svg`
- `figures\model_02_errors.svg`
- `figures\model_02_scatter.svg`
- `figures\model_03_actual_vs_predicted.svg`
- `figures\model_03_errors.svg`
- `figures\model_03_scatter.svg`
- `figures\model_04_actual_vs_predicted.svg`
- `figures\model_04_errors.svg`
- `figures\model_04_scatter.svg`
- `figures\model_05_actual_vs_predicted.svg`
- `figures\model_05_errors.svg`
- `figures\model_05_scatter.svg`
- `figures\model_06_actual_vs_predicted.svg`
- `figures\model_06_errors.svg`
- `figures\model_06_scatter.svg`
- `file_inventory.md`
- `fit_records.csv`
- `forecast_freeze.json`
- `frozen_destination_predictions.csv`
- `frozen_destination_universe.csv`
- `frozen_hs4_predictions.csv.gz`
- `frozen_hs4_universe.csv`
- `frozen_top5_hs4.csv`
- `hs4_predictions.csv`
- `js\comparison.js`
- `mlp_stability_diagnostic.json`
- `model_01_coefficients.csv`
- `model_06_reference_checks.csv`
- `model_diagnostics.json`
- `model_summary.csv`
- `observed_top10_destinations.csv`
- `observed_top5_hs4.csv`
- `quality_checks.json`
- `temporal_2024_frozen.csv.gz`
- `validation_metrics.csv`

## Reproduce

```powershell
python research/harmonized-system/4-hs-data-analysis/build_model_comparison.py
python -m http.server 8000
```

Analysis packages: numpy, pandas, scipy, scikit-learn, xlrd, matplotlib and statsmodels. The supplied read-only vendor folder is an optional fallback for plotting/inference libraries. Exact executed versions are recorded in quality_checks.json.

Set TRADE_RAW_ROOT, TRADE_QUESTION1_REFERENCE and TRADE_TESTING_REFERENCE to override local roots. Use --fresh to rebuild compact data caches; --render-only regenerates publication figures and HTML from completed results. Import HS4 as text when opening CSV files in spreadsheet software.

Primary CAD results freeze the 2024 annual rate 1.3698. The PDF reproduction using realized 2025 1.3978 is a separate ex-post diagnostic.
