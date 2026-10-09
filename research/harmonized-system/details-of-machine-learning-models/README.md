# Inside six machine learning models

Local educational draft created 8 October 2026. Source data, original scripts,
fitted models and the original HTML page are read-only. Only this new project
and one companion entry in the collection index are changed.

The observed 2025 seven-country BACI holdout is unavailable in the supplied archive.
The table has 252 historical source-derived outcomes and 84 explicitly artificial
2025 scenarios. Every 2025 observed value is blank. Scenario values and errors are
separate columns. Do not quote the scenario comparison as actual forecasting accuracy.

## Rebuild from the included verified extract

From the repository root:

```powershell
python research/harmonized-system/details-of-machine-learning-models/dataset_build.py
python research/harmonized-system/details-of-machine-learning-models/scripts/analyze_models.py
python research/harmonized-system/details-of-machine-learning-models/scripts/build_chapter.py
python research/harmonized-system/details-of-machine-learning-models/scripts/verify_project.py
python research/harmonized-system/details-of-machine-learning-models/scripts/verify_reproduction.py
python -m http.server 8000
```

Open http://localhost:8000/research/harmonized-system/details-of-machine-learning-models.html

Use `dataset_build.py --raw` to independently regenerate the compact historical
extract from supplied BACI, WDI and GeoDist at `TRADE_RAW_ROOT`. Raw extraction
requires pandas, NumPy and xlrd. It reads all annual BACI chunks for two headings;
historical source data, metadata and hashes are retained in the small extract/manifest.
No original builder is executed: it would overwrite the protected reference page.

## Dependencies

Python packages: numpy, pandas, scipy, scikit-learn, threadpoolctl, matplotlib;
xlrd additionally for raw GeoDist XLS extraction. Used package versions are saved in
results/quality_checks.json. Install requirements.txt in a separate Python environment
if necessary. build_chapter.py supports the same supplied read-only plotting vendor
fallback as the original project, configurable via TRADE_TESTING_REFERENCE.

The page has no external runtime dependency: MathJax 3.2.2 tex-svg.js is vendored,
with its Apache license; other JavaScript is dependency-free. PNG illustrations are
generated with the built-in imagegen capability, not random downloaded classrooms.
Illustration prompts are retained in images/prompts.md. They illustrate central ideas;
the equations below them define the actual estimators. No API is needed for numerical
reproduction or browsing; illustrations are already included.

## Outputs and limits

- toy_trade_dataset.csv, data_dictionary.csv, data_sources.md, dataset_build.py.
- data/source_observed_subset.csv, source_manifest.json, dataset_audit.json.
- scripts/analyze_models.py: six fits, chronological splits, metrics, exact traces.
- scripts/chapter_content.py: authored model derivations and history.
- scripts/build_chapter.py: renders the new HTML and Matplotlib SVG figures.
- scripts/verify_project.py and browser_checks.py: numerical/static and browser checks.
- scripts/verify_reproduction.py: independent isolated refit; bit-identical numerical artifacts.
- results/predictions.csv/json: every final training fit plus validation/forecast row.
- results/model_comparison.csv, metrics.json: all evaluation populations separated.
- results/fit_records.json: all training IDs, settings, calibration, convergence warnings.
- results/worked_examples.json: coefficients, tree paths/all trees, all neural weights,
  activations, boosting sequences, positive-stage traces and independently estimated lambda.
- results/lambda_estimation.csv, structural_oof_2023.csv: all dynamic estimation records.
- results/frozen_2025_predictions.csv and forecast_manifest.json: common 84-cell forecasts.
- results/figures/: observed 2024 diagnostic plots and clearly labelled scenario comparisons.

Actual observed 2025 follow-through is impossible with this archive; observation 263
is the illustrative Canada→U.S. wheat case and its observed 2024 counterpart is ID179.
Model inventory documents every teaching departure (network, pooled training,
temporal validation, 8/4 MLP and fixed epoch budget) from the large Alberta study.
The MLP reaches its declared budget with a convergence warning; its finite-budget
outputs are not claimed to be optimal. Statistical confidence intervals are not supplied.

Do not commit, push, deploy or add this draft to the main research.html automatically.

### Presentation audit (no fitting)

`scripts/display_numbers.py` centralizes ordinary values, percentages (with explicit
proportion conversion), integer counts, and small nonzero quantities. The chapter
renders numbers in Python; JavaScript only filters existing rows and updates counts.
Ordinary results use one decimal. The complete dataset uses whole-dollar quantities,
and tree-path comparisons retain two decimals to distinguish nearby thresholds.
Rounded substitutions are approximate; inverse transformations use the saved full
precision, with symbolic exponents rather than misleading rounded substitutions.
Exact settings, identifiers and mathematical constants remain exact.

With the repository served using `python -m http.server 8000`, run:

```powershell
python research/harmonized-system/details-of-machine-learning-models/scripts/build_chapter.py
python research/harmonized-system/details-of-machine-learning-models/scripts/audit_display.py
python research/harmonized-system/details-of-machine-learning-models/scripts/verify_presentation_revision.py
```

The renderer reuses saved results and figures. It does not refit or regenerate them.
The Chrome audit expands all six models, examines visible text, equation source,
tooltips, table/paragraph dimensions and overflow, and exercises existing controls.
It uses Chrome's local debugging interface through a small Python standard-library
client, waiting for completed checks before screenshots; no browser package is needed.
It saves screenshots and measurements in `results/browser/`; `--before` saves
diagnostic evidence before editing. Decimal matches are reviewed, never used for
automatic replacements. Exact settings inside code blocks and raw SVG geometry
are intentionally excluded. The integrity check compares the original protected
file hashes, including canonical calculations and unrelated repository files.
