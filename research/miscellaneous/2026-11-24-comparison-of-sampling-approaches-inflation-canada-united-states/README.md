# Measuring Inflation in Canada and the United States

The HTML file beside this folder is the research page. It uses only local assets at runtime and can be hosted directly on GitHub Pages. Serving it over HTTP enables the Web Worker used by the live simulation. Opening the file directly uses the same data and has a bounded synchronous fallback for the worker.

## Architecture

- `css/` contains the responsive article, navigation, comparison tables, equation panels and grouped laboratory controls.
- `js/` contains statistical utilities, linked SVG sampling diagrams, scientific figures, teaching models and the Web Worker.
- `animations/` and `figures/` contain the local opening map assets, projected from Natural Earth 1:50m data.
- `data/raw/` retains official snapshots. `data/processed/` contains harmonized and derived outputs. `data/simulated/` contains the fixed-population experiments.
- `data/metadata/` records original sources, retrievals, dates and hashes.
- `vendor/` contains KaTeX and its license, reused from the adjacent CPI project.
- `scripts/` contains the Python reproduction pipeline and `tests/` holds the mathematical and browser checks.

## Reproduce from Windows PowerShell

From this project folder, using Python 3.11 or newer:

```powershell
python -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
.\.venv\Scripts\python.exe .\scripts\run_all.py
.\.venv\Scripts\python.exe -m unittest discover -s tests -p 'test_*.py'
node .\tests\sampling.test.cjs
```

The reproduction ZIP includes the raw inputs for the geography extension. After extracting it, run `python .\scripts\prepare_geography.py` from the project folder to rebuild those outputs offline. For the full historical pipeline, obtain the remaining raw files with `python .\scripts\acquire.py` and then run `python .\scripts\run_all.py`. To refresh the geographical inputs as well, run `python .\scripts\acquire_geography.py`. Refreshing after the research cutoff can revise agency data and web pages. The scripts cap analysis observations at October 2026.

Serve the project from its parent folder using `python -m http.server 8014` and open the named HTML page on localhost. To run browser checks, install Playwright Core in a temporary location or use an existing installation. Set `CPI_PLAYWRIGHT` to its module folder, and set `CPI_EDGE` to the path to Chromium or Edge if the default Windows Edge path is unavailable. Run `node .\tests\browser.cjs`. Screenshots and downloads used in the test are held in a temporary folder.

## Statistical scope

The empirical data are January 2000 through October 2026. The BLS October 2025 gap is explicit in the calendar and passes through growth calculations. Both national index series use their published vintages and geographic concepts. Comparisons based on a shared rebasing date describe cumulative price movement. Public aggregate data support rate arithmetic and upper-level approximations, while reproducing individual agency production stages requires internal inputs.

The Python simulation fixes one product-level population per scenario using base seed 20261124, then repeats sample selection 1,200 times per design. Processing times in the summary are machine-dependent. The web controls run a separately seeded, lower-cost outlet model. Purposive and hybrid working intervals are evaluated under the synthetic scenarios, where the true target is available.

The live estimator lab builds a finite population explicitly when **Generate population** is selected. Sample size, selected design, draw seed and imputation change how that population is observed and analyzed. They preserve its underlying units and target. For the two-country collection calendar, both schedules observe one synthetic daily price process.

Read [the data dictionary](data/README.md) and [the validation summary](tests/VALIDATION.md) for details.

