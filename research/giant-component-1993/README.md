# The Birth of the Giant Component: reproducible research page

Primary paper: Svante Janson, Donald E. Knuth, Tomasz Łuczak and Boris Pittel
(1993), DOI 10.1002/rsa.3240040303, https://arxiv.org/abs/math/9310236.

The main page remains `../giant-components.html`. No changes are needed to site
configuration or other research pages. It runs as static HTML, CSS and JavaScript;
MathJax 3.2.2 is loaded from jsDelivr. The static tables, SVG charts and source
blocks work without a data fetch, including on `file:` URLs.

## Events and models

* `snapshot`: after **floor(n/2) accepted edges**, every component has
  beta = e - v + 1 <= 2. Limit sqrt(2/3) cosh(sqrt(5/18)) = 0.9325483641301475.
* `trajectory`: never two components with e > v at any accepted edge. Stop at
  the **first failure or connectivity**. Connectivity is an exact success rule,
  not an approximate time cutoff. Limit 5 pi / 18 = 0.8726646259971648.
* `simple`: uniform ordered endpoints, rejecting loops and reused unordered
  pairs. Every remaining unordered pair has two proposals, so the accepted pair
  is uniform over remaining edges. The hash set avoids allocating all n choose 2.
* `multigraph`: accept all independent ordered endpoint pairs. Loops have weight
  1/n^2, non-loop undirected edges 2/n^2; repeated edges retain multiplicity.

## Environment and reproduction (PowerShell)

Python 3.13.1 was used with NumPy 2.4.0 and Numba 0.68.0, on Windows 11,
four Numba workers. Create a virtual environment outside the site asset folder.
From this folder:

```powershell
python -m pip install -r requirements.txt
python -m unittest scripts.test_simulation -v
python -m scripts.run --mode quick --seed 20261012 --out quick-data
python -m scripts.run --mode research --trials 1000 --seed 20261012 --out reproduction
```

The research default covers exactly n = 50, 100, 500, 1000, 5000, 10000,
20000, 30000, 40000, 50000, 60000, 70000, 80000, 90000, 100000, for both
experiments and both models. All published cells have 1,000 trials (60,000 total).
Quick mode defaults to 20 trials at 50, 100 and 500. Optional arguments include
`--ns`, `--models`, `--experiments`, `--threads`, `--batch-size`, `--seed`, `--out`.

To resume the published configuration or increase repetitions:

```powershell
python -m scripts.run --mode research --trials 1000 --seed 20261012 --out data --resume
```

Each completed batch saves `checkpoint.json` atomically and refreshes aggregate
`results.json` and `results.csv`. Resume rejects changed scientific settings and
skips completed trial indices. The default and published research seed
**20261012** passes the full 60,000-seed uniqueness check. The runner refuses any
colliding configuration rather than silently reusing streams.

Trial seed: `SeedSequence([base, n, is_simple, is_trajectory, trial_index])`,
first generated uint32, used to seed Numba MT19937. SeedSequence keys distinguish
all cells and indices. The seed manifest SHA256, library versions and platform
are recorded. Identical scientific settings reproduce aggregates regardless of
batch size or worker count. Runtime will differ; it includes first-call compilation.
Pseudorandom streams are a Monte Carlo approximation to independent random trials.

Browser seeds use a separate Mulberry32 stream and unbiased bounded-integer
rejection. Identical browser settings reproduce the browser sequence; Python
and browser sequences are not identical for the same numeric seed.

## Algorithms and files

`scripts/sampling.py`: lazy exact edge proposals.
`scripts/union_find.py`: path compression, union by size, root vertex/edge counts.
`scripts/experiments.py`: compiled trials and parallel batches.
`scripts/statistics.py`: constants, estimators and Wilson intervals.
`scripts/export.py`: atomic checkpoint and CSV/JSON export.
`scripts/run.py`: configurable execution and resume.
`scripts/test_simulation.py`: independent graph traversal and distribution checks.
`scripts/build_page.py`: article, static SVG charts/tables and real embedded sources.
`scripts/validate_page.py`: asset, source, embedded data and constant verification.

Build and verify the page after replacing `data` with a complete research run:

```powershell
python -m scripts.build_page --assets . --output ../giant-components.html
python -m scripts.validate_page --assets . --page ../giant-components.html
```

The article builder requires both models, both events and all 15 sizes. Aggregate
results are embedded so visitors need no Python or fetch access. Direct downloads
remain the canonical CSV/JSON. Wilson 95% intervals target each finite-n event
probability, not the limiting constant. Reported differences contain both sampling
error and finite-size effects. No fitted theoretical correction is assumed.

## Validation

Independent adjacency traversal verifies each DSU prefix, including edge
multiplicities. Constructed cases cover tree/unicyclic/bicyclic/tricyclic components,
two separate complex components and their merger, internal edges, loops and parallel
edges. Random small graphs validate both events. Tests also check seed reproducibility,
worker independence, uniform proposals, odd-n floor convention, stopping at connectivity,
Wilson boundary behavior and export consistency. Run tests after simulations to include
the export check. `validation.json` records static and browser validation.

## History and retained assets

The earliest actual giant-components file addition is **a5558b1**, March 30, 2026,
08:46:33 -0700, at root `giant-components.html`. `git log --follow` crosses later
renames, but it also reports a similarity-detected C067 copy from `nauty-ramsey.html`
in that addition; following this copy can misleadingly return **71ccd54**, which
created the unrelated nauty page one minute earlier. `git show --name-status a5558b1`
and `git log --diff-filter=A -- '*giant*html'` establish the relevant addition.
Subsequent modifications and moves are not creation dates.

Five existing Python scripts, three MP4s and two PNGs remain unchanged. They are
historical assets, not the computational engine for the new results. In particular,
`giant_component_simulation2.py` estimates the at-most-one-complex **snapshot**
probability near 0.9957, not the full-history event of Theorem 15. All scripts are
accessible through source disclosures at the bottom of the reconstructed article.

The quotation panels transcribe the supplied screenshots verbatim: the complete
Section 11 corollary including Equation (11.7), and the cropped Introduction passage
on page 3. The latter visibly begins and ends mid-sentence, just as supplied.
Links identify the original passages and Theorem 15; surrounding exposition is original.

Model/effort for this revision: session turn_context records `gpt-6.1-sol`, `medium`.
Revised date: October 12, 2026. No automatic staging, committing or pushing.

## Browser checks (optional test tool, not a website dependency)

The saved `tests/browser_check.cjs` uses Playwright with installed Chrome. Serve
this repository locally on port 8765, then run the test with a Playwright module
available in the test environment. Set `PLAYWRIGHT_MODULE` to its installed module
path when it is not on Node's ordinary module path. Pass a scratch output directory:

```powershell
# From the repository root, in a separate terminal:
python -m http.server 8765 --bind 127.0.0.1
# From this asset folder, with Playwright in the test environment:
node tests/browser_check.cjs C:\path\to\scratch-browser-output
```

`tests/python-tests.txt` records the executed nine-test Python suite.
`tests/browser-validation.json` records the graph controls, 200-vertex generation,
seed reproduction, first-failure auto-pause, failure after merger, backward timeline
navigation, MathJax typesetting, 390px/320px responsiveness and real clipboard copy.
The browser report records the exact main-page SHA256. Static validation invalidates
browser evidence if a subsequent rebuild changes that page. No npm package, build
step or server is required by the published website itself.
