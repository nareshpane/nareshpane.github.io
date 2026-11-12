# Visual and teaching revision — October 17, 2026

## Scope and preserved experiment

The existing article was revised rather than replaced. All ten chapter IDs, original calculators/simulations, filter rules, 20/30-return volatility estimators, rates, carry calibration, contract partition, tests A/B and results remain. The original `data/results.json` and `results.js` are byte-for-byte unchanged. Their pre-revision hashes are recorded in `data/revision-provenance.json` and tested.

Design references inspected online and locally:

- `research/harmonized-system/section-338-hs4-hs6-exposure-canada.html` and its `2-section-338-hs4-hs6-exposure-canada/css/style.css`.
- `research/harmonized-system/details-of-machine-learning-models.html` and `details-of-machine-learning-models/css/chapter.css`.
- The collection's `shared/page-provenance.css`.

The revision borrows their blue serif title hierarchy, green subsection headings, warm compact provenance banner, restrained containers and sticky table treatment. The peach/ivory, spring-green, teal and blue Black–Scholes palette remains. The page moves from 1180 to 1320 px, removes the forced hero line break, reduces section/lab spacing, and replaces the blanket 80ch paragraph cap with context-sensitive measures. Mathematical prose remains constrained; charts and tables use the available width. The exact model variant and reasoning configuration are not exposed, so the banner retains the reported GPT-6 identity and states those limits.

## Chapter changes

1. Beginner guide: 12 expandable terminology cards; stock/contract diagram; $100 stock with adjustable strike and two possible terminal prices; four coordinated long/short call/put payoff and profit plots. Original payoff explorer retained.
2. Six-input influence map reports recomputed price changes and selects the original sensitivity axis. Original calculator retained.
3. Brownian increment-resolution explorer uses one seeded realization, with stages from shocks to cumulative increments to GBM. Added the proportional-change explanation; original path/distribution controls retained.
4. Eight numbered bridges use Intuition → Symbols → Equation → Derivation → Interpretation. They expand Itô, the log-stock example, random-term cancellation, self-financing, no arbitrage, boundaries, physical versus Q distributions, integration, d₁/d₂ and parity. Technical proofs remain closed. Worked example: S=K=100, T=1, σ=.20, r=.05, q=.02, call=9.22700551, d₁=.25, d₂=.05.
5. Hedge-cycle diagram connects stock, option, gamma and cash rebalancing with the existing simulated holdings and wealth figures.
6. Limited 20-observation source explorer and full dictionary, plus quote-spread histogram and historical versus market-implied volatility comparison.
7. New moneyness/price plot, signed residual plot and maturity-MAE comparison; original scatter, residual histogram, error heatmap and uncertainty plots retained. The price/residual pair shares side/maturity controls and the existing Test A/B selector.
8. Original empirical smile and volatility heatmap retained. The IV-versus-historical comparison in Chapter 6 makes their interpretation explicit.
9. Existing numerical audit retained and updated to include revision tests.
10. Original sources retained; reproduction instructions include a supplement-only mode that does not overwrite the validated results.

## Exactly 20 genuine explanatory observations

The raw archive, extracted CSVs and intermediate processing remain under Windows TEMP. `revision_outputs.py` runs on the original pipeline's held-out evaluation frame. It chooses five distinct contracts on each of **2022-07-01, 2022-09-19, 2022-11-16 and 2022-12-30**, alternating option side and targeting different moneyness/maturity combinations. Selection is deterministic, sorted by distance then contract symbol; it is explanatory, not a representative statistical subsample.

For each selected contract-day it rereads the original CSV with `csv.DictReader` and retains all **34 source strings**, including missing values. It adds six clearly marked research quantities: midpoint, harmonized SPX close, K/S, calendar maturity, Test A price and Test B price. Original `underlying_close` is not overwritten. Source file SHA-256 and CSV line numbers support independent verification. Friendly headings map to source keys and definitions; source-derived vendor IV/Greeks have a distinct background and are not fed into pricing tests. Empty fields render as N/A; recorded zeros remain zero.

`illustrative-observations.js` contains exactly 20 rows. There is no larger hidden source panel, quote download, pagination to more records, or additional raw extract. The table has a fixed-height scrolling region, sticky column headers and observation labels, keyboard sorting, side/date filters and Clear Filters. All 41 displayed columns (identifier + 34 source + 6 research-derived) have a dictionary entry.

HistoricalData.net's supplied license and current terms §3 were reviewed. They permit limited example rows to explain research but prohibit reconstructable substitutes; there is **no numerical row-count safe harbor**. These 20 non-panel examples explain schema, missing timestamps, settlement, underlying harmonization and source-versus-research calculations. That bounded purpose, the tiny explicitly finite selection, absence of bulk chains and absence of an accumulating quote collection are the basis for using the explanatory-example permission. This is an interpretation of that permission, not provider approval or a guaranteed 20-row allowance. Source: https://historicaldata.net/samples.html; terms: https://historicaldata.net/terms.html.

## New aggregate calculations

`supplement.js` contains no original quote rows. It exports:

- Counts of observed spreads in explicit categorical buckets, including the ≥100-point tail. Unequal-width buckets show counts, not density.
- Counts, mean K/S, observed mean midpoint, mean Test A/B value and signed bias by side × maturity × moneyness, across the unchanged 238,836 evaluation observations.
- Daily historical σ and median independently inverted midpoint IV for held-out near-the-money observations (0.98≤K/S≤1.02), plus counts. It pools maturities and sides and is labeled accordingly. It is not a forecast of realized volatility or an independent self-IV pricing test.

The maturity-MAE figure reads the existing validated `results.js` breakdowns directly. Tests reconcile every new group count to the original sample, every grouped signed bias to the original bias, and historical-volatility values exactly to original outputs. Empirical dates remain July 1–December 30, 2022. No change to conclusions: MAE A=11.6782, B=6.1593; calibration does not prove profitability or improve every metric.

## Reproduce only the supplement

From the project root, using the original external inputs and existing Python environment:

```powershell
$env:PYTHONDONTWRITEBYTECODE = '1'
$raw = Join-Path $env:TEMP 'bsm-research-20261017'
python python/analyze.py `
  --source "$raw\source" --spx "$raw\spx.csv" --fred "$raw\fred.csv" `
  --archive "$raw\options_sample_2022H2.zip" `
  --output data `
  --supplement-output data
python -m unittest discover -s tests -v
node --check js/revision.js
node tests/test_browser_math.cjs
node tests/test_dom.mjs "$raw\linkedom.mjs"
```

The optional DOM test uses the existing external LinkeDOM 0.18.12 standalone worker; no new dependency installation or repository dependency folder is needed. `revise_article.py` records the one-time editorial assembly and refuses to reapply to an already revised page. Editing the current HTML does not require running it.

## Actual verification and limitation

21 Python tests pass, including exact string-for-string comparison of all 20×34 source fields against the four original external CSVs, source hashes, friendly-heading/origin mappings, derived values, aggregate totals, old-result hashes and worked numerical price. Node pricing and syntax checks pass. The DOM integration harness verifies **26 populated SVG charts**, two heatmaps, the 20-row/41-column table, sorting/filtering, 12 closed terminology cards, 10 chapter links, all new controls, shared chart selectors and Reset. It checks no invalid SVG coordinates and all collapsible material starts closed.

The browser connector returned `apps: [], browsers: []` on this revision. No repeated browser attempts or dependency installations followed. **Actual visual browser rendering, responsive overflow/overlap, MathJax network loading and native keyboard/animation behavior could not be visually verified.** DOM checks and responsive CSS are not a substitute for that review. Mathematical typesetting still requires the pinned external MathJax resource.

Only the specified HTML and dedicated asset folder were changed. No raw archive, sandbox, worktree, virtual environment, node_modules, temporary processing directory, staging, commit or push was introduced. Pre-existing unrelated repository changes were preserved.

## Subsequent targeted visual refinements

The four introductory SVG panels are now six semantic story cards in a responsive 3×2 grid, matching the unchanged six story-navigation links. Their bid–ask, scatter and smile motifs are explicitly schematic; no new market observations were invented. All borders are green. The call-pricing formula uses the existing MathJax configuration.

The twelve closed terminology cards retain their examples and native expansion behavior. Green borders and aligned symbol badges now define S, C, P, K, T, premium, positive-part notation, S/K, exercise schedule, position roles and payoff minus premium. S/K is explicitly distinguished from this study's K/S convention; positive-part notation avoids using IV for intrinsic value. A calendar SVG denotes exercise timing rather than incorrectly assigning the expiration symbol to exercise style.

The existing two-future chart's calculation and plotting JavaScript is unchanged. Its three sliders have purple/teal/blue tracks and thumbs, separately from its explicit blue-call/teal-put payoff legend. The payoff-versus-profit note has a neutral surface, blue outer border and faint-green equation inset; its equations and explanatory sentences are preserved exactly.

The five-row proportional-change bridge now defines all stochastic symbols, explains physical drift versus diffusion, the $0.10/$10 effect of a common 1% shock, Brownian variance addition, square-root time scaling, conditional small-step moments and the integral meaning of the SDE. It never treats dW/dt as an ordinary derivative. Its new SVG separates the drift and diffusion contributions. Subsequent log-return/Itô/GBM discussion, all eight complete derivation bridges, and the worked pricing example are unchanged.

The mathematical roadmap and empirical pipeline are each six larger cards in two rows of three on desktop, two columns at medium widths and one on narrow screens. Roadmap destinations are unchanged. The empirical workflow now explicitly includes the already implemented calculation/evaluation stage between sample establishment and publication. Its descriptions follow the actual rounded-K/5 partition, historical/fitted-volatility tests and permitted outputs.

Whitespace changes target unaccompanied direct section prose and structured derivation rows. They remove inherited reading caps where these caused an unintended empty right column; hero copy, terminology cards, split-column content, tables and chart containers retain their independent constraints. No justification, page-wide minimum width, new animation or dependency was introduced.

Verification: all 21 Python tests, Node pricing/syntax checks and the extended 26-chart DOM integration suite pass. Added structural assertions cover all three six-card groups, twelve defined terminology symbols, the two payoff legend entries, five stochastic rows, the equation-note content and valid navigation destinations. A source audit compares the payoff note, all eight existing derivation articles, metadata, original calculations and every data-bundle hash with the pre-refinement version. The browser connector again returned no apps or browsers; actual responsive visual rendering and MathJax display could not be verified. No repeated attempts or package installations were made.

Files changed in this targeted pass: the main HTML, `css/revision.css`, `tests/test_dom.mjs`, and this revision report. Pricing/analysis/plotting JavaScript, empirical source examples, aggregate bundles, selection rules, estimators, findings and metadata values are unchanged.

## Connected pricing system and mathematical visual explanations

The input-influence selector and calculator now share one green-bordered `#influence.pricing-system` container. The main heading is unchanged; “A living pricing formula” is a smaller internal heading beside a subtle downward cue. Original introductory text, all six buttons, influence readout, Figure C caption, six sliders, Reset, results, sensitivity control, chart, legend and explanatory notes remain. Reset follows the six sliders. No control ID, parameter, calculation, chart implementation or event handler changed.

Six static interludes supplement the existing derivation without replacing or shortening it:

1. Itô's time, stock-response and curvature contributions. The caption explicitly states that V_S dS contains both drift and Brownian exposure.
2. The option's +σSV_S dW^P and short stock's −σSΔ dW^P exposures combine to zero at Δ=V_S. Dividends and financing remain in the deterministic gain.
3. The locally riskless gain, financing condition and resulting dividend-consistent PDE are connected visually; μ disappears through replication, not because physical returns are assumed risk-free.
4. Physical P and pricing Q use distinct drifts and Brownian superscripts. Changing valuation weights does not rewrite historical returns.
5. Terminal call payoff → conditional Q expectation → discount to present value. This is not a historical sample average or a return forecast.
6. Log moneyness plus risk-neutral log drift, divided by σ√T, gives d₂; adding σ√T gives d₁. The call's stock-value and strike-payment components are distinguished, and N(d₁) is never presented as a physical exercise probability.

Concise adjacent definitions expand V_S, V_SS, V_t, hedge ratio Δ versus price change ΔS, portfolio Π versus trading gain dG, terminal payoff H, conditional expectation, e^(−rT), and the standard normal CDF N(·). Original equations, algebraic passages, collapsible proofs, numerical examples, sources and empirical content remain. All additions are static HTML/MathJax with small SVG connectors; no new animation, JavaScript dependency or motion preference handling was needed. At narrow widths, diagram components stack with vertical connectors and equations scroll within their own boxes.

Verification: 21 Python tests pass; Node syntax, pricing and the 26-chart DOM suite pass. The integration suite now exercises **every one of the six input buttons**, verifies its selected sensitivity axis and active state, checks all six sliders, and dispatches a full Reset click to restore all six defaults and the S axis. It also verifies the single pricing container, retained control IDs, six new interludes and existing interactions. A source-level before/after audit checks that original pricing text, original derivation equations/proofs and all data/implementation hashes are retained. The browser connector returned no apps or browsers; actual MathJax rendering and responsive visual layout remain unverified. No repeated browser attempts or dependency installation occurred.

This pass modifies only the main HTML, `css/revision.css`, `tests/test_dom.mjs`, and this report. Scientific inputs, outputs, implementation scripts, metadata, unrelated chapters and navigation are preserved. No staging, commit or push was performed.
