# RCT monograph refinement report — November 4, 2026

The existing six-part monograph remains intact. All 40 original components, including 30 lettered A–AD sections, are retained. Only the named HTML and its dedicated asset folder were edited; research.html and unrelated tracked/untracked work were not edited. Nothing was staged, committed or pushed.

## 1. Panel design

The clinical overview metrics, intervention definitions, outcome-category cards, interpretation pair, general .grid boxes/metrics, flow blocks, equation panels, decision matrix and parameter controls now share coordinated padding, radius and borders. Adjacent cards retain equal-width grid tracks and matching row heights without fixed empty-height cards. Blue, peach, green and lavender alternate across related panels; blush marks cautions and grey supports technical disclosures. Trial arm colors remain blue/terracotta. Browser inspection found no unequal heights in adjacent same-row box/metric pairs at any tested width.

## 2. Layout corrections

The actual restrictions were `.chapter>p{max-width:88ch}`, the inherited `.intro{max-width:850px}`, and `.part-heading p{max-width:78ch}`, combined with a 1,550 px desktop page ceiling, navigation allocation and generous sheet padding. The refinement removes those specific prose caps, raises the ceiling to 1,720 px, narrows the desktop navigation and moderates internal padding. Shrinkable tracks and min-width:0 remain in place. Full-width paragraphs now equal chapter width: 998 px at 1366, 1057 at 1440 and 1352 at 1920. Body text remains 16 px on desktop and 15 px on mobile. Dense tables, charts and equations scroll locally on mobile; the page itself does not overflow horizontally.

The h1 maximum changes from 64 to 56.32 px (12% smaller), with a similar reduction through responsive sizes; subtitle sizing is proportionate. Long-distance chapter links move immediately rather than taking a prolonged animated scroll through the monograph. Keyboard focus, active tracking and group expansion remain intact.

## 3. Original illustrations and empirical figures

Five editable compositions were drawn from original SVG primitives: neonatal screening (caregiver, infant scale and clinician), balanced oral strategies with an age-scaled follow-up timeline, clinical/microbiology classifications, an infant counterfactual, and the reproducibility chain. They borrow only the reference screenshots’ broad visual vocabulary: clean outlines, simple human forms, pale accents and open space. No reference image, recoloring, tracing, adult infusion scene or third-party figure asset is inserted.

Each illustration sits beside its relevant explanation and has a direct caption, accessible title/description and HTML alt text. Intrinsic image dimensions reserve layout space before lazy loading. The empirical forest adds events per 2,278, interval endcaps, a continuous RR=1 reference, log ticks and adverse-event direction labels. Primary risk percentages generally use one decimal; calculations and meaningful interval precision are retained.

## 4. Mathematical content

The new methods crosswalk separates documented allocation, ITT, binary testing, SAS 9.4, sample-size assumptions and proposed adjustment from this page’s explanatory reconstructions. It adds Pearson expected-cell arithmetic, the conditional hypergeometric Fisher model, equal-arm sample-size planning, and a power equation linked to the density plot. The potential-outcome illustration precedes the existing finite-population derivation; population ATE and finite-enrolled-population effects are distinguished. Existing assumptions, log-RR delta-method derivation, Wald RD and reciprocal/NNT cautions remain.

Original SAS programs and the exact publication CI algorithm are unavailable. No new equation is falsely presented as a quotation from the inaccessible article body. The explanatory planning formula gives about 4,096 per arm for risks .08 and .064; the protocol planned 4,221. No correction is invented to force equality.

## 5. Statistical interactivity

- The existing balance lab now selects N=40–540, retains exact half-arm complete randomization, rerolls assignments, accumulates 200 seeded repeats per click and plots age-SMD frequency. Changing N resets its distribution. At 201 allocations, RMS imbalance was .084 at N=540 and .309 at N=40; a single draw need not follow that pattern.
- The existing 2×2 event sliders drive the same table, two 100-person icon arrays, a shared-scale risk bar comparison, substituted equations, approximate Katz CI and effect summary. Extreme and equal-risk states were checked; no zero-cell correction is silently introduced.
- The repeated-interval lab retains 60 seeded normal estimates and adds 90/95/99% confidence, per-arm sample size, displayed critical value/SE and reset. Changing confidence holds draws fixed. Its known-SE model uses SE=.055 sqrt(270/n), rather than pretending to simulate individual binomial patients.
- The existing power lab now shares alpha, baseline risk, reduction and n with null/alternative densities, rejection regions and live alpha/beta/power. It remains fixed-sample and synthetic. All limits render finite SVG coordinates.
- Preserved assignment animation, potential-outcome toggles, measure buttons, Fisher sharp-null reassignment and missing-outcome controls were tested. Animation is short and respects reduced-motion preferences.

## 6. Independent empirical verification

The original pipeline was rerun on the unchanged 4,556×74 CSV. A separate csv/standard-library implementation imports no first-pass statistical functions: it reproduces 12 published outcome numerator pairs, six continuous baseline mean/SD pairs and three alternate-population checks. Raw CSV SHA-256 remains `7358326b071203978fb575e7bc4eb4fa8eb9985bdda1a2eb1dd5154afb52076a`; all three deposited files retain their Dryad MD5s.

Primary: 123/2,278 synbiotic versus 206/2,278 placebo; RR .5970874, Katz 95% CI [.4812610,.7407901]; RD −.03643547, Wald CI [−.05143019,−.02144075]; uncorrected Pearson p=2.0273265e−6. The primary conclusion is unchanged. Published-versus-before-refinement-versus-recomputed/status tables are visible on the page and in the independent report.

New discrepancy: mortality Table 2 gives p=.526 labeled Fisher; independent two-sided Fisher is .7536358, Pearson .5266352. Omphalitis published p=.014 differs from Pearson .0122659 and Fisher .0210517. Existing culture-positive, LRTI and abscess interval upper-bound differences remain. Follow-up discrepancies, repeated identifiers, maternal missingness and village-prefix uncertainty are not silently harmonized. The corrected 88 suspect cases comprise 33 culture-positive and 55 culture-negative cases; the additional 94 negative evaluations lack a dedicated release flag.

## 7. Publication citation

A prominent lightly blue panel immediately follows the header, with the full paper title in academic serif type, all 16 full author names at a readable size slightly larger than body text, Nature 548(7668):407–412, August 24, 2017 (online August 16), DOI, Dryad dataset and corrigendum links, and an independent educational-replication statement. The public journal record and PubMed 28813414 were checked; the requested J. Glenn Morris Jr. byline form is retained.

## 8. Metadata

The compact five-row panel displays exactly **Model: GPT-6.1 Sol** and **Reasoning: High**, as explicitly requested. Created August 9, 2026, modified November 4, 2026 and Harness Codex on Windows remain. Emoji, label colors, tint, border and nominal 280 px desktop width are preserved; the panel fits down to 320 px viewports. The outdated placeholder explanation is superseded.

## 9. Files changed in this refinement

The list below is generated by comparing current bytes with `refinement-baseline-manifest.json`. Paths are relative to research/miscellaneous. Snapshots intentionally preserve the pre-refinement page and summary for the requested comparison; unchanged raw files and unchanged assets are omitted.

- **Modified** — `2026-08-09-rct-visual-guide.html`
- **Modified** — `2026-08-09-rct-visual-guide/analysis/build_page.py`
- **New** — `2026-08-09-rct-visual-guide/analysis/independent-verification.md`
- **New** — `2026-08-09-rct-visual-guide/analysis/independent_verification.py`
- **Modified** — `2026-08-09-rct-visual-guide/analysis/monograph-sections.html`
- **New** — `2026-08-09-rct-visual-guide/analysis/refine_content.py`
- **New** — `2026-08-09-rct-visual-guide/analysis/refine_figures.py`
- **Modified** — `2026-08-09-rct-visual-guide/analysis/reproduce_trial.py`
- **Modified** — `2026-08-09-rct-visual-guide/analysis/validate_project.py`
- **New** — `2026-08-09-rct-visual-guide/data/independent_verification.json`
- **New** — `2026-08-09-rct-visual-guide/documentation/browser-validation.json`
- **New** — `2026-08-09-rct-visual-guide/documentation/pre-refinement-guide.html`
- **New** — `2026-08-09-rct-visual-guide/documentation/pre-refinement-summary.json`
- **New** — `2026-08-09-rct-visual-guide/documentation/refinement-audit.md`
- **New** — `2026-08-09-rct-visual-guide/documentation/refinement-baseline-manifest.json`
- **New** — `2026-08-09-rct-visual-guide/documentation/refinement-report.md`
- **Modified** — `2026-08-09-rct-visual-guide/documentation/scientific-methodology.md`
- **Modified** — `2026-08-09-rct-visual-guide/documentation/validation-results.json`
- **New** — `2026-08-09-rct-visual-guide/figures/clinical-classification.svg`
- **New** — `2026-08-09-rct-visual-guide/figures/infant-counterfactual.svg`
- **New** — `2026-08-09-rct-visual-guide/figures/neonatal-screening.svg`
- **New** — `2026-08-09-rct-visual-guide/figures/oral-trial-timeline.svg`
- **Modified** — `2026-08-09-rct-visual-guide/figures/primary-risk.svg`
- **New** — `2026-08-09-rct-visual-guide/figures/reproducibility-chain.svg`
- **Modified** — `2026-08-09-rct-visual-guide/figures/secondary-forest.svg`
- **Modified** — `2026-08-09-rct-visual-guide/monograph.js`
- **New** — `2026-08-09-rct-visual-guide/refinement-labs.js`
- **New** — `2026-08-09-rct-visual-guide/refinement.css`
- **Modified** — `2026-08-09-rct-visual-guide/sources.md`
- **New** — `2026-08-09-rct-visual-guide/statistical-math.js`

## 10. Testing and remaining limitations

Automated validation passes: 179 unique IDs, 135 local resource/anchor links, 13 images with alt text, 13 parseable SVG files, three source checksum matches, all original sections retained, independent numerical agreement and plus-one randomization p. The shared JavaScript planning calculation agrees with SciPy across six risk/sample/alpha scenarios to maximum absolute error 5.2e−15.

Browser QA covered desktop 1366/1440/1920, tablet 1024/768 and mobile 390/360/320. No page-wide horizontal overflow, unequal same-row cards, MathJax errors or broken loaded images were found. All 68 contents links were activated and their hash, focus and visible target position checked. All 21 buttons, ten sliders, the confidence selector, 11 technical disclosures and six navigation groups were exercised; changed state and resets were inspected. A fresh monograph tab had zero console errors/warnings, including the retained animation. Direct standalone SVG viewing caused six preview-host promise errors; those script-free SVG assets rendered correctly and the errors did not occur in the monograph.

The original full article body remains subscription-restricted. Original SAS/CI code, full interim datasets/DSMB decisions, dedicated flags for the 94 further culture-negative evaluations, complete last-contact/censoring times and resolution of follow-up/identifier discrepancies are unavailable. This is an unadjusted fixed-sample numerical reconstruction, not an exact reproduction of sequential inference or event adjudication. No survival curve, mechanism proof or mortality benefit is claimed. Fonts and enhanced equation typesetting retain existing external CDN dependencies; static HTML, tables, local SVGs and local labs are preserved. Browser verification used a supported temporary loopback HTTP preview; no deployment occurred.

Reproduce from the dedicated asset folder:

```powershell
python -m pip install -r analysis/requirements.txt
python analysis/reproduce_trial.py
python analysis/independent_verification.py
python analysis/build_page.py
python analysis/validate_project.py
```

Build imports `refine_content.py`; analysis imports `refine_figures.py`. The preserved pre-refinement summary supports the historical comparison. Node.js is used only by the validation script for JavaScript syntax/numerical checks. Detailed numerical results: `analysis/independent-verification.md`; browser evidence: `documentation/browser-validation.json`; automated checks: `documentation/validation-results.json`.
