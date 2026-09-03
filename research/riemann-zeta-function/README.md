# Riemann Zeta Function Guide

Supporting files for `../riemann-zeta-function.html`, a progressively layered
visual guide to the Riemann zeta function, its zeros, prime distribution, and
carefully classified recent research.

## Structure

- `styles.css` contains the page-specific responsive visual system.
- `scripts/page.js` implements the native SVG/Canvas/HTML interactions.
- `scripts/generate_numerical_assets.py` regenerates JSON datasets and static
  fallback figures with deterministic numerical checks.
- `data/` contains generated zeta, prime-counting, zero-count, spacing, and
  random-matrix datasets.
- `images/` contains generated dense visualizations and static fallbacks.
- `references/` records the bibliography and source audit for recent results.

## Regenerate Numerical Assets

From this directory, create an isolated environment if desired, then run:

```bash
python3 -m pip install -r requirements.txt
python3 scripts/generate_numerical_assets.py
```

The generator checks the values of zeta at 2 and 4, trusted initial zero
ordinates, conjugation behavior, zero counts, dataset metadata, and the fixed
random seed. Generated JSON stores ordinary double-precision values after
high-precision computation where applicable.

## Interactive Components

The page contains laboratories for p-series partial sums, real zeta values,
finite Euler products, complex summand vectors, analytic-continuation domains,
the critical strip, prime-counting approximations, truncated explicit formulas,
Hardy's Z function, zero counting, spacing statistics, and random matrices.
Text, equations, captions, fallback figures, and result qualifications remain
available without JavaScript.

## Local Test

From the repository root, serve the static site so JSON requests work:

```bash
python3 -m http.server 8000
```

Then open
`http://localhost:8000/research/riemann-zeta-function.html`. Check a narrow
mobile viewport, keyboard focus, browser console output, and the no-JavaScript
fallback if practical.

## Source Policy

Classic claims are tied to primary papers or authoritative references such as
DLMF. Recent theorem statements are included only after checking a primary
paper or journal record. Preprints are labeled as preprints, conditional results
are labeled conditional, and technical progress is never described as a proof
of the Riemann Hypothesis. See `references/recent-results-audit.md` for the
claim-by-claim record and cutoff date.
