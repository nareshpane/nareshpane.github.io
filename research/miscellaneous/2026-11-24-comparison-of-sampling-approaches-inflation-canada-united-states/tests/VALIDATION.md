# Validation

## Comparative design refinement

- All 38 files under `data/` matched their pre-edit SHA-256 hashes. The official observations, weight vintages and Python simulation outputs are unchanged.
- Seven Python and five JavaScript tests passed. The fixed-population test checks that using different designs and sample sizes preserves the generated population and its target.
- Playwright with Edge covered 1920×1080, 1536×864, 1440×900, 1366×768, 1024×768, 390×844 and 320×740. It checked pointer and keyboard seeking, playback speed, linked SVG nodes, unavailable benchmark data, estimator changes, sample draws, population generation, chart controls, 53 navigation links and 43 downloads.
- Visual review covered the opening, methodology comparison, sampling diagrams, basket, randomization, collection calendar, budget, weights, housing, BLS standard errors and simulation panels. Responsive corrections addressed overlapping SVG labels and tables exceeding the narrow phone width.
- The localhost and direct-file tests reported no browser errors or failed requests. The tested layouts had no horizontal page overflow.

## Initial statistical and data checks

- Seven Python tests and four JavaScript tests passed.
- Checks cover fixed and geometric price relatives, zero growth at constant prices, normalization, rebasing, link continuity, missing calendar endpoints, the finite-population census limit, seeded reproducibility, donor imputation and the Monte Carlo bias–variance identity.
- October 2026 arithmetic reproduces the published 3.1% Canadian annual index change, both 0.2% Canadian monthly changes and the 0.4% U.S. seasonally adjusted monthly change at one-decimal publication precision.
- The output preserves the U.S. headline gap for October 2025 and the missing year-over-year change for October 2026. Gasoline retains its official October 2025 value and reproduces the 37.4% U.S. October 2026 annual increase.
- All 11 unadjusted BLS series were checked against the archived October 2026 release tables. Raw snapshots were verified against the retrieval manifest byte counts and SHA-256 hashes.
- The Python simulation completed 1,200 replications per design and scenario for 63 cells. Empirical summaries and weights were exported with calculated precision.

## Initial static page checks

- Playwright Core with Edge tested desktop (1440×1000), laptop (1200×800), tablet (768×1024), mobile (390×844) and narrow phone (320×740).
- Exercised 24 laboratories, 60 sliders, 27 selectors, 10 checkbox controls, 40 laboratory buttons, chart inspection, animation controls, 53 keyboard navigation links, downloadable links and code copying. An actual CSV transfer was checked.
- The localhost run returned zero page or console errors, failed requests or HTTP failures. The separate local-file run rendered the study and completed a simulation without JavaScript errors.
- Reviewed screenshots of the opening, simulation, equations and empirical plots. Automated viewport assertions checked for page-level horizontal overflow. An internal link audit verified unique IDs and existing local targets.

## Source access

The main Canadian reference PDF was reused from the adjacent study after the source returned HTTP 403 during direct retrieval. The 2026 Canadian basket paper was consulted online. BLS direct bulk files returned HTTP 403, so the study uses the official API and published table snapshots. Public files omit underlying quote records and production replicate weights.

Reproduction commands are in the [README](../README.md). Acquisition errors and hashes are in the [retrieval log](../data/metadata/retrievals.json).

