# Initial implementation browser verification

This file preserves the original implementation's QA record. For the completed refinement, including the shared three-product scenario, six players, nine supplied images and seven viewport sizes, see [the refinement report](../REFINEMENT_REPORT.md) and the adjacent JSON audits. Earlier example values below describe the original version.

Tested through the Codex in-app Chromium browser over a local HTTP server. Desktop viewport: 1440 × 1000. Mobile viewport: 390 × 844. Default panel width was also inspected (714 pixels). Tests operate the actual page controls and inspect rendered output; they do not merely repeat data-loading success.

## Completed checks

- All 18 chapters render, all 15 SVG plot areas populate, and all mathematical boxes render with KaTeX without parse errors.
- Browser console had no warnings or errors during the inspected loads and interactions.
- The headline cards display 170.0, +3.1%, +0.1% unadjusted and +0.4% adjusted.
- Basket input: changing milk’s second-month price to $5 changes the later cost from $29.70 to $30.90 and inflation from 10.0000% to 14.4444%. The input rule initially had a step mismatch; it was corrected and retested.
- Link-price weight selection displays food 16.93%, shelter 28.30% and transport 18.54%; reference-price selection remains distinct.
- Price shock slider recalculates weighted influence. The reference share slider uses 0.01 percentage-point steps so the initial 16.83% is retained.
- Package slider at 250 g yields $20/kg and +100% compared with the $10/kg reference. Default 450 g gives +11.1111%.
- Reciprocal elementary relatives produce Jevons 1.0000 with the unchanged third offer; Carli remains 1.1667.
- Formula editing: changing the first later price from 3 to 4 produces Laspeyres 152.5000, Paasche 130.9091, Fisher 141.2927 and Lowe 143.0000. Default values were separately checked arithmetically.
- Aggregation trace switches to the rent branch and reports its 0.8000 pp contribution in the synthetic basket.
- New-basket chain final stage: 121.9200. Switching to old basket at the same stage: 127.6800. Both use link level 120.
- Energy annual-history selection yields September 2026 18.9539%; metric and series controls update the plotted observations and table.
- Setting synthetic seasonal amplitude to zero makes observed and adjusted values coincide; clothing selection updates actual seasonal/unadjusted comparisons.
- A −40% synthetic gasoline shock at 5% share gives −2.0000 pp; complete synthetic aggregate falls 0.8500%.
- Raising the core-distribution high-tail observation to 30% changes the weighted mean to 2.3530% while median stays 0.3000% and the middle-60% trimmed mean stays 0.2950%.
- Selecting Iqaluit updates the headline, regional chart and available components. Its special December 2002 reference is recorded explicitly rather than inferred from the generic SDMX UOM label.
- Purchasing-power example: $1,000 at September 2025 requires $1,030.93 at September 2026; an unchanged $1,000 has $970.00 in earlier-period purchasing power.
- Audit search, level and metric filters work together; filtering “gasoline”, Level A and YoY returns the appropriate three annual rows (including the separately checked Table 5 gasoline observation).
- All five animations were individually tested at their final timeline stage, restarted to stage 1, played (button becomes Pause) and paused (button becomes Play). Stage totals are 5, 5, 4, 4 and 10 respectively.
- Increasing sample size to 180 changes the simulated estimator dispersion. Drawing again changes the sample seed and estimate. These are explicitly synthetic.
- At 390 pixels, equation overflow originally expanded the document. Equation boxes were corrected to contain scrollable mathematics. A second check caught a grid child’s intrinsic table width; setting its minimum width to zero resolved the last 11 pixels of overflow. Retest: document scroll width equals client width (375 pixels inside the 390-pixel viewport with its scrollbar). Widths 768, 1024 and 1440 likewise have no document overflow. Wider categorical plots use explicitly labelled contained scrolling; line and bar plots render at mobile width.
- Short equation boxes were tightened to content width with symbol definitions immediately below. Narrow treemap labels were resized and checked for clipping. On phones the accurate 100% stack and numerical table replace the treemap; the treemap is not converted to misleading equal-area tiles.
- Static checks found all 24 unique local page dependencies/linked artifacts and all 60 referenced font files. All 30 original input files (including the SDMX structure) pass their SHA-256 integrity checks.
- All 12 distinct external reference URLs returned HTTP 200 after redirects in the recorded HEAD-request smoke test (`reference-links.json`). The originally supplied methodology index URL was unavailable; the page links the working Chapter 8 and preserved complete PDF instead.

## Verification boundaries

This is manual, browser-assisted functional QA, not an automated cross-browser compatibility suite or an assistive-technology certification. Numerical empirical checks and original-file integrity checks are reproducible Python scripts. External source availability may change after the recorded acquisition. Static references and data remain local.

The preview browser blocks `file://` navigation by policy and permits HTTP/HTTPS only. No alternate browser or policy workaround was used. Direct-file rendering is therefore not browser-verified; the complete static page and relative assets were tested over local HTTP. No public deployment was made or tested.
