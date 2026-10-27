# Development report — October 27, 2026

Created the requested primary HTML beside the project asset directory. Added the approved bottom Graph Theory details/summary section to the root research.html, with the requested entry title and description. Its URL includes research/ because the index is at the repository root.

The four source groups became:

1. Spanning forests directed toward roots.
2. Ring-sum and edge-disjoint star products.
3. Generating directed trees by cancellation.
4. Generating k rooted components.

Transcribed 1a, 1b, 1c, 1d, 1f, 2a, 2b, 3a, 3b, 3c, 3d, 4a and 4b into HTML/LaTeX. All 13 PNG hashes are unchanged. No 1e exists. All notation was readable. The cropped ending of 3d remains cropped and the PDF continuation is separately labelled. Figure 1's topology and all edge labels are represented in SVG and an audit table.

Created three introductory graph comparisons, reversible-cycle demonstration, incremental proof, six-stage generation animation, six-forest source gallery and an editable graph laboratory. The lab supports 3–6 vertices, preset graphs, arbitrary prescribed root sets, edge toggles, candidate browsing, symmetric completion and exact determinant comparisons.

Python verification passed 643 exact graph/root cases using edge-subset/union-find enumeration, parity construction and Fraction-based determinants. Node passed 463 exhaustive/source cases, all 18 printed edge products, and a two-disjoint-cycle cancellation test. The source examples produce 12 trees and 6 two-trees.

Browser QA checked desktop (1280×900), mobile (390×844), 327 MathJax expressions without errors, zero console warnings/errors, controls and keyboard operations, original scan loading, source graph counts, six-vertex complete graph (1296 rooted trees), empty-edge boundary cases, and the expandable index link. No page-wide horizontal overflow was observed. The screenshot is retained in figures/laboratory-preview.jpg.

Static checks verified 35 local links/anchors in the article and 40 local links in research.html, balanced HTML, unique IDs and preservation of every pre-existing index entry. JavaScript syntax and git diff --check passed.

## Limitations disclosed in the article

- The paper's abstract star algebra requires an explicit operational interpretation: edge identifiers may not repeat, vertices may be shared, and expansion occurrences are retained until parity cancellation.
- Multiple disjoint cycles can yield multiplicity 2^q rather than exactly two; the added involution argument explains the cancellation.
- Experiments are finite, unit-weight and loopless. They do not prove all general claims or implement the later weighted electrical-network applications.
- MathJax loads from a CDN; graphs/counts require JavaScript. Exact model and reasoning settings were not reliably exposed.

## Files and Git state

Modified tracked file: research.html (14 added lines). New untracked output: the main HTML and scoped css/, js/, data/, figures/ and validation/ files. Supplied Verbatim/ and chen1966.pdf were already untracked before this task and are preserved. Other pre-existing untracked projects were untouched. Nothing was staged, committed, pushed or deployed.

## Open locally

From research/graph-theory:

    python -m http.server 8000 --directory ../..

Open http://localhost:8000/research/graph-theory/2026-10-27-directed-k-trees-digraph-empirical-demonstration.html.

The temporary QA server was stopped after verification. See README.md for exact rerun commands and data/ for machine-readable results.
