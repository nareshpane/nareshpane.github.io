# Empirical Demonstration of Graph Automorphism and Isomorphism

Local review draft, expanded October 29, 2026. The public page remains one level
above this directory, under its original filename. No index entry or publication
was added; no build step or npm dependency is required.

## Local preview

From the repository root, run `python -m http.server 8765 --bind 127.0.0.1` and open:

`http://127.0.0.1:8765/research/graph-theory/2026-04-22-empirical-demonstration-graph-isomorphism.html`

From the graph-theory directory, the same command serves the page at
`http://127.0.0.1:8765/2026-04-22-empirical-demonstration-graph-isomorphism.html`.
Parent navigation to Home/Research needs the repository-root server.

Text, figures, MathJax, and synchronous demonstrations can also be read directly
from the HTML file. Workers and catalogue fetches need HTTP. Browser security
restrictions on `file:` are reported rather than silently replaced with fake data.

## Structure and preservation

- Both original PDFs remain directly here under their exact original filenames.
- `figures/historical/graph-views/`: 29 original numbered graph PNGs.
- `figures/historical/partition-views/`: their 29 original paired PNGs.
- `python/historical/`: both original generator scripts, unchanged. The older
  `generate_automorphism_demo.py` is a cycle prototype; the `_fixed.py` script is
  the prism generator. They are retained as distinct historical sources.
- `data/historical/`: original prism automorphism and colouring data.
- `js/`: dependency-free graph kernel, renderers, UI, worker, and animation.
- `css/research-page.css`: the responsive cream/green/peach research design.
- `js/vendor/tex-svg.js`: MathJax 3.2.2 combined TeX/SVG distribution, with its
  license retained alongside it. Mathematical typesetting works offline.
- `data/generated-graphs/graph{6,7,8}.g6`: unmodified official McKay catalogues.
- `data/validation/`: original asset inventory and actual validation reports.
- `python/`: independent standard-library computation and validation scripts.

`original-asset-inventory.json` records hashes and original/new paths for all 64
original root files (58 PNGs, 2 PDFs, 2 scripts, 2 data files). The migration moved
assets only within this directory. The original HTML's 29 notebook entries remain
in the page as individually expandable sections. All original PNGs, scripts,
data, and PDFs are preserved byte-for-byte. No duplicate old-location PNGs remain.

## Mathematical contracts

Internal graph vertices are zero-based; visible labels are one-based. Graphs are
undirected, simple, loopless, and represented by binary adjacency matrices.
Permutations map old vertex labels to new labels, and composition is a right
action: `compose(p,q)[v] = q[p[v]]`.

The matrix convention is explicitly **source rows, target columns**:
`P[i][f(i)] = 1`, so `A_G = P A_H P^T`. Therefore
`P_p P_q = P_(q composed with p)`. This is deliberately distinct from the
column-vector convention. `matrix-operations.js` computes products, differences,
walk counts, equitable quotient counts, and prism operations. The renderer uses
these same graph arrays; no stored animation values stand in for calculations.

The six-vertex worked witness is `(1,5,6,3,2,4)`. Graph G's sixth row is
`(0,0,1,1,1,0)`; H's fourth row is `(0,1,1,0,0,1)` and becomes the former when
its columns follow the witness. All 15 unordered pairs are checked. Square,
triangular-prism, hexagonal-prism, composition, block-matrix, walk-count, clique,
and matrix-action demonstrations extend this language through the existing
sections. Layer colours are explanatory, not prescribed fixed colours for the
uncoloured automorphism counts.

The larger matrices use labelled SVG heatmaps with exact-entry tooltips and an
expandable numerical table. The two C4 pruning trees are stacked and fit their
containers; a deliberate readable-size/pan control is available on narrow screens.

Run `node js/validate-matrices.cjs --write` for the 4,033 additional mathematical
and renderer assertions. The existing 51,830 kernel assertions and independent
Python validation also pass. See `data/validation/paper-coverage-audit.md` for the
source-by-source refinement audit and `data/validation/refinement-browser-validation.json`
for the new interaction and responsive checks.

Colour cells are ordered. Transport sends each cell through a permutation.
Ordered refinement preserves all old colour inequalities, not only containment.
The active-cell refiner reconstructs II Algorithm 1 using FIFO splitters,
ascending neighbour-count fragments, and first-largest omission on ties. Root
refinement activates every cell; incremental refinement after individualization
of an equitable state activates the new singleton only. These choices are
label-equivariant and tested independently against synchronous closure.

Isomorphism search uses necessary invariants and degree/colour-compatible
backtracking, checking edges and non-edges on every partial assignment. A witness
is verified; an exhausted search rejects; a bound interruption is undetermined.

There are **two different exact canonical conventions**:

1. Maximum upper-triangle code over all permutations (baseline, n <= 8).
2. Maximum code over every discrete leaf of the specified equivariant IR tree.

Their outputs need not match each other, but each is invariant across all
relabellings of a graph. Browser comparisons use the same convention on both
inputs. Canonical tools canonize uncoloured graphs; formal coloured-pair
definitions are developed in the text without claiming the UI implements a full
coloured-graph canonizer. The result is canonical only when search completes.

P_C pruning uses verified supplied automorphisms in the prefix's pointwise
stabilizer. A deleted sibling carries its actual permutation witness. Group
discovery cost is separate. Full small groups are enumerated, and generating
sets are verified by exact closure. P_A and P_B are explained with their formal
conditions but are not executed by this kernel.

## Generation and limits

- Live A: enumerate every labelled candidate and reject via pairwise exact GI.
- Live B: enumerate candidates and deduplicate exact IR canonical keys.
- Both support n=1..5 (at most 1024 candidates), in cancellable Web Workers.
- n=6..8 loads official catalogues: 156, 1044, and 12346 representatives.
- Atlas pages contain at most 24 cards; filters use edge count/connectedness.
- Canonical augmentation is a rigorous conceptual edge-construction scheme,
  **not an implementation of geng**.
- Imported graphs: at most 12 vertices; 250000 search nodes per job.
- Worker jobs: 20-second wall-clock cutoff; cancellation terminates the worker.
- Displayed trees: at most 350 nodes (strategy comparison: 120), with truncation
  explicitly labelled. Advanced 16-vertex examples use selected refinement paths.
- The 90-second overview is a seekable deterministic original SVG timeline,
  starts paused, has exactly one timeline slider, speed controls, chapter markers,
  and optional fullscreen. It is labelled explanatory rather than a solver video.

## Source fidelity and substantive corrections

The first PDF has 43 scanned pages (printed pp.45-87). It was read through OCR
with mathematical definitions checked against rendered pages. The second local
PDF has 22 pages and a **27 November 2024 revision footer** despite its filename.
References use its internal page numbers and distinguish the 2014 publication
from the 8 January 2013 arXiv submission.

The unified crosswalk preserves I's partition nests, Lambda indicator, original
graph-valued C and its C1-C3 properties; II uses sequence-based R, target T,
node invariant phi, and pair-valued C. I's refiner R and II's F are distinguished.

Two printed issues are documented rather than silently propagated:

- II section 2.2(c) prints a colouring action without the inverse needed for
  forward cell transport. The page explicitly declares the cell-transport
  convention `pi^g(v^g)=pi(v)` and records the printed discrepancy.
- The local II Theorem 5(b)'s restriction uses a different node variable from
  that invoked by its proof's P_B argument. The page states the sufficient
  reference-class preservation condition and executes only P_C.

The original notebook's quoted-looking paragraphs are now labelled paraphrases.
Its entries 10-11 gain the ordered-refinement requirement; entry 22 is annotated
for the transport convention; entries 27-29 explicitly concern colourings of a
fixed labelled prism under Aut(P), not a general S_12 canonizer. Historical PNGs
are unchanged, so their inherited annotations must be read with these corrections.

II Figure 1 is reconstructed as a 9-vertex, 16-edge graph with 8 automorphisms;
source labels 0-8 are shifted to 1-9. The root initial cell order in our live
tree is degree-based; its numbering/selector is not claimed to duplicate the
paper's exact tree. I's Q5 and C5[C5] examples are separately reconstructed and
verified structurally. The 12-vertex prism is the user's original educational
example, not attributed to a figure in either paper.

Full production nauty/Traces, Algorithm 2.31, implicit automorphism cases,
Schreier maintenance, low-degree shortcuts, trace-dominance aborts, experimental
path scheduling, and geng are not implemented. The prose explains those
dependencies and their proof obligations. The traversal comparison isolates DFS
versus level order; it is not an actual solver benchmark.

## Reproduce validation

Run from this directory:

```text
python python/validate_examples.py
node js/validate-mathematics.cjs --write
python python/audit_page.py
```

The Python script has no external dependencies. It independently computes
Burnside counts through n=8, exhaustive permutation class counts through n=5,
S6 canonical uniqueness for every order-six catalogue entry, small-family group
orders, the Figure 1 quotient, the rook/Shrikhande properties, all 3840 cube
coordinate/translation symmetries, and the lexicographic-product block argument.

The JavaScript suite tests every labelled graph through n=5 for refinement,
automorphisms, seeded relabelling invariance, exact GI, and canonical preservation
under P_C. It checks both live generation methods, independent synchronous
refinement, coloured-state equivariance, target selections, exact group closure,
invalid input and incomplete searches. Every catalogue representative through
n=8 has a completed unique IR key. The recorded initial suite passed 51830
assertions. See reports for exact results and versions.

`python/regenerate_historical_figures.py OUTPUT_DIRECTORY` optionally regenerates
the old prism PNGs into a **separate review directory**, using the preserved fixed
generator. Only this optional historical workflow needs matplotlib, numpy, and
networkx. The site and new validators do not depend on them. Reorganized source
locations therefore do not break the route to regenerating the old figures.

## External data and licenses

Official graph source: https://users.cecs.anu.edu.au/~bdm/data/graphs.html
Direct catalogue URLs and SHA-256 values are recorded in `python-validation.json`.
The independent count and uniqueness tests validate the contents as small graph
catalogues, rather than trusting filenames alone. MathJax is distributed under
Apache License 2.0; see `js/vendor/LICENSE.mathjax.txt`.
