# Implementation report — October 29, 2026

## Refinement revision

The existing 20-section structure, 29 notebook entries, 58 historical figures,
primary PDFs, educational algorithms, and controls are retained. Only the named
HTML page and its dedicated asset directory were changed. Nothing was staged,
committed, pushed, added to the research index, or published.

**Files.** Updated the HTML, `css/research-page.css`, and existing
`js/graph-renderer.js`, `js/interactions.js`, and `js/animations.js`.
Added `js/matrix-operations.js`, `js/matrix-renderer.js`, `js/matrix-labs.js`,
and `js/validate-matrices.cjs`. Updated the README and validation reports;
added the detailed paper coverage audit and refinement browser report under
`data/validation/`. The source PDFs remain byte-for-byte unchanged.

**Opening layout.** Applied the exact requested title, linked subtitle, and five
metadata fields. Reduced title offsets, enlarged the metadata to 270px/12px,
removed the introductory line/dot decoration, and retained the cream, spring-green,
and peach palette. The heading balances into two desktop lines and wraps naturally
on mobile; the metadata moves into normal flow.

**Matrix foundation.** Explained adjacency entries, symmetry, zero diagonal,
neighbour rows, and degree sums on P4. All permutation calculations use source
rows and target columns, `P[i,f(i)]=1`, so `A_G=P A_H P^T`. The six-vertex witness
has the requested row-6 / row-4 calculation. Every proposal recalculates all pairs,
P, the product, and the signed difference; graph vertices, matrix rows/columns,
selected entries, mapping dropdowns, and row-comparison playback stay synchronized.
Identity, rows-only swapping, both-axis swapping, and composition are worked live.

**Automorphisms.** Introduced the square's eight operations intuitively before
the prism counts; its selector includes an invalid bijection. Added the complete
12-operation triangular-prism atlas, layer block matrix, animation, and all twelve
distinct/exhaustive isomorphisms to a fixed H via h∘a. Added five-step hexagonal-prism
construction and all 24 rotation/reflection/layer choices with computed matrix
actions. Explained Cartesian products, dihedral order, completeness, and the cube
exception to naive factor counting. Layer colours are illustrative, not fixed
colours for these uncoloured group counts.

**Integration.** Added meaningful matrix interpretation in invariants, groups,
partitions, refinement, individualization, selected search states, canonical
labelling, pruning, and generation. Added A² walk counts, trace(A³)/6, spectral
reasoning, a clickable rook board, synchronized 16-vertex heatmaps, principal
submatrices, and all-1820-subset clique checking. Large matrices offer detailed
entries; small tables remain visible. Refinement now has calculated intermediate
states, before-pass cells, play/pause, restart, previous/next, speed, and one slider.
The original overview now includes a brief graph/matrix/permutation beat.

**Pruning and typography.** Stacked both C4 trees at full content width, added
sequence correspondence and explicit absent-branch explanations, and separated
expanded internal states (2), retained leaves (1), and pruned witnesses (4) in the
seven-state supplied-group run. A readable-size/pan mode complements fit-to-view.
Removed accidental equation overflow caused by assistive MathML while retaining
screen-reader mathematics. Wrapped long equations at logical boundaries; C1–C3
stay aligned and fully visible. Moved matrix captions outside narrow table widths
and removed restrictive prose widths. The long historical permutation display
retains necessary horizontal scrolling only on a phone.

**Source audit.** Re-examined original definitions and implementation discussions.
Added I Theorem 2.34's generator-output qualifications, coloured incidence-matrix
and Hadamard reductions, II's quotient edge-total versus per-vertex-count
distinction, early automorphism extensions, group-only discrete-child reference
strategy, and actual low-degree target exclusions. The source-by-source table is
`data/validation/paper-coverage-audit.md`. Independent teaching examples remain
clearly distinguished from source examples and full production algorithms.

**Verification.** All 51,830 existing kernel assertions and 4,033 new matrix and
renderer assertions pass (55,863 total). New checks cover all 720 six-vertex
proposals, exact prism groups, all block entries, all square products, twelve h∘a
witnesses, walk/triangle oracles, SRG identities, clique checks, C4 pruning, and
selected-row rendering. Independent Python checks still pass, including original
asset hashes. Browser tests verified all six selected-vertex correspondences,
valid/invalid mapping updates, matrix products, playback, canonical and candidate
synchronization, groups/stabilizers, pruning correspondence, worker generation
(34 classes at n=5), catalogue loading (156 at n=6), and imported-graph GI.
Responsive checks cover 1920, 1440, 1280, 1024, 768, and 390px: no page overflow,
vertical equation scrollbar, math error, or clipped fitted pruning tree. All ten
overview chapters fit the mobile player; console checks were clean.

**Limits.** Native 125%/150% zoom shortcuts are unavailable in the in-app preview;
the corresponding 1440px-layout equivalents (1152px and 960px) were checked.
Production nauty/Traces machinery remains outside this educational implementation,
as detailed below. Worker and catalogue tools need an HTTP preview; direct-file
reading supports text, figures, typesetting, and synchronous laboratories.

The earlier expansion record follows for provenance.

## 1. Mathematical sections

Expanded the existing page into 20 connected sections: the paper crosswalk,
isomorphism and invariants, automorphism groups, ordered colourings, equitable
refinement, individualization, search trees, canonical forms, pruning, the 1981
algorithmic foundations, the later framework and Traces, direct comparison,
non-isomorphic generation, a graph builder, empirical evidence, computational
limits, the retained prism notebook, glossary, and primary references.

Definitions are interspersed with worked examples and live tools. Collapsible
formal derivations explain equitable closure, equivariance, stabilizers,
canonical selection, pruning preservation, and canonical augmentation.

## 2. Connection between the papers

A 19-row crosswalk translates partition nests and Lambda indicators into the
later sequence-based R/T/phi framework. It distinguishes I's graph-valued
canonical C with three properties from II's coloured-pair C with two properties,
and distinguishes I's active-cell R from II's F. Exact source locations and
source-to-implementation correspondence are in the page. Printed convention and
node-variable issues in the supplied later PDF are explicitly discussed.

The local second PDF's 2024 revision footer is recorded separately from its
filename, the 2013 preprint history, and the 2014 journal publication.

## 3. Animation and interactions

- Original deterministic 90-second, ten-scene SVG introduction: play/pause,
  restart, one seek slider, clock, speed, chapters, and optional fullscreen.
- Graph/matrix/mapping checker, exact group/orbit/stabilizer explorer, ordered
  partition editor, active-cell refinement with signatures and queue events.
- Vertex individualization with actual propagated states, clickable search
  nodes and collapse/expand, explicit P_C permutation witnesses.
- Exact IR canonical comparison and exhaustive S_n baseline; two live generation
  methods; paginated, filterable graph atlas and candidate-to-class inspection.
- Graph builder with edge toggling, edge/matrix imports, worker-based exact GI,
  canonical calculations, cancellation, and incomplete-result safeguards.
- Synchronized playable DFS/level-order comparison on an actual computed tree.

## 4. Asset organization and preservation

Both PDFs remain unchanged at the dedicated asset root. All 58 original PNGs
were moved into graph-view and partition-view subfolders under
`figures/historical/`. Original scripts and data moved into `python/historical/`
and `data/historical/`. All 64 original root assets retain their hashes.
The 29 original HTML notebook entries remain in expandable sections; genuine
scope/formula problems have explicit corrections. No old-location PNG copies
remain. Historical regeneration has a wrapper for a separate review output.

New source modules, CSS, a local MathJax bundle/license, official catalogues,
independent Python scripts, and validation reports are retained in this folder.

## 5. Generation methods and limits

A enumerates labelled candidates and uses pairwise exact GI; B deduplicates exact
IR canonical keys. Both run live at n=1..5 (at most 1024 candidates). Official
McKay graph6 catalogues serve n=6..8 (156, 1044, 12346 representatives), with only
24 atlas cards rendered per page. Canonical augmentation is explained through
an invariant deletion-orbit construction, not claimed as implemented geng.

User inputs are limited to 12 vertices, searches to 250000 nodes, and worker jobs
to 20 seconds. Displayed trees have 350 nodes maximum. Interrupted work is never
presented as an exact canonical form or a non-isomorphism proof.

## 6. Mathematical validation

The JavaScript suite passed **51830 assertions**. Every labelled graph through
n=5 was checked for refinement, equivariance, exact GI, automorphisms, and P_C
canonical preservation; both generation methods match known counts. Every
official catalogue representative n=6..8 has a completed distinct IR key.

Independent Python standard-library calculations verified Burnside counts
1,2,4,11,34,156,1044,12346; exhaustive S_n class counts through n=5; S6 uniqueness
of all 156 order-six representatives; nine graph-family group orders; II Figure
1's quotient; rook/Shrikhande SRG parameters and 4-clique distinction; all 3840
cube coordinate/translation symmetries; and the C5[C5] block argument.

The Figure 1 reconstruction initially failed both group checks; correcting the
edge pairing against the source diagram resolved it. The saved reports contain
actual successful results, rather than the initial expectation.

## 7. Deliberate implementation boundaries

This is an exact educational kernel, not production nauty or Traces. Full
Algorithm 2.31, implicit-automorphism shortcuts, Schreier group maintenance,
low-degree special cases, P_A/P_B execution, full trace dominance and aborts,
experimental-path scheduling, and geng/canonical augmentation are explained but
not implemented. These require substantial solver engineering and additional
verification beyond the bounded interactive reconstruction.

The live canonical tools operate on uncoloured inputs. The text develops the
full coloured-pair definition and its proofs; the UI does not silently claim
to implement it. Exhaustive S_n and exact IR use different canonical selection
conventions and are not incorrectly required to return the same representative.

## 8. Files changed

Only `../2026-04-22-empirical-demonstration-graph-isomorphism.html` and this
dedicated directory were changed. Major additions are `js/*.js`,
`js/validate-mathematics.cjs`, `css/research-page.css`, `python/*.py`,
`data/generated-graphs/*.g6`, and `data/validation/*`, plus this report and README.
Original asset moves are recorded in `original-asset-inventory.json`.
Other projects and repository index pages were untouched. Nothing was staged,
committed, pushed, or published.

## 9. Browser verification and remaining technical limits

Tested over local HTTP in the Codex browser at desktop and 390-pixel mobile
widths. Local MathJax rendered 422 math containers with no typesetting errors.
Representative tests exercised mapping changes, keyboard-selected graph
vertices, all major mathematical controls, actual worker completion and
cancellation, invalid matrix rejection, candidate classification, atlas filters,
and animation playback/seeking. Both live n=5 methods produced 34 classes; the
n=8 connected filter produced 11117 graphs. Formula containment fixed the
mobile horizontal overflow found during QA.

Direct-file mode cannot load worker/catalogue features in browsers that restrict
file URLs; the README supplies the local HTTP command. Wide tables, long
equations, and trees intentionally scroll within their own panels. The overview
is a self-contained animation, not an MP4/WebM recording. Production solver
benchmark timings and later complexity-theory surveys are not fabricated.

See `data/validation/asset-audit.json`, `http-audit.json`,
`javascript-validation.json`, `python-validation.json`, and `browser-validation.json`
for the saved final checks and source provenance.
