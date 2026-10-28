# Hypergraph tutorial refinement

Implemented locally on 2026-10-27. Scope is the tutorial HTML and its existing asset folder. Nothing was staged, committed, pushed, or published. `research.html` is unchanged. The ten original PNGs, original generators, datasets, earlier reports, and earlier preview files remain intact.

## Files and implementation

- `../2026-04-21-hypergraphs-visualization.html`: animation markup, precise introductory definitions, membership flow, numerical derivations, and supplementary explanations.
- `style.css`: compact story panels, consistent edge colours, matrix and theorem treatments, responsive cards, and corrected paragraph widths.
- `math.js`: deduplicated `twoSection` and exact `edgeIntersections` helpers.
- `tutorial.js`: shared positions and colours, exact incidence selection, draft-edge cardinality, live projection counts and optimum witnesses, compact examples, and removal of the old animation controller.
- New `story.js`: pure, independently testable timeline derived from `HyperMath.example()`. Mathematical snapshots are frozen and isolated from builder edits.
- New `intro.js`: animation controller using cached SVG elements and one requestAnimationFrame loop. Annotation HTML updates only when the stage or active column changes; geometry is not rebuilt per frame.
- Updated `validate_browser.cjs` and `validate_site.py`; added `validate_story.cjs` and `validate_refinement.cjs`.
- New `refinement-*-audit.json` files and `refinement-*` screenshots record this refinement without replacing earlier evidence.

## Timeline and controls

The timeline is exactly **18 seconds at 1×**. Installed Edge measured **18.0304 seconds from the actual click to the concluding frame**, including frame scheduling. All eight stages were observed in order.

| Time | Mathematical development |
|---|---|
| 0–1.5 s | Six fixed vertices, empty edge family |
| 1.5–3.5 s | One triple versus its three projected pairs |
| 3.5–5.5 s | Add the pair BD; distinguish edge cardinalities |
| 5.5–8 s | Add CDE; show shared vertices |
| 8–10.5 s | Add AEF; complete the mixed-size family |
| 10.5–13 s | Scan the four incidence columns and matching spokes |
| 13–15.5 s | Highlight each edge's contribution to the deduplicated 2-section |
| 15.5–18 s | Contrast one triple with three pair edges having the same triangle projection |

The exact title “Animation Video for Hypergraphs” uses bold peach Garamond with serif fallbacks, centered above the panels. Desktop computed size is 26.4 px, matching the requested suggested clamp. Graph panels have a maximum width of 440 px and SVG height of 320 px.

There is one 0–18 slider, Play/Pause, Restart, Previous/Next step, and speeds 0.5×, 1×, 1.5×, and 2×. No autoplay. Manual steps show complete, readable states. Playback pauses offscreen or when the document becomes hidden. Reduced-motion mode removes progressive drawing while retaining manual controls. Reverse scrubbing reconstructs the appropriate sets, lines, captions, matrix visibility, and column highlights.

## Validated running example

V={A,B,C,D,E,F}, with e1={A,B,C}, e2={B,D}, e3={C,D,E}, e4={A,E,F}.

- Edge sizes: (3,2,3,3).
- Vertex degrees: (2,2,2,2,2,1).
- Incidences: 11 spokes, 11 matrix ones, and both degree/cardinality sums equal 11.
- Distinct projected pairs: AB, AC, AE, AF, BC, BD, CD, CE, DE, EF.
- Matrix rows: A=(1,0,0,1), B=(1,1,0,0), C=(1,0,1,0), D=(0,1,1,0), E=(0,0,1,1), F=(0,0,0,1).

These values derive from the shared mathematical object. Independent tests check the expected matrix, pair list, and cardinalities. Projection is also checked on all 128 simple hypergraphs on three vertices. The separate family {ABC,ABD} verifies that six pair contributions can collapse to five distinct pairs.

## Mathematical audit and additions

| Section | Audit outcome and refinement |
|---|---|
| 1. Ordinary graphs and hypergraphs | Defines C(V,2), the power set, distinct nonempty edges, and permitted singletons. Adds pair, triple, overlapping-group examples and the projection counterexample. |
| 2. Builder | Same default vertices, positions, notation, and colours as the animation. Draft set and cardinality are explicitly separate from committed edges. Dense-layout switching is explained. |
| 3. Definitions | Retains the stated simple/indexed conventions. Adds a worked induced-subhypergraph example distinguishing retention from edge truncation. |
| 4. Incidence | Preserves double counting. Adds the vertex → subset → incidence → matrix flow and expanded row/column sums. Selecting C∈e3 identifies exactly C, its hub, its spoke, and its matrix cell. |
| 5. Representations | Defines the 2-section precisely, derives adjacency from off-diagonal entries of BBᵀ, reports live pairs, and distinguishes multiplicity from simple adjacency. |
| 6. Complete uniform families | Expands C(5,3), C(4,2), and the 30-incidence identity. Explicitly restricts equal-degree formulas to the complete family. |
| 7. Intersections | Derives (BᵀB)ij=|ei∩ej| with concrete intersections; retains the pairwise/global-intersection counterexample. |
| 8. Duality | Adds a row-to-column calculation for C and the degree/cardinality exchange. Retains indexed duplicate and empty dual edges and the double-transpose caveat. |
| 9. Paths and cycles | Adds a verified length-two Berge path and its four incidence links. Retains the distinctness requirements and connectivity proof. |
| 10. Matching/transversals | Preserves ν≤τ≤rν and its proof. Lists computed optimum witnesses; verifies {A,D} using disjoint e2/e4 as a lower bound. |
| 11. Property B | Expands the Fano contradiction numerically: nine same-colour pairs versus seven mixed lines. Retains the unique-pair hypothesis and distinction from incidence-graph bipartiteness. |
| 12. Sperner/LYM | Preserves the chain-counting proof. Adds 6·(2!2!)=4! and a non-antichain with Lubell sum 5/12, so the inequality is not misused as a converse. |
| 13. Erdős–Ko–Rado | Retains n≥2k prominently and the accurately labelled circle-method proof sketch. Adds n=5,k=2 equality and n=3,k=2 failure outside the hypothesis. |
| 14. Interval Helly | Retains finite, nonempty, closed-interval assumptions and the endpoint proof. Adds L=3,R=6 and explains why L>R produces a disjoint pair. |
| 15. Random hypergraphs | Expands the indicator calculation through expectation and finite summation. Distinguishes exact expectations 6 and 3 from individual samples and seeded empirical means. |
| 16. Applications | Reviewed and retained: examples explain why group identity matters without making additional theorem or empirical claims. |

All existing general proofs remain. Finite exhaustive checks remain identified as verification of their instances; the EKR circle lemma is not presented as proved by enumeration. No new external paper claims were introduced.

## Visual and wrapping changes

The broad `p { max-width:78ch }` rule imposed a second width limit inside already constrained section and grid layouts, leaving unused horizontal space. The refinement overrides that cap and lets each containing card or column determine paragraph width. Formula panels remain bounded, centered, and able to wrap. Grid children use zero minimum widths so long content does not force the page wider. Animation annotations and matrices have explicit responsive grids; mobile panels and comparison cards stack. Dense matrices retain local scrolling rather than clipping content or hiding page overflow.

New illustrations are SVGs computed from exact set membership: compact pair/triple/overlap examples, the three-panel information-loss comparison, and the interactive membership chain. Existing useful figures remain; shared edge colours and focused incidence highlighting improve their consistency. Desktop, tablet, and mobile screenshots were visually inspected, including the story, matrix section, Sperner section, and both sets of new comparison figures.

## Validation

Run from the graph-theory directory with an existing repository-root HTTP server on `127.0.0.1:8765`:

```powershell
node .\2026-04-21-hypergraphs-visualization\validate_math.js
node .\2026-04-21-hypergraphs-visualization\validate_story.cjs
python .\2026-04-21-hypergraphs-visualization\validate_site.py
$env:NODE_PATH='C:\Users\user\.cache\codex-runtimes\codex-primary-runtime\dependencies\node\node_modules'
node .\2026-04-21-hypergraphs-visualization\validate_browser.cjs
node .\2026-04-21-hypergraphs-visualization\validate_refinement.cjs
```

- Existing mathematics: **69,461 assertions passed**, including all 128 three-vertex hypergraphs, all 65,536 four-element set families, Fano colourings, and seeded simulations.
- New story/projection mathematics: **3,084 assertions passed**. Combined total: **72,545**.
- Browser regression: builder create/replace/remove, invalid inputs, labels, empty families, incidence counts, duality, paths, exact/greedy optimization, uniform enumeration, overlap, Fano, lattice, three EKR instances, interval endpoints, and random experiments passed. Keyboard focus and all ten gallery image decodes passed. No console/runtime or missing-resource errors.
- Animation audit: every stage and matrix-column scan checked forward and backward; actual 1× completion measured; all four speeds and pause checked with a virtual clock; offscreen pause and reduced motion checked. Builder edits do not mutate the animation.
- Responsive audit: **1920, 1440, 1366, 1024, 768, and 390 px** all passed checks for page overflow, text/formula clipping, and SVG text outside its viewBox. The regression also checks a twelve-edge builder and expanded gallery at 320 px.
- Static audit: balanced HTML, **133 unique IDs**, **41 local links**, original ten PNG hashes unchanged, `research.html` unchanged, no old demo filename, and no external runtime dependency.
- Relevant diffs and whitespace checked. Earlier untracked research directories were left untouched.

Limitations: browser automation used installed Microsoft Edge, not a cross-browser matrix. Hidden-tab handling was verified with a simulated visibility event; offscreen pausing used actual scrolling. Interactive figures require JavaScript; static definitions, proofs, and the original gallery remain readable without it. These changes have not been published.

## Git status

Final scoped `git status --short` output (paths relative to `research/graph-theory`):

```text
 M 2026-04-21-hypergraphs-visualization.html
 M 2026-04-21-hypergraphs-visualization/math.js
 M 2026-04-21-hypergraphs-visualization/style.css
 M 2026-04-21-hypergraphs-visualization/tutorial.js
 M 2026-04-21-hypergraphs-visualization/validate_browser.cjs
 M 2026-04-21-hypergraphs-visualization/validate_site.py
?? 2026-04-21-hypergraphs-visualization/REFINEMENT_REPORT.md
?? 2026-04-21-hypergraphs-visualization/intro.js
?? 2026-04-21-hypergraphs-visualization/refinement-animation-audit.json
?? 2026-04-21-hypergraphs-visualization/refinement-counterexample.png
?? 2026-04-21-hypergraphs-visualization/refinement-examples.png
?? 2026-04-21-hypergraphs-visualization/refinement-incidence-1440.png
?? 2026-04-21-hypergraphs-visualization/refinement-incidence-390.png
?? 2026-04-21-hypergraphs-visualization/refinement-incidence-768.png
?? 2026-04-21-hypergraphs-visualization/refinement-preview-1440.jpg
?? 2026-04-21-hypergraphs-visualization/refinement-preview-390.jpg
?? 2026-04-21-hypergraphs-visualization/refinement-preview-768.jpg
?? 2026-04-21-hypergraphs-visualization/refinement-preview-builder.jpg
?? 2026-04-21-hypergraphs-visualization/refinement-preview-colouring.jpg
?? 2026-04-21-hypergraphs-visualization/refinement-preview-sperner.jpg
?? 2026-04-21-hypergraphs-visualization/refinement-regression-audit.json
?? 2026-04-21-hypergraphs-visualization/refinement-site-audit.json
?? 2026-04-21-hypergraphs-visualization/refinement-story-1440.png
?? 2026-04-21-hypergraphs-visualization/refinement-story-390.png
?? 2026-04-21-hypergraphs-visualization/refinement-story-768.png
?? 2026-04-21-hypergraphs-visualization/refinement-story-desktop.png
?? 2026-04-21-hypergraphs-visualization/story.js
?? 2026-04-21-hypergraphs-visualization/validate_refinement.cjs
?? 2026-04-21-hypergraphs-visualization/validate_story.cjs
```
