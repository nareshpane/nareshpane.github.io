# Hypergraphs redesign audit

Final page: `research/graph-theory/2026-04-21-hypergraphs-visualization.html`.
Assets: `research/graph-theory/2026-04-21-hypergraphs-visualization/`.

## Scope and preservation

The original page contained ten static examples and no interactive JavaScript algorithms. Its mathematical material covered set-family notation, singleton edges, varying edge sizes, overlaps, isolation, incidence representations, and clique expansion. These remain in the collapsible original gallery and the expanded tutorial. All ten original PNG files are byte-for-byte unchanged. Their original generating Python script is retained; its output directory now resolves to its own renamed asset directory rather than a working-directory-dependent old name.

The old HTML and asset directory no longer exist. The existing Hypergraph entry remains in the same collapsible Graph Theory subsection of `research.html`, with the requested title, description, and new relative link. A programmatic comparison confirms the rest of that index is unchanged. No other tracked research pages or settings were modified. Five unrelated untracked research directories present before this work were preserved. The branch was `main`, initially synchronized with `origin/main`, with no pre-existing staged changes or unpushed commits. A fresh fetch before committing also reported zero divergence.

## Educational sequence

1. Ordinary edges versus group relationships, using the same four vertices.
2. A six-vertex editable hypergraph with exactly the requested four default edges.
3. Order, size, degrees, rank, uniformity, isolation, subhypergraphs, and explicit conventions.
4. Incidence matrices and a double-counting proof of the incidence identity.
5. Synchronized set, matrix, and bipartite representations, plus a noninjective 2-section counterexample.
6. Complete uniform families, exact binomial counts, and degrees.
7. Disjoint, common-vertex, pairwise-only, and nested intersection configurations.
8. Duality as transposition, retaining indexed repeated and empty dual edges.
9. Berge paths, Berge cycles, components, and shortest-path tracing.
10. Exact maximum matchings, exact minimum transversals, greedy comparison, and proofs of both rank bounds.
11. Property B, direct colouring, Fano pair uniqueness, and a same-colour-pair counting proof of non-2-colourability.
12. Boolean lattice, antichains, LYM chain-counting proof, and Sperner’s bound.
13. Erdős–Ko–Rado, stars, small exact maxima, boundary nonuniqueness, and a clearly labelled Katona proof sketch.
14. Interval Helly theorem and its endpoint proof, with finite-family and geometry limitations.
15. Seeded binomial random uniform hypergraphs, exact expectation derivations, and 500-trial estimates.
16. Synthetic coauthorship, projects, group communications, biology, and resource-cover models.

## Interaction and animation

The 54-second introduction has play/pause, restart, one timeline, speed selection, six synchronized explanations, and offscreen/document-hidden pausing. All other visual progressions use inspectable manual steps: edge construction, incidence counting, incidence-link reveal, dual row-to-column transformation, uniform enumeration, path tracing, matching/cover views, binary colouring, lattice levels, interval movement, and random-edge reveal. Motion does not autoplay; reduced-motion CSS removes transitions and smooth scrolling.

`math.js` contains pure mathematical routines; `tutorial.js` contains topic controllers and SVG rendering. One running state supplies every live set, matrix, degree, dual, path, optimization, and colouring display. Dense cases switch to taller bipartite diagrams, preventing hub-label overlap. Region membership in the original gallery is explicitly qualified; exact incidence is authoritative. Native controls and focusable SVG selections support keyboard operation, with focus retained during shared-state redraws.

## Validation results

- **69,461 mathematical assertions passed** (`math-audit.json`). Independent bitmask checks cover all 128 simple hypergraphs on three vertices; all 65,536 subset families on four elements were checked, including 168 antichains and the exact LYM inequality using integer weights. Complete uniform counts and degrees, rank bounds, dual matrices, paths, greedy feasibility, and Fano pair uniqueness were checked. All 128 Fano colourings fail. Independent exhaustive EKR maxima equal 3, 4, and 5 for the offered cases. Interval endpoint cases and seeded monotonicity also pass.
- Default incidence totals are **11**, with degrees **2,2,2,2,2,1**. Default matching and transversal optima are both **2**.
- Seed 42, n=6, k=3, p=0.3, 500 trials: mean edge count **5.96** versus expectation **6**; mean degree of vertex 1 **2.984** versus expectation **3**. These are simulation estimates.
- Browser audit passed in installed Microsoft Edge using the existing bundled Playwright package. It exercised creation, replacement, deletion, invalid input, cross-highlighting, label toggle, reset, matrix counting, duality, paths, both optimizers, colouring, lattice selection, EKR, intervals, random endpoints/trials, animation controls, keyboard activation, the 12-edge cap, and the empty-family case.
- Layout passed at **1440, 768, 390, and 320 px** with no page-level horizontal overflow. The expanded gallery and a 12-edge family also pass at 320 px. Desktop, tablet, mobile, builder, colouring, and lattice screenshots are saved alongside this report and were visually inspected.
- Browser result: **zero console/runtime errors and zero missing resources**. All ten gallery PNGs decoded successfully.
- Static audit: balanced HTML, **118 unique IDs**, **39 local links checked**, exact preservation of all ten original image hashes, one scoped index entry, old paths absent, no external runtime dependencies.

## Reproduction

From the repository root:

```powershell
node research/graph-theory/2026-04-21-hypergraphs-visualization/validate_math.js
python research/graph-theory/2026-04-21-hypergraphs-visualization/validate_site.py
python -m http.server 8765 --bind 127.0.0.1
```

In another terminal, make the already installed `playwright` package available through `NODE_PATH`, then run `validate_browser.cjs` in the asset directory. It uses installed Edge; no package or browser installation is necessary. Browser screenshots and audit JSON files are regenerated by that script.

## Deliberate limits

No unresolved mathematical failures or browser errors were found. EKR’s circle lemma is referenced rather than proved in full; the section explicitly labels its derivation a proof sketch. Original hull figures are preserved with their geometric limitations explained. Exhaustive computations are intentionally small; simulation does not establish a theorem. The random diagram caps visible hubs at 16 while full counts and trial means include all sampled edges. Unicode and semantic HTML replace an external mathematical renderer, keeping the complete tutorial functional offline. No empirical datasets or new original research claims are asserted.

Deployment status and the resulting commit hash are reported separately after the requested commit and push, so this report does not assume a successful publication in advance.
