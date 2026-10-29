# Paper coverage audit — October 29, 2026 refinement

Sources: the two unchanged PDFs at the asset-directory root. I uses printed pages
45–87; II uses the supplied PDF's pages 1–22 (revision footer November 27, 2024).
The existing full-reading extracts were re-examined for definitions, algorithms,
proofs, implementation assumptions, and examples. Scans of I pp.49 and 74 were
visually checked again for C1–C3 and the generator-structure theorem. This is a
coverage audit, not a claim that the browser implements every source algorithm.

| Paper | Actual source location | Existing HTML coverage | Missing or weak explanation | Revision / disposition |
|---|---|---|---|---|
| I | §§1.1–1.4, pp.45–48 | Bridge, definitions, groups, notebook | Graph-to-matrix interpretation too abrupt | Added adjacency primer, row sums, source-row permutation convention; kept right-action notation distinct |
| I | §2.1 and Theorem 2.2, p.49 | Graph-valued C1–C3 and coloured-isomorphism criterion | Condition box could scroll; distinction from II easy to miss | Rechecked scan; retained all three implications and aligned them visibly |
| I | §§2.3–2.5, pp.50–51 | Equitability and active refiner | Counts not linked to actual matrix blocks | Added active-column sums, grouped matrix, before-pass cells, calculated playback |
| I | Theorems 2.6–2.7 and Lemma 2.8, pp.51–53 | Closure, incremental active sets, equivariance proof | No new theoretical omission found | Kept hypotheses and source-specific queue differences explicit; regression tests retained |
| I | §2.9, p.53 | Individualization and partition nests | Could appear to modify edges | Added fixed-order versus grouped-order matrices from the identical graph |
| I | §§2.10–2.15, pp.54–56 | Target policy, tree action, equivalent subtrees | Selected tree state lacked a matrix | Added node's original-graph matrix and partition; distinguished search-tree edges |
| I | §§2.16–2.20, pp.57–59 | Indicators, prefix vector, canonical-node survival | Ordering list versus permutation needs a numerical bridge | Added exact old-to-new permutation matrix for each selected terminal order |
| I | §§2.21–2.25, pp.59–62 | Explicit automorphisms, fixed sets, implicit cases | Small pruning equivalence insufficiently visible | Added C4 miniature and matrix certificate; kept implicit cases unimplemented |
| I | §§2.26–2.29, pp.62–65 | Indicator comparisons, sibling ordering, two terminal records | No new omitted guarantee found | Preserved cost/strength and traversal distinctions; did not identify historical sibling selection with full Traces |
| I | §§2.30–2.31, pp.68–72 | Full algorithm's roles, dig/lab/storage discussion | Implementation scope must stay clear | Kept full Algorithm 2.31 unimplemented; exact educational kernel separately identified |
| I | Theorem 2.33, pp.73–74 | Stabilizer-chain completeness mechanism | Output structure was only mentioned generally | Added distinction between source generator guarantees and our greedy closure basis |
| I | Theorem 2.34, p.74 | Previously only broad coverage | Disjoint-support and stabilizer generator properties absent | Added source-specific explanation; no false claim about greedy browser generators |
| I | §§3.1–3.6, pp.75–78 | Arrays, adjacency bit vectors, counting-cost assumptions | Matrices could obscure packed-bit implementation differences | Explained mathematical table versus bit-vector engineering and conditional nest bound |
| I | §§3.7–3.11, pp.78–82 | Performance and hard regular families | Risk of reading old empirical results as universal bounds | Retained historical scope; no new production benchmark claims |
| I | §3.12, p.82 | Design-derived hard examples | Incidence reduction not concrete | Added coloured point/block graph and block adjacency matrix [0 M; Mᵀ 0] |
| I | §3.13 / Theorem 3.14, pp.82–84 | Hadamard examples mentioned | Equivalence operations and colour condition not explicit | Added row/column permutation and sign-change reduction, vertex-type requirement, and stabilizer-depth warning |
| I | §§4.1–4.2, pp.84–86 | Q5 and C5[C5], structural group proofs | No new omission found | Preserved 3840 and 1,000,000 derivations and independent validation |
| II | Introduction / §2.1, pp.2–3 | Canonical deduplication; ordered colour surjection | Beginner progression needed numerical context | Added miniature matrix examples without deleting formal crosswalk |
| II | §2.2, p.4 | Pair-valued canonical output, forward graph action | Printed colour-action issue already documented | Retained explicit cell transport and the printed inverse discrepancy; new P convention is an independently declared matrix convention |
| II | §2.3, pp.4–5, Lemmas 1–3 | R/T axioms, tree and pointwise stabilizer | Graph and search-tree matrices could be confused | Added current graph matrix per selected node and equivariance-based pruning explanation |
| II | §2.4, p.6, Lemma 4 | φ1–φ3; equivalent leaves; canonical form | Needed a complete matrix calculation | Added terminal-order permutation, canonical matrix, relabelled-copy verification |
| II | §2.4, pp.6–7, Theorem 5 | P_A/P_B/P_C and printed-variable caution | Trees clipped, counts easy to misinterpret | Stacked full-width trees; matching sequence selection; separate expanded states, witnesses, retained leaves, absent descendants |
| II | §3.1, pp.7–9, Algorithm 1 / Figures 1–2 | Actual refiner and nine-vertex reconstruction | Splits lacked shared matrix interpretation | Added sums into active W, equitable quotient and synchronized partitions/matrices; no stronger invariant pruning inferred |
| II | §3.2, p.9 | Nauty choices and Traces parent-target fallback | No new omission found | Preserved distinction from simple largest-anywhere educational selector |
| II | §3.3, pp.9–10 | Prefix invariants, incremental event traces | Quotient edge totals versus per-vertex counts could be conflated | Added explicit conversion using cell sizes, including factor 1/2 within a cell |
| II | §3.4, pp.10–13 / Figures 3–4 | DFS, level dominance, experimental paths, random Schreier | Teaching animation must not stand in for full solver | Kept educational shared-tree label and all omitted production machinery explicit |
| II | §3.5, pp.13–14 / Figure 5 | Early automorphisms mentioned | Group-only reference set and extension heuristic insufficiently explained | Added singleton-map extension and group-only discrete-child reference strategy |
| II | §3.6, p.14 | Low-degree and preprocessing mentioned | Actual degree exclusions missing | Added 0,1,2,n−1 target exclusions and special-purpose completion; not executed in browser |
| II | §4, pp.14–15 / Figures 6–10, pp.18–22 | Experimental scope and difficult families | Risk of universal performance interpretation | Reaffirmed solver versions, measurement conditions, and distinction from our finite experiments |

The square, two prism studies, source-row permutation-matrix exercises,
rook/Shrikhande comparison, and live graph-generation atlas are independent
educational additions. They are not attributed as examples from either paper.

Mathematical audit results are recorded in `matrix-validation.json`,
`javascript-validation.json`, and `python-validation.json`. PDF hashes remain in
`asset-audit.json`; historical figures remain unchanged.
