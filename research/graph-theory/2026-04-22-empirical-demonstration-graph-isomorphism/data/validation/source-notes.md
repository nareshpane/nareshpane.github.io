# Primary-source reading and translation notes

The local PDFs were inspected in full. The first is scanned (43 pages), so Windows
OCR supplied a reading aid, with definitions and principal formal statements
checked against rendered page images. OCR is not used as mathematical authority.
The later PDF has a text layer (22 pages); its graph figure and action convention
were also inspected visually. Extracted text and scan renders are temporary QA
artifacts outside the repository, not additional distributed paper copies.

## I: printed pagination, PDF page = printed page minus 44

- pp.45-48: graph universe, neighbourhoods, partition lattice, ordered consistency,
  fixed vertices/support, minimum representatives, groups and stabilizers.
- p.49: graph-valued canonical map with C1-C3 and coloured-GI criterion.
- pp.50-53: equitable closure, active-cell Algorithm 2.5, closure theorems,
  equivariance, individualization and partition nests.
- pp.54-56: first-smallest target search, backtracking generator, group action
  and equivalent subtrees.
- pp.57-59: indicator functions, prefix vector, terminal graph, canonical proof,
  identity/canonical nodes, first explicit-automorphism discussion.
- pp.60-62: orbit summaries, stored fixed sets, pruning and implicit cases.
- pp.63-67: indicator-based pruning, ordering heuristics, two terminal nodes,
  backtracking variables and implementation synthesis.
- pp.68-74: Algorithm 2.31, checkpoint stabilizer-chain theorem, generator
  completeness and structure.
- pp.75-79: array/bit-vector representation, storage/time choices, refinement
  cost assumptions, indicator engineering.
- pp.80-84: performance plot, hard examples, design and Hadamard reductions.
- pp.84-86: Q5 and C5[C5] examples; orders 3840 and 1000000.
- pp.86-87: bibliography.

## II: actual supplied PDF, revision footer 27 November 2024

- pp.1-3: abstract, historical setting, motivation, generic method, colourings.
- pp.4-5: action conventions, canonical pair, R/T axioms, tree and stabilizer
  lemmas. Colour-action line 2.2(c) printed without transport inverse, documented.
- pp.6-7: phi axioms, equivalent leaves, canonical selection, P_A/B/C, Theorem 5.
  The displayed node-variable mismatch in part (b) is documented in the page.
- pp.7-9: equitable closure, Algorithm 1, I/R recursion, Figures 1-2.
  Figure 1 reconstructed from its visible edges, independently checked for an
  8-element group and the stated quotient. Source labels shifted 0..8 -> 1..9.
- pp.9-10: target strategies, invariants, event traces, quotient interpretation.
- pp.10-13: DFS, level-oriented selection, experimental paths, Schreier methods,
  Figures 3-5 and separate group/canonical objectives.
- pp.13-14: early automorphisms and low-degree structure.
- pp.14-15: experimental conditions, solver versions, conclusions.
- pp.16-17: references; I listed as McKay (1980), distinguished from scan's 1981.
- pp.18-22: benchmark Figures 6-10, including random/cubic, transitive,
  strongly regular, Hadamard, tree, CFI, Miyazaki, projective-plane families.
  No plotted runtimes were transcribed as newly measured results.

## Additional verification

- Official McKay graph data page read in browser, October 29, 2026: simple graph
  counts n=2..8 and connected counts including n=8 -> 11117.
- Official graph6 files n=6..8 downloaded unchanged. Exact checksums are in the
  Python validation report; independent Burnside and uniqueness checks passed.
- arXiv 1301.1493 abstract/submission record read: first submitted January 8, 2013.
- Crossref DOI metadata checked for the 1998 additional canonical-construction
  reference. The formal augmentation scheme on the page is an original
  sufficient-condition explanation, not a transcription of a geng algorithm.

The HTML contains the full source-to-demonstration correspondence table, and the
README lists mathematical conventions, reproducible computations, and omissions.
