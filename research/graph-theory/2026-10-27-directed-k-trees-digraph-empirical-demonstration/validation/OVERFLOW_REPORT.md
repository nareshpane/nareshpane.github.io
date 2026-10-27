# Horizontal overflow correction

The old inline formula wrappers used `overflow-x: auto`, and equation, table,
and code containers also allowed horizontal scrolling. Long MathJax 3 SVG
expressions were indivisible. MathJax 3 does not provide automatic line breaking
([official documentation](https://docs.mathjax.org/en/stable/output/linebreaks.html)).

`math_layout.py` now formats long formulas into short mathematical units with
natural browser wrapping at semantic boundaries. Prose qualifications wrap as
ordinary text. This covers directed paths, Cartesian products and subset
conditions, M1/M2/M3, lemmas, theorems, and explanatory equations. Matrices,
cases, grouped indices, and endpoint pairs stay intact. The original MathJax 3
renderer remains in use. No content is clipped, hidden, or reduced to tiny text.

The dedicated CSS removes scrolling wrappers, lets code wrap, and stacks ordinary
tables with their column labels on narrow screens. The adjacency matrix fits six
vertices on mobile. Cream cards, peach manuscript boxes, typography, diagrams,
animations, and computation controls are retained.

## Verification

- Browser widths: 1920, 1440, 1366, 1024, 768, and 390 pixels.
- At every width: zero page horizontal overflow, zero horizontal scrolling
  panels, zero mathematical SVGs extending outside their containing panel,
  and zero MathJax errors. All four manuscript panels fit their contents.
- Expandable content and the six-vertex laboratory were included. Small internal
  MathJax painting overhangs remain visible inside panel padding; they produce
  neither scrolling nor clipping. Details: `../data/overflow-browser-qa.json`.
- Content tests preserve the canonical tokens and order of 290 source formulas;
  matrix and cases entries are preserved exactly. All 13 source transcriptions
  and original scan hashes pass verification.
- Computational checks: 643 Python graph/root pairs, 463 JavaScript cases,
  and the teaching-example tests pass. Animation, root changes, matrix editing,
  and cycle tracing also pass browser interaction checks.
- HTML structure and local asset links pass. No remaining responsive limitation
  was observed at the six tested widths.

Original PDF, Verbatim scans, and raw transcription data are unchanged. The
authorized commit includes only the primary HTML and this dedicated directory,
including the preceding uncommitted refinement work in those paths. Unrelated
research directories, research.html, and repository configuration are excluded.

## Reproduce

Run the commands in README.md, including `test_math_layout.py`, then serve the
repository root with Python's HTTP server. Expand all disclosures, exercise the
laboratory, and inspect the six viewport widths. The saved screenshot
`../figures/scrollbar-free-verbatim.jpg` shows the corrected source typesetting.
