# Hypergraphs presentation refinement — October 28, 2026

The existing tutorial is retained, with a smaller header, compact reader navigation, a source-backed historical archive, three exact uniform examples, and linked Bitcoin mathematics. Changes remain local and uncommitted.

1. **Files changed.** Main page: `../2026-04-21-hypergraphs-visualization.html`. New components: `presentation.css`, `presentation.js`, and `uniform-math.js`. Updated `bitcoin.js` and `validate_site.py`. Added `prepare_archive.py`, `refine_presentation.py`, `validate_presentation.cjs`, `validate_presentation_scope.py`, the four `archive/` files, this report, and `qa/presentation-*` evidence. Existing mathematical/browser audit outputs were refreshed. The prior research extension and its earlier preview cleanup were already present in the starting working tree; their status is preserved. `research.html`, other projects, the original PDF, original ten PNGs, figure generator, and nine Verbatim images were not edited.

2. **Main title.** The exact title and Georgia typeface remain. Only `.hypergraph-main-title` uses `font-size:clamp(1.56rem,2.964vw,2.535rem)`, line height 1.13, and letter spacing −0.015em. This is a measured 22% reduction at every requested width: 52→40.56px on large screens and 32→24.96px on small screens. Mathematical section headings keep their existing sizes.

3. **Subtitle and targets.** The exact rendered subtitle is: “An expanded study retaining the [April 21, 2026 original exploration](../2026-04-21-hypergraphs-visualization.html#original-april-2026) and investigating [Exchange Pattern Mining in the Bitcoin Transaction Directed Hypergraph](../2026-04-21-hypergraphs-visualization.html#bitcoin-replication).” Both are actual anchors. The existing `#bitcoin` target remains available alongside the new alias.

4. **Historical construction.** Git commit `859ef8a7e2899211e1b96c52455999557e6e993c`, dated April 21, 2026, contains `research/hypergraph_demo.html`. Its exact 15,633 bytes are retained in `archive/april-21-2026-original.html.txt`, SHA-256 `a541c69f879c37532bc8bb4428a79f6d8669814a67a5a28eabb465876b55438c`. The recovered original contains static examples and anchor navigation with MathJax typesetting, no animated builder, and no model/reasoning metadata. Those facts are stated explicitly rather than inventing historical interactivity or a model name. A deterministic generator preserves its wording and all ten original examples in an offline reading view: old asset paths are updated, IDs avoid collisions, original CSS is scoped, and simple set notation uses HTML subscripts. Original limitations and even historical placeholder wording remain. The unchanged source download preserves the original head, dependencies, and formatting for inspection. The collapsed semantic details section sits before Bitcoin, replacing the earlier duplicate gallery; `#gallery` and `#sec01`–`#sec10` remain valid. Direct/deep links open the archive automatically.

5. **Sticky navigation.** One 52px row follows the header, using spring green and peach accents. It contains a small brand link, active section, grouped Sections menu, and thin progress indicator. On mobile the brand/progress disappear. CSS sticky positioning takes effect after the header scrolls away; an IntersectionObserver controls the stronger stuck appearance. There are no stacked fixed bars. Anchor alignment reserves 64px, opens ancestor details before measuring, and supports clicks, keyboard activation, initial fragments, and back/forward navigation. Escape closes the menu and restores focus to its summary.

6. **Active-section detection.** Major section positions are cached on layout changes. A passive scroll listener schedules one animation frame and binary-searches the cached positions; it does not measure every section on every scroll. ResizeObserver, resizing, details toggles, and font readiness refresh the cache. The current label, `aria-current="location"`, and reading progress update without changing the URL or moving the reader. Bitcoin’s four source groups have distinct active labels. Manual scrolling was tested independently of anchor navigation.

7. **Compacted navigation.** The old permanently visible 19-link header grid is replaced by the single reader bar. Its dropdown groups Foundations, Representations and Combinatorics, Theorems and Applications, Original April 2026 Study, and Bitcoin Research. At 1440px the old header occupied about 555.6px, including 166.8px of navigation; the new header occupies 246.4px, followed by the 52px bar. Existing mathematical section IDs and numbering remain.

8. **Uniform definition and examples.** New Section 06A directly precedes Section 07. It states `H=(V,ℰ)`, `|e|=k` for every edge, `Σdeg(v)=Σ|e|=k|ℰ|`, and the simple-family bound `|ℰ|≤C(n,k)` with equality for the complete family. Uniformity is distinguished from vertex degree; the previous complete-uniform explorer stays in Section 06. One JavaScript data source defines all examples on `V={A,B,C,D,E,F}`:

   | k | Three hyperedges | Degrees A,…,F | Incidences |
   |---|---|---|---|
   | 2 | {A,B}, {B,C}, {D,E} | 1,2,1,1,1,0 | 6 |
   | 3 | {A,B,C}, {C,D,E}, {A,E,F} | 2,1,2,1,2,1 | 9 |
   | 4 | {A,B,C,D}, {B,C,D,E}, {C,D,E,F} | 1,2,3,3,2,1 | 12 |

9. **Uniform figures and validation.** Three equal cards use peach, green, and blue backgrounds with matching darker borders. The 2-uniform SVG is an ordinary graph with isolated F. The other two use labelled incidence hubs and exactly nine/twelve spokes, avoiding false contour membership. Diagrams, edge buttons, degrees, and totals all derive from the same data. Edge selection traces exact memberships. Cards stack below 900px so labels remain readable. Tests verify every member, cardinality, distinctness, degree, incidence total, SVG spoke, highlight/reset action, equal card dimensions, and vertical clearance between wrapped sets and captions.

10. **Four Bitcoin research groups.** The four groups remain: transaction representations; consolidation and hyperarcs; six two-transaction roles; short thick bands. The brief exact excerpts retain Garamond, peach backgrounds, red borders, citations and source links. Each is followed by a clearly labelled Source-Based Research Explanation. Attribution is visually quieter. Figures 1–5 remain attributed vector/topology reconstructions, with captions distinguishing added labels and source-reported counts from synthetic examples. Selecting a source Figure 2 transaction now derives its tail/head sets and input/output quantity sums. Motif presets highlight exactly the union of their nonempty intersection sets. Selecting a lab transaction synchronizes arrows, both binary matrix columns, ordered set pairs, and the editable draft; selecting a matrix cell keeps the matching transaction context. Enlarged figure copies remain static accessible vectors, without inert button roles.

11. **Exact version and reuse review.** The local 16-page PDF is byte-identical to the [official BITCOIN 2017 conference manuscript](https://fc17.ifca.ai/bitcoin/papers/bitcoin17-final17.pdf): 1,198,276 bytes, SHA-256 `d4c0ad3adc8e0749b8b07cd148343a9a049170c9839d703f604fcbb5590d9ef3`. This distinguishes it from the later [Springer chapter](https://doi.org/10.1007/978-3-319-70278-0_16), pp. 248–263. No explicit open reuse license was established in the manuscript or conference listing; the publisher record identifies IFCA copyright, 2017. Accordingly, no extended verbatim reuse was added: the four short source excerpts, paraphrased explanations, and attributed mathematical reconstructions are retained. Source screenshots are not embedded, and source filenames are not used as explanatory body text. The page includes the version/reuse findings in an expandable source note.

12. **Wrapping causes and corrections.** The root `p{max-width:78ch}` was already overridden before this task, so it was not an active cause. The actual source-specific `96ch` rule also matched 13px captions, shrinking some to about 673px in a 1090px container. Narrative prose now has a 1000px measure independent of caption font size; source captions/notes use their available container width. Quote and attribution widths increase from 860→1000px, and wide source figures from 800→940px. Grid children retain `min-width:0`. Uniform set rows reserve enough space when buttons wrap. No blanket clipping, overflow hiding, nonbreaking prose, or equation truncation was introduced. The audit distinguishes deliberate membership-flow connector arrows from text overflow, excludes unpainted closed-details content, and records intentional local matrix scrolling separately.

13. **Tests and evidence.** All requested widths passed in installed Microsoft Edge: 1920, 1440, 1366, 1024, 768, and 390px. `qa/presentation-audit.json` records 207 new assertions, including both subtitle links, collapsed/default and direct/deep archive navigation, browser history, keyboard menu/Escape, active-section scrolling, sticky timing, equal cards, SVG text bounds, expanded archive widths, nested overflow, and vertical set-caption clearance. There are no console/page errors or missing/failed requests in the final run. The existing mathematical validators passed 69,461 + 3,084 + 12,128 checks: **84,673 existing checks, 84,880 combined**. Existing tutorial and research browser regressions passed, covering the builder, counts, duality, paths, optimization, colouring, lattice, EKR, Helly, random samples, twelve representations, Bitcoin editing, motif/STB detection, phase animation, reduced motion/offscreen pause, and magnifier. Static validation confirms balanced HTML, 202 unique static IDs, 98 local links, ten original PNG hashes, and unchanged `research.html`. Screenshots and the before-change measurements are in `qa/`. `git diff --check` passes.

14. **Remaining limitations.** Validation used installed Edge; a second browser engine was not tested. The historical view adapts paths and typesetting for offline reading, so it is explicitly distinguished from the exact source download. Its original model/reasoning metadata is unavailable. No broader source reuse permission was established. The Bitcoin laboratory remains synthetic and does not claim a new empirical replication without the paper’s transaction dataset and exchange labels. No packages, build system, external runtime dependencies, staging, commits, or pushes were introduced.

<!-- FINAL_GIT_STATUS -->

Final `git status --short` (includes pre-existing work):

```text
 M 2026-04-21-hypergraphs-visualization.html
 D 2026-04-21-hypergraphs-visualization/preview-1440.jpg
 D 2026-04-21-hypergraphs-visualization/preview-390.jpg
 D 2026-04-21-hypergraphs-visualization/preview-768.jpg
 D 2026-04-21-hypergraphs-visualization/preview-builder.jpg
 D 2026-04-21-hypergraphs-visualization/preview-colouring.jpg
 D 2026-04-21-hypergraphs-visualization/preview-sperner.jpg
 D 2026-04-21-hypergraphs-visualization/refinement-counterexample.png
 D 2026-04-21-hypergraphs-visualization/refinement-examples.png
 D 2026-04-21-hypergraphs-visualization/refinement-incidence-1440.png
 D 2026-04-21-hypergraphs-visualization/refinement-incidence-390.png
 D 2026-04-21-hypergraphs-visualization/refinement-incidence-768.png
 D 2026-04-21-hypergraphs-visualization/refinement-preview-1440.jpg
 D 2026-04-21-hypergraphs-visualization/refinement-preview-390.jpg
 D 2026-04-21-hypergraphs-visualization/refinement-preview-768.jpg
 D 2026-04-21-hypergraphs-visualization/refinement-preview-builder.jpg
 D 2026-04-21-hypergraphs-visualization/refinement-preview-colouring.jpg
 D 2026-04-21-hypergraphs-visualization/refinement-preview-sperner.jpg
 D 2026-04-21-hypergraphs-visualization/refinement-story-1440.png
 D 2026-04-21-hypergraphs-visualization/refinement-story-390.png
 D 2026-04-21-hypergraphs-visualization/refinement-story-768.png
 D 2026-04-21-hypergraphs-visualization/refinement-story-desktop.png
 M 2026-04-21-hypergraphs-visualization/validate_browser.cjs
 M 2026-04-21-hypergraphs-visualization/validate_refinement.cjs
 M 2026-04-21-hypergraphs-visualization/validate_site.py
?? ../2026-10-21-causal-inference-three-datasets-three-econometric-techniques/
?? ../gdp-seasonality-project/
?? 2026-04-21-hypergraphs-visualization/CLEANUP_INVENTORY.md
?? 2026-04-21-hypergraphs-visualization/PRESENTATION_REFINEMENT_REPORT.md
?? 2026-04-21-hypergraphs-visualization/RESEARCH_EXTENSION_REPORT.md
?? 2026-04-21-hypergraphs-visualization/SOURCE_INVENTORY.md
?? 2026-04-21-hypergraphs-visualization/Verbatim/
?? 2026-04-21-hypergraphs-visualization/archive/
?? 2026-04-21-hypergraphs-visualization/bitcoin.js
?? 2026-04-21-hypergraphs-visualization/cleanup-inventory.json
?? 2026-04-21-hypergraphs-visualization/hypergraph_bitcoin.pdf
?? 2026-04-21-hypergraphs-visualization/prepare_archive.py
?? 2026-04-21-hypergraphs-visualization/presentation.css
?? 2026-04-21-hypergraphs-visualization/presentation.js
?? 2026-04-21-hypergraphs-visualization/qa/
?? 2026-04-21-hypergraphs-visualization/refine_presentation.py
?? 2026-04-21-hypergraphs-visualization/representations.js
?? 2026-04-21-hypergraphs-visualization/research-math.js
?? 2026-04-21-hypergraphs-visualization/research-svg.js
?? 2026-04-21-hypergraphs-visualization/research.css
?? 2026-04-21-hypergraphs-visualization/source-inventory.json
?? 2026-04-21-hypergraphs-visualization/synthetic-transactions.json
?? 2026-04-21-hypergraphs-visualization/uniform-math.js
?? 2026-04-21-hypergraphs-visualization/validate_presentation.cjs
?? 2026-04-21-hypergraphs-visualization/validate_presentation_scope.py
?? 2026-04-21-hypergraphs-visualization/validate_research.cjs
?? 2026-04-21-hypergraphs-visualization/validate_research_browser.cjs
?? ../input-output-tables/
?? ../machine-learning/
?? ../mathematics/
```
