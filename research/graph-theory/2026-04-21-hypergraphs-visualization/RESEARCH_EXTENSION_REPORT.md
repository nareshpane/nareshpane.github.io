# Twelve representations and Bitcoin directed-hypergraph study

Completed locally on October 28, 2026. Changes are confined to `2026-04-21-hypergraphs-visualization.html` and its dedicated folder. No files were staged, committed, or pushed. The existing sixteen numbered tutorial sections retain their numbering and functionality. `research.html` is unchanged and already links to the canonical page.

## A. Twelve-panel gallery

The subsection **Four Representations of the Same Hypergraph** is at the end of Section 03, immediately before Section 04. It contains exactly three rows of four panels:

1. Enclosed regions, peach background/border.
2. Static PAOH membership tracks, green background/border.
3. Venn/Euler membership regions or explicit atoms, red background/border.
4. Simple ordinary 2-section, blue background/border.

| Object | Vertex set | Hyperedge family | Degrees | Sizes | Incidences | Projected pairs |
|---|---|---|---|---|---|---|
| H1 | A,B,C,D | AB; BC | 1,2,1,0 | 2,2 | 4 | AB,BC |
| H2 | A,B,C,D,E | ABC; BD; CDE | 1,2,2,2,1 | 3,2,3 | 8 | AB,AC,BC,BD,CD,CE,DE |
| H3 | v1,…,v7 | v1v2v3; v2v3; v3v5v6; v4 | 1,2,3,1,1,1,0 | 3,2,3,1 | 9 | v1v2,v1v3,v2v3,v3v5,v3v6,v5v6 |

Each row is defined once in `research-math.js`; all four drawings derive their memberships and pairs from that object. Summary values and nonempty intersections are calculated, not maintained as a separate list of display constants. Independent expected values appear only in tests.

Edge selection works through row buttons, enclosure regions, PAOH marks/columns, Euler circles/atoms, and projected lines. The selected edge is traced across the row; reset clears it. Keyboard selection retains focus. Selecting H3's e2 highlights one pair already supplied by e1 and explicitly explains the redundancy. The distinction between singleton-only v4 and truly isolated v7 is retained in the hypergraph views and lost in the projection.

H1 and H2 use conventional membership circles with every displayed region labelled by its actual contents, including ∅. H3 uses six disjoint nonempty membership atoms, including the outside-all-sets atom {v7}. This avoids claiming that four arbitrary ellipses form a full Venn diagram. The spatial view is different: padded convex set contours enclose exactly the member vertex centres, with e2 visibly nested inside e1 and e4 around v4 alone.

Browser checks independently recover vertex inclusion from actual polygon geometry, check dot/row incidences, verify the circle-region locations, compare atom contents, and check deduplicated projected pairs. All twelve cards measure **263.5 × 420 px** at desktop widths 1920, 1440, and 1366. At 1024/768 they wrap to two columns, and at 390 to one column; heights remain 420 px.

## B. Bitcoin source study

The primary input is the supplied 16-page manuscript, **Exchange Pattern Mining in the Bitcoin Transaction Directed Hypergraph**, by Stephen Ranshous, Cliff A. Joslyn, Sean Kreyling, Kathleen Nowak, Nagiza F. Samatova, Curtis L. West, and Samuel Winters. The title, all seven authors, DOI, local source link, and manuscript locators appear in the page. Source inventory was completed before composition.

| Actual source group | Manuscript locator | New subsection |
|---|---|---|
| 1a, 1b | pp. 2–3, Fig. 1 | Addresses and transaction vertices |
| 2a, 2b, 2c | p. 3, Figs. 2–3 | Consolidated incidences and graph motifs |
| 3a, 3b, 3c | pp. 4–5, Fig. 4, Table 1, Fig. 5 | Six address-role intersections |
| 4 | p. 5; Definition 1 immediately below the excerpt | Linear/circular STB predicates |

`SOURCE_INVENTORY.md` gives the full filename mapping and mathematical reconstruction decisions. Source filenames are not used as labels in the article's explanatory text. The original PDF and all nine source images remain byte-for-byte unchanged; SHA-256 checks also protect the ten original PNG gallery figures.

### Verbatim scope and attribution

There are exactly four dark-red-bordered, light-peach verbatim boxes, each preceded by the complete paper title and all authors. The quotations use Garamond/EB Garamond with serif fallbacks. They are short, exact excerpts from the supplied material. Longer original prose is explained in newly written text outside those boxes.

Extended republication permission was not established: neither the supplied material nor the publisher record provides a reuse license for the selected passages. The [publisher record](https://link.springer.com/chapter/10.1007/978-3-319-70278-0_16) identifies IFCA copyright and subscription access. The request explicitly allowed brief quotations and faithful explanatory paraphrases in this situation, so complete extended transcriptions were not placed in the public HTML. No permission or open-access status is asserted. No source screenshot is embedded.

No mathematical symbol used in the implementation remains unreadable. Fig. 4 uses different palettes in its generic and STB panels; this is documented, and one labelled role palette is used consistently in the reconstructions. Tail/head set identities govern the new palette and algorithms. The informal STB wording is read together with Definition 1, which explicitly makes the circular case mixed α/α′.

### Reconstructed figures

- **Fig. 1 left:** SRC, E0–E3, UTXO, twelve directed weighted/address-labelled arcs. SRC and UTXO are explained as source/sink sentinels, not ordinary spendable addresses.
- **Fig. 1 right:** circles for addresses and squares for transactions, eighteen parallel directed quantity arcs, source address identities and UTXO markings retained.
- **Fig. 2:** fourteen consolidated incidences with the original T0–T3 labels and quantities. The toggle switches between unconsolidated and consolidated views. Examples verify 7+20+8=35, 20+8=28, 5+50=55, and 83=73+10 at T2.
- **Fig. 3:** the two-edge path and its α/α′, β, γ orientations; the three-edge path, star, and triangle with the direction classes shown in the source.
- **Fig. 4 left:** the complete ten-address generic motif, with all nineteen directed incidences and three source exchange markings. Added b/c/d labels identify the originally unlabelled circles. The source was inspected at full embedded-image resolution to trace every coloured arc.
- **Fig. 4 center/right:** the exact seven-address linear and circular topologies, exchange markings, and all ten directed incidences per example. Letter labels are added to identify source vertices that were originally unlabelled.
- **Fig. 5 left/right:** six address-role nodes and two transaction nodes; source-reported 1058M, 1064M, 942M, 927M, 2147M, and 1076M counts retain their role identities. The source's aggregation restriction and historical period are explained.

All reconstructions are vector objects generated from nodes and directed incidences, with attribution. A vector magnifier permits reading quantity labels on phones; it preserves SVG data and arrowheads and introduces no raster copy. Mathematical reconstructions are distinguished from smaller educational presets.

### Mathematics and interactivity

The study bridges H=(V,ℰ) with 𝓗=(V,𝒜), E_j=⟨T_j,H_j⟩, explains the address-level abstraction, and derives separate input/output incidence matrices and cardinality/degree sums. Links connect it to incidence counting and intersection products in the existing tutorial.

The six role sets are α=H1∩T2, α′=T1∩H2, β=H1∩H2, γ=T1∩T2, L1=T1∩H1, and L2=T2∩H2. All applicable roles are reported. A cross-transaction intersection is required for a connected pair; isolated self-loops do not suffice. The matrix product calculation explains each intersection count.

The STB implementation tests both transaction orientations exactly. Linear requires a pure chain plus exchange endpoints and a non-exchange intermediate intersection. Circular requires exactly α and α′, an exchange in one cross-intersection and none in the other. β, γ, L1, and L2 invalidate either STB type. Endpoint or intermediate-label changes can falsify a previously valid example.

The laboratory provides five presets: a single two-input/two-output transaction, a two-transaction forward chain, a three-transaction reuse/mixed-pattern network, a linear STB, and a circular STB. Readers can edit tail/head sets, control within-transaction address reuse, save or add transactions, add addresses, change synthetic exchange labels, choose the ordered transaction comparison, and select a matrix cell to highlight its exact arrow. Empty tails/heads are rejected. Display caps are eight addresses and four transactions. A manual/automatic phase sequence traces input and output incidences, pauses offscreen/hidden, and advances a single complete phase under reduced motion. The original introductory animation still completes in **18.0202 seconds** from click to final frame in the final check.

### Computational results and replication limits

On {A1,A2,A3}, enumerate every nonempty tail/head choice for each of two indexed transactions: 49²=2401 ordered pairs. With exchange set {A1,A3}, the independently checked result is:

| Quantity | Exact synthetic count |
|---|---|
| Ordered transaction pairs | 2401 |
| Connected pairs | 2353 |
| Connected pure patterns | 24 |
| Linear STBs | 4 |
| Circular STBs | 22 |

Identical address sets on distinct indexed transactions are allowed. This is a finite combinatorial experiment, not an empirical frequency estimate or significance test. `synthetic-transactions.json` records the presets and distinguishes the transcribed source schematic from synthetic examples.

The page separately labels the paper's historical counts and F1 values (0.9973 with label-derived features; 0.8200 without them). They are not new measurements. No blockchain records, exchange-label snapshot, feature matrix, or train/test split is supplied, so independent empirical replication and classifier fitting are not claimed. The validation plan specifies the data and evaluation controls that a real replication would require, including time/entity separation and avoidance of label-feature leakage.

## C. Layout and validation

New styling is component-specific in `research.css`. The existing global layout and mathematical proofs are preserved. Card widths are set by responsive grids with zero-minimum-width children; captions remain inside fixed-size cards while longer explanations sit outside. Source prose has an academic reading measure, and verbatim panels have a separate 860 px maximum rather than forcing a small width on every paragraph. Matrix wrappers scroll locally when necessary. No page overflow is hidden to mask defects.

Test commands from the graph-theory directory:

```powershell
node .\2026-04-21-hypergraphs-visualization\validate_math.js
node .\2026-04-21-hypergraphs-visualization\validate_story.cjs
node .\2026-04-21-hypergraphs-visualization\validate_research.cjs --write
python .\2026-04-21-hypergraphs-visualization\validate_site.py
$env:NODE_PATH='C:\Users\user\.cache\codex-runtimes\codex-primary-runtime\dependencies\node\node_modules'
node .\2026-04-21-hypergraphs-visualization\validate_browser.cjs
node .\2026-04-21-hypergraphs-visualization\validate_refinement.cjs
node .\2026-04-21-hypergraphs-visualization\validate_research_browser.cjs
```

The browser tests use the locally HTTP-served page on port 8765 and the already-installed Edge/Playwright, without adding project dependencies.

- Existing mathematics: **69,461 assertions pass**.
- Existing story/projection invariants: **3,084 assertions pass**.
- New research mathematics: **12,128 assertions pass**, including an independent bit-mask oracle for all 2401 pairs, positive/negative STBs, both orientations, disjoint pairs, loops, repeated addresses, source quantities, and protected source hashes.
- Combined mathematical total: **84,673 assertions**.
- New browser audit: twelve-card geometry and memberships, source topology, matrix cells, editing, reuse validation, exact arrows, exchange exclusions, all motif presets, finite experiment output, phase playback, reduced motion, offscreen pause, and vector magnification pass.
- Existing full tutorial regression passes, including empty/dense builders, duality, paths, exact/greedy optimization, Fano, Sperner, EKR, interval Helly, random experiments, keyboard focus, and all ten gallery image decodes.
- **1920, 1440, 1366, 1024, 768, 390 px** pass: no document overflow, clipped prose/formulas, or SVG text outside its viewBox; twelve-panel dimensions agree within each layout. The older regression also checks 320 px and expanded gallery/dense matrices.
- Static audit: balanced HTML, **190 static unique IDs**, **69 local links**, original images unchanged, research index unchanged, obsolete filename absent, no external runtime dependency. Dynamic IDs are also checked, including while the magnifier is open.
- No console/runtime or HTTP resource errors. Screenshots under `qa/` were inspected on desktop and mobile.

Limits: automated browser checks used installed Edge, not a cross-browser matrix. Hidden-document handling is covered by the existing simulated visibility event; offscreen pause uses actual scrolling. Source-image transcription is deliberately limited to the brief quotations described above. Some reconstructed source figures are dense on phones; the vector magnifier provides a full-size view.

## D. Cleanup

`CLEANUP_INVENTORY.md` and `cleanup-inventory.json` list every deletion, reason, original hash, and current replacement hash. Exactly **21 obsolete raster previews** were removed individually after replacement and passing-audit checks, eliminating **2,282,483 bytes**. All current generated validation captures are organized under `qa/`. The reproduction scripts now write there, so rerunning them does not recreate obsolete root-level clutter.

No source PDF, original Verbatim image, original gallery PNG, runtime file, generator, test script, or dataset was deleted. Historical reports and small JSON audits remain; the earlier site audit was restored byte-for-byte after current output was redirected. The legacy `hypergraph_demo.html` is absent, so no redirect was removed or created. Canonical HTML and all linked assets resolve locally; `research.html` still points directly to it.

## E. Files and Git status

Main HTML is extended. New runtime modules: `research-math.js`, `research-svg.js`, `representations.js`, `bitcoin.js`, and `research.css`. Existing `math.js`, `story.js`, `intro.js`, `tutorial.js`, and `style.css` are unchanged. Existing validation scripts only change current output locations and supplementary captures. New tests, provenance/cleanup manifests, synthetic data, this report, and current QA evidence are inside the dedicated folder. The PDF and Verbatim folder were already untracked user inputs before this work.

The final scope check passed; the staging area is empty. Exact `git status --short` follows. The five untracked research directories outside this project predate this work and remain untouched. The PDF and Verbatim inputs also predate this work.

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
?? 2026-04-21-hypergraphs-visualization/RESEARCH_EXTENSION_REPORT.md
?? 2026-04-21-hypergraphs-visualization/SOURCE_INVENTORY.md
?? 2026-04-21-hypergraphs-visualization/Verbatim/
?? 2026-04-21-hypergraphs-visualization/bitcoin.js
?? 2026-04-21-hypergraphs-visualization/cleanup-inventory.json
?? 2026-04-21-hypergraphs-visualization/hypergraph_bitcoin.pdf
?? 2026-04-21-hypergraphs-visualization/qa/
?? 2026-04-21-hypergraphs-visualization/representations.js
?? 2026-04-21-hypergraphs-visualization/research-math.js
?? 2026-04-21-hypergraphs-visualization/research-svg.js
?? 2026-04-21-hypergraphs-visualization/research.css
?? 2026-04-21-hypergraphs-visualization/source-inventory.json
?? 2026-04-21-hypergraphs-visualization/synthetic-transactions.json
?? 2026-04-21-hypergraphs-visualization/validate_research.cjs
?? 2026-04-21-hypergraphs-visualization/validate_research_browser.cjs
?? ../input-output-tables/
?? ../machine-learning/
?? ../mathematics/
```
