# Source inventory and reconstruction decisions

Inspected 2026-10-28 before composing the extension. `source-inventory.json` records every initial file, byte size, and SHA-256 hash. The PDF and all nine source screenshots are protected inputs.

| Source group | Files | Location in local 16-page PDF | Theme and reconstruction |
|---|---|---|---|
| 1 | `Verbatim/1a.png`, `1b.png` | PDF pp. 2–3, §2, Fig. 1 | Weighted transaction multigraph and address/transaction bipartite multigraph; join the split paragraph naturally. |
| 2 | `Verbatim/2a.png`, `2b.png`, `2c.png` | PDF p. 3, §2, Figs. 2–3 | Consolidate repeated address arcs; distinguish directed hyperarcs from ordinary arcs; graph motif topologies. |
| 3 | `Verbatim/3a.png`, `3b.png`, `3c.png` | PDF pp. 4–5, Fig. 4, Table 1, Fig. 5 | Generic two-transaction incidence, linear/circular examples, six intersections, role aggregation and reported counts. |
| 4 | `Verbatim/4.png` | PDF p. 5, before Definition 1 | STB motivation and endpoint restrictions. Read Definition 1 immediately below the screenshot to obtain the full mathematical criteria. |

All filenames are provenance labels here, not labels in the article body. PDF page locators refer to the local manuscript; the published chapter occupies pp. 248–263.

## Exact source details that affect implementation

- Table 1: α=H1∩T2, α′=T1∩H2, β=H1∩H2, γ=T1∩T2, L1=T1∩H1, L2=T2∩H2. A connected pair requires a cross-transaction intersection; self-loops alone do not connect two transactions.
- A pure pattern has exactly one of the six conditions. Linear STB: pure α (or reversed α′), exchange at each outer endpoint, no exchange in the intervening intersection.
- Circular STB: exactly α and α′, an exchange in one cross-intersection and none in the other; no β, γ, L1, or L2. Definition 1 specifies one and only one orientation. The implementation tests these explicitly, rather than extrapolating from the earlier informal bullet list.
- Fig. 4 uses different role palettes in its generic and STB panels. Its generic panel's T1 blue/H1 yellow differs from the STB panels' T1 red/H1 blue. The new diagrams use one explicit four-role palette; set identities remain unchanged.
- Fig. 4 left was inspected at its embedded 2179×539-pixel resolution. Its ten address circles and nineteen directed incidences are reconstructed, with added b/c/d labels. Three source exchange markings are retained; the c3 address belongs to H1, T2, and H2 simultaneously.
- Fig. 5's roles represent exactly one of the six intersections. Footnote 5 excludes addresses in more than two intersections and addresses with only one membership. Its reported numbers are not ordinary vertex degrees or toy-lab counts.
- Fig. 2 labels transactions T0–T3, whereas Fig. 1 uses E0–E3; reconstruction preserves those original labels. Educational notation E_j=⟨T_j,H_j⟩ is explained separately to avoid confusing a transaction label with its tail set.
- Fig. 2 consolidated quantities include T1→a2=35, a2→T2=28, a3→T2=55. a3 is both an input and an output of T2. This is address reuse, not a temporal cycle in the underlying UTXO graph.

## Rights and attribution check

The local manuscript and screenshots contain no reuse license. The [publisher record](https://link.springer.com/chapter/10.1007/978-3-319-70278-0_16) identifies © 2017 International Financial Cryptography Association and subscription access; extended republication permission was not established. Following the request's fallback, the webpage uses four short quotations transcribed from the supplied material, with the remainder explained in original prose. It does not reproduce the complete excerpts, embed source screenshots, or present extended transcriptions as authorized. Mathematical structures are reconstructed with attribution and source locators. Source inputs remain unchanged.

PAOH notation was cross-checked against [Valdivia et al., DOI 10.1109/TVCG.2019.2933196](https://doi.org/10.1109/TVCG.2019.2933196), using the [authors' institutional postprint](https://ricerca.uniba.it/retrieve/22b124fe-0ff3-4900-8aa9-b5d60582e752/Paohvis.pdf). Here it is a static membership view; columns are not time slots.

## Data integrity

The only Bitcoin research input is the paper and its excerpt images. There are no blockchain records, WalletExplorer label snapshots, original feature matrices, or train/test splits in this project. Source-reported historical counts and classifier scores must be labelled as such. New computations use synthetic address labels, with no claim of empirical replication or identity inference.

## Existing assets

Runtime: `math.js`, `story.js`, `intro.js`, `tutorial.js`, `style.css`, `favicon.svg`. The ten `hypergraph_*.png` gallery figures and their Python generator remain. Existing validation scripts remain reproducibility inputs. Historical reports and small JSON audits remain evidence, not runtime dependencies. Generated `preview-*` and `refinement-*` raster previews are cleanup candidates only after current replacements exist; see the separate deletion inventory before any removal. The legacy `hypergraph_demo.html` was checked and is absent. The canonical research-index link is retained.
