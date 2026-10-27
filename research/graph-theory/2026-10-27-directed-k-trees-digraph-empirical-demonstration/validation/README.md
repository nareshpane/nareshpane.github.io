# Directed k-tree research demonstration

The primary page is one directory above the asset directory. No build dependencies are required to serve it.

## Reproduce

From research/graph-theory, run:

    python 2026-10-27-directed-k-trees-digraph-empirical-demonstration/validation/validate.py
    node 2026-10-27-directed-k-trees-digraph-empirical-demonstration/validation/test_math.js
    python 2026-10-27-directed-k-trees-digraph-empirical-demonstration/validation/build_page.py
    python -m http.server 8000 --directory ../..

Open http://localhost:8000/research/graph-theory/2026-10-27-directed-k-trees-digraph-empirical-demonstration.html.

MathJax is the only external runtime resource; its CDN is needed to render LaTeX. Graphs and counts use local JavaScript and no fetch calls. No npm installation is needed.

## Independent methods

1. Python enumerates subsets of n-k original arcs. Union-find rejects undirected cycles; degree constraints enforce root orientation.
2. Python's independent operational reconstruction symmetrizes the graph, enumerates one outgoing choice per non-root, rejects repeated underlying edge identifiers, takes parity, orients surviving forests toward roots and filters added arcs.
3. Python builds an outgoing Laplacian principal minor and uses Fraction-based Gaussian elimination.
4. Browser/Node direct enumeration instead enumerates outgoing choices and explicitly follows paths. Its determinant uses the Leibniz formula with BigInt. Its parity code retains directed expansion terms, groups by underlying IDs and filters artificial arcs after parity.

All family comparisons use full sorted edge sets. The Python domain is every loopless labelled digraph on n=1..3 and every nonempty root set, plus 60 deterministic random digraphs for each n=4..6 (independent arc probability 0.4), with one uniform-size nonempty sampled root set per graph. The seed is 1966. Two paper examples are added: 643 graph/root pairs total.

The Node audit covers the 461 exhaustive small cases and the two source cases (463 total), compares against the Python source-family snapshots, verifies all 18 displayed source products, and separately tests multiplicity four for two disjoint cycles.

The determinant counts fixed-root directed forests; it does not count forests with unspecified roots. The derivation is supplied in Section 4. Weighted electrical network formulas beyond the unit-weight case are not implemented.

## Source audit

All 13 excerpts are preserved unchanged. transcriptions.json is the editable transcription source, with filename anchors mapped in data/content-inventory.json. The builder assembles the main page; retain both its explanatory text and the source JSON.

Only physical line-break hyphenation is normalized. Images 3d and 4a cut text mid-sentence; the source boundaries remain visible. No 1e file was present. All notation was readable after comparison with the full PDF; no missing equation numbers were invented.

The paper's set-valued star definition and its operational multiplicity cancellation require a distinction: this implementation retains expansion occurrences until ring-sum. Edge-disjointness in products means edge identifiers, not vertex-disjoint subgraphs. These interpretive issues are explicitly disclosed on the page. The multi-cycle involution explanation strengthens the single-cycle wording in the PDF's Lemma 2 proof.

## Provenance and integrity

data/validation-results.json contains actual Python outputs and SHA-256 hashes of all supplied PNGs. The full paper is chen1966.pdf, with printed pages 550-560. Figure 1's topology comes from 4a; its ten original arcs and four artificial reversals are listed on the page. The implementation excludes loops and parallel copies.

No finite experiment is claimed as a general proof. No files are staged, committed or pushed.
