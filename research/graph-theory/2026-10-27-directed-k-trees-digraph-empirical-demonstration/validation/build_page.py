"""Assemble accessible HTML from audited source text and editorial explanations."""
from pathlib import Path
import json, html
from learning_content import refine_page
from math_layout import fit_math
BASE=Path(__file__).resolve().parents[1]
SLUG=BASE.name
sources=json.loads((BASE/'validation/transcriptions.json').read_text(encoding='utf-8'))
groups=[['1a','1b','1c','1d','1f'],['2a','2b'],['3a','3b','3c','3d'],['4a','4b']]
topics=['Spanning forests directed toward roots','Ring-sum and edge-disjoint star products','Generating directed trees by cancellation','Generating k rooted components']
inventory=[]
AUDIT={
 '1a':dict(definitions=['digraph','initial/terminal nodes','outgoing/incoming degree','regular','symmetric','subgraph (continued)'],equations=['rho(i)=rho*(i)=k'],claims=[],assumptions=['ordered endpoint pairs'],references=[],figures=[],boundary='Final sentence continues in 1b.'),
 '1b':dict(definitions=['empty graph','isolated node','null graph','directed circuit','disjoint subgraphs','spanning','directed path','simple path','length'],equations=['P_ij edge sequence'],claims=[],assumptions=['paths may repeat nodes'],references=[],figures=[],boundary='Begins with completion of 1a.'),
 '1c':dict(definitions=['undirected edge/graph','underlying graph','duplication','connected','component','sectional graph (continued)'],equations=['V_u=V','(i,j)=(j,i) in E_u'],claims=['Multiplicity extension asserted by author, not implemented'],assumptions=['edge multiplicity excluded'],references=[],figures=[],boundary='Sectional-graph sentence continues in 1d.'),
 '1d':dict(definitions=['sectional graph','tree','k-tree','reference node','directed tree','directed 2-tree','directed k-tree'],equations=['rho(i)=1 outside r','rho(r)=0'],claims=['extension from 2 to k components'],assumptions=['spanning','acyclic underlying graph','one reference per component'],references=[],figures=[],boundary='Complete end of definition paragraph.'),
 '1f':dict(definitions=['mu cardinality','edge product','union','intersection'],equations=['g1 union g2=e1e2e3e4e5e6','g1 intersection g2=e1e4'],claims=[],assumptions=['edge products omit isolated nodes'],references=[],figures=[],boundary='No 1e file supplied; the full PDF confirms the paragraph order.'),
 '2a':dict(definitions=['D','H','ring-sum','star (continued)'],equations=[],claims=[],assumptions=['families of subgraphs'],references=[],figures=[],boundary='Continues in 2b.'),
 '2b':dict(definitions=['ring-sum','star','zero','identity'],equations=['symmetric difference','disjoint-union star','h0*h=h0','h xor h=h0'],claims=['commutativity','associativity','abelian group','commutative semigroup with unity'],assumptions=['intersection-empty condition; interpretation discussed separately'],references=[],figures=[],boundary='All symbols readable; set-valued versus occurrence-valued expansion is an interpretive limitation, not an illegible symbol.'),
 '3a':dict(definitions=['S_k','C1','M1','M2','M3'],equations=['C1=product S_k over k not r'],claims=['Lemma 1: repeated underlying edges vanish','Lemma 2: longer cycles cancel'],assumptions=['symmetric digraph','fixed root r','identify opposite orientations'],references=['[7]'],figures=[],boundary='Lemma 2 proof not included in images; read from full PDF p.554.'),
 '3b':dict(definitions=['X1','T^r'],equations=['T^r=ring-sum of expansion occurrences'],claims=['Theorem 1','proof using Lemmas 1 and 2'],assumptions=['symmetric digraph','opposite arcs identified'],references=['Lemmas 1 and 2'],figures=[],boundary='The next supplied image skips PDF subsection 4(b).'),
 '3c':dict(definitions=['labelled completion example'],equations=['S1*S2*S3*S4','first six T-prime products'],claims=[],assumptions=['Fig.1','r fixed'],references=['Fig.1a','Fig.1b'],figures=[],boundary='T-prime list continues in 3d; arbitrary-digraph construction supplied separately from PDF section 4(b).'),
 '3d':dict(definitions=[],equations=['last ten T-prime products','12 T products'],claims=['root outgoing arcs irrelevant','early repeated-index rejection','early cancellation'],assumptions=['edge-label expansion'],references=['S1*S2 and S3*S4'],figures=[],boundary='Raw excerpt ends at greatly, mid-sentence. Displayed paragraph completed from the supplied PDF p.555; disclosed below panel; raw transcription unchanged.'),
 '4a':dict(definitions=['C2'],equations=['C2=product S_j outside k prescribed roots'],claims=[],assumptions=['symmetric digraph','specified root set'],references=['Fig.1'],figures=['Fig.1a original G','Fig.1b symmetric completion G-prime'],boundary='Text interrupted by figure; sentence continues in 4b. All figure topology and edge labels audited.'),
 '4b':dict(definitions=['X2','T_k^r'],equations=['root-excluded expansion','8 completion 2-trees','6 original 2-trees'],claims=['Theorem 2','arbitrary-digraph filtering'],assumptions=['fixed nonempty root set','opposite orientations identified before parity','remove double-primed arcs after generation'],references=['section 4(b)','Theorem 2','Fig.1'],figures=[],boundary='Complete end of section 5; no unreadable notation remains.')
}
def verbatim(i):
    names=groups[i-1]
    text='<div class="verbatim"><h3>Original Research — Verbatim</h3><p class="source-list">From <cite>On Directed Trees and Directed k-Trees of a Digraph and Their Generation</cite> (1966), by Wai-Kai Chen</p><div class="transcript">'
    for name in names:
        inventory.append(dict(filename=name+'.png',group=i,topic=topics[i-1],anchor='source-'+name,paragraphs=len(sources[name]),**AUDIT[name]))
    parts=[]
    for name in names:
        chunks=list(sources[name])
        if name in ('1b','1d','2b'):
            # These image boundaries split an actual sentence, not a paragraph.
            parts[-1]+=' '+chunks.pop(0)
        if name=='3d':
            # The completed-tree list spans two scanned images.
            first=chunks.pop(0).removeprefix(r'\[\begin{gathered}').removesuffix(r'\end{gathered}\]')
            previous=parts.pop().removeprefix(r'\[').removesuffix(r'\]')
            parts.append(r'\[\begin{gathered}'+previous+r'\\'+first+r'\end{gathered}\]')
        if name=='4a':
            # Place the original caption after the continued source paragraph.
            caption=chunks.pop()
        if name=='4b':
            parts[-1]+=' '+chunks.pop(0)
        parts.extend(chunks)
    if i==3:
        # Completed from the supplied full PDF, visually checked on printed p.555.
        parts[-1]+=r' simplified by eliminating those elements with same subscripts at an earlier stage; for example, if in \(S_1*S_2\) there are two elements with the same subscripts then they can be eliminated from \(S_1*S_2\) before one forms the star operation with \(S_3*S_4\).'
    if i==4:
        parts.append(caption.replace('<span class=\'editorial\'>[Original Fig. 1 is reproduced below the transcription. Its printed caption reads:]</span> ',''))
    for part in parts:
        if part.startswith(r'\['):
            text+='<div class="math-scroll source-equation">'+part+'</div>'
        else:
            text+='<p>'+part+'</p>'
    # Keep source anchors machine-readable without visible filename dividers.
    text+=''.join(f'<span id="source-{name}" hidden></span>' for name in names)
    text+='</div></div>'
    if i==3:
        text+='<p class="small source-completion">The final source paragraph is completed from the supplied full paper, printed page 555, where the cropped excerpt ends mid-sentence. The original excerpt data remain unchanged.</p>'
    return text

INTRO=r'''
<a class="skip" href="#introduction">Skip to article</a><main>
<header><div class="hero"><div><div class="eyebrow">Graph theory · A source-based research demonstration</div>
<h1>Directed k-Trees in Digraphs<span>Mathematical Foundations and Empirical Demonstration</span></h1>
<p class="subtitle">From original graph-theoretic research to rigorous derivations, interactive directed graphs, algorithmic constructions, and computational validation.</p></div>
<aside class="meta" aria-label="Research metadata"><div>◷ <b>Created:</b> October 27, 2026</div><div>▣ <b>Harness:</b> Codex on Windows</div><div>◇ <b>Model:</b> Not reliably exposed</div><div>◌ <b>Reasoning:</b> Not reliably exposed</div></aside></div>
<nav class="toc" aria-label="Contents"><a href="#introduction">Introduction</a><a href="#notation">Notation</a><a href="#section1">1 · Rooted forests</a><a href="#section2">2 · The algebra</a><a href="#section3">3 · Directed trees</a><a href="#section4">4 · Directed k-trees</a><a href="#laboratory">Computational laboratory</a><a href="#references">Reproducibility &amp; references</a></nav></header>
<section class="card" id="introduction"><div class="eyebrow">One arrow from every non-root</div><h2>Follow the arrows. Where do you arrive?</h2>
<p>In Wai-Kai Chen’s 1966 paper, a directed tree sends every non-reference vertex along a unique path <em>toward</em> the reference vertex. A directed <i>k</i>-tree allows <i>k</i> such components. Every vertex is retained, including isolated reference vertices.</p>
<div class="intro-grid">
<figure><div id="intro-tree"></div><figcaption><strong>A directed tree · k = 1</strong>Four arrows, five vertices. Every route ends at the square reference vertex r.</figcaption></figure>
<figure><div id="intro-cycle"></div><figcaption><strong>An invalid candidate</strong>Vertices 1, 2 and 3 circle forever. The outgoing-degree rule alone does not imply a tree.</figcaption></figure>
<figure><div id="intro-forest"></div><figcaption><strong>A directed 2-tree · k = 2</strong>Three arrows, two components. Each contains exactly one square reference vertex.</figcaption></figure>
</div><div class="note"><b>Terminology matters.</b> Here k is the number of components. Chen’s object is a spanning forest oriented toward specified roots. It differs from the clique-based “k-tree” used in structural graph theory. Arrowheads indicate direction; squares mark roots; circles mark other vertices.</div>
<p class="small">Reading layers: peach panels preserve the original excerpts; cream panels contain educational reconstructions; green results report exact computations on specified finite graphs. A computational check is evidence for those cases, not a proof for arbitrary size.</p></section>
<section class="card" id="notation"><h2>A notation map</h2><div class="table-scroll"><table><thead><tr><th>Source symbol</th><th>Meaning and intuition</th><th>Role / first appearance</th></tr></thead><tbody>
<tr><td>\(G(V,E),G_u\)</td><td>A digraph and its underlying undirected graph; forgetting arrows also merges reciprocal pairs.</td><td>Topology and symmetry · 1a, 1c</td></tr>
<tr><td>\(\rho(i),\rho^*(i)\)</td><td>Outgoing and incoming degree. These are degrees in the selected subgraph when testing a tree.</td><td>One choice per non-root · 1a</td></tr>
<tr><td>\(G[A],P_{ij}\)</td><td>Induced (“sectional”) subgraph on A; a directed path from i to j, with repeated nodes permitted in the source’s convention.</td><td>Components and reachability · 1b–1d</td></tr>
<tr><td>\(\mu(A),e_i\)</td><td>Cardinality of a set; a named edge. A product of edge letters records selected edges.</td><td>Counting and expansion · 1f</td></tr>
<tr><td>\(t^r,t^r_{r,i}\)</td><td>A tree rooted at r; a two-component directed forest with roots r and i.</td><td>Objects generated · 1d</td></tr>
<tr><td>\(h_0,\{\varnothing\}\)</td><td>No terms versus one empty term. The empty product represents all retained isolated roots when k = n.</td><td>Zero and multiplicative identity · 2b</td></tr>
<tr><td>\(D,H,h_1,h_2,x,y\)</td><td>The set of subgraphs; its families; two such families; individual subgraph terms combined in the algebra.</td><td>Algebra domain and operands · 2a–2b</td></tr>
<tr><td>\(\oplus,*\)</td><td>Symmetric difference (cancel even multiplicity); combine terms only if they use distinct underlying edges.</td><td>Algebraic generation · 2a–2b</td></tr>
<tr><td>\(S_j,C_1,C_2\)</td><td>Outgoing choices at vertex j; Cartesian products before cycle elimination for one or several roots.</td><td>Candidate generation · 3a, 4a</td></tr>
<tr><td>\(M_1,M_2,M_3\)</td><td>Trees, candidates with a 2-cycle, and remaining non-trees, respectively.</td><td>Cancellation argument · 3a</td></tr>
<tr><td>\(X_1,X_2,T^r,T_k^r\)</td><td>Star expansions and resulting tree/forest families after cancellation.</td><td>Theorems 1 and 2 · 3b, 4b</td></tr>
<tr><td>\(e_i',e_i'',G'\)</td><td>An existing reverse arc, an artificially added reverse arc, and the symmetric completion.</td><td>Arbitrary digraph extension · 3c; PDF §4(b)</td></tr>
</tbody></table></div>
<details><summary>Added notation used in our explanations</summary><p>\(n=|V|\), \(R\) is the prescribed root set, \(k=|R|\), \(U=V\setminus R\). \(L\) is the outgoing Laplacian; \(L_{U,U}\) retains rows and columns indexed by non-roots. These explanatory symbols do not replace the source notation.</p></details></section>
'''
SECTIONS=[
r'''<h3>Intuition → a precise membership test</h3>
<p>Take all n vertices. At each vertex outside R choose exactly one outgoing edge; choose none at a root. Now follow each chosen arrow. If a walk enters a directed cycle, the selection fails. If every walk reaches a root, it is a directed k-tree with exactly one root per component.</p>
<div class="equation">\(d_F^+(v)=\begin{cases}0&v\in R\\1&v\notin R,\end{cases}\qquad |E(F)|=n-k.\)</div>
<div class="two"><div><h3>Why this test is sufficient</h3><ol>
<li>In a finite graph, repeated arrow-following either stops at a root or repeats a vertex, creating a cycle.</li>
<li>If all walks stop, every vertex has a uniquely determined terminal root. The endpoints of any selected edge have the same terminal root.</li>
<li>Every component therefore has exactly one root. A component on s vertices has s − 1 edges, and is connected, hence is an undirected tree.</li></ol></div>
<div><h3>A repeated example</h3><p>Use vertices 1, 2, 3, 4, r. Select 1 → 2, 2 → 3, 3 → 4, 4 → r. There are four selected edges and one root, so k = 1.</p><p>Remove 3 → 4 and declare vertex 3 a second root. Now {1,2,3} and {4,r} are the two components. Merely removing the edge without changing the prescribed root set would fail the degree rule.</p><p>The invalid introduction instead selects 1 → 2 → 3 → 1 and 4 → r. Its four edges satisfy the degree rule but leave an unrooted cyclic component.</p></div></div>
<h3>Boundary cases and an exact check</h3><p>For k = n, every vertex is a root: the only spanning forest is the null graph on all n vertices. It has zero edges. For k = 1, all vertices must reach the single reference vertex. If a non-root has no available outgoing edge, no forest exists.</p>
<div class="note">In the laboratory’s three-vertex complete symmetric graph, with root 3, there are four outgoing-choice assignments. One is the 1 ↔ 2 cycle; three are valid rooted trees. Browse all four under “Candidates” to check the definition directly.</div>
<p class="summary">Established: degree constraints plus root reachability characterize Chen’s forests. The edge count n − k follows from the definition; it is not by itself a sufficient test.</p>''',
r'''<h3>Two separate operations</h3>
<p>Ring-sum cancels a term when it appears twice. Star rejects repeated underlying edge identifiers within a product. For the reciprocal pair u → v and v → u, both arcs use the same identifier {u,v}. Their product is rejected: that is Lemma 1’s elimination of a directed 2-cycle.</p>
<div class="equation">\(\{a\}\oplus\{a\}=h_0,\qquad \{a\}*\{a\}=h_0,\qquad \{\varnothing\}*\{a\}=\{a\}.\)</div>
<div class="note warning"><b>Implementation convention and a source ambiguity.</b> The paper first defines intersection for subgraphs, including nodes, but its actual edge-letter expansion permits different edges to share endpoints. We therefore interpret disjointness in the generation algorithm as <em>no repeated edge identifier</em>, matching 3d. Literal vertex-disjointness would reject ordinary trees. Also, a set-valued star product can erase multiplicities before ring-sum; our implementation retains expansion occurrences until parity is taken. This makes the paper’s operational cancellation precise without silently rewriting its verbatim algebra.</div>
<h3>A transparent cancellation example</h3>
<div id="algebra-triangle"></div>
<p>In a complete symmetric graph on vertices 1, 2, 3, r, choose the cycle 1 → 2 → 3 → 1 while leaving r isolated. Reverse that cycle. Both orientations produce the same undirected product \(a_{12}a_{23}a_{13}\). Their two occurrences cancel.</p>
<div class="equation">\(a_{12}a_{23}a_{13}\ \oplus\ a_{13}a_{23}a_{12}=0.\)</div>
<div class="controls"><button id="flip-cycle" type="button">Reverse the cycle</button><span id="cycle-caption" aria-live="polite">Orientation 1 of the same underlying triangle.</span></div>
<h3>Why multiple cycles also cancel</h3><p>This is an explanatory proof refinement. For a candidate with q ≥ 1 disjoint directed cycles of length at least three, each cycle has two orientations in a symmetric digraph. A fixed underlying edge selection therefore has \(2^q\) occurrences, an even number. Alternatively, choose the cycle containing the smallest-labelled cyclic vertex and reverse it; this gives a fixed-point-free involution. Rooted tree components have unique orientations toward their roots and contribute one occurrence.</p>
<p>The full PDF’s Lemma 2 proof says each element appears “exactly twice.” That describes the single-cycle case. The even-multiplicity argument above covers several disjoint cycles and supplies the needed general cancellation explanation.</p>
<h3>Computational interpretation</h3><p>Our parity implementation never asks whether a term is acyclic during expansion. It rejects repeated edge IDs and toggles a set entry for every occurrence. Direct enumeration separately follows paths in JavaScript and uses undirected union-find in Python. Equality of the resulting <em>families</em>, rather than just their sizes, is checked.</p>
<p class="summary">Established: repeated-edge rejection removes 2-cycles; even-multiplicity cancellation removes longer cycles in the symmetric completion. Keeping expansion multiplicities is essential.</p>''',
r'''<h3>From local choices to a global tree family</h3>
<p>Each factor \(S_j\) contributes one outgoing edge from j. There is no factor for r. A Cartesian expansion enforces local degrees but allows cycles; the algebra removes precisely those cyclic candidates.</p>
<div class="proof"><ol id="tree-derivation">
<li>\(C_1=\prod_{j\ne r}S_j\). Every assignment appears once before identifying opposite directions.</li>
<li class="next-step">Identify both orientations of each undirected edge. Reject every product that uses an edge identifier twice; this removes \(M_2\).</li>
<li class="next-step">Group the remaining terms by their unordered edge sets. Terms in \(M_3\) occur an even number of times and vanish under ring-sum.</li>
<li class="next-step">A spanning undirected tree has a unique orientation toward r. Its coefficient is one, so exactly the rooted trees remain.</li></ol>
<button type="button" id="derive-next">Show the next derivation step</button></div>
<h3>Arbitrary digraphs: the missing bridge</h3>
<div class="note"><b>Supplement from the full supplied PDF, §4(b), pp. 554–555; explanatory paraphrase.</b> Add any missing reverse arcs to make G′ symmetric. Existing reverse arcs receive primes; artificially added arcs receive double primes. Perform generation after identifying the two orientations of each underlying edge. Orient the resulting trees toward r, then discard any tree using an artificial arc. A single prime does <em>not</em> mean an artificial edge.</div>
<p>The source example contains ten original directed arcs on five vertices. Its symmetric completion contains fourteen arcs on seven underlying edge pairs. There are sixteen trees rooted at r in G′; four use 2 → 4 (e3″) and are removed. The original G has twelve.</p>
<div class="equation">\(\#T^r(G')=16,\qquad \#T^r(G)=16-4=12.\)</div>
<h3>Step through Chen’s construction</h3>
<div class="laboratory animation"><div class="two"><div id="algorithm-graph"></div><div><p class="group">Symmetric completion → expansion → cancellation → filtering</p><h4 id="algorithm-title"></h4><p id="algorithm-text" aria-live="polite"></p><div id="algorithm-expression" class="equation"></div></div></div>
<div class="controls"><button id="alg-prev" type="button">Previous step</button><button id="alg-play" type="button">Play</button><button id="alg-next" type="button">Next step</button><button id="alg-reset" type="button">Restart</button><label>Speed<select id="alg-speed"><option value="2400">Slow</option><option value="1400" selected>Medium</option><option value="700">Fast</option></select></label><label>Progress<input id="alg-progress" type="range" min="0" max="5" value="0"></label><span id="alg-counter"></span></div></div>
<h3>Check a determinant independently</h3><p>The full supplied paper’s Theorem 3 (p. 558) expresses a network determinant as a weighted sum over directed trees. For unit arc weights, form the outgoing Laplacian L, with diagonal equal to outgoing degree and off-diagonal −1 for each available arc. Delete the row and column of r. The example gives:</p>
<div class="equation">\(L_{U,U}=\begin{pmatrix}1&-1&0&0\\-1&3&-1&0\\0&-1&3&-1\\0&-1&-1&3\end{pmatrix},\qquad \det L_{U,U}=12.\)</div>
<p>This determinant does not use the enumeration code. In the laboratory you can toggle an arc in the adjacency table and see its graph arrow, its Laplacian contribution, and all three computed counts change together.</p>
<p class="summary">Established: the original directed-tree list has twelve elements. Symmetric completion, parity cancellation and artificial-edge filtering reconstruct that list; an exact determinant checks the count.</p>''',
r'''<h3>One omitted factor for each reference node</h3><p>Replace the single root r by \(R=\{r,i_1,\ldots,i_{k-1}\}\). The only change to the outgoing-choice construction is to omit every factor indexed by a root. Rooted components can now terminate at any vertex in R.</p>
<div class="equation">\(C_R=\prod_{j\notin R}S_j,\qquad |E(F)|=n-|R|=n-k.\)</div>
<p>The same cycle-reversal argument applies: a component without a root contains a directed cycle and cancels; a tree component with its specified root has exactly one orientation. A component cannot contain two roots: in a connected component with s vertices and t roots, the degree sum gives s − t edges; a connected acyclic component requires s − 1, hence t = 1.</p>
<h3>Reconstructing the actual Figure 1</h3><p>The original Fig. 1 belongs to the tree example although it is printed among the Group 4 material. The SVGs below preserve its five vertices, all original arc directions, seven edge identifiers and the four added reverse arcs. They are geometric redrawings with equivalent topology, rather than tracings of the old curves.</p>
<div class="two"><figure><div id="paper-original"></div><figcaption>(a) G: ten original arcs.</figcaption></figure><figure><div id="paper-completed"></div><figcaption>(b) G′: fourteen arcs. Dashed double-primed arcs are artificial.</figcaption></figure></div>
<div class="legend"><span><i></i> Original arc</span><span><i class="dash"></i> Added reverse arc</span><span>□ Reference vertex</span></div>
<details><summary>Figure topology and edge-label audit</summary><div class="table-scroll"><table><thead><tr><th>Identifier</th><th>Original arcs</th><th>Added arc in G′</th></tr></thead><tbody>
<tr><td>e1</td><td>2 → 3 (e1); 3 → 2 (e1′)</td><td>None</td></tr><tr><td>e2</td><td>3 → 4 (e2); 4 → 3 (e2′)</td><td>None</td></tr><tr><td>e3</td><td>4 → 2</td><td>2 → 4 (e3″)</td></tr><tr><td>e4</td><td>2 → r</td><td>r → 2 (e4″)</td></tr><tr><td>e5</td><td>4 → r</td><td>r → 4 (e5″)</td></tr><tr><td>e6</td><td>3 → r</td><td>r → 3 (e6″)</td></tr><tr><td>e7</td><td>2 → 1 (e7); 1 → 2 (e7′)</td><td>None</td></tr>
</tbody></table></div></details>
<h3>Worked 2-tree: roots r and 3</h3><p>Omit \(S_3\) as well as \(S_r\). There are eight surviving rooted forests in the completion, of which two use the artificial edge e3″. The six remaining products match 4b. Select each forest to see exactly which vertices reach which reference node.</p>
<div class="laboratory"><div class="two"><div id="forest-gallery"></div><div><h4>All six source forests</h4><div id="forest-buttons" class="controls"></div><p id="forest-product" aria-live="polite"></p><p id="forest-components"></p></div></div></div>
<div class="equation">\(L_{U,U}=\begin{pmatrix}1&-1&0\\-1&3&0\\0&-1&3\end{pmatrix},\qquad\det L_{U,U}=6,\quad U=\{1,2,4\}.\)</div>
<h3>Why the principal minor counts these forests</h3><p>This is an added derivation for unit-weight loopless digraphs. Write row i of the retained Laplacian as a sum over outgoing arcs: an arc i → j contributes \(\mathbf e_i^\mathsf T-\mathbf e_j^\mathsf T\) if j is a non-root, or just \(\mathbf e_i^\mathsf T\) if j is a root. Multilinearity of the determinant expands over one outgoing choice per non-root.</p>
<ol><li>A selected directed cycle makes the associated rows dependent: summing the rows around that cycle gives zero. Its determinant contribution is zero.</li><li>For a valid rooted forest, order non-roots by distance from their terminal root. A simultaneous row-and-column permutation gives a triangular matrix with all diagonal entries one. Its determinant contribution is one.</li><li>Summing these contributions gives exactly the number of forests rooted at the <em>fixed</em> set R.</li></ol>
<div class="equation">\(\#\{F:\text{roots exactly }R\}=\det L_{V\setminus R,V\setminus R}.\)</div>
<p>For R = V the empty determinant is one, matching the unique null graph. This is not the number of all k-component forests with unspecified roots: each unrooted forest would otherwise be counted once for every choice of one root per component. Weighted network applications and general cofactors from the later paper are outside these experiments.</p>
<p class="summary">Established: the same local construction generates k components by omitting k factors. The source’s rooted 2-tree example gives six forests, and the principal-minor derivation explains the independent count.</p>'''
]
LAB=r'''
<section class="card" id="laboratory"><div class="eyebrow">Exact finite experiments</div><h2>Computational Laboratory: Testing the Mathematical Results</h2>
<p>Choose a graph and a prescribed root set. Compare Chen’s operational parity construction with direct outgoing-choice enumeration and an exact integer determinant. The default triangle has only four candidates, so every case is inspectable.</p>
<div class="laboratory">
<div class="controls"><label>Graph<select id="preset"><option value="triangle">Complete symmetric triangle</option><option value="paper">Chen’s Figure 1</option><option value="cycle">Directed cycle</option><option value="empty">No edges</option><option value="complete">Complete symmetric graph</option></select></label>
<label>Vertices<select id="size"><option>3</option><option>4</option><option>5</option><option>6</option></select></label><label>k components<select id="root-count"><option>1</option><option>2</option><option>3</option></select></label>
<label>Primary root<select id="primary-root"></select></label><button id="lab-reset" type="button">Reset example</button></div>
<p class="small" id="root-help"></p><div id="root-choices" class="controls" aria-label="Choose prescribed roots"></div>
<div class="controls" aria-label="Experiment selector"><button class="active" data-mode="counts" type="button" aria-pressed="true">Compare methods</button><button data-mode="candidates" type="button" aria-pressed="false">Candidates</button><button data-mode="completion" type="button" aria-pressed="false">Symmetric completion</button></div>
<div class="metrics"><div class="metric"><b id="parity-count"></b><span>Chen parity + filtering</span></div><div class="metric"><b id="direct-count"></b><span>Direct enumeration</span></div><div class="metric"><b id="minor-count"></b><span>Exact determinant</span></div></div>
<p id="agreement" class="status" role="status"></p>
<div class="two"><div><div id="lab-graph"></div><p class="small" id="graph-description"></p><div class="controls" id="candidate-controls"><button id="candidate-prev" type="button">Previous</button><button id="candidate-next" type="button">Next</button><span id="candidate-index"></span></div><p id="candidate-explanation" aria-live="polite"></p></div>
<div><h4>Toggle an arc in the adjacency table</h4><p class="small">Rows are initial vertices; columns are terminal vertices. A 1 means an arc is present. Focus or hover on an entry to locate its arrow. Loops are excluded.</p><div id="adjacency"></div><h4>Retained outgoing Laplacian</h4><div id="lab-matrix" class="equation"></div><p class="small">Each i → j adds 1 to Lᵢᵢ and −1 to Lᵢⱼ before deleting all root rows and columns.</p></div></div>
<details><summary>Show full computational trace</summary><pre id="lab-trace"></pre></details>
</div><div class="note">Limits: 3–6 labelled vertices, simple loopless arcs, unit weights, and a fixed nonempty root set. The largest outgoing-choice space is 5⁵ = 3,125 assignments. Determinants use BigInt with at most 5! terms. No approximate arithmetic or randomized browser computation is used.</div></section>
<section class="card" id="references"><h2>Reproducibility &amp; references</h2>
<h3>What was actually tested</h3><p id="offline-summary">Python verification: 643 cases, zero discrepancies. This includes every loopless labelled digraph on n = 1, 2, 3 with every nonempty root set; 60 seeded graphs for each n = 4, 5, 6, each with one sampled nonempty root set; and both source examples. Seed: 1966.</p>
<div class="table-scroll"><table><thead><tr><th>Claim / experiment</th><th>Independent checks</th><th>Scope</th></tr></thead><tbody>
<tr><td>Theorems 1–2 operational construction</td><td>Family equality: parity expansion versus edge-subset / union-find enumeration</td><td>643 exact finite cases</td></tr>
<tr><td>Rooted forest count</td><td>Exact rational Gaussian elimination versus enumerated family size</td><td>All the same cases</td></tr>
<tr><td>Source trees and 2-trees</td><td>12 and 6, including comparison of all displayed edge products</td><td>Figure 1, fixed root sets {r} and {3,r}</td></tr>
<tr><td>Boundary cases</td><td>Empty graph, isolated non-root, all vertices roots, single vertex, disconnected graphs</td><td>Included in exhaustive n ≤ 3 domain</td></tr></tbody></table></div>
<p>All reported checks are exact; the larger graph collection is sampled, not exhaustive. The experiments support the implemented operational reading of the source. They do not resolve every abstract algebra claim in §3, nor establish a runtime improvement over modern algorithms.</p>
<h3>Source audit and editorial boundaries</h3>
<p>All thirteen supplied PNG files are transcribed above and linked to their source panels. No file named 1e was supplied. No equation numbers are visible in these excerpts, so none have been invented. Wrapped words have been joined across printed line breaks; mathematical content and paragraph order are retained.</p>
<p>3d ends mid-sentence at “greatly”; this is preserved. The full PDF continues: “simplified by eliminating those elements with same subscripts at an earlier stage; for example, if in S1*S2 there are two elements with the same subscripts then they can be eliminated from S1*S2 before one forms the star operation with S3*S4.” This plain-text continuation is supplemental context, not part of the excerpt transcription. 3c–3d and 4a–4b are continuations across images. The source’s path notation uses membership in E for an edge sequence; we retain it rather than silently correcting it.</p>
<p>The Group 4 image includes Fig. 1. Its printed graph labels are captured in the topology table and the SVG redrawings; its caption is transcribed in the source panel. Repeated-cycle cancellation, the principal-minor proof, all interface text and computational methods are our explanatory additions. The page does not transcribe the entire eleven-page paper.</p>
<h3>Download and rerun</h3>
<ul><li><a href="ASSET/chen1966.pdf">Supplied full paper (PDF)</a></li><li><a href="ASSET/validation/transcriptions.json">Audited HTML/LaTeX transcription data</a> · <a href="ASSET/data/content-inventory.json">Source-to-section inventory</a></li>
<li><a href="ASSET/validation/validate.py">Independent Python verification</a> · <a href="ASSET/data/validation-results.json">Actual results and source SHA-256 hashes</a></li>
<li><a href="ASSET/js/math.js">Browser mathematical methods</a> · <a href="ASSET/validation/test_math.js">Node cross-checks</a> · <a href="ASSET/validation/README.md">Methods and local testing instructions</a></li></ul>
<pre>python ASSET/validation/validate.py
node ASSET/validation/test_math.js
python -m http.server 8000 --directory ../..</pre>
<p class="small">Run these commands from research/graph-theory. Then open http://localhost:8000/research/graph-theory/ASSET.html. The graph laboratory works locally without fetching data; MathJax loads from a CDN and requires connectivity. With JavaScript disabled the transcribed article remains readable as HTML and LaTeX, while the interactive graphs require JavaScript.</p>
<h3>Bibliographic source</h3><p>Wai-Kai Chen, <cite>On Directed Trees and Directed k-Trees of a Digraph and Their Generation</cite>, <i>Journal of the Society for Industrial and Applied Mathematics, Series A: Control / Applied Mathematics</i>: the supplied scan’s masthead reads <i>J. SIAM Appl. Math.</i>, volume 14, number 3 (May 1966), pp. 550–560. Citation details are taken from the supplied scan; no DOI is asserted.</p>
</section>
<footer><a href="../../research.html">← Research collection</a> · <a href="../../index.html">Home</a><p class="small">Original research: Wai-Kai Chen (1966). Explanations and exact finite experiments: this research demonstration.</p></footer></main>
'''
def build():
    head=f'''<!doctype html><html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><meta name="description" content="Chen's directed k-trees: audited source excerpts, mathematical derivations and exact interactive forest enumeration."><title>Directed k-Trees in Digraphs: Mathematical Foundations and Empirical Demonstration</title><link rel="stylesheet" href="{SLUG}/css/style.css">
<script>window.MathJax={{tex:{{inlineMath:[['\\\\(','\\\\)']],displayMath:[['\\\\[','\\\\]']]}},options:{{skipHtmlTags:['script','noscript','style','textarea','pre','code']}}}};</script><script defer src="https://cdn.jsdelivr.net/npm/mathjax@3/es5/tex-svg.js"></script>
<script defer src="{SLUG}/js/math.js"></script><script defer src="{SLUG}/js/app.js"></script></head><body>'''
    page=head+INTRO
    for i,topic in enumerate(topics,1):
        page+=f'<section class="major" id="section{i}"><span class="group">Original research group {i}</span><h2>{i}. {topic}</h2>'+verbatim(i)+SECTIONS[i-1]+'</section>'
    page+=LAB.replace('ASSET',SLUG)+'</body></html>'
    # The title as printed in the source masthead is authoritative.
    page=page.replace('<i>Journal of the Society for Industrial and Applied Mathematics, Series A: Control / Applied Mathematics</i>: the supplied scan’s masthead reads <i>J. SIAM Appl. Math.</i>', '<i>J. SIAM Appl. Math.</i> (journal abbreviation as printed)')
    page=fit_math(refine_page(page,SLUG))
    (BASE.parent/(SLUG+'.html')).write_text(page,encoding='utf-8')
    (BASE/'data/content-inventory.json').write_text(json.dumps(inventory,indent=2),encoding='utf-8')
    print('Built page; inventoried',len(inventory),'source excerpts.')
if __name__=='__main__':build()
