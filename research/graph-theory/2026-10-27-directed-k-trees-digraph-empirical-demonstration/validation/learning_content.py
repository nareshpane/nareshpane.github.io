"""Scoped editorial additions and layout refinements; original source JSON is unchanged."""
import re

MINI=r'''
<section class="card visual-primer" id="visual-primer" aria-labelledby="primer-title">
<div class="eyebrow">Five vertices · one defining rule</div><h2 id="primer-title">One component, two components, three components</h2>
<p>Keep every vertex. Choose one outgoing arc at every non-reference vertex, and none at a reference vertex. Every chosen path must end at a reference vertex.</p>
<div class="primer-grid">
<figure><div id="mini-one"></div><figcaption><strong>Directed tree · k = 1</strong><span>n = 5 · 4 arcs · 1 component</span>All paths reach one square reference vertex.</figcaption></figure>
<figure><div id="mini-two"></div><figcaption><strong>Directed 2-tree · k = 2</strong><span>n = 5 · 3 arcs · 2 components</span>Each component reaches its own reference vertex.</figcaption></figure>
<figure><div id="mini-three"></div><figcaption><strong>Directed 3-tree · k = 3</strong><span>n = 5 · 2 arcs · 3 components</span>An isolated reference vertex is a valid component.</figcaption></figure>
<figure><div id="mini-invalid"></div><figcaption><strong>Invalid · a directed cycle</strong><span>n = 5 · 4 arcs · fails reachability</span>The cycle never reaches the designated reference vertex.</figcaption></figure>
</div><p class="small">Squares are reference vertices. Component colours aid comparison; shapes, labels and arrowheads carry the same information. For valid structures, the number of selected arcs is n − k.</p></section>
'''
TRACE=r'''
<div class="trace-panel" aria-labelledby="trace-title"><h3 id="trace-title">Try following one path</h3>
<p>Use the three diagrams above. Each step follows exactly one selected arc; reaching a square stops the walk, while revisiting a vertex exposes a cycle.</p>
<div class="controls"><label>Trace example<select id="trace-example"><option value="tree">Directed tree</option><option value="cycle">Invalid cycle</option><option value="forest">Directed 2-tree</option></select></label><label>Starting vertex<select id="trace-start"><option value="0">1</option><option value="1">2</option><option value="2">3</option><option value="3">4</option><option value="4">r</option></select></label>
<button id="trace-prev" type="button">Previous</button><button id="trace-play" type="button">Play</button><button id="trace-next" type="button">Next arc</button><button id="trace-reset" type="button">Restart</button><label>Trace speed<select id="trace-speed"><option value="1800">Slow</option><option value="1000" selected>Medium</option><option value="550">Fast</option></select></label></div>
<p id="trace-status" class="status" role="status"></p></div>
'''
FOUNDATIONS=r'''
<div class="learning-block"><h3>Start with three vertices: what changes when a root is added?</h3>
<p>A root is a stopping point, not an extra arrow. In the first graph, both non-roots must make an outgoing choice. In the second, making vertex 2 a root removes its outgoing choice. All three vertices are still present.</p>
<div class="comparison-row"><figure><div id="foundation-one"></div><figcaption><strong>One root: {3}</strong>1 → 2 → 3. Three vertices, two arcs, one component.</figcaption></figure><figure><div id="foundation-two"></div><figcaption><strong>Two roots: {2,3}</strong>1 → 2, with 3 isolated. Three vertices, one arc, two components.</figcaption></figure></div>
<p>The same rule extends to any fixed root set R: every vertex outside R has outgoing degree one in the selected forest, every vertex in R has outgoing degree zero, and every vertex reaches a root.</p></div>
'''
ALGEBRA=r'''
<div class="learning-block"><h3>Why does choosing an outgoing arc not finish the job?</h3>
<p>Start with the symmetric triangle on vertices 1, 2, 3, with reference vertex 3. The choices are S₁ = {1 → 2, 1 → 3} and S₂ = {2 → 1, 2 → 3}. Their product has only four assignments. Inspect each before reading the general algebra.</p>
<div class="two"><div id="choice-graph"></div><div><div id="choice-buttons" class="controls" aria-label="Choose a triangle assignment"></div><p id="choice-expression" class="equation"></p><p id="choice-explanation" aria-live="polite"></p><p class="small">Reciprocal arcs share one underlying edge identifier a₁₂. The other three assignments use two different identifiers and lead to root 3.</p></div></div></div>
'''
SMALL_DET=r'''
<div class="learning-block" id="small-determinant"><h3>Before the larger determinant: count the three-vertex trees</h3>
<p>Use the same complete symmetric triangle and reference vertex 3. Its outgoing Laplacian has Lᵢᵢ equal to the number of arcs leaving i, and Lᵢⱼ = −1 when i → j is present. This convention counts paths directed <em>toward</em> the roots.</p>
<div class="determinant-flow">
<figure><div id="det-small-graph"></div><figcaption><strong>1 · The graph</strong>Each vertex has two available outgoing arcs.</figcaption></figure>
<div class="flow-step"><h4>2 · Full outgoing Laplacian</h4><div class="equation">\(L=\begin{pmatrix}2&-1&-1\\-1&2&-1\\-1&-1&2\end{pmatrix}\)</div><p>Every row sums to zero. The full determinant is zero; it is not the tree count.</p></div>
<div class="flow-step"><h4>3 · Delete root 3</h4><div class="equation">\(L_{U,U}=\begin{pmatrix}2&-1\\-1&2\end{pmatrix}\)</div><p>Keep rows and columns 1 and 2. The diagonal still includes arcs going to root 3.</p></div>
</div><div class="controls"><button id="det-prev" type="button">Previous determinant step</button><button id="det-next" type="button">Next determinant step</button><button id="det-reset" type="button">Restart determinant</button><span id="det-counter"></span></div>
<p id="det-step" class="equation" aria-live="polite"></p><p id="det-explanation"></p>
<div class="two"><div id="det-result-graph"></div><div><h4>4 · Inspect the counted structures</h4><div class="controls"><button id="det-forest-prev" type="button">Previous small tree</button><button id="det-forest-next" type="button">Next small tree</button></div><p id="det-enumeration" aria-live="polite"></p><p>Direct enumeration follows every outgoing-choice assignment and rejects cycles. The determinant uses a separate BigInt calculation. Agreement on this example illustrates the fixed-root counting identity, not a rule for arbitrary directed subgraphs.</p></div></div></div>
'''
MATRIX=r'''
<div class="learning-block matrix-lesson" id="matrix-lesson"><h3>From one arrow to two matrix entries</h3>
<p>Select an arc in the triangle. A present arc i → j adds +1 to Lᵢᵢ and −1 to Lᵢⱼ. Removing that arc subtracts those contributions. Root deletion happens <em>after</em> this construction, so arcs into the root still contribute to the retained diagonal.</p>
<div class="controls"><label>Arc to inspect<select id="matrix-edge"></select></label><button id="matrix-toggle" type="button">Remove selected arc</button><button id="matrix-reset" type="button">Reset triangle</button></div>
<div class="matrix-triptych"><figure><div id="matrix-graph"></div><figcaption><strong>A · Directed graph</strong>The orange arc is the selected contribution.</figcaption></figure><div><h4>B · Construct L</h4><div id="matrix-full" class="table-scroll"></div><p id="matrix-contribution" aria-live="polite"></p></div><div><h4>C · Delete root 3</h4><div id="matrix-reduced" class="table-scroll"></div><p id="matrix-result" class="status" role="status"></p></div></div></div>
'''
ROOTS=r'''
<div class="learning-block"><h3>A tiny generalization: from one root to three</h3>
<p>For the complete symmetric triangle, choose the roots before choosing any arcs. Each additional root omits one factor and creates one more component in every valid selection.</p>
<div class="controls"><label>Triangle components<select id="tiny-k"><option value="1">k = 1 · roots {3}</option><option value="2">k = 2 · roots {2,3}</option><option value="3">k = 3 · roots {1,2,3}</option></select></label><button id="tiny-prev" type="button">Previous tiny forest</button><button id="tiny-next" type="button">Next tiny forest</button></div>
<div class="two"><div id="tiny-forest"></div><div><p id="tiny-formula" class="equation"></p><p id="tiny-count" class="status" role="status"></p><p id="tiny-description"></p></div></div></div>
'''
SYNTHESIS=r'''
<div class="synthesis"><h3>How the pieces fit together</h3>
<ol><li><strong>Prescribe the roots.</strong> This fixes which outgoing factors are omitted and which rows and columns are deleted.</li><li><strong>Generate candidates.</strong> Select one arc at every non-root. Local degrees alone still allow cycles.</li><li><strong>Remove cyclic terms.</strong> Repeated-edge rejection and parity cancellation leave rooted forests in the symmetric completion; artificial-arc filtering restricts them to G.</li><li><strong>Count independently.</strong> Direct enumeration and the outgoing Laplacian principal minor give the same finite family size. The source examples give 12 for {r} and 6 for {3,r}.</li></ol>
<p>Try these same steps on an edited graph in the laboratory below. The general proof and the exact finite checks remain distinct.</p></div>
'''

def refine_page(page,slug):
    page=page.replace('From original graph-theoretic research to rigorous derivations, interactive directed graphs, algorithmic constructions, and computational validation.',
      'Empirical Demonstration of <cite>On Directed Trees and Directed k-Trees of a Digraph and Their Generation</cite> (1966) by Wai-Kai Chen')
    a=page.index('<aside class="meta"');b=page.index('</aside>',a)+len('</aside>')
    page=page[:a]+'''<aside class="meta" aria-label="Research metadata"><div><span aria-hidden="true">📅</span> <b class="created">Created:</b> <span>October 27, 2026</span></div><div><span aria-hidden="true">🧰</span> <b class="harness">Harness:</b> <span>Codex on Windows</span></div><div><span aria-hidden="true">🖥️</span> <b class="model">Model:</b> <span>GPT-6.1 Sol</span></div><div><span aria-hidden="true">🧠</span> <b class="reasoning">Reasoning:</b> <span>Medium</span></div></aside>'''+page[b:]
    page=page.replace('<section class="card" id="introduction">',MINI+'<section class="card" id="introduction">')
    page=page.replace('</div><div class="note"><b>Terminology matters.', '</div>'+TRACE+'<div class="note"><b>Terminology matters.')
    page=page.replace('<section class="card" id="notation"><h2>A notation map</h2>', '<section class="card notation-card" id="notation"><details class="notation-disclosure"><summary>A notation map</summary><h3>Notation from the original paper</h3>')
    page=page.replace('<th>Role / first appearance</th>', '<th>Role</th>')
    # Keep roles, remove only the trailing source-image first-appearance values.
    a=page.index('id="notation"');b=page.index('</section>',a)
    notation=page[a:b]
    notation=re.sub(r' · (?:[^<]*)</td>', '</td>',notation)
    notation=notation.replace('<details><summary>Added notation used in our explanations</summary>', '<h3>Additional notation used in our explanations</h3>')
    page=page[:a]+notation+page[b:]
    for anchor,content in [
      ('<h3>Intuition → a precise membership test</h3>',FOUNDATIONS),
      ('<h3>Two separate operations</h3>',ALGEBRA),
      ('<h3>Check a determinant independently</h3>',SMALL_DET+MATRIX),
      ('<h3>One omitted factor for each reference node</h3>',ROOTS)]:
        page=page.replace(anchor,content+anchor)
    page=page.replace('<section class="card" id="laboratory">',SYNTHESIS+'<section class="card" id="laboratory">')
    page=page.replace('All thirteen supplied PNG files are transcribed above and linked to their source panels. No file named 1e was supplied.',
      'The supplied excerpts are transcribed above as continuous academic text. The original source images remain unchanged in the project assets; the downloadable inventory maps their provenance.')
    page=re.sub(r'<p>3d ends mid-sentence.*?</p>', '<p>The final directed-tree paragraph is completed from the supplied full PDF, printed page 555; this is disclosed immediately below its verbatim panel. Sentence continuations across image boundaries are joined without changing their wording. The source’s path notation uses membership in E for an edge sequence; we retain it rather than silently correcting it.</p>', page, flags=re.S)
    page=page.replace('3c–3d and 4a–4b are continuations across images.', 'Sentences and the long tree list continue seamlessly across the supplied image boundaries.')
    page=page.replace('The Group 4 image includes Fig. 1.', 'The fourth source group includes Fig. 1.')
    page=page.replace('matching 3d.', 'matching the paper’s edge-product expansion.')
    page=page.replace('<a href="#introduction">Introduction</a>','<a href="#visual-primer">Introduction</a>')
    page=page.replace('<h3>Bibliographic source</h3>',f'<p class="small"><a href="{slug}/js/learning.js">Small-example and path-tracing code</a> · <a href="{slug}/validation/test_learning.js">Learning-example tests</a> · <a href="{slug}/validation/test_transcription.py">Continuous-transcription audit</a></p><h3>Bibliographic source</h3>')
    page=page.replace('</head>',f'<script defer src="{slug}/js/learning.js"></script></head>')
    return page
