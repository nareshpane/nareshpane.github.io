"""Idempotent structural edits for the October 28 presentation refinement."""
from pathlib import Path
import re
ASSETS=Path(__file__).resolve().parent
page=ASSETS.with_suffix('.html')
s=page.read_text(encoding='utf-8')
groups=[('Foundations',[('ordinary','01 · Beyond pairs'),('builder','02 · Build a hypergraph'),('definitions','03 · Definitions'),('incidence','04 · Count incidences')]),('Representations and combinatorics',[('four-representations','Twelve views · compare'),('representations','05 · Representations'),('uniform','06 · Uniform families'),('uniform-introduction','06A · Uniform hypergraphs'),('intersections','07 · Intersections'),('duality','08 · Duality'),('paths','09 · Paths & cycles'),('optimization','10 · Match & cover')]),('Theorems and applications',[('colouring','11 · Property B'),('sperner','12 · Sperner’s theorem'),('ekr','13 · Erdős–Ko–Rado'),('helly','14 · Interval Helly'),('random','15 · Random hypergraphs'),('applications','16 · Applications')]),('Original April 2026 study',[('original-april-2026','April 21 · Historical snapshot')]),('Bitcoin research',[('bitcoin-replication','Bitcoin · Research introduction'),('bitcoin-source-1','Bitcoin · 1 · Transactions'),('bitcoin-source-2','Bitcoin · 2 · Directed hypergraphs'),('bitcoin-source-3','Bitcoin · 3 · Motifs'),('bitcoin-source-4','Bitcoin · 4 · Short thick bands'),('bitcoin-lab','Bitcoin · Transaction laboratory'),('bitcoin-evidence','Bitcoin · Evidence & limits'),('references','References')])]
nav='<nav id="reading-nav" class="reading-nav" aria-label="Research section navigation">\n<a class="reading-brand" href="#top">Hypergraphs</a>\n<a id="current-section" class="current-section" href="#ordinary">01 · Beyond pairs</a>\n<details id="section-menu" class="section-menu"><summary>Sections</summary><div class="section-links">\n'
for title,links in groups:
    nav+='<div><strong>'+title+'</strong>'+''.join('<a href="#'+id+'">'+label.replace('&','&amp;')+'</a>' for id,label in links)+'</div>\n'
nav+='</div></details><progress id="reading-progress" max="100" value="0" aria-label="Page reading progress"></progress>\n</nav>'
if 'id="reading-nav"' not in s:
    s=re.sub(r'<nav class="toc".*?</nav>\s*</header>','</header>\n<div id="nav-sentinel" aria-hidden="true"></div>\n'+nav,s,count=1,flags=re.S)
if 'uniform-math.js' not in s:s=s.replace('<script defer src="'+ASSETS.name+'/presentation.js">','<script defer src="'+ASSETS.name+'/uniform-math.js"></script>\n<script defer src="'+ASSETS.name+'/presentation.js">')
if 'id="uniform-introduction"' not in s:
    block='''<section class="lesson" id="uniform-introduction">
<div class="eyebrow">06A · Uniform Hypergraphs</div>
<h2>When Every Hyperedge Has the Same Size</h2>
<p class="subtitle">From pairwise relationships to uniform three-member and four-member groups.</p>
<p>A finite hypergraph <span class="math">H=(V,ℰ)</span> is <em>k-uniform</em> when every hyperedge contains exactly k vertices:</p>
<div class="formula">|e|=k for every e∈ℰ.</div>
<p>Uniformity concerns the number of vertices in each hyperedge, not the number of hyperedges or the number incident to each vertex. In particular, 3-uniform does not mean degree three. The complete-uniform explorer in Section 06 considers all k-subsets; these smaller families need not be complete.</p>
<div class="formula">Σ<sub>e∈ℰ</sub>|e| = k|ℰ| = Σ<sub>v∈V</sub>deg(v).</div>
<p>For a simple k-uniform hypergraph on n vertices, <span class="math">|ℰ|≤C(n,k)=n!/[k!(n−k)!]</span> (1≤k≤n). Equality holds for the complete k-uniform hypergraph.</p>
<div id="uniform-cards" class="uniform-cards" aria-label="Three exact uniform hypergraph examples"></div>
<div id="uniform-totals" class="uniform-totals" aria-label="Computed uniform incidence totals"></div>
<p class="caption">Each family has three hyperedges on the same universe V={A,B,C,D,E,F}. Select an edge label to trace precisely its incidences; select it again to show the whole family. Coloured squares in the 3- and 4-uniform diagrams are edge hubs, not additional vertices.</p>
</section>
'''
    s=s.replace('<section class="lesson" id="intersections">',block+'<section class="lesson" id="intersections">')
if 'id="bitcoin-replication"' not in s:s=s.replace('<section class="lesson bitcoin-study" id="bitcoin">','<section class="lesson bitcoin-study" id="bitcoin">\n<div id="bitcoin-replication"></div>')
if 'Exact source version' not in s:
    pos=s.index('<div class="two bitcoin-bridge">')
    note='''<details class="source-version"><summary>Exact source version and reuse review</summary>
<p>The local 16-page PDF is byte-for-byte identical to the <a href="https://fc17.ifca.ai/bitcoin/papers/bitcoin17-final17.pdf">official BITCOIN 2017 conference manuscript</a> (1,198,276 bytes; SHA-256 <code>d4c0ad3adc8e0749b8b07cd148343a9a049170c9839d703f604fcbb5590d9ef3</code>). It is the conference-hosted version, distinguished from the later Springer chapter, pp. 248–263.</p>
<p>No explicit open reuse license appears in this PDF or its conference listing. The <a href="https://doi.org/10.1007/978-3-319-70278-0_16">publisher record</a> identifies IFCA copyright, 2017. Public availability does not establish permission for extended verbatim reuse. Accordingly, the four boxes below retain only brief exact excerpts. The surrounding explanations are source-based paraphrases; the SVGs are attributed mathematical reconstructions with added labels documented in their captions. No extended passages or source screenshots are republished.</p>
</details>
'''
    s=s[:pos]+note+s[pos:]
s=s.replace('</blockquote>','</blockquote>\n<div class="source-explanation-label">Source-Based Research Explanation</div>') if 'source-explanation-label' not in s else s
page.write_text(s,encoding='utf-8',newline='\n')
print('Presentation structure updated.')
