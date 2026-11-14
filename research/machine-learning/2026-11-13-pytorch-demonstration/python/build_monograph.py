"""Deterministic page compiler: one content specification, canonical source, live result tables."""
from __future__ import annotations
import ast
import html
import io
import json
import keyword
from pathlib import Path
import re
import sys
import tokenize
import hashlib
import os
from math_typesetting import m,save_sources
sys.dont_write_bytecode = True
if __name__ == '__main__':
    sys.modules['build_monograph'] = sys.modules[__name__]
ROOT = Path(__file__).resolve().parents[1]
PAGE = ROOT.parent/(ROOT.name+'.html')
equation_number = 0


def canonical(filename, symbol=''):
    source = (ROOT/'python'/filename).read_text(encoding='utf-8')
    if symbol:
        node = next(n for n in ast.walk(ast.parse(source)) if isinstance(n,(ast.FunctionDef, ast.ClassDef)) and n.name == symbol)
        source = ''.join(source.splitlines(keepends=True)[node.lineno-1:node.end_lineno])
    return source


def highlight(source, language='Python'):
    if language != 'Python':
        result=[];cursor=0
        for match in re.finditer(r'#[^\n]*|"[^"\n]*"|\x27[^\x27\n]*\x27|\$[A-Za-z_][\w:]*',source):
            token=match.group()
            style='comment' if token.startswith('#') else 'keyword' if token.startswith('$') else 'string'
            result.extend([html.escape(source[cursor:match.start()]),f'<span class="code-{style}">{html.escape(token)}</span>'])
            cursor=match.end()
        result.append(html.escape(source[cursor:]))
        return ''.join(result)
    lines = source.splitlines(keepends=True)
    offsets = [0]
    for line in lines:
        offsets.append(offsets[-1]+len(line))
    pos = lambda p: offsets[min(p[0]-1,len(offsets)-1)]+p[1]
    cursor, result = 0, []
    for token in tokenize.generate_tokens(io.StringIO(source).readline):
        style = {tokenize.COMMENT:'comment',tokenize.STRING:'string',tokenize.NUMBER:'number'}.get(token.type)
        if token.type == tokenize.NAME and keyword.iskeyword(token.string):
            style = 'keyword'
        if style:
            start,end = pos(token.start),pos(token.end)
            result.extend([html.escape(source[cursor:start]),f'<span class="code-{style}">{html.escape(source[start:end])}</span>'])
            cursor=end
    result.append(html.escape(source[cursor:]))
    return ''.join(result)


def code(filename,symbol='',title=None,opened=False,compact=False,framework=''):
    language='PowerShell' if filename.endswith('.ps1') else 'Python'
    source=canonical(filename,symbol)
    title=title or ('Complete executable program · '+filename if not symbol else symbol.replace('_',' ').capitalize())
    note=('Canonical function/class excerpt; not independently executable. The complete program supplies imports and supporting definitions.' if symbol else
          'Complete executable file. Application programs require the supplied sibling common.py and the documented dependencies; keep these files together.')
    cls='code-panel'+(' comparison-code '+framework if compact else '')
    return (f'<details class="{cls}" data-source="{filename}" data-symbol="{symbol}"'+(' open' if opened else '')+'>'
            f'<summary><span class="code-icon" aria-hidden="true">〈/〉</span><span class="code-title">{html.escape(title)}</span>'
            f'<span class="language">{language}</span><span class="code-action"><span class="expand-label">Expand Code ▾</span><span class="collapse-label">Collapse Code ▴</span></span></summary>'
            f'<p class="code-note">{note} <a href="{ROOT.name}/python/python-source.zip" download>Download all Python sources and setup</a>.</p><div class="code-toolbar"><button class="copy-code">Copy Code</button>'
            '<button class="expand-code" aria-expanded="false">Expand code fully</button><span class="copy-status" aria-live="polite"></span>'
            f'<a href="{ROOT.name}/python/{filename}" download>Download {filename}</a></div><pre><code>{highlight(source,language)}</code></pre></details>')


def eq(tex,fallback,kind='probability'):
    global equation_number
    equation_number+=1
    key='eq-'+hashlib.sha256(tex.encode()).hexdigest()[:12]
    baseline=ROOT/'data/typesetting-baseline.json'
    old=next((r['old_number'] for r in json.loads(baseline.read_text(encoding='utf-8'))['equations'] if r['explanation']==fallback),None) if baseline.exists() else None
    audit=f' data-audit-old-number="{old}"' if old is not None else ''
    from typesetting_content import EQUATION_NOTES
    explanation=EQUATION_NOTES.get(old,fallback)
    return (f'<div class="equation {kind}" id="{key}"{audit} tabindex="0" role="region" aria-label="Mathematical equation">'
            f'<a class="eq-number" href="#{key}" aria-label="Link to this equation">({equation_number})</a>{m(tex,True)}</div>'
            f'<p class="equation-explanation">{explanation}</p>')


def p(text):
    return '<p>'+text+'</p>'


def panel(title,text,kind='blue'):
    return f'<aside class="panel {kind}"><h4>{title}</h4>{p(text)}</aside>'


def table(headers,rows,caption='',wide=False):
    def numeric(value):
        text=html.unescape(re.sub('<[^>]+>','',str(value))).strip().replace('−','-')
        return bool(re.fullmatch(r'[+-]?[\d,]+(?:\.\d+)?(?:[eE][+-]?\d+)?(?:\s*(?:%|×|s|ms|bytes))?',text))
    columns=[sum(numeric(row[i]) for row in rows if len(row)>i)>len(rows)/2 for i in range(len(headers))]
    classes='table-wrap'+(' wide' if wide else '')+(' sensitivity' if headers[:2]==['Sensors','Noise SD'] else '')
    head=''.join(f'<th scope="col"'+(' class="numeric"' if columns[i] else '')+'>'+x+'</th>' for i,x in enumerate(headers))
    body=''
    for row in rows:
        cells=[]
        for i,value in enumerate(row):
            tag='th' if i==0 else 'td'
            attributes=(' scope="row"' if i==0 else '')+(' class="numeric"' if numeric(value) else '')
            cells.append(f'<{tag}{attributes}>{value}</{tag}>')
        body+='<tr>'+''.join(cells)+'</tr>'
    return '<div class="'+classes+'"><table>'+('<caption>'+caption+'</caption>' if caption else '')+'<thead><tr>'+head+'</tr></thead><tbody>'+body+'</tbody></table></div>'


def art(name,alt,caption='',cls='context-art'):
    if name in {'ad-products.svg','competing-risk-budget.svg','message-aggregation.svg','inverse-workflow.svg','cvar-tail.svg','soft-vs-feasible.svg'}:
        from math_diagrams import diagram
        return diagram(name)
    import xml.etree.ElementTree as ET
    path=ROOT/'images'/name
    node=ET.parse(path).getroot()
    _,_,width,height=node.attrib['viewBox'].split()
    image=f'<img src="{ROOT.name}/images/{name}" width="{width}" height="{height}" alt="{html.escape(alt)}">'
    if cls=='teaching-figure':image='<div class="scientific-scroll" tabindex="0" role="region" aria-label="Scrollable mathematical diagram">'+image+'</div>'
    return f'<figure class="{cls}">'+image+('<figcaption>'+caption+'</figcaption>' if caption else '')+'</figure>'


def plot(name,alt,caption=''):
    return f'<figure class="empirical-figure"><img src="{ROOT.name}/outputs/seed-42/{name}" width="880" height="350" loading="lazy" alt="{alt}"><figcaption>{caption}</figcaption></figure>'


def playback_controls():
    return '<div class="playback"><button data-action="play" aria-pressed="false">Play</button><button data-action="pause">Pause</button><button data-action="previous">Previous step</button><button data-action="next">Next step</button><button data-action="restart">Restart</button></div><p class="motion-note small" aria-live="polite"></p>'


def select(id,label,options):
    return f'<label for="{id}">{label}<select id="{id}">'+''.join(f'<option value="{value}">{text}</option>' for value,text in options)+'</select></label>'


def slider(id,label,minimum,maximum,value,step=1,output=True):
    return f'<label for="{id}">{label}'+(f'<output id="{id}-value" for="{id}"></output>' if output else '')+f'<input id="{id}" type="range" min="{minimum}" max="{maximum}" step="{step}" value="{value}"></label>'


def lab(id,title,provenance,body,kind=''):
    return f'<div class="lab {kind}" id="{id}"><div class="lab-heading"><h4>{title}</h4><span class="badge browser">Interactive laboratory</span></div><p class="provenance">{provenance}</p>{body}</div>'


def result(domain):
    return json.loads((ROOT/'outputs'/'seed-42'/f'{domain}.json').read_text(encoding='utf-8'))


def build():
    global equation_number
    equation_number=0
    import monograph_content
    from typesetting_content import enhance_chapters
    chapters=enhance_chapters(monograph_content.chapters())
    toc=''
    body=''
    for i,(id,title,short,intro,sections) in enumerate(chapters,1):
        toc+=f'<details data-chapter="{id}"'+(' open' if i==1 else '')+f'><summary><a href="#{id}">{i}. {short}</a></summary><div class="subsections">'
        body+=f'<section class="chapter" id="{id}"><div class="chapter-kicker">Chapter {i} · {short}</div><h2>{title}</h2><p class="chapter-intro">{intro}</p>'
        for j,(sid,stitle,content) in enumerate(sections,1):
            toc+=f'<a href="#{sid}">{i}.{j} {stitle}</a>'
            body+=f'<section class="subsection" id="{sid}"><h3>{i}.{j} {stitle}</h3>{content}</section>'
        toc+='</div></details>'
        body+='</section>'
    toc+='<a class="toc-reference" href="#references">References</a>'
    body+='<section class="references-section" id="references"><h2>References</h2>'+monograph_content.references()+'</section>'
    intro=monograph_content.hero()
    # Added derivations keep sequential equation numbers in document order.
    equation_number=0
    def renumber(match):
        global equation_number
        equation_number+=1
        return match.group(1)+f'({equation_number})</a>'
    body=re.sub(r'(<a class="eq-number"[^>]*>)\(\d+\)</a>',renumber,body)
    from typesetting_content import polish_inline_notation
    body=polish_inline_notation(body)
    intro=intro.replace('Six chapters','Seven chapters')
    save_sources()
    if os.environ.get('MONOGRAPH_COLLECT_MATH')=='1':
        print('Collected mathematical sources; article not overwritten.')
        return
    config=r'''<script>window.MathJax={loader:{paths:{tex:'2026-11-13-pytorch-demonstration/js/mathjax-extensions'},load:['[tex]/boldsymbol']},tex:{packages:{'[-]':['noundefined','noerrors'],'[+]':['boldsymbol']},inlineMath:[['\\(','\\)']],displayMath:[['\\[','\\]']]},options:{ignoreHtmlClass:'math-fallback|code-panel|tex2jax_ignore'},svg:{fontCache:'global'},startup:{ready(){MathJax.startup.defaultReady();MathJax.startup.promise.then(()=>document.documentElement.classList.add('math-ready'))}}};</script>'''
    text='<!doctype html><html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><meta name="description" content="A mathematical PyTorch monograph with genuine CPU experiments in competing risks, transaction graphs, river inverse inference, and risk-aware resource allocation."><title>PyTorch: From Computational Graphs to Scientific Inference and Intelligent Decisions</title><link rel="stylesheet" href="'+ROOT.name+'/css/styles.css">'+config+'<script defer src="../miscellaneous/harmonized-system/details-of-machine-learning-models/js/vendor/tex-svg.js"></script><script type="module" src="'+ROOT.name+'/js/main.js"></script></head><body><a href="#article" class="skip-link">Skip to article</a><button class="menu-toggle" id="menu-toggle" aria-controls="sidebar" aria-expanded="false">Contents</button><div id="drawer-backdrop" class="drawer-backdrop"></div><aside id="sidebar" class="sidebar" aria-label="Monograph navigation"><button class="drawer-close">Close contents</button><a href="#beginning" class="toc-heading">PyTorch · a research monograph</a><div class="progress-track" role="presentation"><div id="reading-progress"></div></div><div class="toc-actions"><button id="expand-toc">Expand all</button><button id="collapse-toc">Collapse all</button></div><nav id="toc-tree" class="toc-scroll" aria-label="Chapters and subsections">'+toc+'</nav><div class="sidebar-footer">Synthetic data · executed CPU evidence<br>November 13, 2026 · seven chapters</div></aside><main id="article">'+intro+'<p id="math-status" class="math-status"></p><noscript><p class="panel">The full article, mathematical equations, figures, source downloads, contents, and code disclosures work without JavaScript. Laboratories require JavaScript and an HTTP server.</p></noscript>'+body+'<footer><p>Original synthetic experiments, mathematical explanations, and editable illustrations. Created November 13, 2026. No clinical, financial, environmental, or emergency-service deployment claim is made.</p><p><a href="'+ROOT.name+'/README.md">Reproduction guide</a> · <a href="'+ROOT.name+'/qa-report.md">QA and limitations</a> · <a href="#beginning">Back to beginning</a></p></footer></main></body></html>'
    nojs='<noscript><style>.toc-actions,.progress-track,.drawer-close{display:none!important}@media(max-width:820px){.menu-toggle{display:none!important}.sidebar{position:static!important;transform:none!important;visibility:visible!important;width:auto!important;max-height:360px;box-shadow:none!important}.toc-scroll{max-height:270px}main{padding-top:25px!important}}</style></noscript>'
    text=text.replace('</head>',nojs+'</head>')
    PAGE.write_text(text,encoding='utf-8')
    structure=[dict(chapter=id,title=title,subsections=[dict(id=sid,title=stitle) for sid,stitle,_ in sections]) for id,title,short,intro,sections in chapters]
    (ROOT/'data').mkdir(parents=True,exist_ok=True)
    (ROOT/'data'/'structure.json').write_text(json.dumps(structure,indent=2)+'\n',encoding='utf-8')
    print(f'Built {PAGE.name}: {len(chapters)} chapters, {sum(len(x[4]) for x in chapters)} subsections, {equation_number} equations')


if __name__ == '__main__':
    build()
