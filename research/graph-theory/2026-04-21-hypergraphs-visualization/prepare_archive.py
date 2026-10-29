"""Build an offline reading view from the exact recovered April 21 Git source.
The .html.txt source remains unchanged; only paths, typesetting and scoped styling adapt.
Run with --apply to replace the marked archive block in the canonical page.
"""
from pathlib import Path
import hashlib, html, json, re, struct, sys

ASSETS = Path(__file__).resolve().parent
ARCHIVE = ASSETS / 'archive'
PAGE = ASSETS.with_suffix('.html')
source_bytes = (ARCHIVE / 'april-21-2026-original.html.txt').read_bytes()
source = source_bytes.decode('utf-8-sig')
provenance = json.loads((ARCHIVE / 'provenance.json').read_text(encoding='utf-8'))
assert hashlib.sha256(source_bytes).hexdigest() == provenance['originalSha256']

def scoped_css(css):
    result, pos = [], 0
    while pos < len(css):
        start = css.find('{', pos)
        if start < 0: break
        selector = css[pos:start].strip()
        depth, end = 1, start + 1
        while depth:
            if css[end] == '{': depth += 1
            if css[end] == '}': depth -= 1
            end += 1
        content = css[start+1:end-1]
        if selector.startswith('@media'):
            result.append(selector + '{' + scoped_css(content) + '}')
        else:
            selectors = [('#april-snapshot' if x.strip() in {':root','body','html'} else '#april-snapshot '+x.strip()) for x in selector.split(',')]
            result.append(','.join(selectors) + '{' + content + '}')
        pos = end
    return '\n'.join(result)

css = scoped_css(re.search(r'<style>(.*?)</style>', source, re.S).group(1))
css = css.replace('"Inter", sans-serif', '"Inter", "Segoe UI", Arial, sans-serif').replace('"Cormorant Garamond", serif', '"Cormorant Garamond", Garamond, Georgia, serif').replace('"Source Code Pro", monospace', '"Source Code Pro", Consolas, monospace')
css += '''
/* Reading-view adaptations; the exact original CSS is retained in the source download. */
#april-snapshot{min-width:0;margin-top:18px}
#april-snapshot .page{padding:0;max-width:none}
#april-snapshot .page>.card{padding:14px 0;border:0;box-shadow:none}
#april-snapshot .toc{display:block;border:0;padding:0;margin:10px 0 14px 20px;columns:2;column-gap:30px}
#april-snapshot .example-card{border-top:0}
#april-snapshot .april-math,#april-snapshot .april-equation{font-family:Cambria,"Times New Roman",serif}
#april-snapshot .april-equation{margin:12px 0;line-height:1.8;overflow-wrap:break-word}
#april-snapshot .april-math{overflow-wrap:break-word}
#april-snapshot .center-figure img{height:auto}
#april-snapshot .figcaption{width:auto;max-width:none}
@media(max-width:720px){#april-snapshot .toc{columns:1}#april-snapshot .panel{padding:13px}#april-snapshot .card{padding:0}}
'''
(ARCHIVE / 'april-21-2026.css').write_text(css, encoding='utf-8')
body = re.search(r'<body>(.*?)</body>', source, re.S).group(1).strip()
body = body.replace('id="top"', 'id="april-top"').replace('href="#top"', 'href="#april-top"')
body = re.sub(r'\s+onclick="[^"]*"', '', body)
body = body.replace('href="index.html"', 'href="../../index.html"')

def image(match):
    name = match.group(1)
    image_path = ASSETS / name
    assert image_path.exists()
    width, height = struct.unpack('>II', image_path.read_bytes()[16:24])
    return f'src="{ASSETS.name}/{name}" loading="lazy" width="{width}" height="{height}"'
body = re.sub(r'src="hypergraph_outputs/([^"]+\.png)"', image, body)

def math_markup(text, display=False):
    text = text.replace(r'\{','{').replace(r'\}','}').replace(r'\;', ' ')
    parts = text.split(r'\qquad')
    parts = [re.sub(r'([A-Za-z])_(\d+)', r'\1<sub>\2</sub>', html.escape(p.strip())) for p in parts]
    rendered = ' <wbr> · '.join(parts)
    return ('<div class="april-equation">'+rendered+'</div>') if display else '<span class="april-math">'+rendered+'</span>'
body = re.sub(r'\$\$(.*?)\$\$', lambda m:math_markup(m.group(1),True), body, flags=re.S)
body = re.sub(r'\$([^$]+)\$', lambda m:math_markup(m.group(1)), body)
assert '$' not in body and r'\qquad' not in body
body = '\n'.join(line.rstrip() for line in body.splitlines())
fragment = '''<!-- ARCHIVE_SNAPSHOT_START -->
<section class="lesson archive-section" id="gallery">
<div class="eyebrow">Research history · recovered source</div>
<details id="original-april-2026">
<summary><h2>Original Hypergraph Exploration — April 21, 2026</h2></summary>
<div class="archive-provenance">
<p>An archived presentation of the original study, preserving its mathematical explanations, visual demonstrations, and model-era approach. Recovered from Git commit <a href="https://github.com/nareshpane/nareshpane.github.io/commit/859ef8a7e2899211e1b96c52455999557e6e993c">859ef8a, April 21, 2026</a>.</p>
<p>The recovered page contains ten static examples, in-page navigation, and mathematical typesetting. It has no animated builder and records no model name or reasoning configuration. Its original wording and limitations are retained here. This offline reading view updates asset paths, scopes the original styling, and renders its simple set formulas in HTML; the preserved source download is exact.</p>
<p><a href="2026-04-21-hypergraphs-visualization/archive/april-21-2026-original.html.txt" download>Download the unchanged April 21 HTML source</a> · <a href="2026-04-21-hypergraphs-visualization/archive/provenance.json">Source provenance</a> · <a href="2026-04-21-hypergraphs-visualization/hypergraph_gallery_animation(3).py">Original figure generator</a></p>
</div>
<div id="april-snapshot">
''' + body + '''
</div>
</details>
</section>
<!-- ARCHIVE_SNAPSHOT_END -->'''
(ARCHIVE / 'april-21-2026-fragment.html').write_text(fragment, encoding='utf-8')
if '--apply' in sys.argv:
    page = PAGE.read_text(encoding='utf-8')
    if '<!-- ARCHIVE_SNAPSHOT_START -->' in page:
        page = re.sub(r'<!-- ARCHIVE_SNAPSHOT_START -->.*?<!-- ARCHIVE_SNAPSHOT_END -->', fragment, page, flags=re.S)
    else:
        start = page.index('<section class="lesson" id="gallery">')
        end = page.index('<section class="lesson" id="references">', start)
        page = page[:start] + page[end:]
        marker = '<section class="lesson bitcoin-study" id="bitcoin">'
        page = page.replace(marker,fragment+'\n'+marker,1)
    PAGE.write_text(page, encoding='utf-8')
print('Recovered source preserved; built offline archive reading view.')
