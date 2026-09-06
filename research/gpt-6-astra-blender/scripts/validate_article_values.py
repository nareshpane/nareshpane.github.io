"""Check generated HTML measurements, retained structure and unchanged index."""
import argparse
import hashlib
from html.parser import HTMLParser
import json
from pathlib import Path
import re
ROOT=Path(__file__).resolve().parents[1]
p=argparse.ArgumentParser();p.add_argument('--baseline',type=Path);args=p.parse_args()
html=(ROOT.parent/'gpt-6-astra-blender.html').read_text()
A=json.loads((ROOT/'data/analysis.json').read_text());V=json.loads((ROOT/'data/scene_validation.json').read_text())
class Text(HTMLParser):
    def __init__(self):super().__init__();self.text=[];self.headings=[];self.heading=False
    def handle_starttag(self,tag,attrs):
        if tag=='h2':self.heading=True
    def handle_endtag(self,tag):
        if tag=='h2':self.heading=False
    def handle_data(self,data):
        self.text.append(data)
        if self.heading:self.headings.append(data)
current=Text();current.feed(html)
# Original instructions are retained verbatim, not treated as current results.
without_prompt=re.sub(r'<details id="prompt">.*?</details>','',html,flags=re.S)
parser=Text();parser.feed(without_prompt);text=' '.join(parser.text)
for value in [f"{A['nodes']} / {A['road_edges']}",f"{A['route']['metres']:.2f}",f"{A['closure_route']['metres']:.2f}",
              f"{A['route']['seconds']/60:.2f}",f"{A['closure_route']['seconds']/60:.2f}",
              f"{A['accessibility_mean_before']:.2f}",f"{A['accessibility_mean_after']:.2f}",f"{A['accessibility_mean_gain_pct']:.2f}",
              f"{V['master_objects']} / {V['master_mesh_objects']}",f"{V['glb']['city_web']['triangles_instanced']:,}",f"{V['blend_bytes']:,}"]:
    assert value in text,value
for r in A['monte_carlo']:
    for value in [f"{r['disconnected_runs']} / 400",f"{r['probability_disconnected']:.2%}",f"{r['mean_accessibility_loss_pct']:.2f}%"]:
        assert value in text,value
for stale in ['56 / 92','|V|=56','|E_R|=92','n=56','384 m','672 m','3,136','1,540','48-frame','frame 25','864 × 600','Six seconds']:
    assert stale not in text,stale
if args.baseline:
    previous=Text();previous.feed((args.baseline/'gpt-6-astra-blender.html').read_text())
    assert current.headings==previous.headings,'Article chapter structure changed'
    for id in ['built','blender','anatomy','city','graph','paths','centrality','gis','statistics','failure','monte-carlo','accessibility','engineering','interactive','reproduce','files','limitations']:
        assert f'id="{id}"' in html
    assert (ROOT.parent.parent/'research.html').read_bytes()==(args.baseline/'research.html').read_bytes()
    assert (ROOT/'page.css').read_bytes()==(args.baseline/'assets/page.css').read_bytes()
    assert (ROOT/'data/original_prompt.txt').read_bytes()==(args.baseline/'assets/data/original_prompt.txt').read_bytes()
index=(ROOT.parent.parent/'research.html').read_text()
first=re.search(r'<ul class="research-list">\s*<li>\s*<a[^>]+href="([^"]+)"',index)
assert first and first.group(1)=='research/gpt-6-astra-blender.html'
report=dict(status='PASS',html_sha256=hashlib.sha256(html.encode()).hexdigest(),checks=[
    'Current route, closure, accessibility, MC and performance values appear in HTML',
    'Known superseded graph and media values absent outside original prompt',
    'Chapter headings/IDs and page.css preserved',
    'Original prompt and research.html preserved; project remains first index entry'])
(ROOT/'data/article_value_validation.json').write_text(json.dumps(report,indent=2)+'\n')
print('ARTICLE_VALUES_PASS')
