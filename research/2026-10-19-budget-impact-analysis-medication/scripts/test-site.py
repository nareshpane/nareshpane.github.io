"""Dependency-free asset, markup and source-preservation audit."""
from pathlib import Path
from html.parser import HTMLParser
from urllib.parse import unquote, urlsplit
import hashlib, json, re
import sys
root = Path(__file__).resolve().parents[1]
html = root / '2026-10-19-budget-impact-analysis-medication.html'
class Audit(HTMLParser):
    def __init__(self):
        super().__init__(); self.ids=[]; self.links=[]; self.sections=0
    def handle_starttag(self, tag, attrs):
        a=dict(attrs)
        if 'id' in a: self.ids.append(a['id'])
        if tag=='section': self.sections+=1
        for key in ('href','src'):
            if key in a:self.links.append(a[key])
a=Audit();a.feed(html.read_text(encoding='utf-8'))
assert len(a.ids)==len(set(a.ids)), 'Duplicate IDs'
checked=[]
for link in a.links:
    if link.startswith('#'):
        assert link[1:] in a.ids, link
    elif not urlsplit(link).scheme:
        file=(root / unquote(urlsplit(link).path)).resolve()
        assert file.exists(), str(file)
        checked.append(link)
js=(root/'js/app.js').read_text(encoding='utf-8')
dynamic=set(re.findall(r'\bid="([^"$]+)"',js))
for id in re.findall(r"\$\('([^']+)'\)",js):
    assert id in a.ids or id in dynamic, f'Unresolved DOM ID {id}'
for path in root.glob('js/*.js'):
    for dep in re.findall(r"from\s+'(\.[^']+)'",path.read_text(encoding='utf-8')):
        assert (path.parent/dep).exists(),dep
# Include local assets referenced by dynamically rendered SVG artwork.
for path in root.glob('js/*.js'):
    for link in re.findall(r'(?:href|src)="([^"$]+)"',path.read_text(encoding='utf-8')):
        if not urlsplit(link).scheme and not link.startswith('#'):
            assert (root / link.split('#')[0]).exists(), link
            checked.append(link)
assert '<svg' in (root/'figures/bc-outline.svg').read_text(encoding='utf-8')
engine=hashlib.sha256((root/'js/model.js').read_bytes()).hexdigest().upper()
assert engine=='4AD3930F3EEB7BC11177BD47C97731935F0718EAEFFACC3A64C7D7886DDA91C0', 'Calculation engine changed'
original=root/'SR0781-Cosentyx_Combined_Review CADTH Pharmacoeconomic Review - BIA.pdf'
digest=hashlib.sha256(original.read_bytes()).hexdigest().upper()
assert digest=='A457422B67B5792E57854BE310084DAA53D0D183D325B48B72E671A3B0439A45','Original PDF changed'
assert not re.search(r'[A-Z]:[\\/]',html.read_text(encoding='utf-8')), 'Machine-specific URL'
browser_report=root/'data/browser-validation.json'
browser_status='PASS: actual desktop/tablet/mobile browser regression and screenshot capture' if browser_report.exists() and all(r.get('status')=='PASS' for r in json.loads(browser_report.read_text())) else 'Not yet verified by browser regression'
report={'status':'PASS','htmlIDs':len(a.ids),'sections':a.sections,'localLinksChecked':checked,'originalPDFSHA256':digest,'calculationEngineSHA256':engine,'browserVisualQA':browser_status}
(root/'data/static-validation.json').write_text(json.dumps(report,indent=2)+'\n',encoding='utf-8')
print(json.dumps(report,indent=2))
if '--http' in sys.argv:
    from urllib.request import urlopen
    prefix='http://127.0.0.1:8000/research/2026-10-19-budget-impact-analysis-medication/'
    assets=['2026-10-19-budget-impact-analysis-medication.html','css/style.css','js/app.js','js/model.js','js/art.js','js/assumptions.js','figures/bc-outline.svg','css/revision.css','js/illustrations.js','js/storyboard.js','js/video.js','js/education.js','figures/canada-provinces.svg']
    responses=[]
    for asset in assets:
        with urlopen(prefix+asset,timeout=5) as response:
            assert response.status==200
            responses.append({'path':asset,'status':response.status,'type':response.headers.get('Content-Type'),'bytes':len(response.read())})
    (root/'data/http-validation.json').write_text(json.dumps({'status':'PASS','assets':responses},indent=2)+'\n',encoding='utf-8')
    print('HTTP PASS:',len(assets),'local page/module/graphic requests; correct serving from repository root.')
