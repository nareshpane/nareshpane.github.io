"""Verify generated annual data and local page assets without browser packages."""
from collections import defaultdict
from html.parser import HTMLParser
import json
from pathlib import Path
import re
from urllib.error import URLError
from urllib.parse import urlparse
from urllib.request import urlopen

BASE = Path(__file__).resolve().parents[1]
ROOT = BASE.parents[2]
PAGE = BASE.parent / 'section-338-hs4-hs6-exposure-canada.html'
LEGACY_PAGE = BASE.parent / 'preferential-tariffs-canada.html'

def read(name):
    return json.loads((BASE / 'data' / name).read_text(encoding='utf-8'))

meta, hs4, hs6, search, policy = map(read, ['metadata.json','exports-hs4-2025-us.json',
    'exports-hs6-2025-us.json','search-index.json','section338-hs6.json'])
assert len(meta['origins']) == 13
assert len({p[0] for p in meta['origins']}) == 13
assert {p[1] for p in meta['origins']} == {'Alberta','British Columbia','Manitoba','New Brunswick',
    'Newfoundland and Labrador','Nova Scotia','Ontario','Prince Edward Island','Quebec','Saskatchewan',
    'Northwest Territories','Nunavut','Yukon'}
for data, length in [(hs4,4),(hs6,6)]:
    for code, values in data.items():
        assert isinstance(code,str) and len(code) == length and code.isdigit()
        assert len(values) == 13 and all(isinstance(v,int) and v >= 0 for v in values)
    assert sum(map(sum,data.values())) == meta['validation']['annual_value_total']
    for i, (province, _) in enumerate(meta['origins']):
        assert sum(v[i] for v in data.values()) == meta['validation']['province_totals'][province]
independent4 = defaultdict(lambda:[0]*13)
for code, values in hs6.items():
    for i,value in enumerate(values):
        independent4[code[:4]][i] += value
assert dict(independent4) == hs4
assert {r[0] for r in search} == hs4.keys() | hs6.keys()
assert len(search) == len(hs4) + len(hs6)
assert len(set(policy['hs6'])) == len(policy['hs6'])
assert policy['matched_exports'] == sum(sum(v) for c,v in hs6.items() if c in set(policy['hs6']))
assert policy['positive_export_matches'] == sum(sum(v) > 0 for c,v in hs6.items() if c in set(policy['hs6']))
assert abs(meta['validation']['annual_value_total'] / 1e9 - 517.28) < .005
assert abs(policy['matched_exports'] / 1e9 - 36.36) < .005
assert policy['copies_identical']
for bundle in BASE.glob('data/tariffs-*.json'):
    data = json.loads(bundle.read_text(encoding='utf-8'))
    for code, item in data['products'].items():
        for row in item['rows']:
            assert row[0].replace('.','').startswith(code)
            assert len(row) == 6
            assert all(isinstance(i,int) and 0 <= i < len(data['strings']) for i in row[2:])

class Assets(HTMLParser):
    def __init__(self):
        super().__init__(); self.links = []; self.ids = []; self.references = []; self.nav = []
        self.refresh = []; self.canonical = []
    def handle_starttag(self,tag,attrs):
        attrs = dict(attrs)
        if 'id' in attrs: self.ids.append(attrs['id'])
        for key in ['src','href']:
            if key in attrs: self.links.append(attrs[key])
        for key in ['aria-controls','aria-describedby','aria-labelledby','for']:
            self.references.extend(attrs.get(key,'').split())
        if tag == 'meta' and attrs.get('http-equiv','').lower() == 'refresh':
            self.refresh.append(attrs.get('content'))
        if tag == 'link' and attrs.get('rel') == 'canonical':
            self.canonical.append(attrs.get('href'))

parser = Assets(); text = PAGE.read_text(encoding='utf-8'); parser.feed(text)
assert len(parser.ids) == len(set(parser.ids)), 'Duplicate HTML IDs'
assert set(parser.references) <= set(parser.ids), 'Dangling accessibility references'
assert not re.search('Planned|Under development|future study',text)
assert 'prefers-reduced-motion:reduce' in (BASE / 'css/style.css').read_text(encoding='utf-8')
assert 'matchMedia' in (BASE / 'js/app.js').read_text(encoding='utf-8')
local_assets = []
for link in parser.links:
    if urlparse(link).scheme or link.startswith('//'): continue
    file, _, fragment = link.partition('#')
    target = (PAGE.parent / file).resolve() if file else PAGE
    assert target.is_file(), f'Missing asset: {link}'
    if target == PAGE and fragment: assert fragment in parser.ids
    if file: local_assets.append(target)
assert BASE.name == '2-section-338-hs4-hs6-exposure-canada', 'Unexpected asset folder'
redirect = Assets(); redirect_text = LEGACY_PAGE.read_text(encoding='utf-8'); redirect.feed(redirect_text)
assert LEGACY_PAGE.stat().st_size < 1200, 'Redirect must remain small'
assert redirect.refresh == [f'0; url={PAGE.name}']
assert redirect.canonical == ['https://nareshpane.github.io/research/harmonized-system/' + PAGE.name]
assert PAGE.name in redirect.links, 'Missing normal fallback link'
assert f'location.replace("{PAGE.name}"+location.search+location.hash)' in redirect_text
assert not redirect.ids and '<script src=' not in redirect_text, 'Do not duplicate the application'
for code in ['8414','841490','9403','8537','060110']:
    assert code in (hs4 if len(code) == 4 else hs6)
print('Annual HS4/HS6, all 13 origins, independent child sums, policy matching, tariff prefixes, accessibility references and local asset paths: PASS')
print('Legacy route: tiny meta-refresh redirect, canonical URL, clickable fallback and query/fragment preservation: PASS')
try:
    for target in [PAGE,LEGACY_PAGE,*local_assets,*BASE.glob('data/*.json')]:
        relative = target.relative_to(ROOT).as_posix()
        with urlopen('http://localhost:8000/' + relative,timeout=5) as response:
            assert response.status == 200
    print('Local HTTP page, navigation, stylesheet, JavaScript and all 15 JSON assets: PASS')
except URLError as error:
    print('HTTP check unavailable (start python -m http.server 8000):',error)
print(json.dumps(meta['validation'],indent=2))
