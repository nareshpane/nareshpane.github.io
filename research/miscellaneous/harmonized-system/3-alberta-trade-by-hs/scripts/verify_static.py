"""Verify changed HTML references, JSON manifest sizes and linked annual data.

Run from repository root. This checks local files without making external requests.
"""
from html.parser import HTMLParser
import json
from pathlib import Path
from urllib.parse import unquote, urlsplit

BASE = Path(__file__).resolve().parents[1]
ROOT = BASE.parents[2]


class References(HTMLParser):
    def __init__(self):
        super().__init__()
        self.paths = []
        self.ids = []

    def handle_starttag(self, tag, attrs):
        values = dict(attrs)
        if 'id' in values:
            self.ids.append(values['id'])
        for field in ['src', 'href']:
            if field in values:
                self.paths.append(values[field])


paths_checked = 0
for name in ['alberta-trade-by-hs.html', 'harmonized-system-index.html']:
    path = BASE.parent / name
    parser = References()
    parser.feed(path.read_text(encoding='utf-8'))
    assert len(parser.ids) == len(set(parser.ids)), 'Duplicate HTML id'
    for href in parser.paths:
        url = urlsplit(href)
        if url.scheme or url.netloc:
            continue
        if url.path:
            target = (path.parent / unquote(url.path)).resolve()
            assert target.is_relative_to(ROOT) and target.is_file(), (name, href)
            paths_checked += 1
        elif url.fragment:
            assert url.fragment in parser.ids, (name, href)
index = (BASE.parent / 'harmonized-system-index.html').read_text(encoding='utf-8')
assert '>Alberta’s Export Atlas</a>' in index
assert 'href="alberta-trade-by-hs.html">Alberta’s Export Atlas</a> <span class="planned">' not in index
report = json.loads((BASE / 'scripts/build-validation.json').read_text(encoding='utf-8'))
for name, size in report['json_bytes'].items():
    assert (BASE / 'data' / name).stat().st_size == size, name
assert sum(c['total'] for c in report['summary']['countries']) == report['summary']['overall']['total']
assert all(isinstance(code,str) and len(code)==6 for code,_ in json.loads((BASE / 'data/alberta-products.json').read_text(encoding='utf-8'))['products'])
print(f'PASS: {paths_checked} local HTML references, unique IDs/fragments, Page 3 collection status, {len(report["json_bytes"])} JSON files/sizes and annual total.')
