"""Check browser assets, serialized data and read-only source hashes."""
from collections import Counter
import argparse
import ast
import hashlib
from html.parser import HTMLParser
import json
from pathlib import Path
import re
from urllib.parse import unquote, urlsplit
import xml.etree.ElementTree as ET

BASE = Path(__file__).resolve().parent
PAGE = BASE.parent/'canada-and-provinces-trade-by-hs.html'
SOURCE = Path('D:/Trade_Data_Scientist_Gov_Alberta/raw_data/statcan/CIMT-CICM_Dom_Exp_2025')

class References(HTMLParser):
    def __init__(self):
        super().__init__()
        self.links = []
        self.ids = []
    def handle_starttag(self, tag, attrs):
        a = dict(attrs)
        if 'id' in a:
            self.ids.append(a['id'])
        for key in ('src', 'href'):
            if key in a:
                self.links.append(a[key])

def read(name):
    return json.loads((BASE/name).read_text(encoding='utf-8'))

def main(source=SOURCE):
    for script in [*BASE.glob('*.py'), *BASE.glob('animation-assets/*.py')]:
        ast.parse(script.read_text(encoding='utf-8'), filename=str(script))
    checked = []
    for html in [PAGE, BASE.parent/'harmonized-system-index.html']:
        content = html.read_text(encoding='utf-8')
        assert not re.search(r'[A-Z]:[\\/]', content), 'Machine-specific path in HTML'
        parser = References()
        parser.feed(content)
        assert len(parser.ids) == len(set(parser.ids)), ('Duplicate ids', html.name)
        for ref in parser.links:
            url = urlsplit(ref)
            if url.scheme or url.netloc:
                continue
            assert not url.path.startswith('/'), ('Absolute path', ref)
            target = (html.parent/unquote(url.path)).resolve() if url.path else html
            assert target.exists(), ('Missing reference', html.name, ref)
            if url.fragment:
                if target.suffix == '.svg':
                    ids = {n.attrib.get('id') for n in ET.parse(target).getroot().iter()}
                    assert url.fragment in ids, ('Missing SVG id', ref)
                elif target.suffix == '.html':
                    target_parser = References()
                    target_parser.feed(target.read_text(encoding='utf-8'))
                    assert url.fragment in target_parser.ids, ('Missing anchor', ref)
            checked.append(ref)
    landing = (BASE.parent/'harmonized-system-index.html').read_text(encoding='utf-8')
    listing = landing.split('<ul class="research-list">', 1)[1].split('</ul>', 1)[0]
    links = re.findall(r'class="project-title" href="([^"]+)"', listing)
    assert links[links.index('section-338-hs4-hs6-exposure-canada.html')+1] == PAGE.name
    meta = read('geography-summary-2025.json')
    descriptions = read('product-descriptions.json')
    totals = {}
    product_values = Counter()
    destination_values = Counter()
    country_names = read('destination-countries-2025.json')['countries']
    destination_pairs = 0
    for g in meta['geographies']:
        trade = read(g['file'])
        destinations = read(f"destinations-{g['id']}-2025.json")
        assert destinations['year'] == 2025 and destinations['geography'] == g['id']
        assert set(destinations['headings']) == {h[0] for h in trade['headings']}
        assert trade['year'] == 2025 and trade['geography'] == g['id']
        assert trade['total'] == g['total'] == sum(h[1] for h in trade['headings'])
        assert len(trade['headings']) == g['hs4Count']
        assert sum(len(h[3]) for h in trade['headings']) == g['hs6Count']
        assert trade['headings'] == sorted(trade['headings'], key=lambda h: (-h[1], h[0]))
        totals[g['id']] = g['total']
        for rank, h in enumerate(trade['headings'], 1):
            assert h[0] in descriptions['hs4'] and h[0][:2] in descriptions['hs2']
            assert h[1] == sum(c[1] for c in h[3])
            countries = destinations['headings'][h[0]]
            assert h[1] == sum(v for c, v in countries), ('Destination/HS4 difference', g['id'], h[0])
            assert countries == sorted(countries, key=lambda r: (-r[1], r[0]))
            assert len(countries) == len({c for c, v in countries})
            assert all(c in country_names and c not in {'ZX', 'ZZ', 'CA', 'PC'} and isinstance(v, int) and v > 0 for c, v in countries)
            destination_pairs += len(countries)
            if g['id'] != 'CANADA':
                for c, v in countries:
                    destination_values[(h[0], c)] += v
            assert h[3] == sorted(h[3], key=lambda c: (-c[1], c[0]))
            assert abs(h[2] - h[1]/g['total']*100) < 1e-7
            assert meta['comparisons'][h[0]][g['id']][:2] == [h[1], rank]
            for c in h[3]:
                assert isinstance(c[0], str) and len(c[0]) == 6 and c[0][:4] == h[0]
                assert c[0] in descriptions['hs6'] and isinstance(c[1], int) and 0 < c[1] < 2**53
                assert abs(c[2] - c[1]/h[1]*100) < 1e-7
                assert abs(c[3] - c[1]/g['total']*100) < 1e-7
                if g['id'] != 'CANADA':
                    product_values[c[0]] += c[1]
    canada = read('trade-CANADA-2025.json')
    assert totals['CANADA'] == sum(v for g, v in totals.items() if g != 'CANADA')
    assert product_values == Counter({c[0]: c[1] for h in canada['headings'] for c in h[3]})
    national_destinations = read('destinations-CANADA-2025.json')['headings']
    assert destination_values == Counter({(h, c): v for h, rows in national_destinations.items() for c, v in rows})
    before = BASE/'.qa/refinement-before/hashes.json'
    if before.exists():
        for name, expected in json.loads(before.read_text(encoding='utf-8')).items():
            assert hashlib.sha256((BASE/name).read_bytes()).hexdigest() == expected, ('Existing aggregate asset changed', name)
    ET.parse(BASE/'animation-assets/world-map.svg')
    animation = read('animation-data-2025.json')
    assert animation['headings'] == read('trade-AB-2025.json')['headings'][:3]
    assert not re.search(r'Section\s*338|matched.scope', (BASE/'js/origin-exposure-animation.js').read_text(encoding='utf-8'), re.I)
    inspection = read('source-inspection.json')
    unchanged = []
    for entry in inspection['files']:
        with (source/entry['file']).open('rb') as f:
            digest = hashlib.file_digest(f, 'sha256').hexdigest()
        assert digest == entry['sha256'], ('External source changed', entry['file'])
        unchanged.append(entry['file'])
    report = {'html_references_checked': len(checked), 'geographies': len(meta['geographies']),
              'hs4': len(canada['headings']), 'hs6': len(product_values),
              'source_files_unchanged': unchanged, 'landing_entry_position': 'Immediately after Section 338',
              'serialized_hierarchy_ranks_shares_totals': 'PASS', 'animation_uses_actual_alberta_values': True,
              'relative_paths_and_fragments': 'PASS', 'python_script_syntax': 'PASS',
              'destination_country_or_territory_categories': len(country_names),
              'destination_origin_heading_country_pairs': destination_pairs,
              'destination_vs_hs4_difference': 0, 'national_destination_vs_13_origins_difference': 0,
              'destination_sort_codes_integer_values': 'PASS', 'existing_aggregate_assets_unchanged': True}
    (BASE/'static-validation.json').write_text(json.dumps(report, indent=2)+'\n', encoding='utf-8')
    print(json.dumps(report, indent=2))

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source', type=Path, default=SOURCE)
    main(parser.parse_args().source)
