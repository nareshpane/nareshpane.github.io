"""Build Alberta's observed 2025 domestic-export atlas (standard library only).

Run: python research/harmonized-system/3-alberta-trade-by-hs/scripts/build_data.py
External source files are opened read-only. No monthly observations are discarded.
Assertions run before output is written; unexpected duplicate keys stop the build.
"""
from __future__ import annotations
import argparse
from collections import Counter, defaultdict
import csv
import hashlib
import json
from pathlib import Path
import re

# CONFIGURATION — override --source on another machine; never copy raw data here.
SOURCE_DIR = Path('D:/Trade_Data_Scientist_Gov_Alberta/raw_data/statcan/CIMT-CICM_Dom_Exp_2025')
BASE = Path(__file__).resolve().parents[1]
HIERARCHY = BASE.parent / '1-harmonized-system-canada/data/hs-t2026-2.json'
YEAR = '2025'
PROVINCE = 'AB'
NON_MARKETS = {'ZX': 'Unknown or unspecified: not an identified destination market',
               'ZZ': 'High Seas: not a country or territorial destination',
               'PC': 'Pacific Islands: legacy grouped/ambiguous label, not an individually identified market; excluded conservatively (zero Alberta value)'}
MARKET_REVIEW = {'AQ': 'Retain separately reported Antarctica geographic destination (not a sovereign country)',
                 'EA': 'Retain separately reported Ceuta and Melilla territorial destination (zero Alberta value)',
                 'HK': 'Retain separately reported Hong Kong destination',
                 'MO': 'Retain separately reported Macao destination',
                 'TW': 'Retain separately reported Taiwan destination',
                 'XK': 'Retain separately reported Kosovo destination; nonstandard ISO-style code is not a reason to discard',
                 'CA': 'Recognized country lookup record, but zero Alberta exports; not an active market'}


def lookup(path, product=False):
    result = {}
    for line in path.read_text(encoding='cp1252').splitlines():
        start, end = line[11:17], line[18:24]
        assert start.isdigit() and end.isdigit()
        if start <= YEAR + '12' and end >= YEAR + '01':
            code = line[:6].strip() if product else line[:11].split()[0]
            text = line[29:111].strip() if product else line[25:107].strip()
            assert text and (code not in result or result[code] == text), code
            result[code] = text
    return result


def ranked(values):
    return sorted(((c, v) for c, v in values.items() if v > 0), key=lambda r: (-r[1], r[0]))


def products(values):
    """Single sparse [HS6 string, integer CAD] array; no redundant HS4/description payload."""
    return [[c, v] for c, v in ranked(values)]


def describe(values):
    headings = Counter()
    for code, value in values.items():
        headings[code[:4]] += value
    total = sum(values.values())
    order = ranked(headings)
    assert sum(headings.values()) == total
    for heading, amount in headings.items():
        assert sum(v for c, v in values.items() if c.startswith(heading)) == amount
    return {'total': total, 'hs4Count': len(order), 'hs6Count': len(ranked(values)),
            'top4': [[c, v] for c, v in order[:4]],
            'top4Share': sum(v for _, v in order[:4]) / total * 100 if total else 0}


def build(folder):
    schemas = []
    candidates = []
    for path in sorted(folder.glob('*.csv')):
        with path.open(encoding='utf-8-sig', newline='') as stream:
            header = next(csv.reader(stream))
        schemas.append({'file': path.name, 'columns': header})
        if len(header) == 8 and header[1] == 'HS6' and header[2:4] == ['Country/Pays', 'Province']:
            candidates.append(path)
    assert len(candidates) == 1, candidates
    source = candidates[0]
    countries = lookup(folder / 'ODPF_6_CtyDesc.TXT')
    assert lookup(folder / 'ODPF_8_ProvDesc.TXT')['48'] == 'Alberta'
    # Review every active label, not just a heuristic that might accept aggregates.
    suspicious = {c: n for c, n in countries.items() if re.search(
        r'world|total|aggregate|unspecified|high seas|other countries|pacific islands', n, re.I)}
    assert set(suspicious) <= set(NON_MARKETS), suspicious
    flows = defaultdict(Counter)
    excluded = Counter()
    before = Counter()
    seen_keys = {}
    months = set()
    all_codes = set()
    raw_rows = selected_rows = 0
    with source.open(encoding='utf-8-sig', newline='') as stream:
        reader = csv.reader(stream)
        next(reader)
        for row in reader:
            raw_rows += 1
            assert len(row) == 8
            month, code, country, province, state, value, quantity, unit = row
            assert re.fullmatch(r'\d{6}', code), code  # NEVER int(code).
            all_codes.add(code)
            if province != PROVINCE or not month.startswith(YEAR):
                continue
            selected_rows += 1
            assert month in {YEAR + f'{m:02}' for m in range(1, 13)}
            months.add(month)
            assert country in countries, ('Unmatched destination', country)
            assert re.fullmatch(r'\d+', value), ('Negative/noninteger CAD', row)
            amount = int(value)
            key = (month, code, country, province, state, unit)
            assert key not in seen_keys, ('Repeated complete observation key needs review', key,
                                           seen_keys.get(key), (value, quantity))
            seen_keys[key] = (value, quantity)
            before[country] += amount
            if country in NON_MARKETS:
                excluded[country] += amount
            else:
                flows[country][code] += amount
    assert months == {YEAR + f'{m:02}' for m in range(1, 13)}
    # Countries and separately identified territories are genuine destination markets.
    # Kosovo (XK) is explicitly retained; historical expired country records are excluded.
    global6 = Counter()
    for values in flows.values():
        global6.update(values)
    total = sum(global6.values())
    assert sum(before.values()) == total + sum(excluded.values())
    # A second source, at HS2, independently reconciles the annual province total.
    hs2_check = Counter()
    hs2_source = next(folder.glob('ODPFN020_*.csv'))
    with hs2_source.open(encoding='utf-8-sig', newline='') as stream:
        reader = csv.DictReader(stream)
        assert 'HS2' in reader.fieldnames
        for row in reader:
            if row['Province'] == PROVINCE and row[reader.fieldnames[0]].startswith(YEAR):
                hs2_check[row['Country/Pays']] += int(row['Value/Valeur'])
    assert before == hs2_check, 'Independent HS2 source differs from HS6 source'
    overall = describe(global6)
    destinations = []
    for rank, (code, amount) in enumerate(ranked({c: sum(v.values()) for c, v in flows.items()}), 1):
        info = describe(flows[code])
        assert info['total'] == before[code]
        destinations.append(dict(code=code, name=countries[code], rank=rank,
                                 share=amount / total * 100, **info))
    assert sum(c['total'] for c in destinations) == total == overall['total']
    stat6 = lookup(folder / 'ODPF_4_HS6XDesc.TXT', product=True)
    cbsa = json.loads(HIERARCHY.read_text(encoding='utf-8'))
    cbsa4, cbsa6, safe4, heading_audit = {}, {}, {}, []
    tariff_note = 'Note: The General Tariff rate that applies to goods of this tariff item is the Most-Favoured-Nation Tariff rate.'
    chapters = [ch for section in cbsa['sections'] for ch in section['chapters']] + cbsa['special_chapters']
    for chapter in chapters:
        for heading in chapter['headings']:
            code = heading['code']
            assert isinstance(code, str) and re.fullmatch(r'\d{4}', code)
            cbsa4[code] = heading['description']
            safe = heading['extraction'] == 'explicit_hs4_row' or (
                bool(heading['subheadings']) and all(
                    child['source_description'].split(' - ')[0] == heading['description']
                    for child in heading['subheadings']))
            if safe:
                text = heading['description']
                if text.endswith(tariff_note):
                    heading_audit.append({'code': code, 'action': 'Remove exact appended tariff-rate note from heading label',
                                          'original_description': text, 'removed_note': tariff_note})
                    text = text[:-len(tariff_note)].rstrip()
                safe4[code] = text
            else:
                heading_audit.append({'code': code, 'action': 'Reject unconfirmed inferred heading description; use code-only label'})
            for sub in heading['subheadings']:
                code6 = sub['code']
                assert isinstance(code6, str) and re.fullmatch(r'\d{6}', code6) and code6[:4] == code
                if not sub.get('source_warning') and 'null' not in sub['description'].lower():
                    cbsa6[code6] = sub['description']
    active6 = {c for c, v in global6.items() if v > 0}
    active4 = {c[:4] for c in active6}
    index = {'hs4': {c: safe4[c] for c in sorted(active4) if c in safe4},
             'hs6': {c: stat6.get(c, cbsa6.get(c)) for c in sorted(active6) if c in stat6 or c in cbsa6}}
    coverage = {'active_hs4_missing_cbsa': sorted(active4 - cbsa4.keys()),
                'active_hs6_missing_cbsa': sorted(active6 - cbsa6.keys()),
                'source_hs4_missing_cbsa': sorted({c[:4] for c in all_codes} - cbsa4.keys()),
                'source_hs6_missing_cbsa': sorted(all_codes - cbsa6.keys()),
                'active_hs6_missing_statcan': sorted(active6 - stat6.keys()),
                'hs4_missing_descriptions': sorted(active4 - index['hs4'].keys()),
                'hs6_missing_descriptions': sorted(active6 - index['hs6'].keys()),
                'leading_zero_hs4': sum(c.startswith('0') for c in active4),
                'leading_zero_hs6': sum(c.startswith('0') for c in active6)}
    # Approximate external checks only; none feeds the calculations above.
    checks = {'alberta': (total, 177.96e9), 'US': (before['US'], 152.04e9),
              'CN': (before['CN'], 10.19e9), 'JP': (before['JP'], 2.39e9),
              'top10': (sum(c['total'] for c in destinations[:10]), 171.12e9),
              'US_2709': (sum(v for c, v in flows['US'].items() if c.startswith('2709')), 110.82e9),
              'US_2711': (sum(v for c, v in flows['US'].items() if c.startswith('2711')), 11.57e9)}
    for label, (actual, target) in checks.items():
        assert abs(actual - target) / target < .005, (label, actual, target)
    summary = {'year': 2025, 'province': 'Alberta', 'currency': 'CAD',
               'overall': overall, 'countries': destinations,
               'top10Total': checks['top10'][0], 'top10Share': checks['top10'][0] / total * 100,
               'thresholds': {str(p): next(c['rank'] for i, c in enumerate(destinations)
                                if sum(d['total'] for d in destinations[:i + 1]) / total * 100 >= p)
                              for p in [90, 95, 99]}}
    report = {'source': source.name, 'source_bytes': source.stat().st_size,
              'source_sha256': hashlib.sha256(source.read_bytes()).hexdigest(), 'schemas': schemas,
              'raw_rows_examined': raw_rows, 'alberta_2025_source_rows': selected_rows,
              'complete_key': ['YearMonth', 'HS6', 'Country', 'Province', 'State', 'Unit of Measure'],
              'duplicate_keys': 0, 'months': sorted(months), 'removed_rows': 0,
              'country_filter': 'Active 2025 countries and separately identified territorial destinations; retain XK and AQ; exclude ZX, ZZ and ambiguous legacy grouped label PC. No world/continent totals; zero Alberta value for all excluded codes.',
              'unallocated_nonmarket_value': sum(excluded.values()),
              'country_validation': [{'code': c, 'name': n, 'alberta_value': before[c],
                                      'retained': c not in NON_MARKETS,
                                      'decision': NON_MARKETS.get(c, MARKET_REVIEW.get(c, 'Identified country/territorial destination; retained'))}
                                     for c, n in sorted(countries.items())],
              'coverage': coverage, 'hierarchy_source': cbsa['source'], 'heading_label_audit': heading_audit,
              'description_note': 'HS4 from Page 1 CBSA T2026-2 hierarchy by exact string code: explicit rows or inferred heading paths confirmed across all children; exact appended tariff-rate notes removed and recorded. HS6 from year-active StatCan export descriptions, CBSA fallback only. Code coverage does not certify unchanged wording between tariff editions. Unmatched HS4 remains a code-only label.',
              'reconciliation': 'PASS: exact integer CAD equality: source HS6 vs independent HS2 by destination; valid-country sum; overall HS4/HS6; every country HS4/HS6; every country-HS4 children. Nonnegative values; twelve months; unique complete keys.',
              'checks': {k: {'actual': a, 'approximate_target': t} for k, (a, t) in checks.items()},
              'summary': summary}
    outputs = {'summary.json': summary, 'product-index.json': index,
               'alberta-products.json': {'products': products(global6)}}
    outputs.update({f'countries/{c["code"]}.json': {'products': products(flows[c['code']])} for c in destinations})
    sizes = {}
    for name, content in outputs.items():
        payload = json.dumps(content, ensure_ascii=False, separators=(',', ':')).encode('utf-8')
        path = BASE / 'data' / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(payload)
        sizes[name] = len(payload)
    report['json_bytes'] = sizes
    (BASE / 'scripts/build-validation.json').write_text(json.dumps(report, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')
    print(json.dumps({'source': source.name, 'rows': raw_rows, 'alberta_rows': selected_rows,
                      'markets': len(destinations), **overall, 'coverage': coverage,
                      'nonmarket_value': sum(excluded.values()), 'checks': report['checks'],
                      'initial_json_bytes': sum(sizes[n] for n in ['summary.json', 'product-index.json', 'alberta-products.json']),
                      'all_json_bytes': sum(sizes.values())}, indent=2))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source', type=Path, default=SOURCE_DIR)
    build(parser.parse_args().source)
