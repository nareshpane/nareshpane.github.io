"""Build geography-first 2025 domestic-export assets; Python 3.11+, no packages.

The external archive is read-only. CSVs are streamed; only one month's duplicate
keys and sparse annual product counters are retained. Assertions precede output.
Normal builds use a retained official 2025 classification snapshot, offline.
"""
from __future__ import annotations
import argparse
from collections import Counter, defaultdict
import csv
import hashlib
import json
from pathlib import Path
import re

BASE = Path(__file__).resolve().parent
DEFAULT = Path('D:/Trade_Data_Scientist_Gov_Alberta/raw_data/statcan/CIMT-CICM_Dom_Exp_2025')
YEAR = '2025'
ORIGINS = ['AB', 'BC', 'MB', 'NB', 'NL', 'NS', 'NT', 'NU', 'ON', 'PE', 'QC', 'SK', 'YT']
FIELDS = ['YearMonth/AnnéeMois', 'HS6', 'Country/Pays', 'Province', 'State/État',
          'Value/Valeur', 'Quantity/Quantité', 'Unit of Measure/Unité de Mesure']

def fingerprint(p):
    with p.open('rb') as f:
        digest = hashlib.file_digest(f, 'sha256').hexdigest()
    return {'file': p.name, 'bytes': p.stat().st_size, 'sha256': digest}

def lookup(p, product=False, province=False):
    result = {}
    ids = {}
    with p.open(encoding='cp1252') as f:
        for line in f:
            start, end = line[11:17], line[18:24]
            assert start.isdigit() and end.isdigit(), (p.name, line)
            if start <= YEAR+'12' and end >= YEAR+'01':
                parts = line[:11].split()
                code = parts[1] if province else line[:6].strip() if product else parts[0]
                description = line[29:111].strip() if product else line[25:107].strip()
                assert description and (code not in result or result[code] == description), ('Conflicting active label', code)
                result[code] = description
                if province:
                    ids[code] = parts[0]
    return result, ids

def ordered(values):
    return sorted(((c, v) for c, v in values.items() if v > 0), key=lambda x: (-x[1], x[0]))

def share(value, total):
    return round(value/total*100, 8) if total else None

def main(args):
    folder = args.source
    # This schema has no flow field: the supplied archive is already domestic
    # exports. A similarly-shaped import/re-export archive is not a substitute.
    assert 'Dom_Exp_2025' in folder.name, 'Use the supplied 2025 domestic-export archive (rename a relocated copy accordingly).'
    schemas = []
    files = {}
    for p in sorted(folder.glob('*.csv')):
        with p.open(encoding='utf-8-sig', newline='') as f:
            header = next(csv.reader(f))
        schemas.append({'file': p.name, 'columns': header})
        if len(header) in (6, 8) and header[1] in ('HS2', 'HS6', 'HS8'):
            level = header[1]
            assert level not in files, ('Ambiguous source level', level)
            expected = FIELDS.copy()
            expected[1] = level
            if level == 'HS2':
                expected = expected[:6]
            assert header == expected, (p.name, header)
            files[level] = p
    assert set(files) == {'HS2', 'HS6', 'HS8'}, files
    names, ids = lookup(folder/'ODPF_8_ProvDesc.TXT', province=True)
    countries, _ = lookup(folder/'ODPF_6_CtyDesc.TXT')
    states, _ = lookup(folder/'ODPF_7_StateDesc.TXT')
    desc6, _ = lookup(folder/'ODPF_4_HS6XDesc.TXT', product=True)
    desc2, _ = lookup(folder/'ODPF_5_HS2Desc.TXT')
    assert set(names) == set(ORIGINS) and len(set(ids.values())) == 13
    # None of these active destination labels is a world/country total. ZX, ZZ
    # and PC are reported destination categories, retained once, not discarded.
    assert not {c: n for c, n in countries.items() if re.search(r'\bworld\b|all countries|total countries|aggregate', n, re.I)}
    six = {g: Counter() for g in ORIGINS}
    month_chapters = Counter()
    destination_totals = Counter()
    destinations = {g: defaultdict(Counter) for g in ORIGINS}
    months = Counter()
    seen = set()
    seen_keys = set()
    previous = ''
    rows = zeros = duplicates = repeated_keys = 0
    with files['HS6'].open(encoding='utf-8-sig', newline='') as f:
        reader = csv.reader(f)
        assert next(reader) == FIELDS
        for row in reader:
            assert len(row) == 8, row
            ym, code, country, origin, state, value, quantity, unit = row
            assert re.fullmatch(r'\d{6}', ym) and '01' <= ym[4:] <= '12', ('Annual/invalid period', ym)
            assert ym >= previous, 'Monthly ordering changed: review duplicate-check partitioning.'
            if ym != previous:
                seen.clear()
                seen_keys.clear()
                previous = ym
            if not ym.startswith(YEAR):
                continue
            assert origin in six, ('Unknown/aggregate geography', origin)
            assert country in countries, ('Unknown/aggregate destination', country)
            assert not state or (country == 'US' and state in states), ('Invalid state/destination', state, country)
            assert re.fullmatch(r'\d{6}', code) and code in desc6, ('Invalid/inactive HS6', code)
            assert value.isascii() and value.isdigit(), ('Missing/suppressed/negative value', row)
            amount = int(value)
            complete = tuple(row)
            key = (ym, code, country, origin, state, unit)
            duplicates += int(complete in seen)
            repeated_keys += int(key in seen_keys)
            seen.add(complete)
            seen_keys.add(key)
            rows += 1
            zeros += int(amount == 0)
            six[origin][code] += amount
            month_chapters[(ym, origin, code[:2])] += amount
            destination_totals[country] += amount
            destinations[origin][code[:4]][country] += amount
            months[ym] += 1
    assert not duplicates and not repeated_keys, ('Duplicate records/keys need review', duplicates, repeated_keys)
    assert set(months) == {YEAR+f'{m:02}' for m in range(1, 13)}, 'Incomplete 2025'
    # Independent HS2 source, compared at month × origin × chapter.
    check2 = Counter()
    with files['HS2'].open(encoding='utf-8-sig', newline='') as f:
        reader = csv.reader(f)
        next(reader)
        for row in reader:
            ym, code, country, origin, state, value = row
            assert re.fullmatch(r'\d{6}', ym) and '01' <= ym[4:] <= '12'
            if ym.startswith(YEAR):
                assert origin in six and country in countries and re.fullmatch(r'\d{2}', code) and value.isdigit()
                check2[(ym, origin, code)] += int(value)
    assert check2 == month_chapters, ('HS2 reconciliation differences', (check2-month_chapters).most_common(5), (month_chapters-check2).most_common(5))
    # Independent HS8 source, compared at origin × HS6 (not added to HS6).
    check8 = {g: Counter() for g in ORIGINS}
    with files['HS8'].open(encoding='utf-8-sig', newline='') as f:
        reader = csv.reader(f)
        next(reader)
        for row in reader:
            ym, code, country, origin, state, value, quantity, unit = row
            assert re.fullmatch(r'\d{6}', ym) and '01' <= ym[4:] <= '12'
            if ym.startswith(YEAR):
                assert origin in six and re.fullmatch(r'\d{8}', code) and value.isdigit()
                check8[origin][code[:6]] += int(value)
    assert all(check8[g] == six[g] for g in ORIGINS), 'HS8-to-HS6 reconciliation differs'
    six['CANADA'] = Counter()
    for g in ORIGINS:
        six['CANADA'].update(six[g])
    destinations['CANADA'] = defaultdict(Counter)
    for g in ORIGINS:
        for heading, values in destinations[g].items():
            destinations['CANADA'][heading].update(values)
    # These source-defined categories are not individual international countries.
    # They remain in export totals; any nonzero residual is reported, never scaled.
    excluded_destinations = {'ZX', 'ZZ', 'PC', 'CA'}
    destination_differences = []
    country_pairs = 0
    positive_countries = set()
    hs4codes = {c[:4] for c, v in six['CANADA'].items() if v > 0}
    hs6codes = {c for c, v in six['CANADA'].items() if v > 0}
    local = json.loads((BASE/'hs4-cbsa-reference.json').read_text(encoding='utf-8'))
    official = json.loads((BASE/'hs4-official-2025-extract.json').read_text(encoding='utf-8'))
    # Prefer the exact-year export classification. Later CBSA heading wording is
    # a transparently identified label reference only; no tariff data enter this
    # analysis. Both use the HS 2022 four-digit hierarchy.
    labels = {c: r['description'] for c, r in local['hs4'].items()}
    label_sources = {c: {'source': 'CBSA T2026-2 retained heading label', 'url': r['url'], 'method': r['method']} for c, r in local['hs4'].items()}
    exact_headings = set()
    for chapter in official['chapters'].values():
        for row in chapter['rows']:
            code = row[0].replace('.', '')
            if len(code) == 4:
                labels[code] = row[2]
                exact_headings.add(code)
                label_sources[code] = {'source': 'Statistics Canada Export Classification 2025', 'url': chapter['url'], 'method': 'Explicit HS4 row'}
    for chapter in official['chapters'].values():
        for row in chapter['rows']:
            code = row[0].replace('.', '')
            if len(code) == 6 and code.endswith('00') and code[:4] not in exact_headings and not row[2].startswith('-'):
                assert {c for c in desc6 if c.startswith(code[:4])} == {code}, ('Unsplit heading has multiple active children', code)
                labels[code[:4]] = row[2]
                label_sources[code[:4]] = {'source': 'Statistics Canada Export Classification 2025', 'url': chapter['url'], 'method': 'Sole unsplit .00 row'}
    if '9802' not in labels:
        compatible = {c for c in desc6 if c.startswith('9802')}
        assert compatible == {'980200'}, compatible
        labels['9802'] = desc6['980200']
        label_sources['9802'] = {'source': 'StatCan CIMT export lookup, valid during 2025', 'file': 'ODPF_4_HS6XDesc.TXT', 'method': 'Sole unsplit 980200 subheading; no arbitrary child wording'}
    reference = {'year': 2025, 'source': 'StatCan 2025 export classification; documented CBSA heading-label supplement',
                 'url': official['url'], 'hs4': labels}
    assert not hs4codes - reference['hs4'].keys(), ('Missing official HS4 labels', sorted(hs4codes-reference['hs4'].keys()))
    national = sum(six['CANADA'].values())
    assert national == sum(sum(six[g].values()) for g in ORIGINS) == sum(destination_totals.values())
    outputs = {}
    summaries = []
    comparisons = defaultdict(dict)
    for g in ['CANADA', *ORIGINS]:
        products = six[g]
        total = sum(products.values())
        headings = Counter()
        children = defaultdict(dict)
        for c, v in products.items():
            assert len(c) == 6 and c[:4] in reference['hs4']
            if v > 0:
                headings[c[:4]] += v
                children[c[:4]][c] = v
        ranking = ordered(headings)
        assert all(a[1] >= b[1] for a, b in zip(ranking, ranking[1:]))
        records = []
        destination_records = {}
        for rank, (h, value) in enumerate(ranking, 1):
            child_order = ordered(children[h])
            assert value == sum(v for _, v in child_order), ('HS4/HS6 difference', g, h)
            assert all(a[1] >= b[1] for a, b in zip(child_order, child_order[1:]))
            records.append([h, value, share(value, total), [[c, v, share(v, value), share(v, total)] for c, v in child_order]])
            comparisons[h][g] = [value, rank, share(value, total)]
            all_destinations = destinations[g][h]
            assert sum(all_destinations.values()) == value, ('Destination source reconciliation', g, h)
            eligible = ordered({c: v for c, v in all_destinations.items() if c not in excluded_destinations})
            destination_records[h] = [[c, v] for c, v in eligible]
            country_pairs += len(eligible)
            positive_countries.update(c for c, _ in eligible)
            difference = value - sum(v for _, v in eligible)
            if difference:
                destination_differences.append({'geography': g, 'hs4': h, 'recorded': value,
                                                'countries': value-difference, 'difference': difference})
        assert sum(h[1] for h in records) == total
        summaries.append({'id': g, 'statcan_id': ids.get(g), 'name': 'Canada' if g == 'CANADA' else names[g],
                          'year': 2025, 'total': total, 'hs4Count': len(ranking), 'hs6Count': sum(v > 0 for v in products.values()),
                          'canadaShare': share(total, national), 'largest': list(ranking[0]) if ranking else None,
                          'file': f'trade-{g}-2025.json'})
        outputs[f'destinations-{g}-2025.json'] = {'geography': g, 'year': 2025, 'headings': destination_records}
        outputs[f'trade-{g}-2025.json'] = {'geography': g, 'year': 2025, 'total': total, 'headings': records}
    # Verify all per-product Canada constructions, not only the grand total.
    assert all(six['CANADA'][c] == sum(six[g][c] for g in ORIGINS) for c in six['CANADA'])
    for h in hs4codes:
        assert comparisons[h]['CANADA'][0] == sum(comparisons[h].get(g, [0])[0] for g in ORIGINS)
    audit = {'year': 2025, 'measure': 'Domestic exports', 'currency': 'CAD', 'basis': 'Customs-based',
             'destination': 'All reported destinations, including unknown/unspecified and high seas',
             'sources': [fingerprint(p) for p in files.values()], 'lookup_sources': [fingerprint(p) for p in sorted(folder.glob('*.TXT'))],
             'classification_sources': [fingerprint(BASE/'hs4-official-2025-extract.json'), fingerprint(BASE/'hs4-cbsa-reference.json')], 'inspected_schemas': schemas,
             'hs4_description_sources': {c: label_sources[c] for c in sorted(hs4codes)},
             'later_cbsa_heading_labels': sorted(c for c in hs4codes if label_sources[c]['source'].startswith('CBSA')),
             'source_rows': rows, 'zero_source_rows': zeros, 'duplicates': duplicates, 'repeated_observation_keys': repeated_keys,
             'months': dict(sorted(months.items())), 'geographies': 14, 'origins': 13, 'hs4_positive': len(hs4codes),
             'hs6_positive': len(hs6codes), 'annual_total': national, 'national_record_present': False,
             'national_method': 'Sum of 13 distinct origin identifiers. Active province lookup covers all ten provinces and three territories. No Canada or residual origin records exist.',
             'destination_categories': len(destination_totals), 'special_destination_values': {c: {'name': countries[c], 'value': destination_totals[c]} for c in ['ZX', 'ZZ', 'PC', 'CA']},
             'destination_totals': dict(sorted(destination_totals.items())),
             'validation': {'hs4_equals_hs6': True, 'rankings_descending': True, 'tie_break': 'Ascending string code; ordinal rank',
                            'canada_equals_13_origins': True, 'hs2_month_origin_chapter_difference': 0,
                            'hs8_origin_hs6_difference': 0, 'annual_records_added': 0, 'suppressed_or_missing_value_rows': 0,
                            'missing_hs4_descriptions': [], 'missing_hs6_descriptions': [], 'geographic_reconciliation_difference': 0},
             'notes': ['No trade-flow column: archive selection identifies domestic exports, excluding re-exports.',
                       'Destination categories contain no world total. Every source record is included once.',
                       'U.S. states are summed, not layered with a state total. Other destinations have blank state fields.',
                       'Values are integer Canadian dollars. No quantity sums across incompatible units.',
                       'Zero source values are retained in audits and excluded from positive-export rankings; absent products mean no positive recorded exports.',
                       'Chapters 98 and 99 are Canadian special classification provisions, not internationally harmonized product chapters.',
                       'Source description lookup snapshot is marked 202606; validity intervals are restricted to 2025.',
                       'No independent national-origin observation exists; reconciliation tests internal source consistency, not an external national benchmark.',
                       'Export descriptions are the official abbreviated CIMT labels. Shares are stored to eight decimals; monetary precision remains exact.']}
    audit['destination_validation'] = {
        'positive_country_or_territory_categories': len(positive_countries),
        'geography_heading_country_pairs': country_pairs,
        'excluded_categories': {c: {'name': countries[c], 'value': destination_totals[c]} for c in sorted(excluded_destinations)},
        'all_reported_destination_totals_equal_hs4': True,
        'individual_destination_difference_count': len(destination_differences),
        'individual_destination_differences': destination_differences,
        'maximum_absolute_difference': max((abs(d['difference']) for d in destination_differences), default=0),
        'tie_break': 'Ascending official country code; ordinal rank',
        'denominator': 'Full recorded origin-HS4 total, including any excluded destination categories'}
    outputs['destination-countries-2025.json'] = {'year': 2025,
        'source': 'Statistics Canada ODPF_6_CtyDesc.TXT; validity intervals overlapping 2025',
        'countries': {c: countries[c] for c in sorted(positive_countries)}}
    outputs['geography-summary-2025.json'] = {'year': 2025, 'currency': 'CAD', 'geographies': summaries,
                                            'comparisons': dict(sorted(comparisons.items()))}
    outputs['product-descriptions.json'] = {'hs2': {c: desc2[c] for c in sorted({h[:2] for h in hs4codes})},
                                          'hs4': {c: reference['hs4'][c] for c in sorted(hs4codes)},
                                          'hs6': {c: desc6[c] for c in sorted(hs6codes)}, 'source': reference['source'], 'url': reference['url'],
                                          'hs4_sources': {c: label_sources[c] for c in sorted(hs4codes)}}
    # Animation uses observed Alberta bars; it does not hard-code fictitious widths.
    outputs['animation-data-2025.json'] = {'geography': 'Alberta', 'headings': outputs['trade-AB-2025.json']['headings'][:3],
        'destinations': outputs['destinations-AB-2025.json']['headings']['2711'][:3],
        'countryNames': {c: countries[c] for c, _ in outputs['destinations-AB-2025.json']['headings']['2711'][:3]}}
    for name, value in outputs.items():
        (BASE/name).write_text(json.dumps(value, ensure_ascii=False, separators=(',', ':'))+'\n', encoding='utf-8')
    audit['output_bytes'] = {name: (BASE/name).stat().st_size for name in outputs}
    (BASE/'validation-2025.json').write_text(json.dumps(audit, ensure_ascii=False, indent=2)+'\n', encoding='utf-8')
    print(json.dumps({'total': national, 'rows': rows, 'geographies': 14, 'hs4': len(hs4codes), 'hs6': len(hs6codes),
                      'validation': audit['validation'], 'outputs_bytes': audit['output_bytes']}, indent=2))

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source', type=Path, default=DEFAULT)
    main(parser.parse_args())
