"""Build the annual explorer using Python's standard library only.

Run from the repository root: python research/harmonized-system/2-section-338-hs4-hs6-exposure-canada/scripts/build_data.py
Raw sources are read-only. All assertions pass before any website JSON is written.
"""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import csv
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import re

# CONFIGURATION: change these paths, or use command-line overrides, on another machine.
STATCAN_DIR = Path('D:/Trade_Data_Scientist_Gov_Alberta/raw_data/statcan/CIMT-CICM_Dom_Exp_2025')
SECTION338_DIR = Path('D:/Trade_Data_Scientist_Gov_Alberta/raw_data/section338')
BASE = Path(__file__).resolve().parents[1]
CBSA_FILE = BASE.parent / '1-harmonized-system-canada/data/hs-t2026-2.json'
YEAR = '2025'
DESTINATION = 'US'  # Destination filtering is isolated for a future expansion.
PROVINCES = ['AB', 'BC', 'MB', 'NB', 'NL', 'NS', 'ON', 'PE', 'QC', 'SK', 'NT', 'NU', 'YT']
COLUMNS = ['YearMonth/AnnéeMois', 'HS6', 'Country/Pays', 'Province', 'State/État',
           'Value/Valeur', 'Quantity/Quantité', 'Unit of Measure/Unité de Mesure']
CODE_PATTERN = re.compile(r'\b\d{4}\.\d{2}\.\d{2}(?:\d{2})?\b')


def normalize_code(value):
    """Codes stay strings: punctuation is removed, leading zeroes are retained."""
    code = re.sub(r'[.\s_\-/]', '', value)
    assert code.isascii() and code.isdigit(), f'Invalid product code: {value!r}'
    return code


def fingerprint(path):
    return {'file': path.name, 'bytes': path.stat().st_size,
            'sha256': hashlib.sha256(path.read_bytes()).hexdigest()}


def active_lookup(path, product=False):
    """Verified fixed-width layout; dates restrict descriptions to the trade year.

    HS6: code 0:6, dates 11:17 and 18:24, unit 25:28, English 29:111.
    Geography: identifiers 0:11, same dates, English 25:107.
    Historical descriptions are never blindly joined on code alone.
    """
    result = {}
    for line in path.read_text(encoding='cp1252').splitlines():
        start, end = line[11:17], line[18:24]
        assert start.isdigit() and end.isdigit(), (path.name, line)
        if start <= YEAR + '12' and end >= YEAR + '01':
            key = line[:6].strip() if product else line[:11].split()[0]
            description = line[29:111].strip() if product else line[25:107].strip()
            assert description
            # A changing description within the year needs an explicit decision.
            assert key not in result or result[key] == description, (key, 'multiple active descriptions')
            result[key] = description
    return result


def choose_trade_file(folder):
    """Choose by actual header, never by the ODPFN numeric filename."""
    candidates = []
    schemas = []
    for path in sorted(folder.glob('*.csv')):
        with path.open(encoding='utf-8-sig', newline='') as stream:
            header = next(csv.reader(stream))
        schemas.append({'file': path.name, 'columns': header})
        if header == COLUMNS:
            candidates.append(path)
    assert len(candidates) == 1, f'Expected one detailed HS6 source, found {candidates}'
    return candidates[0], schemas


def aggregate_trade(path):
    """Retain every month and U.S. state, then sum to product × origin × year.

    Only a byte-equivalent complete CSV record can be a duplicate. A repeated
    province/country/product across months or states is never removed.
    Integer CAD amounts give exact reconciliations, better than a float tolerance.
    """
    seen = set()
    hs6 = defaultdict(lambda: [0] * len(PROVINCES))
    before, after = Counter(), Counter()
    raw_total = duplicate_value = selected_total = selected_duplicate_value = 0
    raw_rows = selected_rows = duplicate_rows = selected_duplicate_rows = 0
    months = set()
    key_counts = Counter()
    with path.open(encoding='utf-8-sig', newline='') as stream:
        reader = csv.reader(stream)
        assert next(reader) == COLUMNS
        for row in reader:
            assert len(row) == len(COLUMNS)
            raw_rows += 1
            ym, product, country, province, state, value, quantity, unit = row
            assert re.fullmatch(r'\d{6}', ym) and '01' <= ym[4:] <= '12'
            code = normalize_code(product)
            assert len(code) == 6
            assert re.fullmatch(r'\d+', value), f'Expected nonnegative integer CAD: {value}'
            dollars = int(value)  # Only the monetary measure is converted, never an HS code.
            raw_total += dollars
            selected = ym.startswith(YEAR) and country == DESTINATION
            if selected:
                assert province in PROVINCES, province
                selected_rows += 1
                selected_total += dollars
                before[province] += dollars
                months.add(ym[4:])
                key_counts[(ym, country, province, state, code, unit)] += 1
            complete_record = tuple(row)
            if complete_record in seen:
                duplicate_rows += 1
                duplicate_value += dollars
                if selected:
                    selected_duplicate_rows += 1
                    selected_duplicate_value += dollars
                continue
            seen.add(complete_record)
            if selected:
                hs6[code][PROVINCES.index(province)] += dollars
                after[province] += dollars
    assert months == {f'{m:02}' for m in range(1, 13)}, 'Incomplete calendar year'
    # Exact repeated records require a documented review if present; do not silently
    # treat coincident observations as erroneous duplication.
    assert duplicate_rows == 0, f'{duplicate_rows} exact repeats need source-record review before publishing'
    assert set(before) == set(PROVINCES) == set(after)
    assert before == after
    assert selected_total - selected_duplicate_value == sum(map(sum, hs6.values()))
    hs4 = defaultdict(lambda: [0] * len(PROVINCES))
    for code, values in hs6.items():
        for i, value in enumerate(values):
            hs4[code[:4]][i] += value
    canada6 = {code: sum(values) for code, values in hs6.items()}
    canada4 = {code: sum(values) for code, values in hs4.items()}
    for level, totals in [(hs6, canada6), (hs4, canada4)]:
        for code, values in level.items():
            assert len(values) == 13 and totals[code] == sum(values)
        for i, province in enumerate(PROVINCES):
            assert sum(values[i] for values in level.values()) == after[province]
        assert sum(totals.values()) == sum(after.values())
    audit = dict(raw_rows=raw_rows, raw_value_total=raw_total, selected_rows=selected_rows,
                 selected_value_before_dedup=selected_total, duplicate_rows=duplicate_rows,
                 duplicate_value=duplicate_value, selected_duplicate_rows=selected_duplicate_rows,
                 selected_duplicate_value=selected_duplicate_value, annual_value_total=sum(after.values()),
                 province_totals=dict(after), origins=13, months_aggregated=12,
                 hs6_products=len(hs6), hs4_products=len(hs4),
                 hs6_annual_rows=sum(v > 0 for values in hs6.values() for v in values),
                 hs4_annual_rows=sum(v > 0 for values in hs4.values() for v in values),
                 hs6_dense_rows=len(hs6) * 13, hs4_dense_rows=len(hs4) * 13,
                 repeated_keys_with_different_records=sum(n > 1 for n in key_counts.values()),
                 reconciliation='Exact integer equality for every product, province and Canada total')
    return dict(sorted(hs6.items())), dict(sorted(hs4.items())), audit


def policy_codes(text):
    return {normalize_code(m) for m in CODE_PATTERN.findall(text) if not m.startswith(('98', '99'))}


def section338(folder):
    """Reconstruct a supplied tariff-only snapshot; never certify current law.

    September Annex II replaces entire HTS8 parents with retained HTS10 children.
    Unrestricted ban lines are removed before collapsing to HS6. Packaged-only
    lines remain potentially tariff-covered and are flagged as partially banned.
    """
    canonical = folder / 's338_products_browser.csv'
    twin = folder / 's338_products_browser_extract.csv'
    with canonical.open(encoding='utf-8-sig', newline='') as stream:
        reader = csv.DictReader(stream)
        assert reader.fieldnames == ['hts8', 'program', 'ch99_heading']
        rows = list(reader)
    identical = canonical.read_bytes() == twin.read_bytes()
    assert identical, 'The supplied policy copies differ; review before selecting one'
    july = {normalize_code(r['hts8']) for r in rows}
    assert len(july) == len(rows) and all(len(c) == 8 for c in july)
    sources = [fingerprint(canonical), fingerprint(twin)]
    texts = {}
    for path in sorted(folder.glob('*.txt')):
        text = path.read_text(encoding='utf-8-sig')
        texts[path.stem] = text
        info = fingerprint(path)
        info['urls'] = list(dict.fromkeys(re.findall(r'https://[^\s)]+', text)))
        sources.append(info)
    scope = set(july)
    amendments = []
    try:
        for name in ['alcohol', 'auto']:
            description = texts[f'{name}_scope_browser']
            part_a, part_b = description.split('Part B.', 1)
            assert 'Part A.' in part_a and 'No Longer Covered' in part_b
            removed = policy_codes(part_b)
            annex = texts[f'{name}_hts_browser']
            assert 'September 15, 2026' in annex and 'deleting' in annex and 'inserting' in annex
            if name == 'alcohol':
                deleted_block, inserted_block = annex.split('b. inserting', 1)
                replacements = policy_codes(deleted_block)
                inserted = policy_codes(inserted_block)
            else:
                # The local PDF extraction places both tables after the instructions.
                # Annex I Part B identifies which HTS8 parents must be replaced.
                replacements = {c[:8] for c in removed}
                inserted = policy_codes(annex) - replacements
            assert replacements <= scope, f'Missing amendment parents: {replacements - scope}'
            assert policy_codes(part_a) <= inserted
            assert removed.isdisjoint(inserted)
            scope.difference_update(replacements)
            scope.update(inserted)
            amendments.append({'annex': name, 'deleted_parents': sorted(replacements),
                               'inserted_lines': sorted(inserted), 'removed_detail': sorted(removed)})
        bans, partial_bans = set(), set()
        for name in ['alcohol', 'dairy', 'motor']:
            text = texts[f'{name}_ban_browser']
            matches = list(CODE_PATTERN.finditer(text))
            for i, match in enumerate(matches):
                code = normalize_code(match.group())
                stop = matches[i + 1].start() if i + 1 < len(matches) else len(text)
                row_text = text[match.end():stop]
                # Packaged-only bans do not remove the entire tariff line:
                # unpackaged goods may remain in tariff scope. HS6 cannot split them.
                (partial_bans if 'Packaged' in row_text else bans).add(code)
        assert all('excluded from importation' in texts[f'{n}_ban_browser'] for n in ['alcohol', 'dairy', 'motor'])
        unmatched_bans = bans - scope
        # Some September ban lines (2204.29) were not in the July tariff universe.
        # They cannot remove any Canadian observations or create tariff matches.
        tariff_lines = {c for c in scope if not any(c.startswith(b) for b in bans)}
        mode = 'supplied-september-reconstruction'
        note = ('July lines, September 15 parent replacements/additions/removals, then supplied '
                'September unrestricted ban-listed lines removed at HTS8/HTS10 before HS6 collapse. '
                'Packaged-only bans retain the tariff line because other goods can remain covered. '
                'A local tariff-only scope reconstruction, not independently certified current law. '
                'Ban annexes do not themselves establish September 29 effective dates. '
                'Packaged-only restrictions and shipment exceptions cannot be resolved at Canadian HS6.')
    except (KeyError, ValueError, AssertionError) as error:
        tariff_lines, bans, partial_bans, unmatched_bans = july, set(), set(), set()
        mode = 'supplied-july-list'
        note = 'Appears in supplied Section 338 product list; reconstruction unavailable: ' + str(error)
    hs6 = sorted({c[:6] for c in tariff_lines})
    return dict(mode=mode, note=note, hs6=hs6, hs4=sorted({c[:4] for c in hs6}),
                july_records=len(rows), july_hs6=len({c[:6] for c in july}),
                copies_identical=identical, canonical_file=canonical.name,
                current_detailed_lines=sorted(tariff_lines), amendments=amendments,
                ban_lines=sorted(bans), partial_ban_lines=sorted(partial_bans),
                ban_lines_not_in_tariff_scope=sorted(unmatched_bans),
                sources=sources)


def product_metadata(raw, cbsa, hs6, hs4):
    descriptions = active_lookup(raw / 'ODPF_4_HS6XDesc.TXT', product=True)
    heading_nodes, tariff_nodes, cbsa_codes = {}, {}, set()
    chapters = [c for s in cbsa['sections'] for c in s['chapters']] + cbsa['special_chapters']
    for chapter in chapters:
        for h in chapter['headings']:
            heading_nodes[h['code']] = h
            for child in h['subheadings']:
                cbsa_codes.add(child['code'])
                tariff_nodes[child['code']] = child
    missing_descriptions = set(hs6) - descriptions.keys()
    assert not missing_descriptions, f'Missing StatCan HS6 descriptions: {missing_descriptions}'
    # Heading reuse is restricted to exact CBSA headings or a common heading path. This confirms
    # prefix coverage without assuming that 2026 national extensions describe 2025 exports.
    missing_headings = set(hs4) - heading_nodes.keys()
    search = []
    for code in hs4:
        node = heading_nodes.get(code)
        if node and node['extraction'] == 'explicit_hs4_row':
            description, source = node['description'], 0
        elif node and all(child['source_description'].split(' - ')[0] == node['description']
                          for child in node['subheadings']):
            # Confirm the heading path across ALL children. Never label a single
            # arbitrary HS6 child's commodity description as the entire heading.
            description, source = node['description'], 1
        else:
            description, source = 'Heading description unavailable in compatible local source', 2
        search.append([code, description, source])
    for code in hs6:
        search.append([code, descriptions[code], 3])
    compatibility = dict(statcan_hs6_in_cbsa=len(set(hs6) & cbsa_codes),
                         statcan_hs6_not_in_cbsa=sorted(set(hs6) - cbsa_codes),
                         headings_not_in_cbsa=sorted(missing_headings),
                         headings_without_safe_description=[r[0] for r in search if len(r[0]) == 4 and r[2] == 2],
                         note='Exact string prefixes only. 2026 CBSA import treatment is separate context; no claim of full year-to-year or import/export equivalence.')
    # Lazily loaded tariff bundles: ten first-digit groups, with repeated strings
    # pooled once per bundle. Preserve all national rows and source warnings.
    bundles = {}
    for digit in sorted({c[0] for c in hs6}):
        pool, index, products = [], {}, {}
        def pooled(value):
            if value not in index:
                index[value] = len(pool)
                pool.append(value)
            return index[value]
        for code in hs6:
            if code[0] != digit or code not in tariff_nodes:
                continue
            node = tariff_nodes[code]
            products[code] = {'url': node['source_url'], 'warning': node.get('source_warning'),
                             'rows': [[r['tariff_item'], r['statistical_suffix'],
                                       *[pooled(r[k]) for k in ['description', 'unit', 'mfn_tariff', 'preferential_tariffs']]]
                                      for r in node['canadian_tariff_lines']]}
        bundles[f'tariffs-{digit}.json'] = {'strings': pool, 'products': products}
    return search, bundles, compatibility


def build(args):
    path, schemas = choose_trade_file(args.statcan_dir)
    hs6, hs4, audit = aggregate_trade(path)
    names = {}
    for line in (args.statcan_dir / 'ODPF_8_ProvDesc.TXT').read_text(encoding='cp1252').splitlines():
        if line[11:17] <= YEAR + '12' and line[18:24] >= YEAR + '01':
            names[line[:11].split()[1]] = line[25:107].strip()
    assert set(names) == set(PROVINCES)
    countries = active_lookup(args.statcan_dir / 'ODPF_6_CtyDesc.TXT')
    assert countries[DESTINATION].startswith('United States'), countries[DESTINATION]
    cbsa = json.loads(args.cbsa_file.read_text(encoding='utf-8'))
    assert cbsa['schema_version'] == 2 and not cbsa['validation']['errors']
    search, tariff_bundles, compatibility = product_metadata(args.statcan_dir, cbsa, hs6, hs4)
    policy = section338(args.section338_dir)
    matched = [c for c in policy['hs6'] if c in hs6 and sum(hs6[c]) > 0]
    exposure = sum(sum(hs6[c]) for c in matched)
    policy.update(positive_export_matches=len(matched), matched_exports=exposure,
                  exposure_share=exposure / audit['annual_value_total'] * 100)
    meta = dict(schema_version=1, year=YEAR, destination=DESTINATION,
                destination_name=countries[DESTINATION], currency='CAD', measure='Domestic exports',
                generated_at=datetime.now(timezone.utc).isoformat(),
                origins=[[p, names[p]] for p in PROVINCES], source=fingerprint(path),
                source_columns=COLUMNS, inspected_schemas=schemas, validation=audit,
                description_sources=['CBSA T2026-2 explicit heading', 'CBSA heading path, confirmed across all children',
                                     'Unavailable in compatible local source', 'StatCan HS6 export lookup, valid during 2025'],
                description_compatibility=compatibility,
                cbsa_source={k: cbsa['source'][k] for k in ['title', 'url', 'edition', 'retrieved_at', 'chapter_effective_dates']},
                reference_comparison={'canada_target_approx': 517280000000,
                                      'canada_difference_from_rounded_target': audit['annual_value_total'] - 517280000000,
                                      'exposure_target_approx': 36360000000,
                                      'exposure_difference_from_rounded_target': exposure - 36360000000,
                                      'policy_hs6_reference': 412, 'positive_matches_reference': 396})
    outputs = {'metadata.json': meta, 'search-index.json': search,
               'exports-hs6-2025-us.json': hs6, 'exports-hs4-2025-us.json': hs4,
               'section338-hs6.json': policy, **tariff_bundles}
    # Values are stored only once per product. Canada is derived in the browser
    # from the same 13-element arrays and was independently reconciled above.
    output_dir = BASE / 'data'
    output_dir.mkdir(exist_ok=True)
    for name, value in outputs.items():
        destination = output_dir / name
        temporary = destination.with_suffix('.json.tmp')
        temporary.write_text(json.dumps(value, ensure_ascii=False, separators=(',', ':')), encoding='utf-8')
        temporary.replace(destination)
    report = {'source': path.name, **audit,
                      'section338_mode': policy['mode'], 'section338_hs6': len(policy['hs6']),
                      'section338_positive_matches': len(matched), 'section338_exposure': exposure,
                      'section338_share': policy['exposure_share'], 'description_compatibility': compatibility,
                      'output_bytes': {n: (output_dir / n).stat().st_size for n in outputs}}
    report_text = json.dumps(report, indent=2)
    (BASE / 'scripts/build-validation.json').write_text(report_text + '\n', encoding='utf-8')
    print(report_text)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--statcan-dir', type=Path, default=STATCAN_DIR)
    parser.add_argument('--section338-dir', type=Path, default=SECTION338_DIR)
    parser.add_argument('--cbsa-file', type=Path, default=CBSA_FILE)
    build(parser.parse_args())
