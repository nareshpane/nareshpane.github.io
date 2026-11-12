#!/usr/bin/env python3
"""Build the small browser HS6 bridge from the Budget Lab's s338_products CSV.

Usage: python build_section338_hs6.py PATH/TO/s338_products_browser.csv
Uses Python's standard library only; raw CSV files are not copied into the site.
"""
import argparse
import csv
import hashlib
import json
from pathlib import Path
import re

BASE = Path(__file__).resolve().parent.parent
SOURCE_URL = 'https://github.com/Budget-Lab-Yale/tariff-rate-tracker/blob/master/resources/s338_products.csv'


def normalize_hts8(value):
    # Keep leading zeros; reject letters, ranges, scientific notation and other
    # unexpected input rather than silently stripping everything non-numeric.
    if not re.fullmatch(r'[0-9.\s-]+', value or ''):
        raise ValueError(f'Invalid HTS8 value: {value!r}')
    code = re.sub(r'[.\s-]', '', value)
    if not re.fullmatch(r'[0-9]{8}', code):
        raise ValueError(f'Expected eight digits in hts8: {value!r}')
    if code[:2] in {'77', '98', '99'}:
        raise ValueError(f'Reserved/national chapter cannot provide an international HS6 bridge: {code}')
    return code


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('csv_path', type=Path)
    args = parser.parse_args()
    with args.csv_path.open(encoding='utf-8-sig', newline='') as source:
        reader = csv.DictReader(source)
        if reader.fieldnames != ['hts8', 'program', 'ch99_heading']:
            raise ValueError(f'Unexpected CSV columns: {reader.fieldnames!r}')
        rows = list(reader)
    codes = {normalize_hts8(row['hts8']) for row in rows}
    if not codes:
        raise ValueError('Empty Section 338 source')
    # ch99_heading is the U.S. measure heading, NOT the product classification.
    hs6 = sorted({code[:6] for code in codes})
    canadian = json.loads((BASE / 'data/hs-t2026-2.json').read_text(encoding='utf-8'))
    canadian_codes = {node['code'] for section in canadian['sections']
                      for chapter in section['chapters'] for heading in chapter['headings']
                      for node in heading['subheadings']}
    unmatched = sorted(set(hs6) - canadian_codes)
    result = {
        'source_url': SOURCE_URL,
        'source_file': args.csv_path.name,
        'source_sha256': hashlib.sha256(args.csv_path.read_bytes()).hexdigest(),
        'product_code_column': 'hts8',
        'source_rows': len(rows),
        'source_tariff_code_count': len(codes),
        'hs6_count': len(hs6),
        'matched_hs6_count': len(hs6) - len(unmatched),
        'unmatched_hs6': unmatched,
        # Preserve the originating product codes if any HS6 does not match.
        'unmatched_source_codes': sorted(code for code in codes if code[:6] in unmatched),
        'hs6': hs6,
    }
    output = BASE / 'data/section338-hs6.json'
    output.write_text(json.dumps(result, separators=(',', ':')) + '\n', encoding='utf-8')
    print(f'Source rows: {len(rows)}; unique HTS8: {len(codes)}; unique HS6: {len(hs6)}; '
          f'matched: {len(hs6) - len(unmatched)}; unmatched: {len(unmatched)}')
    print(f'Unmatched HS6: {unmatched}; source codes: {result["unmatched_source_codes"]}')
    print(f'Wrote {output.name} ({output.stat().st_size} bytes)')


if __name__ == '__main__':
    main()
