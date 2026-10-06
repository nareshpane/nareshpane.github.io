"""Explain detailed-line policy reconstruction and reference differences."""
import csv
import json
from collections import Counter
from build_data import BASE, SECTION338_DIR, section338, policy_codes

p = section338(SECTION338_DIR)
exports = json.loads((BASE / 'data/exports-hs6-2025-us.json').read_text())
with (SECTION338_DIR / p['canonical_file']).open() as stream:
    rows = list(csv.DictReader(stream))
scope = {r['hts8'] for r in rows}
print('July programs:', dict(Counter(r['program'] for r in rows)))
for amendment in p['amendments']:
    scope -= set(amendment['deleted_parents'])
    scope |= set(amendment['inserted_lines'])
    print(amendment['annex'], 'parents deleted', len(amendment['deleted_parents']),
          'lines inserted', len(amendment['inserted_lines']))
for label, lines in [('July', {r['hts8'] for r in rows}), ('Before bans', scope),
                     ('After ban-listed line exclusion', set(p['current_detailed_lines']))]:
    codes = {c[:6] for c in lines}
    matches = codes & exports.keys()
    print(label, 'lines', len(lines), 'HS6', len(codes), 'positive matches',
          sum(sum(exports[c]) > 0 for c in matches), 'exports', sum(sum(exports[c]) for c in matches))
before = {c[:6] for c in scope}
after = set(p['hs6'])
print('HS6 removed by bans:', sorted(before - after))
print('Ban lines outside tariff universe:', p['ban_lines_not_in_tariff_scope'])
for program in ['alcohol', 'dairy', 'motor']:
    text = (SECTION338_DIR / f'july_{program}_browser.txt').read_text(encoding='utf-8-sig')
    relevant = policy_codes(text)
    csv_program = 'motor_vehicles' if program == 'motor' else program
    csv_codes = {r['hts8'] for r in rows if r['program'] == csv_program}
    print(program, 'CSV lines not found in July extract:', sorted(csv_codes - relevant),
          'other codes in extract:', len(relevant - csv_codes),
          '(includes exceptions; not an additional policy universe)')
