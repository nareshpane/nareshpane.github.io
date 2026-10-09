"""Refresh the retained official 2025 HS4 lookup (network required only here).

Extract explicit heading rows and unsplit .00 rows from Statistics Canada's
2025 Canadian Export Classification HTML tables. The trade builder independently
checks that a .00 label has only one active child. Normal builds use the retained
snapshot and need no network. Failed downloads leave the current snapshot intact.
"""
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
import hashlib
from html.parser import HTMLParser
import json
from pathlib import Path
import re
import time
import urllib.request

BASE = Path(__file__).resolve().parent
URL = 'https://www150.statcan.gc.ca/n1/pub/65-209-x/2025001/t/tbl_{:02}-eng.htm'

class Rows(HTMLParser):
    def __init__(self):
        super().__init__(convert_charrefs=True)
        self.rows = []
        self.cells = None
        self.cell = None
    def handle_starttag(self, tag, attrs):
        if tag == 'tr':
            self.cells = []
        elif tag in ('td', 'th') and self.cells is not None:
            self.cell = []
        elif tag == 'br' and self.cell is not None:
            self.cell.append(' ')
    def handle_data(self, data):
        if self.cell is not None:
            self.cell.append(data)
    def handle_endtag(self, tag):
        if tag in ('td', 'th') and self.cell is not None:
            self.cells.append(re.sub(r'\s+', ' ', ''.join(self.cell)).strip())
            self.cell = None
        elif tag == 'tr' and self.cells is not None:
            self.rows.append(self.cells)
            self.cells = None

def chapter(number):
    url = URL.format(number)
    for attempt in range(3):
        try:
            request = urllib.request.Request(url, headers={'User-Agent': 'CIMT-educational-research/1.0'})
            with urllib.request.urlopen(request, timeout=60) as f:
                raw = f.read()
            break
        except Exception:
            if attempt == 2:
                raise
            time.sleep(1 + attempt)
    parser = Rows()
    parser.feed(raw.decode('utf-8-sig'))
    headings = {}
    retained = []
    codes6 = []
    for row in parser.rows:
        if len(row) >= 3 and re.fullmatch(r'\d{2}\.\d{2}', row[0]):
            code = row[0].replace('.', '')
            assert row[2] and code[:2] == f'{number:02}'
            assert code not in headings
            headings[code] = row[2]
            retained.append(row)
        elif row and re.fullmatch(r'\d{4}\.\d{2}', row[0]):
            codes6.append(row[0].replace('.', ''))
            if len(row) >= 3 and row[0].endswith('.00') and not row[2].startswith('-'):
                retained.append(row)
    assert retained, ('No usable classification rows extracted', url, parser.rows[:3])
    return headings, codes6, retained, {'chapter': f'{number:02}', 'url': url, 'sha256': hashlib.sha256(raw).hexdigest(), 'bytes': len(raw), 'hs4_rows': len(headings)}

def main():
    headings = {}
    codes6 = set()
    sources = []
    chapters = {}
    with ThreadPoolExecutor(max_workers=4) as pool:
        for h, c, rows, source in pool.map(chapter, [n for n in range(1, 100) if n != 77]):
            headings.update(h)
            codes6.update(c)
            sources.append(source)
            chapters[str(int(source['chapter']))] = {'url': source['url'], 'rows': rows}
    snapshot = {'year': 2025, 'source': 'Statistics Canada, Canadian Export Classification, 2025',
                'url': 'https://www150.statcan.gc.ca/n1/pub/65-209-x/65-209-x2025001-eng.htm',
                'method': 'Explicit HS4 and unsplit .00 rows; descriptions whitespace-normalized, otherwise verbatim.',
                'retrieved': datetime.now(timezone.utc).isoformat(), 'chapters': chapters}
    (BASE / 'hs4-official-2025-extract.json').write_text(json.dumps(snapshot, ensure_ascii=False, separators=(',', ':'))+'\n', encoding='utf-8')
    (BASE / 'classification-provenance.json').write_text(json.dumps({'retrieved_at': datetime.now(timezone.utc).isoformat(), 'sources': sources}, indent=2)+'\n', encoding='utf-8')
    print(f'Official 2025 snapshot: {len(headings)} HS4 headings; {len(codes6)} HS6 codes; {len(sources)} chapter tables.')
    print('Examples:', {c: headings.get(c) for c in ['0601', '2709', '9802', '9901']})

if __name__ == '__main__':
    main()
