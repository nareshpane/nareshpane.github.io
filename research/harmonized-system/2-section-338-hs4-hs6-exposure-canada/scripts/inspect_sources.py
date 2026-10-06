"""Read-only source inventory. Run before changing the analytical builder."""
import csv
import hashlib
import json
from collections import Counter
from pathlib import Path
import re
import zipfile
import sys
sys.stdout.reconfigure(encoding='utf-8')

RAW = Path('D:/Trade_Data_Scientist_Gov_Alberta/raw_data/statcan/CIMT-CICM_Dom_Exp_2025')
POLICY = RAW.parents[1] / 'section338'

def inspect():
    for path in sorted(RAW.iterdir()):
        if path.suffix.lower() == '.docx':
            with zipfile.ZipFile(path) as doc:
                print(path.name, re.sub('<[^>]+>', ' ', doc.read('word/document.xml').decode()))
            continue
        payload = path.read_bytes()
        encoding = 'utf-8-sig'
        try:
            payload.decode(encoding)
        except UnicodeDecodeError:
            encoding = 'cp1252'
        sample = payload[:16000].decode(encoding)
        if path.suffix == '.TXT':
            lines = payload.decode(encoding).splitlines()
            print(json.dumps(dict(file=path.name, bytes=len(payload), encoding=encoding,
                format='fixed-width, no header', rows=len(lines), first=lines[:3]), ensure_ascii=False))
            continue
        delimiter = csv.Sniffer().sniff('\n'.join(sample.splitlines()[:-1]), delimiters=',\t|;').delimiter
        with path.open(encoding=encoding, newline='') as stream:
            reader = csv.reader(stream, delimiter=delimiter)
            header = next(reader)
            first = []
            counts = [Counter() for _ in header]
            n = 0
            for row in reader:
                assert len(row) == len(header), (path.name, row)
                n += 1
                if n <= 3:
                    first.append(row)
                for i, value in enumerate(row):
                    counts[i][value] += 1
            print(json.dumps(dict(file=path.name, bytes=len(payload), encoding=encoding,
                delimiter=delimiter, columns=header, rows=n, first=first,
                distinct={name: dict(c) if len(c) < 25 else {'count': len(c), 'examples': list(c)[:8]}
                          for name, c in zip(header, counts)}), ensure_ascii=False))
    policy_rows = []
    for path in sorted(POLICY.iterdir()):
        payload = path.read_bytes()
        text = payload.decode('utf-8-sig')
        info = dict(file=path.name, sha256=hashlib.sha256(payload).hexdigest(), bytes=len(payload))
        if path.suffix == '.csv':
            records = list(csv.DictReader(text.splitlines()))
            info.update(rows=len(records), columns=list(records[0]), first=records[:2],
                        unique_records=len({tuple(r.items()) for r in records}),
                        hs6=len({r['hts8'][:6] for r in records}))
            policy_rows.append(records)
        else:
            # Inspect every line, retaining relevant structural markers and codes.
            info.update(urls=re.findall(r'https://[^\s)]+', text),
                        lines=len(text.splitlines()),
                        markers=[line for line in text.splitlines() if re.search('Part [AB]|Effective|September|July|deleting|inserting|Yale|Budget', line)],
                        codes=re.findall(r'\b\d{4}\.\d{2}\.\d{2}(?:\d{2})?\b', text))
        print(json.dumps(info))
    print('POLICY RECORDS IDENTICAL:', policy_rows[0] == policy_rows[1])
    cbsa = json.loads((Path(__file__).resolve().parents[2] / '1-harmonized-system-canada/data/hs-t2026-2.json').read_text(encoding='utf-8'))
    print('CBSA KEYS:', list(cbsa))
    print('CBSA METADATA:', json.dumps({k: v for k, v in cbsa.items() if k not in ['sections', 'special_chapters']}))
    print('CBSA SAMPLE:', json.dumps(cbsa['sections'][0])[:5500])

if __name__ == '__main__':
    inspect()
