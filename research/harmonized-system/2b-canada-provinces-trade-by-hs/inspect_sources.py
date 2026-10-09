"""Read-only inventory of the supplied CIMT archive; standard library only."""
import argparse
from collections import Counter
import csv
import hashlib
import json
from pathlib import Path
import xml.etree.ElementTree as ET
import zipfile

BASE = Path(__file__).resolve().parent
DEFAULT = Path('D:/Trade_Data_Scientist_Gov_Alberta/raw_data/statcan/CIMT-CICM_Dom_Exp_2025')

def inspect(folder):
    result = {'files': [], 'notes': {}}
    for p in sorted(folder.iterdir()):
        if not p.is_file():
            continue
        with p.open('rb') as f:
            digest = hashlib.file_digest(f, 'sha256').hexdigest()
        info = {'file': p.name, 'bytes': p.stat().st_size, 'sha256': digest}
        if p.suffix.lower() == '.docx':
            with zipfile.ZipFile(p) as z:
                doc = ET.fromstring(z.read('word/document.xml'))
            ns = {'w': 'http://schemas.openxmlformats.org/wordprocessingml/2006/main'}
            result['notes'][p.name] = '\n'.join(''.join(t.text or '' for t in para.findall('.//w:t', ns)) for para in doc.findall('.//w:p', ns))
        elif p.suffix.lower() == '.txt':
            lines = p.read_text(encoding='cp1252').splitlines()
            info.update(encoding='cp1252', format='fixed-width', rows=len(lines), samples=lines[:2])
        elif p.suffix.lower() == '.csv':
            with p.open(encoding='utf-8-sig', newline='') as f:
                r = csv.reader(f)
                header = next(r)
                distinct = [Counter() for _ in header]
                totals = Counter()
                rows = zeros = 0
                for row in r:
                    assert len(row) == len(header)
                    rows += 1
                    for i in (0, 1, 2, 3, 4):
                        distinct[i][row[i]] += 1
                    assert row[5].isdigit(), ('Non-integer/suppressed value', p.name, rows, row)
                    totals[row[3]] += int(row[5])
                    zeros += int(row[5] == '0')
            info.update(encoding='utf-8-sig', format='CSV, comma separated', columns=header, rows=rows,
                        periods=dict(sorted(distinct[0].items())), product_count=len(distinct[1]),
                        destinations=dict(sorted(distinct[2].items())), origins=dict(sorted(distinct[3].items())),
                        states=dict(sorted(distinct[4].items())), zero_rows=zeros,
                        origin_totals=dict(sorted(totals.items())), total=sum(totals.values()))
        result['files'].append(info)
    (BASE / 'source-inspection.json').write_text(json.dumps(result, ensure_ascii=False, indent=2)+'\n', encoding='utf-8')
    print(json.dumps({'files': [{k: v for k, v in f.items() if k in ('file', 'rows', 'columns', 'total', 'zero_rows', 'product_count', 'origin_totals')} for f in result['files']], 'notes': result['notes']}, ensure_ascii=True, indent=2))

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source', type=Path, default=DEFAULT)
    inspect(parser.parse_args().source)
