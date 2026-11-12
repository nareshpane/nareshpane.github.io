"""Small, read-only checks of fixed-width offsets and CBSA compatibility."""
from pathlib import Path
import json
import re
from inspect_sources import RAW

for name in ['ODPF_4_HS6XDesc.TXT', 'ODPF_8_ProvDesc.TXT', 'ODPF_6_CtyDesc.TXT']:
    line = (RAW / name).read_text(encoding='cp1252').splitlines()[0]
    print(name, [(m.start(), m.end(), m.group()) for m in re.finditer(r'\S(?:.*?\S)?(?= {2,}|$)', line)])
cbsa = json.loads((Path(__file__).resolve().parents[2] / '1-harmonized-system-canada/data/hs-t2026-2.json').read_text(encoding='utf-8'))
for section in cbsa['sections']:
    for chapter in section['chapters']:
        for heading in chapter['headings']:
            if heading['code'] in ['8414', '9403', '8537']:
                print(heading['code'], heading['description'], heading['extraction'])
print('CBSA heading extraction', sorted({h['extraction'] for s in cbsa['sections'] for c in s['chapters'] for h in c['headings']}))
