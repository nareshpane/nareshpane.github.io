from pathlib import Path
import hashlib,json
root=Path('D:/GitHub/nareshpane.github.io'); base=root/'research/labour-force-survey-uncertainty-measurement'; work=Path('C:/Users/user/Documents/Codex/2026-10-13/before-starting-implementation-read-both-pdf/work')
old=json.loads((work/'baseline-hashes.json').read_text(encoding='utf-8-sig'))
for row in old:
 p=Path(row['Path']); current=hashlib.sha256(p.read_bytes()).hexdigest().upper(); print(('UNCHANGED' if current==row['Hash'] else 'CHANGED'),p.name)
# Preserve existing anchors and core mathematical displays, except the expanded bootstrap's replaced Python example.
import re
before=(work/'before.html').read_text(encoding='utf-8'); after=(base.with_suffix('.html')).read_text(encoding='utf-8')
ids=re.findall(r'\bid="([^"]+)"',before);assert all('id="'+i+'"' in after for i in ids)
print('All',len(ids),'original IDs preserved')
for name in ['sampling','variance','cv','intervals','errors','seasonality','provinces','case-study','laboratory','examples']:
 b=re.search(r'<section class="chapter" id="'+name+r'">.*?</section>',before,re.S).group()
 a=re.search(r'<section class="chapter" id="'+name+r'">.*?</section>',after,re.S).group()
 assert a==b,name
print('Ten existing quantitative/technical sections preserved byte for byte')
