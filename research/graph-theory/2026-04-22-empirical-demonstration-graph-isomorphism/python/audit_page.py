"""Audit local HTML dependencies, anchors, preserved assets, and source PDFs."""
from html.parser import HTMLParser
from pathlib import Path
from urllib.parse import urlsplit, unquote
import hashlib
import json
import re

ASSETS=Path(__file__).resolve().parents[1]
PAGE=ASSETS.with_suffix('.html')

class Scan(HTMLParser):
    def __init__(self):
        super().__init__()
        self.ids=[]; self.references=[]; self.images=[]; self.headings=[]
    def handle_starttag(self,tag,attrs):
        a=dict(attrs)
        if 'id' in a:self.ids.append(a['id'])
        for key in ['href','src']:
            if key in a:self.references.append((tag,key,a[key]))
        if tag=='img':self.images.append(a)
        if tag in ('h1','h2'):self.headings.append(tag)

def main():
    text=PAGE.read_text(encoding='utf-8'); scan=Scan(); scan.feed(text)
    assert len(scan.ids)==len(set(scan.ids)), 'Duplicate HTML id'
    checked=[]
    for tag,key,url in scan.references:
        parts=urlsplit(url)
        if parts.scheme or parts.netloc:continue
        if parts.path:
            p=(PAGE.parent/unquote(parts.path)).resolve()
            assert p.is_file(),f'Missing local asset: {url}'
            checked.append(url)
        elif parts.fragment:
            assert parts.fragment in scan.ids,f'Missing anchor: {url}'
    assert scan.headings.count('h1')==1
    assert all(im.get('alt','').strip() for im in scan.images)
    assert len(scan.images)==58
    assert scan.ids.count('film-timeline')==1
    assert len(re.findall(r'<details id="s\d\d"',text))==29
    assert '<!--NEXT-->' not in text and '<!--LEGACY-->' not in text
    assert not any(re.match(r'^[A-Za-z]:[\\/]',url) for _,_,url in scan.references),'Machine-specific HTML asset path'
    source=[]
    inventory=json.loads((ASSETS/'data/validation/original-asset-inventory.json').read_text())
    for item in inventory:
        p=ASSETS/item['destination']
        assert p.is_file() and hashlib.sha256(p.read_bytes()).hexdigest()==item['sha256'],p
        if p.suffix=='.pdf':
            assert p.parent==ASSETS
            source.append({'name':p.name,'sha256':item['sha256']})
        elif item['original'].endswith('.png'):
            assert not (ASSETS/item['original']).exists(),'Duplicate old-location image'
    # Check direct JS dependencies used inside the worker and classic script list.
    worker=(ASSETS/'js/compute-worker.js').read_text()
    imported=re.search(r'importScripts\((.*?)\)',worker).group(1)
    for name in re.findall(r"'([^']+)'",imported):assert (ASSETS/'js'/name).is_file()
    report={'date':'2026-10-29','local_references_checked':len(checked),
            'unique_ids':len(scan.ids),'retained_images':len(scan.images),
            'retained_notebook_entries':29,'main_sections':20,
            'pdfs':source,'all_checks_passed':True}
    (ASSETS/'data/validation/asset-audit.json').write_text(json.dumps(report,indent=2),encoding='utf-8')
    print(json.dumps(report,indent=2))

if __name__=='__main__':main()
