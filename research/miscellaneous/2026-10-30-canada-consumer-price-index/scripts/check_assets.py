"""Check published-page links, local dependencies and untouched original hashes."""
from pathlib import Path
from html.parser import HTMLParser
from urllib.parse import urlsplit, unquote
import hashlib, json, re
ROOT=Path(__file__).resolve().parents[1]
PAGE=ROOT.parent/(ROOT.name+'.html')
class PageParser(HTMLParser):
 def __init__(self):super().__init__();self.links=[];self.ids=[];self.sections=[]
 def handle_starttag(self,tag,attrs):
  a=dict(attrs)
  if 'id' in a:self.ids.append(a['id'])
  if tag=='section' and a.get('id','').startswith('s'):self.sections.append(a['id'])
  for k in ('href','src'):
   if k in a:self.links.append(a[k])
def main():
 parser=PageParser();parser.feed(PAGE.read_text(encoding='utf-8'));errors=[];checked=[]
 for link in sorted(set(parser.links)):
  u=urlsplit(link)
  if u.scheme or u.netloc:continue
  if not u.path:
   if u.fragment and u.fragment not in parser.ids:errors.append('Missing anchor '+link)
   continue
  p=PAGE.parent/unquote(u.path)
  checked.append(str(p.relative_to(PAGE.parent)))
  if not p.is_file():errors.append('Missing file '+link)
 for font in set(re.findall(r'url\(([^)]+)\)',(ROOT/'vendor/katex.min.css').read_text())):
  if not (ROOT/'vendor'/font).is_file():errors.append('Missing font '+font)
 inventory=json.loads((ROOT/'data/metadata/input-inventory.json').read_text(encoding='utf-8'))
 for f in inventory:
  path=ROOT/f['file'];digest=hashlib.sha256(path.read_bytes()).hexdigest()
  if digest!=f['sha256']:errors.append('Original modified: '+f['file'])
 if len(parser.ids)!=len(set(parser.ids)):errors.append('Duplicate HTML id')
 if len(parser.sections)!=18:errors.append('Expected eighteen chapters')
 report={'chapters':len(parser.sections),'unique_local_links_checked':len(checked),'original_files_hash_checked':len(inventory),'font_files_checked':len(set(re.findall(r'url\(([^)]+)\)',(ROOT/'vendor/katex.min.css').read_text()))),'errors':errors,'local_links':checked}
 (ROOT/'validation/asset-check.json').write_text(json.dumps(report,indent=2),encoding='utf-8')
 print(json.dumps({k:v for k,v in report.items() if k!='local_links'},indent=2))
 assert not errors, errors
if __name__=='__main__':main()
