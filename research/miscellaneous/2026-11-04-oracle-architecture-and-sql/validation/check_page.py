"""Static portability, anchor and asset checks; does not execute Oracle."""
from html.parser import HTMLParser
from pathlib import Path
from urllib.parse import unquote,urlsplit
from collections import Counter
import json,re,sys
BASE=Path(__file__).resolve().parents[1]
PAGE=BASE.parent/(BASE.name+'.html')
class PageParser(HTMLParser):
    def __init__(self):super().__init__();self.ids=[];self.refs=[];self.scripts=[];self.images=[]
    def handle_starttag(self,tag,attrs):
        a=dict(attrs)
        if 'id' in a:self.ids.append(a['id'])
        if tag in ('a','link') and 'href' in a:self.refs.append(a['href'])
        if tag in ('script','img') and 'src' in a:self.refs.append(a['src'])
        if tag=='script' and 'src' in a:self.scripts.append(a['src'])
        if tag=='img':self.images.append(a)
p=PageParser();p.feed(PAGE.read_text(encoding='utf-8'))
duplicates=[k for k,n in Counter(p.ids).items() if n>1]
assert not duplicates,duplicates
for ref in p.refs:
    u=urlsplit(ref)
    if u.scheme or u.netloc:continue
    if not u.path:
        assert unquote(u.fragment) in p.ids,ref
    else:
        path=PAGE.parent/unquote(u.path)
        assert path.exists(),ref
for image in p.images:assert image.get('alt'),image
assert all(not urlsplit(ref).scheme for ref in p.scripts)
css=(BASE/'css/monograph.css').read_text(encoding='utf-8')
assert 'prefers-reduced-motion' in css
js=(BASE/'js/monograph.js').read_text(encoding='utf-8')
assert 'IntersectionObserver' in js
assert not re.search(r'\b(fetch|XMLHttpRequest|WebSocket)\s*\(',js)
index=PAGE.parents[2]/'research.html'
assert ('research/miscellaneous/'+PAGE.name) in index.read_text(encoding='utf-8')
report={'static_ids':len(p.ids),'local_references_checked':sum(not urlsplit(r).scheme for r in p.refs),'images_with_alt':len(p.images),'external_runtime_dependencies':0,'index_entry':True,'oracle_executed':False}
if '--http-root' in sys.argv:
    from urllib.parse import urljoin
    from urllib.request import urlopen
    root=sys.argv[sys.argv.index('--http-root')+1]
    paths=sorted(set(r for r in p.refs if not urlsplit(r).scheme and urlsplit(r).path))
    for ref in paths:
        with urlopen(urljoin(root,ref),timeout=10) as response:assert response.status==200,ref
    report['http_resources_returning_200']=len(paths)
(BASE/'validation/static-results.json').write_text(json.dumps(report,indent=2)+'\n')
print(json.dumps(report,indent=2))
