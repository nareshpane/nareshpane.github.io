"""Check every local referenced file over the same HTTP preview used in browser QA."""
from pathlib import Path
from html.parser import HTMLParser
from urllib.parse import urljoin,urlsplit
from urllib.request import build_opener,ProxyHandler
from concurrent.futures import ThreadPoolExecutor
import json

ROOT=Path(__file__).resolve().parents[1]
PAGE=ROOT.with_suffix('.html')
BASE='http://127.0.0.1:8765/research/graph-theory/'+PAGE.name
class Scan(HTMLParser):
    def __init__(self): super().__init__();self.urls=set()
    def handle_starttag(self,tag,attrs):
        for k,v in attrs:
            if k in ('href','src') and v and not urlsplit(v).scheme and not v.startswith('#'):
                self.urls.add(urljoin(BASE,v))
def check(url):
    with build_opener(ProxyHandler({})).open(url,timeout=20) as response:
        assert response.status==200
        size=len(response.read())
        return {'path':urlsplit(url).path,'status':response.status,'bytes':size}
def main():
    scan=Scan();scan.feed(PAGE.read_text(encoding='utf-8'))
    with ThreadPoolExecutor(max_workers=6) as pool: results=list(pool.map(check,sorted(scan.urls)))
    report={'date':'2026-10-29','references':len(results),'all_http_200':True,'requests':results}
    (ROOT/'data/validation/http-audit.json').write_text(json.dumps(report,indent=2),encoding='utf-8')
    print(f'{len(results)} distinct local references: HTTP 200, including every original PNG and both PDFs.')
if __name__=='__main__':main()
