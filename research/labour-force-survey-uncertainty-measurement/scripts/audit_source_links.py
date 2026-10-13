from pathlib import Path
from html.parser import HTMLParser
import urllib.request,json,concurrent.futures
ROOT=Path(__file__).resolve().parents[1]
class P(HTMLParser):
 def __init__(self):super().__init__();self.links=[];self.ids=[]
 def handle_starttag(self,tag,attrs):
  a=dict(attrs)
  if 'id' in a:self.ids.append(a['id'])
  if tag=='a' and 'href'in a:self.links.append(a['href'])
for name in ['chapter-7.html','guide-2025.html']:
 p=P();p.feed((ROOT/'sources'/name).read_text(encoding='utf-8'))
 print(name,[l for l in p.links if 'ref' in l or 'app_ann_b' in l],p.ids[:40])
p=P();p.feed((ROOT.parent/(ROOT.name+'.html')).read_text(encoding='utf-8'))
def check(url):
 try:
  response=urllib.request.urlopen(urllib.request.Request(url,headers={'User-Agent':'Mozilla/5.0'}),timeout=25)
  return dict(url=url,status=response.status,final_url=response.url)
 except Exception as e:return dict(url=url,error=str(e))
with concurrent.futures.ThreadPoolExecutor(max_workers=6) as ex:results=list(ex.map(check,sorted(set(l.split('#')[0] for l in p.links if l.startswith('https:')))))
(ROOT/'sources'/'link-audit.json').write_text(json.dumps(results,indent=2),encoding='utf-8')
print(json.dumps(results,indent=2))
