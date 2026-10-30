"""Optional network smoke test of the main page's external reference links.

Uses HEAD requests and follows redirects, without redownloading source bodies.
Windows curl retains certificate verification. This does not prove future uptime.
"""
from pathlib import Path
from html.parser import HTMLParser
import concurrent.futures, subprocess, shutil, json, datetime
ROOT=Path(__file__).resolve().parents[1]
class Links(HTMLParser):
 def __init__(self):super().__init__();self.urls=set()
 def handle_starttag(self,tag,attrs):
  a=dict(attrs)
  if tag=='a' and a.get('href','').startswith('https://'):self.urls.add(a['href'])
def main():
 parser=Links();parser.feed((ROOT.parent/(ROOT.name+'.html')).read_text(encoding='utf-8'))
 curl=shutil.which('curl.exe') or shutil.which('curl')
 if not curl:raise SystemExit('curl is required for optional reference smoke test')
 def check(url):
  p=subprocess.run([curl,'--silent','--show-error','--head','--location','--max-time','35','--output','NUL' if Path('C:/Windows').exists() else '/dev/null','--write-out','%{http_code}\n%{url_effective}',url],text=True,capture_output=True)
  lines=p.stdout.splitlines();return {'url':url,'status':int(lines[0]) if lines and lines[0].isdigit() else None,'effective_url':lines[1] if len(lines)>1 else None,'error':p.stderr.strip() or None}
 with concurrent.futures.ThreadPoolExecutor(max_workers=5) as pool:rows=list(pool.map(check,sorted(parser.urls)))
 report={'checked_at_utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),'method':'HEAD with redirects; certificate verification enabled','links':rows}
 (ROOT/'validation/reference-links.json').write_text(json.dumps(report,indent=2),encoding='utf-8')
 print(json.dumps(rows,indent=2))
if __name__=='__main__':main()
