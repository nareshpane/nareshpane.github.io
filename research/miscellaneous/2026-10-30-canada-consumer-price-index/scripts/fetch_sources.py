"""Fetch missing primary references once; retain failures without substituting data."""
from pathlib import Path
import urllib.request, json, datetime, hashlib, re, concurrent.futures, subprocess, shutil, sys
ROOT=Path(__file__).resolve().parents[1]
SOURCES={
 'references/basket-update-2026.html':'https://www150.statcan.gc.ca/n1/pub/62f0014m/62f0014m2026005-eng.htm',
 'references/concordance.html':'https://www.statcan.gc.ca/en/statistical-programs/document/2301_D66_V1',
 'references/bank-of-canada-cpi.html':'https://www.bankofcanada.ca/rates/price-indexes/cpi/',
 'references/reference-paper-index.html':'https://www150.statcan.gc.ca/n1/pub/62-553-x/2023001/index-eng.htm',
 'references/reference-paper-chapter8.html':'https://www150.statcan.gc.ca/n1/pub/62-553-x/2023001/chap-8-eng.htm',
 'references/reference-paper-chapter5.html':'https://www150.statcan.gc.ca/n1/pub/62-553-x/2023001/chap-5-eng.htm',
 'references/reference-paper-chapter6.html':'https://www150.statcan.gc.ca/n1/pub/62-553-x/2023001/chap-6-eng.htm',
 'references/geographical-release-2026-09.html':'https://www150.statcan.gc.ca/n1/daily-quotidien/261019/t002a-eng.htm',
 'references/cpi-portal.html':'https://www.statcan.gc.ca/en/subjects-start/prices_and_price_indexes/consumer_price_indexes',
 'references/basket-release-2026.html':'https://www150.statcan.gc.ca/n1/daily-quotidien/260615/dq260615c-eng.htm',
 'references/september-release.html':'https://www150.statcan.gc.ca/n1/daily-quotidien/261019/dq261019a-eng.htm',
 'data/18100007-eng.zip':'https://www150.statcan.gc.ca/n1/tbl/csv/18100007-eng.zip',
 'data/18100006-eng.zip':'https://www150.statcan.gc.ca/n1/tbl/csv/18100006-eng.zip',
 'vendor/katex.min.js':'https://cdn.jsdelivr.net/npm/katex@0.16.11/dist/katex.min.js',
 'vendor/katex.min.css':'https://cdn.jsdelivr.net/npm/katex@0.16.11/dist/katex.min.css',
 'vendor/katex-LICENSE':'https://cdn.jsdelivr.net/npm/katex@0.16.11/LICENSE',
}
def fetch(pair):
 name,url=pair;p=ROOT/name;p.parent.mkdir(parents=True,exist_ok=True)
 result={'file':name,'url':url,'attempted_at':datetime.datetime.now(datetime.timezone.utc).isoformat()}
 try:
  if not p.exists():
   req=urllib.request.Request(url,headers={'User-Agent':'Mozilla/5.0 CPI educational research'})
   try:
    data=urllib.request.urlopen(req,timeout=45).read();p.write_bytes(data)
   except urllib.error.URLError as e:
    # Windows curl uses the OS certificate trust. Never disable certificate checks.
    if sys.platform=='win32' and 'CERTIFICATE_VERIFY_FAILED' in str(e) and shutil.which('curl.exe'):
     subprocess.run([shutil.which('curl.exe'),'--fail','--silent','--show-error','--location','--max-time','45','--output',str(p),url],check=True)
     result['transport']='Windows curl with certificate verification'
    else:raise
  result.update(status='available',bytes=p.stat().st_size,sha256=hashlib.sha256(p.read_bytes()).hexdigest())
 except Exception as e: result.update(status='unavailable',error=str(e))
 return result
def main():
 with concurrent.futures.ThreadPoolExecutor(max_workers=6) as pool: results=list(pool.map(fetch,SOURCES.items()))
 css=ROOT/'vendor/katex.min.css'
 if css.exists():
  fonts=set(re.findall(r'url\((fonts/[^)]+)\)',css.read_text()))
  with concurrent.futures.ThreadPoolExecutor(max_workers=6) as pool:
   results+=list(pool.map(fetch,[(f'vendor/{f}','https://cdn.jsdelivr.net/npm/katex@0.16.11/dist/'+f) for f in fonts]))
 (ROOT/'data/metadata/retrievals.json').write_text(json.dumps(results,indent=2),encoding='utf-8')
 for r in results:print(r['file'],r['status'],r.get('error',''))
if __name__=='__main__': main()
