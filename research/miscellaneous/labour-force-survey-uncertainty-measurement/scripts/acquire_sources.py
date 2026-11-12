"""Download official source snapshots; never replace the frozen extracts silently."""
from pathlib import Path
import urllib.request, json, hashlib, datetime, sys
ROOT = Path(__file__).resolve().parents[1]
SOURCES = {
 'guide-2025.html': 'https://www150.statcan.gc.ca/n1/pub/71-543-g/71-543-g2025001-eng.htm',
 'methodology-2017.html': 'https://www150.statcan.gc.ca/n1/pub/71-526-x/71-526-x2017001-eng.htm',
 'methodology-2017.pdf': 'https://www150.statcan.gc.ca/n1/pub/71-526-x/71-526-x2017001-eng.pdf',
 'guide-2025.pdf': 'https://www150.statcan.gc.ca/n1/pub/71-543-g/71-543-g2025001-eng.pdf',
 'september-2026.html': 'https://www150.statcan.gc.ca/n1/daily-quotidien/261009/dq261009a-eng.htm',
 'survey-3701.html': 'https://www23.statcan.gc.ca/imdb/p2SV.pl?Function=getSurvey&SDDS=3701',
 'table-14100287.zip': 'https://www150.statcan.gc.ca/n1/tbl/csv/14100287-eng.zip',
 'guide-appendix-b.html': 'https://www150.statcan.gc.ca/n1/pub/71-543-g/2025001/app_ann_b-eng.htm',
 'methodology-references.html': 'https://www150.statcan.gc.ca/n1/pub/71-526-x/2017001/ref-biblio-eng.htm',
}
SOURCES.update({f'chapter-{c}.html': f'https://www150.statcan.gc.ca/n1/pub/71-526-x/2017001/chapt{c}-eng.htm' for c in (2,5,6,7,8)})
def main():
 (ROOT/'sources').mkdir(parents=True, exist_ok=True)
 manifest_path=ROOT/'sources'/'manifest.json'
 manifest=json.loads(manifest_path.read_text()) if manifest_path.exists() else []
 for name,url in SOURCES.items():
  if len(sys.argv)>1 and name not in sys.argv[1:]: continue
  manifest=[r for r in manifest if r['file']!=name]
  try:
   data=urllib.request.urlopen(urllib.request.Request(url,headers={'User-Agent':'Mozilla/5.0'}),timeout=90).read()
   (ROOT/'sources'/name).write_bytes(data)
   manifest.append(dict(file=name,url=url,downloaded_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),bytes=len(data),sha256=hashlib.sha256(data).hexdigest()))
   print(name,len(data),flush=True)
  except Exception as error:
   manifest.append(dict(file=name,url=url,error=str(error)))
   print(name,str(error),flush=True)
 (ROOT/'sources'/'manifest.json').write_text(json.dumps(manifest,indent=2),encoding='utf-8')
if __name__=='__main__': main()
