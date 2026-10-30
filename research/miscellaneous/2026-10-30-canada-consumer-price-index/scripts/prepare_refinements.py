"""Retain supplied PNGs and extract additional public weights; never edit raw inputs."""
from pathlib import Path
import csv,io,zipfile,json,hashlib,shutil,struct,datetime
ROOT=Path(__file__).resolve().parents[1]
ILLUSTRATIONS=[
 ('052730','shopper-trolley'),('052738','shopper-market'),('052746','mobile-clothing'),
 ('052754','online-selection'),('052828','food-service'),('052854','residential-towers'),
 ('052904','residential-neighbourhood'),('052910','city-skyline'),('052945','digital-wallet')]
def main():
 (ROOT/'images').mkdir(exist_ok=True)
 records=[]
 for stamp,name in ILLUSTRATIONS:
  source=Path('C:/Users/user/Pictures/Screenshots')/f'Screenshot 2026-10-30 {stamp}.png'
  target=ROOT/'images'/f'{name}.png'
  if not target.exists():shutil.copyfile(source,target)
  raw=target.read_bytes();assert raw[:8]==b'\x89PNG\r\n\x1a\n'
  width,height=struct.unpack('>II',raw[16:24]);digest=hashlib.sha256(raw).hexdigest()
  if source.exists():assert hashlib.sha256(source.read_bytes()).hexdigest()==digest,'Supplied image changed unexpectedly'
  records.append({'source':str(source),'file':str(target.relative_to(ROOT)),'width':width,'height':height,'sha256':digest,'treatment':'Byte-for-byte copy; natural aspect ratio; CSS blend integrates white background.','attribution':'Supplied by the user; original authorship/licensing not independently identified.'})
 weights={};regional={}
 with zipfile.ZipFile(ROOT/'data/18100007-eng.zip') as archive:
  with archive.open('18100007.csv') as f:
   for r in csv.DictReader(io.TextIOWrapper(f,encoding='utf-8-sig')):
    if r['REF_DATE']!='2025' or not r['VALUE']:continue
    kind='link' if r['Price period of weight']=='Weight at basket link month prices' else 'reference'
    if r['GEO']=='Canada' and r['Geographic distribution of weight']=='Distribution to selected geographies':
     weights.setdefault(r['Products and product groups'],{})[kind]={'value':float(r['VALUE']),'vector':r['VECTOR'],'status':r['STATUS']}
    if r['Products and product groups']=='All-items' and r['Geographic distribution of weight']=='Distribution to Canada':
     regional.setdefault(r['GEO'],{})[kind]={'value':float(r['VALUE']),'vector':r['VECTOR'],'status':r['STATUS']}
 data={'basket_year':'2025','national_weights':weights,'regional_weights':regional,'source':'Statistics Canada Table 18-10-0007-01; existing ZIP, no new dataset copy.','illustrations':records,'prepared_at':datetime.datetime.now(datetime.timezone.utc).isoformat()}
 (ROOT/'data/processed/refinement-data.json').write_text(json.dumps(data,indent=2,ensure_ascii=False),encoding='utf-8')
 (ROOT/'data/processed/refinement-data.js').write_text('window.CPI_REFINEMENT_DATA='+json.dumps(data,separators=(',',':'),ensure_ascii=False)+';',encoding='utf-8')
 (ROOT/'images/README.md').write_text('# Supplied illustrations\n\nAll nine PNG originals are preserved byte for byte. The page uses five: shopper-trolley (consumer basket), shopper-market (retail sampling), digital-wallet (expenditure shares), food-service (food classification), and residential-neighbourhood (regional shelter comparisons). The other four originals remain available for provenance and future use.\n\nNo raster editing or optimized variants were needed. CSS preserves natural proportions, reserves intrinsic dimensions, blends white backgrounds, and aligns each image with explanatory text. Repetitive captions are omitted; provenance is retained in ../data/processed/refinement-data.json. None of these illustrations is an official statistical chart or a geographic map.\n',encoding='utf-8')
 print(json.dumps({'images':len(records),'national_weight_groups':len(weights),'regional_weight_geographies':len(regional)}))
if __name__=='__main__':main()
