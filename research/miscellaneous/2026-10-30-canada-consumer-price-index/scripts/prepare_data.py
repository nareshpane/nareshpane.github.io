"""Streaming SDMX extraction; independently transcribe release tables from PDF text.

No network calls. Originals are opened read-only. The first file is a historical
snapshot; numbered later files are monthly patches, not separate geographies.
"""
from pathlib import Path
import csv, io, json, re, zipfile, xml.etree.ElementTree as ET
from collections import Counter
ROOT=Path(__file__).resolve().parents[1]
MAJORS=['Food','Shelter','Household operations, furnishings and equipment','Clothing and footwear','Transportation','Health and personal care','Recreation, education and reading','Alcoholic beverages, tobacco products and recreational cannabis']
def write(name,obj):
 (ROOT/name).write_text(json.dumps(obj,ensure_ascii=False,separators=(',',':')),encoding='utf-8')
def ziprows(table):
 with zipfile.ZipFile(ROOT/f'data/{table}-eng.zip') as z:
  with z.open(f'{table}.csv') as f:
   yield from csv.DictReader(io.TextIOWrapper(f,encoding='utf-8-sig'))
def parse_release():
 text=(ROOT/'references/cpi-september-report.txt').read_text(encoding='utf-8')
 text=text.replace('\u00a0',' ').replace('Y ellowknife','Yellowknife')
 blocks=[text.split('Table 1\n')[1].split('Table 2\n')[0],text.split('Table 2\n')[1].split('Table 3\n')[0],text.split('Table 3\n')[1].split('Table 4\n')[0]]
 out=[]
 for block,kind,page in zip(blocks,['national','geography','seasonal'],[7,8,8]):
  # A few labels wrap; flatten only those known continuations.
  block=block.replace('recreational\ncannabis','recreational cannabis').replace('products\nand smokers',"products and smokers").replace('and\nrecreational','and recreational')
  for line in block.splitlines():
   m=re.match(r'^(.+?)\s+(-?\d+\.\d+)\s+(-?\d+\.\d+)\s+(-?\d+\.\d+)\s+(-?\d+\.\d+)\s+(-?\d+\.\d+)\s*$',line)
   if m:
    label=m[1].strip(); label=re.sub(r'(?<=[A-Za-z])\d+$','',label)
    nums=list(map(float,m.groups()[1:]));out.append(dict(kind=kind,label=label,values=nums,pdf_page=page))
 core={}
 coreblock=text.split('Table 4\n')[1].split('Table 5\n')[0]
 for label in ('CPI-common','CPI-median','CPI-trim'):
  m=re.search(re.escape(label)+r'[^\n]*',coreblock)
  nums=re.findall(r'(?<!\d)\d+\.\d+',m[0]);core[label]=float(nums[5])
 contributors=[];period='YoY'
 for line in text.split('Table 5\n')[1].split('--- PDF PAGE 11 ---')[0].splitlines():
  if 'August to September 2026' in line:period='MoM'
  m=re.match(r'^([A-Za-z][^\n]+?)\s+(-?\d+\.\d+)\s*$',line)
  if m:contributors.append({'label':m[1].strip(),'metric':period,'value':float(m[2]),'pdf_page':10})
 write('data/processed/release-tables.json',{'rows':out,'core_yoy':core,'contributors':contributors,'source':'cpi-september-report.pdf','date':'2026-10-19','core_verification':'Statistics Canada release Table 4; independently compare Bank of Canada HTML table when available'})
 return out,core
def main():
 books=json.loads((ROOT/'data/metadata/codebooks.json').read_text(encoding='utf-8'))
 products=books['CL_Products_and_product_groups'];geos=books['CL_Geography']
 keep={k for k,v in products.items() if v['label'] in ['All-items']+MAJORS}
 series={};count=0;conflicts=[];status=Counter();files={}
 for f in sorted((ROOT/'cpi-stat-canada-18100004-SDMX').glob('18100004_*.xml'),key=lambda p:int(p.stem.split('_')[-1]) if p.stem.split('_')[-1].isdigit() else 999):
  if f.stem.endswith('Structure'):continue
  for event,e in ET.iterparse(f,events=('end',)):
   tag=e.tag.split('}')[-1]
   if tag!='Series':continue
   a=e.attrib;g=a['Geography'];p=a['Products_and_product_groups'];count+=len(e)
   if g=='2' or p in keep:
    key=g+'.'+p
    label=products.get(p,{}).get('label',p);unit=books['CL_UOM'].get(a.get('UOM'),{}).get('label',a.get('UOM'))
    reference=re.search(r'\((\d{4,6}=100)\)',label)
    index_reference='December 2002=100' if g=='31' and p=='2' else reference[1] if reference else unit
    s=series.setdefault(key,dict(geo=g,product=p,label=label,geography=geos.get(g,{}).get('label',g),vector='v'+a['VECTOR_ID'],unit=unit,index_reference=index_reference,index_reference_note='Iqaluit override from release Table 2; raw UOM codelist retained separately.' if g=='31' and p=='2' else '',decimals=int(a.get('NB_DECIMAL',1)),dguid=a.get('DGUID'),values={},flags={},missing={}))
    for o in e:
     t=o.attrib['TIME_PERIOD'];v=o.attrib.get('OBS_VALUE');flags={k:v for k,v in o.attrib.items() if k not in ('TIME_PERIOD','OBS_VALUE')}
     if flags:s['flags'][t]=flags
     for k,vv in flags.items():status[k+'='+vv]+=1
     if v is None or not re.fullmatch(r'-?\d+(\.\d+)?',v):
      s['missing'][t]={'raw_value':v,'attributes':flags};continue
     v=float(v)
     if t in s['values'] and s['values'][t]!=v:conflicts.append([key,t,s['values'][t],v,f.name])
     s['values'][t]=v
    files[key]=f.name
   e.clear()
 # Preserve all observations in chosen series, without converting missing to zero.
 weights=[]
 for r in ziprows('18100007'):
  if r['GEO']=='Canada' and r['Products and product groups'] in MAJORS+['All-items'] and '2021'<=r['REF_DATE']<='2025' and r['Geographic distribution of weight']=='Distribution to selected geographies' and r['VALUE']:
   weights.append(dict(year=r['REF_DATE'],label=r['Products and product groups'],price_period=r['Price period of weight'],value=float(r['VALUE']),vector=r['VECTOR'],status=r['STATUS']))
 seasonal={}
 for r in ziprows('18100006'):
  if r['GEO']=='Canada' and r['REF_DATE']<='2026-09' and r['VALUE']:
   s=seasonal.setdefault(r['Products and product groups'],{'values':{},'vector':r['VECTOR'],'unit':r['UOM'],'flags':{}})
   s['values'][r['REF_DATE']]=float(r['VALUE'])
   if r['STATUS']:s['flags'][r['REF_DATE']]=r['STATUS']
 rows,core=parse_release()
 bank={};bankfile=ROOT/'references/bank-of-canada-cpi.html'
 if bankfile.exists():
  html=bankfile.read_text(encoding='utf-8')
  bankrow=next((r for r in re.findall(r'<tr\b[^>]*>(.*?)</tr>',html,re.S) if '>2026-09</th>' in r),'')
  for key,label in [('cpitrim','CPI-trim'),('cpimedian','CPI-median'),('cpicommon','CPI-common')]:
   match=re.search(r"<td[^>]*headers=['\"][^'\"]*th-"+key+r"\b[^'\"]*['\"][^>]*>([\d.]+)</td>",bankrow)
   if match:bank[label]=float(match[1])
 write('data/processed/bank-core.json',{'period':'2026-09','values':bank,'source':'https://www.bankofcanada.ca/rates/price-indexes/cpi/','method':'Extract the 2026-09 HTML row by semantic column headers; compare with release PDF Table 4.'})
 contributors=json.loads((ROOT/'data/processed/release-tables.json').read_text(encoding='utf-8'))['contributors']
 data={'as_of':'2026-10-19','target':'2026-09','table':'18-10-0004-01','series':series,'geographies':geos,'products':products,'majors':MAJORS,'weights':weights,'seasonal':seasonal,'release':rows,'contributors':contributors,'core':core,'bank_core':bank,'notes':{'raw_observation_count':count,'selected_series':len(series),'conflicts':conflicts,'selected_flags':dict(status),'vintage':'Historical snapshot through 2024-08 plus 25 monthly patches to 2026-09. Revisions in separately retrieved seasonal data may differ from release vintage.'}}
 write('data/processed/cpi.json',data)
 (ROOT/'data/processed/cpi-data.js').write_text('window.CPI_DATA='+json.dumps(data,ensure_ascii=False,separators=(',',':'))+';',encoding='utf-8')
 with (ROOT/'data/processed/series-map.csv').open('w',newline='',encoding='utf-8') as f:
  w=csv.writer(f);w.writerow(['series_key','vector','geography','product','raw_unit','index_reference','first','last','observations'])
  for k,s in series.items():w.writerow([k,s['vector'],s['geography'],s['label'],s['unit'],s['index_reference'],min(s['values']),max(s['values']),len(s['values'])])
 print(json.dumps({'raw_observations':count,'series':len(series),'weights':len(weights),'seasonal':len(seasonal),'release_rows':len(rows),'core':core,'conflicts':len(conflicts)}))
if __name__=='__main__':main()
