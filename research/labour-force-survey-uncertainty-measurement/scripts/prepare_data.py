"""Frozen 2026-10-13 extract of 14-10-0287-01, preserving published rows."""
from pathlib import Path
import csv, io, zipfile, json
ROOT=Path(__file__).resolve().parents[1]
allocation=dict(zip(['Canada','Newfoundland and Labrador','Prince Edward Island','Nova Scotia','New Brunswick','Quebec','Ontario','Manitoba','Saskatchewan','Alberta','British Columbia'],[63596,3724,1600,3392,3912,12043,19519,4145,3903,4095,7262]))
rows=[]
with zipfile.ZipFile(ROOT/'sources'/'table-14100287.zip') as z:
 for row in csv.DictReader(io.TextIOWrapper(z.open('14100287.csv'),encoding='utf-8-sig')):
  if row['REF_DATE'] in ('2026-08','2026-09') and row['GEO'] in allocation and row['Gender']=='Total - Gender' and row['Age group']=='15 years and over' and row['Data type']=='Seasonally adjusted': rows.append(row)
(ROOT/'data').mkdir(exist_ok=True)
with (ROOT/'data'/'official-rows.csv').open('w',newline='',encoding='utf-8') as f:
 w=csv.DictWriter(f,fieldnames=rows[0].keys());w.writeheader();w.writerows(rows)
lookup={(r['GEO'],r['Labour force characteristics'],r['REF_DATE'],r['Statistics']):r for r in rows}
out=[]
for geo,target in allocation.items():
 indicators={}
 for indicator in ['Population','Labour force','Employment','Unemployment','Unemployment rate','Participation rate','Employment rate']:
  def get(date,stat):
   row=lookup.get((geo,indicator,date,stat));return float(row['VALUE']) if row and row['VALUE'] and not row['STATUS'] else None
  value=get('2026-09','Estimate');previous=get('2026-08','Estimate')
  r=lookup[(geo,indicator,'2026-09','Estimate')]
  indicators[indicator]=dict(value=value,previous=previous,change=round(value-previous,4) if value is not None and previous is not None else None,se=get('2026-09','Standard error of estimate'),se_change=get('2026-09','Standard error of month-to-month change'),unit=r['UOM'],vector=r['VECTOR'],status=r['STATUS'])
 out.append(dict(geography=geo,target_households_2025=target,indicators=indicators))
data=dict(reference_period='2026-09',comparison_period='2026-08',release_date='2026-10-09',retrieved_date='2026-10-13',source='https://www150.statcan.gc.ca/t1/tbl1/en/tv.action?pid=1410028701',adjustment='Seasonally adjusted',gender='Total - Gender',age='15 years and over',status='official published inputs; intervals calculated by author',geographies=out)
(ROOT/'data'/'september-2026.json').write_text(json.dumps(data,indent=2),encoding='utf-8')
(ROOT/'js').mkdir(exist_ok=True)
(ROOT/'js'/'official-data.js').write_text('window.LFS_OFFICIAL = '+json.dumps(data)+';\n',encoding='utf-8')
for g in out:
 print(g['geography'],g['indicators']['Employment'],g['indicators']['Unemployment rate'])
print('Rows:',len(rows),'Allocation sum:',sum(v for k,v in allocation.items() if k!='Canada'))
