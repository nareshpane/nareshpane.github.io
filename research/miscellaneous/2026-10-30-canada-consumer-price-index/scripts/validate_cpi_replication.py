"""Compare extracted SDMX/CSV observations with independently extracted release tables.

Tolerance: indices must match at 1e-8. Growth rates use 0.05 percentage points,
the half-width of a published one-decimal rounding bin. Source-rounded index
uncertainty is reported separately and never used to hide a failed comparison.
"""
from pathlib import Path
from collections import Counter
import json, math, csv, hashlib
ROOT=Path(__file__).resolve().parents[1]
def rate(a,b):return 100*(a/b-1)
def rate_interval(a,b):return [rate(a-.05,b+.05),rate(a+.05,b-.05)]
def main():
 d=json.loads((ROOT/'data/processed/cpi.json').read_text(encoding='utf-8'));rows=[];missing=[]
 def add(label,geo,period,official,calculated,level,metric,source,tol=1e-8,**extra):
  diff=None if calculated is None or official is None else calculated-official
  result=('Unavailable' if calculated is None else 'Calculated; no release comparator' if official is None else 'Matched' if abs(diff)<1e-8 else 'Within rounding tolerance' if abs(diff)<=tol+1e-10 else 'Mismatch')
  rows.append(dict(series=label,geography=geo,reference_period=period,official=official,calculated=calculated,difference=diff,tolerance=tol,category=level,metric=metric,result=result,source=source,**extra))
 def find(label,geo='Canada'):
  canonical={'Whitehorse':'Whitehorse, Yukon','Yellowknife':'Yellowknife, Northwest Territories','Iqaluit':'Iqaluit, Nunavut'}.get(geo,geo)
  exact=next((s for s in d['series'].values() if s['label']==label and s['geography']==canonical),None)
  if exact:return exact
  # Release labels sometimes omit an explicit index-reference suffix.
  candidates=[s for s in d['series'].values() if s['label'].startswith(label+' (') and s['geography']==canonical]
  return candidates[0] if len(candidates)==1 else None
 for r in d['release']:
  label=r['label'];geo='Canada';kind=r['kind']
  if kind=='geography':geo=label.replace(' (Dec. 2002=100)','');label='All-items'
  if kind=='seasonal':s=d['seasonal'].get(label);periods=['2026-07','2026-08','2026-09']
  else:s=find(label,geo);periods=['2025-09','2026-08','2026-09']
  source=f"{'18-10-0006-01 CSV' if kind=='seasonal' else '18-10-0004-01 SDMX'} vs release PDF p.{r['pdf_page']}"
  vals=[s['values'].get(t) if s else None for t in periods]
  for t,v,official in zip(periods,vals,r['values'][:3]):add(label,geo,t,official,v,'B','SA index' if kind=='seasonal' else 'NSA index',source)
  a,b,c=vals
  for metric,prior,official in [('MoM',b,r['values'][4] if kind=='seasonal' else r['values'][3]),('YoY',a,r['values'][4])] if kind!='seasonal' else [('SA MoM',b,r['values'][4])]:
   calc=rate(c,prior) if c is not None and prior else None
   add(label,geo,'2026-09',official,calc,'A',metric,source,.05,rounding_interval=rate_interval(c,prior) if calc is not None else None,vector=s.get('vector') if s else None)
 # Table 5 provides component growth rates, never percentage-point contributions.
 for r in d.get('contributors',[]):
  s=find(r['label']);prior_period='2025-09' if r['metric']=='YoY' else '2026-08'
  current=s['values'].get('2026-09') if s else None;prior=s['values'].get(prior_period) if s else None
  add(r['label'],'Canada','2026-09',r['value'],rate(current,prior) if current and prior else None,'A',r['metric'],'18-10-0004-01 SDMX vs release PDF p.10',.05,vector=s['vector'] if s else None)
 # Independent source matching of core rates is Level B, not core estimation.
 for label,value in d.get('bank_core',{}).items():
  add(label,'Canada','2026-09',d['core'][label],value,'B','Core YoY (source match)','Bank of Canada HTML vs Statistics Canada release PDF p.9')
 # All available geographical all-items series: only those in release get comparisons above.
 compared={r['label'].replace(' (Dec. 2002=100)','') for r in d['release'] if r['kind']=='geography'}|{'Whitehorse, Yukon','Yellowknife, Northwest Territories','Iqaluit, Nunavut'}
 for s in d['series'].values():
  if s['label']!='All-items' or s['geography'] in compared:continue
  v=s['values'];c=v.get('2026-09')
  for metric,t in [('MoM','2026-08'),('YoY','2025-09')]:
   prior=v.get(t);add('All-items',s['geography'],'2026-09',None,rate(c,prior) if c and prior else None,'A',metric,'18-10-0004-01 SDMX; no release-table comparator',.05,vector=s['vector'])
 # Product-specific arithmetic beyond the release comparison table.
 for label in ['Food purchased from stores','Rent','Air transportation','Fresh vegetables','Natural gas',"Homeowners' replacement cost"]:
  s=find(label)
  for metric,t in [('MoM','2026-08'),('YoY','2025-09')]:
   c=s['values'].get('2026-09') if s else None; prior=s['values'].get(t) if s else None
   add(label,'Canada','2026-09',None,rate(c,prior) if c and prior else None,'A',metric,'18-10-0004-01 SDMX',.05,vector=s['vector'] if s else None)
 def weights(year):
  a=[next(w['value'] for w in d['weights'] if w['year']==year and w['label']==label and w['price_period']=='Weight at basket link month prices') for label in d['majors']]
  return [v/sum(a) for v in a]
 wnew=weights('2025');wold=weights('2024');ms=[find(l)['values'] for l in d['majors']]
 newfactor=sum(w*s['2026-09']/s['2026-04'] for w,s in zip(wnew,ms))
 oldfactor=lambda t:sum(w*s[t]/s['2025-04'] for w,s in zip(wold,ms))
 allv=find('All-items')['values'];estimate=allv['2026-04']*newfactor
 approx={'method':'Eight rounded major-component link-month weights, normalized; April 2026 headline anchor. Within-component basket changes are not undone.', 'link_month':'2026-04','weights_2025':wnew,'link_index':allv['2026-04'],'factor':newfactor,'estimated_index':estimate,'official_index':allv['2026-09'],'index_difference':estimate-allv['2026-09'],'estimated_yoy':100*(oldfactor('2026-04')/oldfactor('2025-09')*newfactor-1),'official_yoy':3.1,'naive_weighted_levels':sum(w* s['2026-09'] for w,s in zip(wnew,ms))}
 add('Eight-component Lowe reconstruction','Canada','2026-09',170,estimate,'C','NSA index','18-10-0004-01 + 18-10-0007-01',0)
 rows[-1]['result']='Approximate'
 add('Product-offer construction','Canada','2026-09',170,None,'D','NSA index','Unpublished microdata, full internal weights, imputations and quality adjustments')
 rows[-1]['result']='Not independently reproducible'
 assert not d['notes']['conflicts'], 'Conflicting observations must be resolved'
 assert find('All-items')['values']['2026-09']==170
 assert abs(rate(170,164.9)-3.09278350515465)<1e-10
 assert abs(rate(170,169.8)-.1177856301531)<1e-10
 assert abs(rate(170,169.4)-.35419126328217)<1e-10
 assert len(d['majors'])==8
 for year in ['2024','2025']:assert abs(sum(weights(year))-1)<1e-12
 comparisons=[r for r in rows if r['official'] is not None and r['category'] in ('A','B')]
 failures=[r for r in comparisons if r['result'] in ('Mismatch','Unavailable')]
 report={'target':'2026-09','observations_scanned':d['notes']['raw_observation_count'],'series_extracted':len(d['series']),'summary':dict(Counter(r['result'] for r in rows)),'independent_comparisons':len(comparisons),'failures':failures,'aggregation':approx,'core':d['core'],'bank_core':d.get('bank_core',{}),'rows':rows,'tolerance_note':__doc__,'limitations':['Level B validates public series extraction, not retail-price construction.','Core figures match release Table 4 and Bank of Canada HTML; construction is not independently reconstructed.','Seasonal data are retrieved public results; X-12-ARIMA estimation is not rerun.','Public index rounding introduces extra ratio uncertainty; intervals shown separately.']}
 (ROOT/'validation/replication.json').write_text(json.dumps(report,indent=2,ensure_ascii=False),encoding='utf-8')
 (ROOT/'validation/replication-data.js').write_text('window.CPI_VALIDATION='+json.dumps(report,separators=(',',':'),ensure_ascii=False)+';',encoding='utf-8')
 with (ROOT/'validation/replication.csv').open('w',newline='',encoding='utf-8') as f:
  names=['series','geography','reference_period','metric','official','calculated','difference','tolerance','category','result','source'];w=csv.DictWriter(f,names,extrasaction='ignore');w.writeheader();w.writerows(rows)
 lines=['# September 2026 CPI validation','',f"Scanned {report['observations_scanned']:,} raw observations; {len(comparisons)} independent release-table comparisons.",'',json.dumps(report['summary'],indent=2),'','## Approximate aggregation','',json.dumps(approx,indent=2),'','## Limitations','']+['- '+s for s in report['limitations']]
 (ROOT/'validation/replication.md').write_text('\n'.join(lines),encoding='utf-8')
 inventory=json.loads((ROOT/'data/metadata/input-inventory.json').read_text(encoding='utf-8'))
 manifest={'prepared_for':'2026-10-30 research page','snapshot_as_of':'2026-10-19','original_download_dates':'Not known; SDMX Prepared fields retained, not represented as retrieval dates.','inputs':inventory,'retrievals':json.loads((ROOT/'data/metadata/retrievals.json').read_text(encoding='utf-8')),'transformations':['scripts/inspect_sources.py','scripts/prepare_data.py','scripts/validate_cpi_replication.py'],'coverage':'SDMX 1914-01 to 2026-09, uneven by series; seasonal 1992-01 onward, filtered at 2026-09; basket years 2021–2025.','originals_preserved':True}
 (ROOT/'data/metadata/manifest.json').write_text(json.dumps(manifest,indent=2),encoding='utf-8')
 print(json.dumps({k:report[k] for k in ['summary','independent_comparisons','aggregation','failures']},indent=2))
 assert not failures, f'{len(failures)} comparisons failed; inspect report'
if __name__=='__main__':main()
