"""Static integrity/calculation checks. Does NOT execute Oracle SQL."""
import collections
import datetime as dt
import json
import math
import re
import statistics
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
D=json.loads((ROOT/'data/sample.json').read_text(encoding='utf-8'))
E=json.loads((ROOT/'data/exercises.json').read_text(encoding='utf-8'))
P=D['pharmacy'];H=D['hospital'];V=D['physician'];ED=D['emergency']
assert json.loads((ROOT/'data/sample.js').read_text(encoding='utf-8').split('window.HDN_DATA = ',1)[1].rstrip(';\n'))==D
assert json.loads((ROOT/'data/exercises.js').read_text(encoding='utf-8').split('window.HDN_EXERCISES = ',1)[1].rstrip(';\n'))==E
refs={'client_key':'clients','prescriber_key':'prescribers','product_key':'products','pharmacy_key':'pharmacies',
      'geography_key':'geography','facility_key':'facilities','diagnosis_key':'diagnoses','service_key':'services',
      'dispense_date_key':'date','admit_date_key':'date','discharge_date_key':'date','visit_date_key':'date','arrival_date_key':'date',
      'hospital_stay_key':'hospital','residence_geography_key':'geography'}
keys={'clients':'client_key','prescribers':'prescriber_key','products':'product_key','pharmacies':'pharmacy_key',
      'geography':'geography_key','facilities':'facility_key','diagnoses':'diagnosis_key','services':'service_key','date':'date_key',
      'hospital':'hospital_stay_key','pharmacy':'dispense_line_key','physician':'physician_visit_key','emergency':'emergency_visit_key'}
for name,key in keys.items():
    assert len({r[key] for r in D[name]})==len(D[name]),name
for name in keys|{'hospital_diagnosis':'diagnosis_position'}:
    for row in D[name]:
        for col,ref in refs.items():
            if col in row and row[col] is not None:
                assert any(r[keys[ref]]==row[col] for r in D[ref]),(name,col,row[col])
for row in H:
    actual=(dt.datetime.fromisoformat(row['discharged_at'])-dt.datetime.fromisoformat(row['admitted_at'])).total_seconds()/3600
    assert math.isclose(row['length_of_stay_hours'],actual)
for row in ED:
    assert (dt.datetime.fromisoformat(row['departed_at'])-dt.datetime.fromisoformat(row['arrived_at'])).total_seconds()/60==row['ed_length_of_stay_minutes']
    assert (dt.datetime.fromisoformat(row['triaged_at'])-dt.datetime.fromisoformat(row['arrived_at'])).total_seconds()/60==row['waiting_minutes']
    if row['hospital_stay_key'] is not None:
        stay=next(h for h in H if h['hospital_stay_key']==row['hospital_stay_key'])
        assert stay['client_key']==row['client_key']
        assert dt.datetime.fromisoformat(row['departed_at'])<=dt.datetime.fromisoformat(stay['admitted_at'])
assert len({(r['source_system'],r['source_record_id']) for r in P})==len(P)
assert len({(r['hospital_stay_key'],r['diagnosis_position']) for r in D['hospital_diagnosis']})==6

actual={}
geos=sorted(D['geography'],key=lambda r:r['province'])
actual[1]=[dict(province=g['province'],month_label=m,dispensing_lines=sum(r['geography_key']==g['geography_key'] and r['dispensed_at'].startswith(m) for r in P)) for g in geos for m in ['2026-01','2026-02','2026-03']]
selected=[p for p in P if p['product_key']==1]
actual[2]=[dict(product_code='SYN-A',dispensing_lines=len(selected),distinct_clients=len({r['client_key'] for r in selected}))]
actual[3]=[dict(facility_name=f['facility_name'],stays=len(rows),mean_los_hours=round(statistics.mean(r['length_of_stay_hours'] for r in rows),6))
 for f in D['facilities'] if (rows:=[h for h in H if h['facility_key']==f['facility_key']])]
actual[4]=[]
for c in D['clients']:
    stays=sorted([h for h in H if h['client_key']==c['client_key']],key=lambda r:(r['admitted_at'],r['hospital_stay_key']))
    for a,b in zip(stays,stays[1:]):
        gap=(dt.datetime.fromisoformat(b['admitted_at'])-dt.datetime.fromisoformat(a['discharged_at'])).total_seconds()/86400
        if 0<=gap<=30: actual[4].append(dict(client_key=c['client_key'],hospital_stay_key=b['hospital_stay_key'],gap_days=round(gap,6)))
edcounts=collections.Counter(r['client_key'] for r in ED)
actual[5]=[dict(client_key=k,emergency_attendances=n) for k,n in sorted(edcounts.items()) if n>=2]
actual[6]=[dict(province=g['province'],dispensing_lines=len(rows),units=sum(r['quantity'] for r in rows),recorded_cost_lines=sum(r['total_cost'] is not None for r in rows),known_cost_npr=sum(r['total_cost'] for r in rows if r['total_cost'] is not None)) for g in geos if (rows:=[p for p in P if p['geography_key']==g['geography_key']])]
bands=collections.defaultdict(list)
for c in D['clients']:
    birth=dt.date.fromisoformat(c['birth_date']);asof=dt.date(2026,3,31)
    age=asof.year-birth.year-((asof.month,asof.day)<(birth.month,birth.day))
    band='00-17' if age<18 else '18-44' if age<45 else '45-64' if age<65 else '65+'
    bands[band].extend([p for p in P if p['client_key']==c['client_key']])
actual[7]=[dict(age_band=b,dispensing_lines=len(rows),distinct_clients=len({r['client_key'] for r in rows})) for b,rows in sorted(bands.items())]
facilitycounts=collections.Counter(h['facility_key'] for h in H)
actual[8]=[dict(facility_name=f['facility_name'],admissions=facilitycounts[f['facility_key']],utilization_rank=1+sum(n>facilitycounts[f['facility_key']] for n in facilitycounts.values())) for f in D['facilities']]
actual[9]=[dict(client_id=c['client_id'],pharmacy_lines=sum(r['client_key']==c['client_key'] for r in P),admissions=sum(r['client_key']==c['client_key'] for r in H),ed_visits=sum(r['client_key']==c['client_key'] for r in ED),physician_visits=sum(r['client_key']==c['client_key'] for r in V)) for c in D['clients']]
actual[10]=actual[1]
monthcounts=collections.Counter(r['dispensed_at'][:7] for r in P if r['client_key']==1)
months=sorted({r['month_label'] for r in D['date']})
actual[11]=[dict(month_label=m,lines=monthcounts[m],trailing_three_month_lines=sum(monthcounts[x] for x in months[max(0,i-2):i+1])) for i,m in enumerate(months)]
dates=collections.defaultdict(set)
for v in V:
    if v['diagnosis_key']==1:dates[v['client_key']].add(v['visit_date_key'])
for b in D['hospital_diagnosis']:
    if b['diagnosis_key']==1:
        h=next(h for h in H if h['hospital_stay_key']==b['hospital_stay_key'])
        dates[h['client_key']].add(h['admit_date_key'])
actual[12]=[dict(client_key=c,diagnosis_dates=len(ds),exposure_lines=sum(r['client_key']==c and r['product_key']==1 for r in P)) for c,ds in sorted(dates.items()) if len(ds)>=2 and any(r['client_key']==c and r['product_key']==1 for r in P)]
for e in E: assert e['output']==actual[e['id']],(e['id'],e['output'],actual[e['id']])
ddl=(ROOT/'sql/01_schema.sql').read_text(encoding='utf-8')
created=set()
for table,body in re.findall(r'CREATE TABLE (\w+) \((.*?)\n\);',ddl,re.S):
    for ref in re.findall(r'REFERENCES (\w+)',body):assert ref in created,(table,ref)
    created.add(table)
assert len(created)==14
seed=(ROOT/'sql/02_seed.sql').read_text(encoding='utf-8')
for key,table in D['meta']['tables'].items():
    assert len(re.findall('INSERT INTO '+table+r'\b',seed))==len(D[key]),key
costs=[p['total_cost'] for p in P if p['total_cost'] is not None]
quantities=[p['quantity'] for p in P if p['total_cost'] is not None]
qmean=statistics.mean(quantities); cmean=statistics.mean(costs)
slope=sum((x-qmean)*(y-cmean) for x,y in zip(quantities,costs))/sum((x-qmean)**2 for x in quantities)
summary=dict(validation='Static checks and independent Python calculations passed; Oracle not executed.',
    checked_exercises=len(E),checked_tables=len(created),checks=D['meta']['checks'],
    costs=dict(count=len(costs),sum=sum(costs),mean=statistics.mean(costs),median=statistics.median(costs),
        min=min(costs),max=max(costs),sample_sd=statistics.stdev(costs),sample_variance=statistics.variance(costs),population_variance=statistics.pvariance(costs),
        correlation=statistics.correlation(quantities,costs),slope_cost_on_quantity=slope,intercept_cost_on_quantity=cmean-slope*qmean),
    fanout=dict(admission_rows=4,diagnosis_join_rows=6,correct_los_sum=sum(h['length_of_stay_hours'] for h in H),
        inflated_los_sum=sum(next(h['length_of_stay_hours'] for h in H if h['hospital_stay_key']==b['hospital_stay_key']) for b in D['hospital_diagnosis'])))
(ROOT/'data/validation.json').write_text(json.dumps(summary,indent=2)+'\n',encoding='utf-8')
print(json.dumps(summary,indent=2))
