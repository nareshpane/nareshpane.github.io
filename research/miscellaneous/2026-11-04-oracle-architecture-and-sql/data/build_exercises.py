"""Rebuild 12 reproducible Oracle 19c analyst exercises, standard library only."""
import datetime as dt
import json
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
D=json.loads((ROOT/'data/sample.json').read_text(encoding='utf-8'))
E=[]
def add(n,title,inputs,grain,keys,filters,sql,output,pitfall,interpretation):
    E.append(dict(id=n,title=title,inputs=inputs,grain=grain,join_keys=keys,filters=filters,sql=sql,
        output=output,pitfall=pitfall,interpretation=interpretation))
P=D['pharmacy'];H=D['hospital'];G={g['geography_key']:g for g in D['geography']}
add(1,'Count dispensing lines by province and month', ['cb_dtl_ft_pharmacy_dispense','cb_dtl_dm_date','cb_dtl_dm_geography'],
    'One province-month','dispense_date_key = date_key; geography_key = geography_key',
    'January through March 2026; dispensing date in Nepal local calendar',
'''SELECT g.province, d.month_label, COUNT(*) AS dispensing_lines
FROM cb_dtl_ft_pharmacy_dispense f
JOIN cb_dtl_dm_date d ON d.date_key = f.dispense_date_key
JOIN cb_dtl_dm_geography g ON g.geography_key = f.geography_key
WHERE d.calendar_date >= DATE '2026-01-01' AND d.calendar_date < DATE '2026-04-01'
GROUP BY g.province, d.month_label ORDER BY g.province, d.month_label;''',
    [dict(province=geo['province'],month_label=m,dispensing_lines=sum(r['geography_key']==g and r['dispensed_at'][:7]==m for r in P))
        for g,geo in sorted(G.items(),key=lambda t:t[1]['province']) for m in ['2026-01','2026-02','2026-03']],
    'A dispensing line count is not a prescription count, patient count, or population rate.',
    'Facility geography is recorded at dispensing; it is not necessarily patient residence.')
add(2,'Count distinct recipients of product A',['cb_dtl_ft_pharmacy_dispense','cb_dtl_dm_health_product'],
    'One selected product','product_key',"Product code SYN-A; all fixture dates",
'''SELECT p.product_code, COUNT(*) AS dispensing_lines,
       COUNT(DISTINCT f.client_key) AS distinct_clients
FROM cb_dtl_ft_pharmacy_dispense f
JOIN cb_dtl_dm_health_product p ON p.product_key = f.product_key
WHERE p.product_code = 'SYN-A'
GROUP BY p.product_code;''',
    [dict(product_code='SYN-A',dispensing_lines=8,distinct_clients=2)],
    'Summing monthly distinct counts double counts patients observed in several months.',
    'Two synthetic clients received product A; dispensing does not establish adherence.')
add(3,'Average exact hospitalization duration by facility',['hdn_v_hospital_duration','cb_dtl_dm_facility'],
    'One facility','facility_key','Completed stays; all fixture dates',
'''SELECT f.facility_name, COUNT(*) AS stays,
       ROUND(AVG(h.computed_los_hours), 6) AS mean_los_hours
FROM hdn_v_hospital_duration h
JOIN cb_dtl_dm_facility f ON f.facility_key = h.facility_key
GROUP BY f.facility_name ORDER BY f.facility_name;''',
    [dict(facility_name=f['facility_name'],stays=2,mean_los_hours=round(sum(h['length_of_stay_hours'] for h in H if h['facility_key']==f['facility_key'])/2,6)) for f in D['facilities']],
    'A diagnosis bridge join repeats stays; averaging the repeated rows weights multi-diagnosis stays more heavily.',
    'Duration in hours differs from calendar bed-days; open stays need an explicit censoring rule.')
gap=(dt.datetime.fromisoformat(H[2]['admitted_at'])-dt.datetime.fromisoformat(H[0]['discharged_at'])).total_seconds()/86400
add(4,'Find the next admission within 30 days of prior discharge',['cb_dtl_ft_hospital_stay'],
    'One qualifying subsequent admission','Partition client_key; order admitted_at, hospital_stay_key',
    'Nonnegative gap no greater than 30 elapsed days; sequential prior admission',
'''WITH sequenced AS (
 SELECT h.*, LAG(discharged_at) OVER
  (PARTITION BY client_key ORDER BY admitted_at, hospital_stay_key) AS prior_discharge
 FROM cb_dtl_ft_hospital_stay h
)
SELECT client_key, hospital_stay_key,
 ROUND((CAST(SYS_EXTRACT_UTC(admitted_at) AS DATE)
      - CAST(SYS_EXTRACT_UTC(prior_discharge) AS DATE)), 6) AS gap_days
FROM sequenced
WHERE admitted_at >= prior_discharge
  AND admitted_at <= prior_discharge + NUMTODSINTERVAL(30, 'DAY')
ORDER BY client_key, admitted_at, hospital_stay_key;''',
    [dict(client_key=1,hospital_stay_key=3,gap_days=round(gap,6))],
    'This simple sequence is insufficient for overlapping stays, transfers, or exclusions for planned readmissions.',
    'A descriptive subsequent admission flag; not a validated readmission quality indicator.')
add(5,'Find clients with repeated emergency attendances',['cb_dtl_ft_emergency_visit'],
    'One client','client_key','At least two attendances during January–March',
'''SELECT client_key, COUNT(*) AS emergency_attendances
FROM cb_dtl_ft_emergency_visit
WHERE arrival_date_key >= 20260101 AND arrival_date_key < 20260401
GROUP BY client_key HAVING COUNT(*) >= 2 ORDER BY client_key;''',
    [dict(client_key=2,emergency_attendances=2)],
    'An emergency attendance followed by admission is still one ED attendance and one separate inpatient stay.',
    'The fixture identifies one frequent user under this illustrative threshold.')
add(6,'Compare recorded pharmacy utilization by geography',['cb_dtl_ft_pharmacy_dispense','cb_dtl_dm_geography'],
    'One dispensing province','geography_key','All fixture dates; retain unknown cost',
'''SELECT g.province, COUNT(*) AS dispensing_lines, SUM(f.quantity) AS units,
 COUNT(f.total_cost) AS recorded_cost_lines, SUM(f.total_cost) AS known_cost_npr
FROM cb_dtl_ft_pharmacy_dispense f
JOIN cb_dtl_dm_geography g ON g.geography_key = f.geography_key
GROUP BY g.province ORDER BY g.province;''',
    [dict(province=g['province'],dispensing_lines=8,units=sum(r['quantity'] for r in P if r['geography_key']==k),
        recorded_cost_lines=sum(r['total_cost'] is not None for r in P if r['geography_key']==k),
        known_cost_npr=sum(r['total_cost'] or 0 for r in P if r['geography_key']==k)) for k,g in sorted(G.items(),key=lambda t:t[1]['province'])],
    'Units across different products are not equipotent doses. Missing costs do not mean zero cost.',
    'Counts describe these samples; no resident population denominator or sampling weights exist.')
add(7,'Construct age bands at a fixed reporting date',['cb_dtl_dm_client','cb_dtl_ft_pharmacy_dispense'],
    'One age band','client_key','Age at 2026-03-31; completed years using Oracle MONTHS_BETWEEN convention',
'''WITH ages AS (
 SELECT client_key, FLOOR(MONTHS_BETWEEN(DATE '2026-03-31', birth_date)/12) AS age_years
 FROM cb_dtl_dm_client
), banded AS (
 SELECT client_key, CASE WHEN age_years < 18 THEN '00-17'
 WHEN age_years < 45 THEN '18-44' WHEN age_years < 65 THEN '45-64'
 ELSE '65+' END AS age_band FROM ages
)
SELECT b.age_band, COUNT(*) AS dispensing_lines,
 COUNT(DISTINCT f.client_key) AS distinct_clients
FROM cb_dtl_ft_pharmacy_dispense f JOIN banded b ON b.client_key = f.client_key
GROUP BY b.age_band ORDER BY b.age_band;''',
    [dict(age_band='18-44',dispensing_lines=8,distinct_clients=2),dict(age_band='45-64',dispensing_lines=4,distinct_clients=1),dict(age_band='65+',dispensing_lines=4,distinct_clients=1)],
    'Age at reporting date differs from age at service; leap-day and source birth-date rules require governance.',
    'A person-level fixed-age classification is reused consistently across the reporting period.')
add(8,'Rank facilities by admission count',['cb_dtl_ft_hospital_stay','cb_dtl_dm_facility'],
    'One facility','facility_key','All four admissions; count admissions, not discharge or diagnosis rows',
'''WITH utilization AS (
 SELECT facility_key, COUNT(*) AS admissions
 FROM cb_dtl_ft_hospital_stay GROUP BY facility_key
)
SELECT f.facility_name, u.admissions,
 RANK() OVER (ORDER BY u.admissions DESC) AS utilization_rank
FROM utilization u JOIN cb_dtl_dm_facility f ON f.facility_key = u.facility_key
ORDER BY utilization_rank, f.facility_name;''',
    [dict(facility_name=f['facility_name'],admissions=2,utilization_rank=1) for f in D['facilities']],
    'Tie breaking inside RANK would remove the intentional tie; use a separate outer sort for display.',
    'Equal fixture counts reveal nothing about quality, need, capacity, or risk-adjusted performance.')
add(9,'Create a client-level longitudinal utilization dataset',['cb_dtl_dm_client','cb_dtl_ft_pharmacy_dispense','cb_dtl_ft_hospital_stay','cb_dtl_ft_emergency_visit','cb_dtl_ft_physician_visit'],
    'One client','client_key after separate fact aggregation','All fixture dates; include clients with zero encounters in a setting',
'''WITH p AS (SELECT client_key, COUNT(*) AS pharmacy_lines
 FROM cb_dtl_ft_pharmacy_dispense GROUP BY client_key),
h AS (SELECT client_key, COUNT(*) AS admissions
 FROM cb_dtl_ft_hospital_stay GROUP BY client_key),
e AS (SELECT client_key, COUNT(*) AS ed_visits
 FROM cb_dtl_ft_emergency_visit GROUP BY client_key),
v AS (SELECT client_key, COUNT(*) AS physician_visits
 FROM cb_dtl_ft_physician_visit GROUP BY client_key)
SELECT c.client_id, NVL(p.pharmacy_lines,0) AS pharmacy_lines,
 NVL(h.admissions,0) AS admissions, NVL(e.ed_visits,0) AS ed_visits,
 NVL(v.physician_visits,0) AS physician_visits
FROM cb_dtl_dm_client c LEFT JOIN p ON p.client_key=c.client_key
LEFT JOIN h ON h.client_key=c.client_key LEFT JOIN e ON e.client_key=c.client_key
LEFT JOIN v ON v.client_key=c.client_key ORDER BY c.client_id;''',
    [dict(client_id=c['client_id'],pharmacy_lines=4,admissions=sum(r['client_key']==c['client_key'] for r in H),
        ed_visits=sum(r['client_key']==c['client_key'] for r in D['emergency']),physician_visits=1) for c in D['clients']],
    'Joining raw facts on client_key multiplies the counts across settings.',
    'Zero means no row in this complete synthetic extract; in real data it may instead mean missing coverage.')
add(10,'Build a balanced province-month panel',['cb_dtl_dm_geography','cb_dtl_dm_date','cb_dtl_ft_pharmacy_dispense'],
    'One province-month including zero-use cells','Cross join geography and distinct months; left join fact summary',
    'January–March 2026; panel geography dimension contains one row per illustrated province',
'''WITH months AS (SELECT DISTINCT month_label FROM cb_dtl_dm_date
 WHERE calendar_date >= DATE '2026-01-01' AND calendar_date < DATE '2026-04-01'),
counts AS (SELECT f.geography_key, d.month_label, COUNT(*) AS dispensing_lines
 FROM cb_dtl_ft_pharmacy_dispense f JOIN cb_dtl_dm_date d ON d.date_key=f.dispense_date_key
 GROUP BY f.geography_key, d.month_label)
SELECT g.province, m.month_label, NVL(c.dispensing_lines,0) AS dispensing_lines
FROM cb_dtl_dm_geography g CROSS JOIN months m
LEFT JOIN counts c ON c.geography_key=g.geography_key AND c.month_label=m.month_label
ORDER BY g.province, m.month_label;''',
    E[0]['output'],
    'Do not infer zero utilization for an unreported facility or inactive coverage month; add coverage flags in a real panel.',
    'The six balanced sample cells illustrate a panel structure, not a causal econometric design.')
add(11,'Calculate a rolling three-calendar-month utilization count',['cb_dtl_dm_date','cb_dtl_ft_pharmacy_dispense'],
    'One calendar month for selected client 1','month_label after month spine and fact summary','Client 1; January–March 2026, no earlier history',
'''WITH months AS (SELECT DISTINCT month_label FROM cb_dtl_dm_date),
counts AS (SELECT d.month_label, COUNT(*) AS lines
 FROM cb_dtl_ft_pharmacy_dispense f JOIN cb_dtl_dm_date d ON d.date_key=f.dispense_date_key
 WHERE f.client_key=1 GROUP BY d.month_label),
complete_months AS (SELECT m.month_label, NVL(c.lines,0) AS lines
 FROM months m LEFT JOIN counts c ON c.month_label=m.month_label)
SELECT month_label, lines,
 SUM(lines) OVER (ORDER BY month_label ROWS BETWEEN 2 PRECEDING AND CURRENT ROW)
 AS trailing_three_month_lines
FROM complete_months ORDER BY month_label;''',
    [dict(month_label='2026-01',lines=2,trailing_three_month_lines=2),dict(month_label='2026-02',lines=1,trailing_three_month_lines=3),dict(month_label='2026-03',lines=1,trailing_three_month_lines=4)],
    'ROWS counts months here only because the spine guarantees exactly one row for every consecutive month.',
    'January and February have incomplete preceding windows; the fixture has no pre-January observation history.')
add(12,'Form an illustrative chronic-condition cohort candidate',['cb_dtl_ft_physician_visit','cb_dtl_ft_hospital_stay','cb_dtl_br_stay_diagnosis','cb_dtl_dm_diagnosis','cb_dtl_ft_pharmacy_dispense','cb_dtl_dm_health_product'],
    'One candidate client','diagnosis_key, hospital_stay_key, client_key, product_key',
    "At least two distinct local dates with illustrative ICD-10 E11.9 plus SYN-A exposure during fixture dates",
'''WITH diagnosis_dates AS (
 SELECT v.client_key, v.visit_date_key AS diagnosis_date_key
 FROM cb_dtl_ft_physician_visit v JOIN cb_dtl_dm_diagnosis d ON d.diagnosis_key=v.diagnosis_key
 WHERE d.code_system='ICD-10' AND d.diagnosis_code='E11.9'
 UNION
 SELECT h.client_key, h.admit_date_key
 FROM cb_dtl_ft_hospital_stay h
 JOIN cb_dtl_br_stay_diagnosis b ON b.hospital_stay_key=h.hospital_stay_key
 JOIN cb_dtl_dm_diagnosis d ON d.diagnosis_key=b.diagnosis_key
 WHERE d.code_system='ICD-10' AND d.diagnosis_code='E11.9'
), evidence AS (SELECT client_key, COUNT(*) AS diagnosis_dates
 FROM diagnosis_dates GROUP BY client_key HAVING COUNT(*)>=2),
exposure AS (SELECT f.client_key, COUNT(*) AS exposure_lines
 FROM cb_dtl_ft_pharmacy_dispense f JOIN cb_dtl_dm_health_product p ON p.product_key=f.product_key
 WHERE p.product_code='SYN-A' GROUP BY f.client_key)
SELECT e.client_key, e.diagnosis_dates, x.exposure_lines
FROM evidence e JOIN exposure x ON x.client_key=e.client_key ORDER BY e.client_key;''',
    [dict(client_key=1,diagnosis_dates=3,exposure_lines=4)],
    'Codes and medication proxies alone are not a clinically validated disease definition; incidental use and coding errors matter.',
    'A synthetic case-finding exercise requiring clinical validation, code-version governance, sensitivity analyses, and a defined lookback period.')
(ROOT/'data/exercises.json').write_text(json.dumps(E,indent=2)+'\n',encoding='utf-8')
(ROOT/'data/exercises.js').write_text('window.HDN_EXERCISES = '+json.dumps(E,indent=2)+';\n',encoding='utf-8')
(ROOT/'sql/07_analyst_exercises.sql').write_text('-- Oracle 19c analyst exercises. Outputs and assumptions: ../data/exercises.json\n\n'+'\n\n'.join('-- '+str(e['id'])+'. '+e['title']+'\n-- Grain: '+e['grain']+'\n-- Filter: '+e['filters']+'\n'+e['sql'] for e in E)+'\n',encoding='utf-8')
print(f'Wrote {len(E)} exercises with matching synthetic outputs.')
