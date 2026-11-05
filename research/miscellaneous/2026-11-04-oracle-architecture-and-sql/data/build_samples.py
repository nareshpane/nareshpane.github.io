"""Rebuild checked-in synthetic records and their Oracle 19c DDL/inserts.

Standard library only. All identities, providers, facilities, and products are fictional.
This generator does not connect to Oracle.
"""
import calendar
import datetime as dt
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SQL = ROOT / 'sql'
SQL.mkdir(exist_ok=True)
DATA = {'meta': {'name': 'HealthDataNepal', 'synthetic': True,
    'time_zone': '+05:45', 'currency': 'NPR', 'version': '2026-11-04',
    'execution': 'Illustrative browser-side simulation; Oracle SQL has not been executed.',
    'geography_note': 'Real province/district labels provide context; municipalities, wards, identities and facilities are illustrative.'}}
DATA['date'] = []
for n in range(90):
    d = dt.date(2026, 1, 1) + dt.timedelta(days=n)
    DATA['date'].append(dict(date_key=int(d.strftime('%Y%m%d')), calendar_date=d.isoformat(),
        calendar_year=d.year, month_number=d.month, month_label=d.strftime('%Y-%m'),
        quarter_number=(d.month-1)//3+1, day_of_week=d.isoweekday(),
        fiscal_period='Illustrative FY2025/26'))
DATA['geography'] = [dict(geography_key=1, province='Gandaki', district='Kaski', municipality='Synthetic Lakeside Municipality', ward=1),
    dict(geography_key=2, province='Bagmati', district='Chitwan', municipality='Synthetic Riverside Municipality', ward=2)]
DATA['clients'] = [dict(client_key=i, client_id=f'CLIENT_SYN_{i:03}', birth_date=b, residence_geography_key=g,
    effective_from='2020-01-01', effective_to=None, current_flag='Y')
    for i,b,g in [(1,'1980-06-15',1),(2,'1955-02-28',2),(3,'2005-09-01',1),(4,'1990-12-10',2)]]
DATA['prescribers'] = [dict(prescriber_key=i, prescriber_id=f'SYN-PR{i:02}', prescriber_name=name,
    specialty='General practice', status='ACTIVE', geography_key=i)
    for i,name in [(1,'Synthetic Provider One'),(2,'Synthetic Provider Two')]]
DATA['products'] = [dict(product_key=i, product_code=f'SYN-{code}', product_name=name,
    ingredient=f'Fictional ingredient {code}', strength='Illustrative 10 mg', dosage_form='Tablet',
    classification=category, brand=f'Synthetic Brand {code}', effective_from='2020-01-01', effective_to=None)
    for i,code,name,category in [(1,'A','Synthetic product A','Glucose-lowering illustration'),(2,'B','Synthetic product B','Respiratory illustration')]]
DATA['pharmacies'] = [dict(pharmacy_key=i, pharmacy_id=f'SYN-PH{i:02}', pharmacy_name=name, geography_key=i)
    for i,name in [(1,'Synthetic Lakeside Pharmacy'),(2,'Synthetic Riverside Pharmacy')]]
DATA['facilities'] = [dict(facility_key=i, facility_id=f'SYN-F{i:02}', facility_name=name,
    facility_type='Illustrative hospital', ownership_category='Fictional public', service_capability='Inpatient, outpatient, emergency', geography_key=i)
    for i,name in [(1,'Synthetic Lakeside Hospital'),(2,'Synthetic Riverside Hospital')]]
DATA['diagnoses'] = [dict(diagnosis_key=i, diagnosis_code=code, code_system='ICD-10', code_version='Illustrative 2019', description=desc)
    for i,code,desc in [(1,'E11.9','Illustrative type 2 diabetes code'),(2,'I10','Illustrative essential hypertension code'),(3,'J45.9','Illustrative asthma code')]]
DATA['services'] = [dict(service_key=1, service_code='SYN-CONSULT', service_description='Completed general consultation')]

lines = [(1,1,1,'01-10',30,120),(2,2,2,'01-11',10,80),(3,3,2,'01-15',20,None),(4,4,1,'01-16',30,120),
    (5,1,1,'01-20',30,120),(6,2,2,'01-21',10,80),(7,3,2,'02-01',20,90),(8,4,1,'02-02',30,120),
    (9,1,1,'02-10',30,150),(10,2,2,'02-11',15,80),(11,3,2,'02-15',20,90),(12,4,1,'02-16',30,120),
    (13,1,1,'03-10',30,150),(14,2,2,'03-11',15,80),(15,3,2,'03-15',20,90),(16,4,1,'03-16',30,120)]
DATA['pharmacy'] = []
for key,client,product,md,qty,cost in lines:
    geo=1 if client in (1,3) else 2
    DATA['pharmacy'].append(dict(dispense_line_key=key,source_system='PHARMA-NEPAL-SYN',source_record_id=f'DISPENSE_SYN_{key:03}',
        prescription_id=f'SYN-RX{key:03}', dispensing_id=f'SYN-DSP{key:03}', client_key=client,prescriber_key=geo,
        product_key=product,pharmacy_key=geo,geography_key=geo,dispense_date_key=int('2026'+md.replace('-','')),
        dispensed_at=f'2026-{md}T10:15:00+05:45',quantity=qty,days_supply=30 if product==1 else 10,
        ingredient_cost=None if cost is None else cost-10,dispensing_fee=None if cost is None else 10,total_cost=cost,
        payment_category='Synthetic public plan'))

stays=[(1,1,1,'01-12T23:50:00','01-15T06:20:30'),(2,2,2,'01-22T08:00:00','01-23T08:00:00'),
    (3,1,1,'02-01T12:00:00','02-03T12:00:00'),(4,3,2,'03-03T10:00:00','03-05T10:00:00')]
DATA['hospital']=[]
for key,client,facility,a,b in stays:
    admission=f'2026-{a}+05:45';discharge=f'2026-{b}+05:45'
    hours=(dt.datetime.fromisoformat(discharge)-dt.datetime.fromisoformat(admission)).total_seconds()/3600
    DATA['hospital'].append(dict(hospital_stay_key=key,encounter_id=f'HOSPITAL_SYN_{key:03}',client_key=client,facility_key=facility,
        admit_date_key=int(admission[:10].replace('-','')),discharge_date_key=int(discharge[:10].replace('-','')),
        admitted_at=admission,discharged_at=discharge,length_of_stay_hours=hours,
        admission_type='Unplanned',referral_source='Synthetic ED' if key in(1,4) else 'Synthetic clinic',discharge_disposition_code='HOME'))
DATA['hospital_diagnosis']=[dict(hospital_stay_key=s,diagnosis_key=d,diagnosis_position=p)
    for s,d,p in [(1,1,1),(1,2,2),(2,3,1),(3,1,1),(3,2,2),(4,3,1)]]
DATA['physician']=[]
for key,client,geo,md,diag in [(1,1,1,'01-10',1),(2,2,2,'01-11',3),(3,3,1,'02-01',3),(4,4,2,'02-02',2)]:
    DATA['physician'].append(dict(physician_visit_key=key,encounter_id=f'VISIT_SYN_{key:03}',client_key=client,prescriber_key=geo,
        facility_key=geo,geography_key=geo,service_key=1,diagnosis_key=diag,visit_date_key=int('2026'+md.replace('-','')),
        scheduled_at=f'2026-{md}T08:45:00+05:45',encountered_at=f'2026-{md}T09:00:00+05:45',
        encounter_status='COMPLETED',consultation_mode='In person',service_amount=200,claim_amount=None))
DATA['emergency']=[]
for key,client,facility,md,hour,wait,duration,stay,diag in [(1,1,1,'01-12',21,15,140,1,1),(2,2,2,'01-21',14,30,180,None,3),
        (3,2,2,'02-05',14,20,120,None,3),(4,3,2,'03-03',8,10,90,4,3)]:
    arrival=dt.datetime.fromisoformat(f'2026-{md}T{hour:02}:30:00+05:45')
    departure=arrival+dt.timedelta(minutes=duration)
    DATA['emergency'].append(dict(emergency_visit_key=key,encounter_id=f'EMERGENCY_SYN_{key:03}',client_key=client,facility_key=facility,
        diagnosis_key=diag,arrival_date_key=int('2026'+md.replace('-','')),arrived_at=arrival.isoformat(),
        triaged_at=(arrival+dt.timedelta(minutes=wait)).isoformat(),departed_at=departure.isoformat(),
        triage_category=2 if stay else 3,waiting_minutes=wait,ed_length_of_stay_minutes=duration,
        disposition='ADMITTED' if stay else 'HOME',hospital_stay_key=stay))
# Align first ED departure with admission while retaining separate encounter grains.
DATA['emergency'][0]['departed_at']=DATA['hospital'][0]['admitted_at']
DATA['emergency'][0]['ed_length_of_stay_minutes']=140
DATA['meta']['checks'] = dict(pharmacy_rows=len(DATA['pharmacy']), hospital_rows=4, diagnosis_bridge_rows=6,
    physician_rows=4, emergency_rows=4, known_cost_rows=sum(r['total_cost'] is not None for r in DATA['pharmacy']),
    known_cost_total=sum(r['total_cost'] or 0 for r in DATA['pharmacy']),quantity_total=sum(r['quantity'] for r in DATA['pharmacy']))

schema={
'date':('cb_dtl_dm_date',[('date_key','NUMBER(8)'),('calendar_date','DATE'),('calendar_year','NUMBER(4)'),('month_number','NUMBER(2)'),('month_label','VARCHAR2(7)'),('quarter_number','NUMBER(1)'),('day_of_week','NUMBER(1)'),('fiscal_period','VARCHAR2(40)')]),
'geography':('cb_dtl_dm_geography',[('geography_key','NUMBER'),('province','VARCHAR2(40)'),('district','VARCHAR2(40)'),('municipality','VARCHAR2(80)'),('ward','NUMBER(2)')]),
'clients':('cb_dtl_dm_client',[('client_key','NUMBER'),('client_id','VARCHAR2(30)'),('birth_date','DATE'),('residence_geography_key','NUMBER'),('effective_from','DATE'),('effective_to','DATE'),('current_flag','CHAR(1)')]),
'prescribers':('cb_dtl_dm_prescriber',[('prescriber_key','NUMBER'),('prescriber_id','VARCHAR2(30)'),('prescriber_name','VARCHAR2(80)'),('specialty','VARCHAR2(60)'),('status','VARCHAR2(12)'),('geography_key','NUMBER')]),
'products':('cb_dtl_dm_health_product',[('product_key','NUMBER'),('product_code','VARCHAR2(30)'),('product_name','VARCHAR2(80)'),('ingredient','VARCHAR2(80)'),('strength','VARCHAR2(40)'),('dosage_form','VARCHAR2(30)'),('classification','VARCHAR2(80)'),('brand','VARCHAR2(50)'),('effective_from','DATE'),('effective_to','DATE')]),
'pharmacies':('cb_dtl_dm_pharmacy',[('pharmacy_key','NUMBER'),('pharmacy_id','VARCHAR2(30)'),('pharmacy_name','VARCHAR2(80)'),('geography_key','NUMBER')]),
'facilities':('cb_dtl_dm_facility',[('facility_key','NUMBER'),('facility_id','VARCHAR2(30)'),('facility_name','VARCHAR2(80)'),('facility_type','VARCHAR2(40)'),('ownership_category','VARCHAR2(40)'),('service_capability','VARCHAR2(100)'),('geography_key','NUMBER')]),
'diagnoses':('cb_dtl_dm_diagnosis',[('diagnosis_key','NUMBER'),('diagnosis_code','VARCHAR2(16)'),('code_system','VARCHAR2(20)'),('code_version','VARCHAR2(30)'),('description','VARCHAR2(120)')]),
'services':('cb_dtl_dm_service',[('service_key','NUMBER'),('service_code','VARCHAR2(30)'),('service_description','VARCHAR2(100)')]),
'pharmacy':('cb_dtl_ft_pharmacy_dispense',[(k,('TIMESTAMP(0) WITH TIME ZONE' if k=='dispensed_at' else 'VARCHAR2(40)' if k in ('source_system','source_record_id','prescription_id','dispensing_id','payment_category') else 'NUMBER(12,2)' if k in('ingredient_cost','dispensing_fee','total_cost','quantity') else 'NUMBER')) for k in DATA['pharmacy'][0]]),
'hospital':('cb_dtl_ft_hospital_stay',[(k,'TIMESTAMP(0) WITH TIME ZONE' if k in('admitted_at','discharged_at') else 'VARCHAR2(40)' if k in('encounter_id','admission_type','referral_source','discharge_disposition_code') else 'NUMBER(16,8)' if k=='length_of_stay_hours' else 'NUMBER') for k in DATA['hospital'][0]]),
'hospital_diagnosis':('cb_dtl_br_stay_diagnosis',[('hospital_stay_key','NUMBER'),('diagnosis_key','NUMBER'),('diagnosis_position','NUMBER(3)')]),
'physician':('cb_dtl_ft_physician_visit',[(k,'TIMESTAMP(0) WITH TIME ZONE' if k.endswith('_at') else 'VARCHAR2(40)' if k in('encounter_id','encounter_status','consultation_mode') else 'NUMBER(12,2)' if k in('service_amount','claim_amount') else 'NUMBER') for k in DATA['physician'][0]]),
'emergency':('cb_dtl_ft_emergency_visit',[(k,'TIMESTAMP(0) WITH TIME ZONE' if k.endswith('_at') else 'VARCHAR2(40)' if k in('encounter_id','disposition') else 'NUMBER') for k in DATA['emergency'][0]])}

primary={key:fields[0][0] for key,(_,fields) in schema.items()}
primary['hospital_diagnosis']='hospital_stay_key, diagnosis_position'
nullable={'effective_to','ingredient_cost','dispensing_fee','total_cost','claim_amount','hospital_stay_key'}
foreign={
'clients':{'residence_geography_key':'geography'},'prescribers':{'geography_key':'geography'},
'pharmacies':{'geography_key':'geography'},'facilities':{'geography_key':'geography'},
'pharmacy':{'client_key':'clients','prescriber_key':'prescribers','product_key':'products','pharmacy_key':'pharmacies','geography_key':'geography','dispense_date_key':'date'},
'hospital':{'client_key':'clients','facility_key':'facilities','admit_date_key':'date','discharge_date_key':'date'},
'hospital_diagnosis':{'hospital_stay_key':'hospital','diagnosis_key':'diagnoses'},
'physician':{'client_key':'clients','prescriber_key':'prescribers','facility_key':'facilities','geography_key':'geography','service_key':'services','diagnosis_key':'diagnoses','visit_date_key':'date'},
'emergency':{'client_key':'clients','facility_key':'facilities','diagnosis_key':'diagnoses','arrival_date_key':'date','hospital_stay_key':'hospital'}}
uniques={'date':['calendar_date'],'clients':['client_id, effective_from'],'prescribers':['prescriber_id'],
'products':['product_code'],'pharmacies':['pharmacy_id'],'facilities':['facility_id'],
'diagnoses':['diagnosis_code, code_system, code_version'],'services':['service_code'],
'pharmacy':['source_system, source_record_id'],'hospital':['encounter_id'],'hospital_diagnosis':['hospital_stay_key, diagnosis_key'],
'physician':['encounter_id'],'emergency':['encounter_id']}
checks={'clients':["current_flag IN ('Y','N')",'effective_to IS NULL OR effective_to > effective_from'],
'pharmacy':['quantity > 0','days_supply > 0','total_cost IS NULL OR total_cost >= 0'],
'hospital':['discharged_at >= admitted_at','length_of_stay_hours >= 0'],
'hospital_diagnosis':['diagnosis_position > 0'],
'physician':["encounter_status = 'COMPLETED'"],
'emergency':['departed_at >= arrived_at','triaged_at >= arrived_at','triaged_at <= departed_at','triage_category BETWEEN 1 AND 5']}

ddl=['-- Oracle 19c. Run once in an EMPTY dedicated educational schema.','-- Synthetic teaching design; no health-data security or production provisioning is implied.']
inserts=['-- Deterministic synthetic fixtures. All timestamps explicitly include Nepal +05:45.','SET DEFINE OFF;']
for key,(table,fields) in schema.items():
    definitions=[]
    for col,typ in fields:
        # Only the ED hospital link is nullable; stay and bridge keys remain mandatory.
        can_null=col in nullable and (col!='hospital_stay_key' or key=='emergency')
        definitions.append(f'  {col} {typ}'+('' if can_null else ' NOT NULL'))
    definitions.append(f'  PRIMARY KEY ({primary[key]})')
    definitions += [f'  UNIQUE ({cols})' for cols in uniques.get(key,[])]
    definitions += [f'  FOREIGN KEY ({col}) REFERENCES {schema[ref][0]} ({primary[ref]})' for col,ref in foreign.get(key,{}).items()]
    definitions += [f'  CHECK ({condition})' for condition in checks.get(key,[])]
    ddl.append(f'CREATE TABLE {table} (\n'+',\n'.join(definitions)+'\n);\n')
    for row in DATA[key]:
        values=[]
        for col,typ in fields:
            val=row[col]
            if val is None: v='NULL'
            elif typ=='DATE': v=f"DATE '{val}'"
            elif typ.startswith('TIMESTAMP'): v="TO_TIMESTAMP_TZ('"+val+"', 'YYYY-MM-DD\"T\"HH24:MI:SSTZH:TZM')"
            elif isinstance(val,str):v="'"+val.replace("'","''")+"'"
            else:v=str(val)
            values.append(v)
        inserts.append(f"INSERT INTO {table} ({', '.join(c for c,_ in fields)})\nVALUES ({', '.join(values)});")
inserts.append('COMMIT;')
(SQL/'01_schema.sql').write_text('\n'.join(ddl)+'\n',encoding='utf-8')
(SQL/'02_seed.sql').write_text('\n'.join(inserts)+'\n',encoding='utf-8')
DATA['meta']['tables']={k:v[0] for k,v in schema.items()}
(ROOT/'data'/'sample.json').write_text(json.dumps(DATA,indent=2)+'\n',encoding='utf-8')
(ROOT/'data'/'sample.js').write_text('/* Deterministic synthetic fixture. Oracle is not executed by this file. */\nwindow.HDN_DATA = '+json.dumps(DATA,ensure_ascii=False,indent=2)+';\n',encoding='utf-8')
print(json.dumps(DATA['meta']['checks'],indent=2))
