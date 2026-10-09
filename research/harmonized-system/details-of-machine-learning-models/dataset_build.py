"""Rebuild the balanced chapter dataset; never modify supplied raw data.

Default: rebuild from the committed, source-derived 252-row input extract.
--raw: independently rescan BACI, WDI and GeoDist at TRADE_RAW_ROOT.
Only the two selected headings are scanned into memory. Hashes trace snapshots.
"""
from pathlib import Path
import os, sys, json, hashlib, argparse
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent
NAMES = {'CAN':'Canada','CHN':'China','DEU':'Germany','JPN':'Japan','MEX':'Mexico','GBR':'United Kingdom','USA':'United States'}
CODES = ['1001','8703']
INDICATORS = {'NY.GDP.MKTP.CD':'gdp','SP.POP.TOTL':'population','NV.IND.MANF.ZS':'manufacturing','IT.NET.USER.ZS':'internet'}
SEED = 338

def sha(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as f:
        for chunk in iter(lambda:f.read(1048576),b''):h.update(chunk)
    return h.hexdigest()

def extract(raw):
    """Absence is a BACI-derived grid zero, not evidence of a customs-reported zero."""
    bp=raw/'baci'
    countries=pd.read_csv(bp/'country_codes_V202601.csv',dtype=str,keep_default_na=False).drop_duplicates('country_iso3')
    lookup=countries.set_index('country_iso3').country_code.to_dict()
    ids={lookup[k] for k in NAMES}
    labels=json.loads((ROOT.parent/'1-harmonized-system-canada/data/hs-t2026-2.json').read_text(encoding='utf-8'))
    descriptions={h['code']:h['description'] for s in labels['sections'] for c in s['chapters'] for h in c['headings'] if h['code'] in CODES}
    products=pd.read_csv(bp/'product_codes_HS22_V202601.csv',dtype=str)
    assert set(CODES)<=set(products.code.str.zfill(6).str[:4])
    wp=next((raw/'wdi').glob('API_Download_DS2_EN*.csv'))
    wdi=pd.read_csv(wp,skiprows=4)
    geo=pd.read_excel(raw/'geographic/dist_cepii.xls').set_index(['iso_o','iso_d'])
    files=[bp/'country_codes_V202601.csv',bp/'product_codes_HS22_V202601.csv',bp/'Readme.txt',wp,raw/'geographic/dist_cepii.xls']
    rows=[]; scans=[]
    for year in [2022,2023,2024]:
        p=bp/f'BACI_HS22_Y{year}_V202601.csv';files.append(p)
        print('Read-only BACI scan',year,flush=True)
        pieces=[];count=0
        for chunk in pd.read_csv(p,dtype={'i':str,'j':str,'k':str},chunksize=500000):
            assert chunk.t.eq(year).all()
            count+=len(chunk);chunk['hs4']=chunk.k.str.zfill(6).str[:4]
            c=chunk[chunk.hs4.isin(CODES)].copy();c['usd']=c.v*1000
            assert np.isfinite(c.usd).all() and c.usd.ge(0).all()
            pieces.append(c[['i','j','hs4','usd']])
        trade=pd.concat(pieces,ignore_index=True)
        outside=trade[~(trade.i.isin(ids)&trade.j.isin(ids))]
        bilateral=trade.groupby(['i','j','hs4']).usd.sum()
        supply=outside.groupby(['i','hs4']).usd.sum();demand=outside.groupby(['j','hs4']).usd.sum();world=outside.groupby('hs4').usd.sum()
        macro=wdi[wdi['Indicator Code'].isin(INDICATORS)&wdi['Country Code'].isin(NAMES)].pivot(index='Country Code',columns='Indicator Code',values=str(year)).rename(columns=INDICATORS)
        for ex in NAMES:
            for dest in NAMES:
                if ex==dest:continue
                for hs in CODES:
                    key=(lookup[ex],lookup[dest],hs);g=geo.loc[(ex,dest)]
                    r={'year':year,'exporter':NAMES[ex],'destination':NAMES[dest],'exporter_iso3':ex,'destination_iso3':dest,
                       'hs4_code':hs,'hs4_description':descriptions[hs],
                       'observed_exports_usd':float(bilateral.get(key,0)),
                       'distance_km':float(g.distw),'common_language':int(g.comlang_off),'contiguity':int(g.contig),
                       'external_supply_usd':float(supply.get((lookup[ex],hs),0)),
                       'external_demand_usd':float(demand.get((lookup[dest],hs),0)),
                       'world_external_demand_usd':float(world[hs]),
                       'trade_status':'observed_baci_reconciled' if key in bilateral else 'baci_grid_zero',
                       'data_source':f'BACI HS2022 V202601 Y{year}; WDI 2026-07-13; CEPII GeoDist; CBSA T2026-2 label',
                       'macro_year':year,'data_status':'observed'}
                    missing=[]
                    for role,iso in [('exporter',ex),('destination',dest)]:
                        for col in INDICATORS.values():
                            r[role+'_'+col]=float(macro.loc[iso,col])
                            if pd.isna(r[role+'_'+col]):missing.append(role+'_'+col)
                    r['covariate_status']='missing '+','.join(missing) if missing else 'observed'
                    rows.append(r)
        scans.append({'year':year,'raw_rows_scanned':count,'selected_hs6_rows':len(trade)})
    d=pd.DataFrame(rows).sort_values(['year','exporter','destination','hs4_code']).reset_index(drop=True)
    (ROOT/'data').mkdir(exist_ok=True)
    d.to_csv(ROOT/'data/source_observed_subset.csv',index=False,float_format='%.17g')
    manifest={'sources':[{'file':p.name,'bytes':p.stat().st_size,'sha256':sha(p)} for p in files],
              'subset_sha256':sha(ROOT/'data/source_observed_subset.csv'),'scans':scans,
              'network':NAMES,'hs_revision':'HS2022','label_vintage':'CBSA T2026-2; heading-only descriptive wording',
              'units':'BACI v*1000 -> current USD; no currency mixing',
              'outside_network_rule':'NOT(exporter in network AND destination in network)',
              'available_baci_years':[2022,2023,2024]}
    (ROOT/'data/source_manifest.json').write_text(json.dumps(manifest,indent=2),encoding='utf-8')

def build():
    p=ROOT/'data/source_observed_subset.csv'
    manifest=json.loads((ROOT/'data/source_manifest.json').read_text())
    assert sha(p)==manifest['subset_sha256'],'Source extract changed; rebuild --raw or restore verified extract'
    d=pd.read_csv(p,dtype={'hs4_code':str}).sort_values(['year','exporter','destination','hs4_code']).reset_index(drop=True)
    # A source-file absence is an explicitly imputed grid zero, not a recorded shipment.
    d.loc[d.trade_status.eq('baci_grid_zero'),'data_status']='imputed'
    d['illustrative_exports_usd']=np.nan;d['scenario_procedure']='none'
    scenario=d[d.year==2024].copy();scenario['year']=2025
    # Predeclared artificial sector shock and independent fixed-seed shocks.
    # Not fitted to any research model, not a substitute for observed 2025 trade.
    rng=np.random.default_rng(SEED)
    shock=np.where(scenario.hs4_code.eq('1001'),-0.08,0.06)+rng.normal(0,0.10,len(scenario))
    scenario['illustrative_exports_usd']=scenario.observed_exports_usd*np.exp(shock)
    scenario['observed_exports_usd']=np.nan
    scenario['data_status']='illustrative';scenario['trade_status']='unobserved_2025_scenario'
    scenario['covariate_status']='2024 covariates carried as forecast-origin inputs; '+scenario.covariate_status
    scenario['data_source']='Illustrative scenario seed 338; historical inputs BACI/WDI/GeoDist 2024'
    scenario['scenario_procedure']='2024 trade * exp(sector shock + N(0,0.10)); wheat -0.08; cars +0.06; seed 338; preserve zeros'
    d=pd.concat([d,scenario],ignore_index=True).sort_values(['year','exporter','destination','hs4_code']).reset_index(drop=True)
    d.insert(0,'observation_id',np.arange(1,337))
    validate(d)
    d.to_csv(ROOT/'toy_trade_dataset.csv',index=False,float_format='%.17g')
    dictionary=[]
    for c in d:
        unit='USD' if c.endswith('_usd') or c.endswith('_gdp') else 'persons' if c.endswith('_population') else 'km' if c=='distance_km' else 'percent' if c.endswith(('_manufacturing','_internet')) else 'binary' if c in ['common_language','contiguity'] else ''
        meaning={'observed_exports_usd':'BACI reconciled HS6 sum; blank for 2025; absent valid source cell is grid zero',
                 'illustrative_exports_usd':'Artificial 2025 scenario only; never observed and never a model training target',
                 'macro_year':'Covariate year; 2025 record intentionally retains 2024 macro/product inputs',
                 'data_status':'Outcome status; inspect covariate_status separately',
                 'trade_status':'Distinguishes BACI positive reconciliation, absent-grid zero, and unobserved scenario',
                 'covariate_status':'Raw missing values retained; fit-specific medians are learned later',
                 'external_supply_usd':'Exporter-HS4 exports outside complete seven-country internal network',
                 'external_demand_usd':'Destination-HS4 imports outside complete seven-country internal network',
                 'world_external_demand_usd':'All selected-HS4 flows outside complete internal network',
                 'hs4_code':'Four-character HS2022 heading; codes are categories, not numeric magnitudes',
                 'hs4_description':'Official CBSA heading label, T2026-2 wording; data remain HS2022',
                 'observation_id':'1–336 ordered year, alphabetic exporter, alphabetic destination, HS4',
                 'common_language':'CEPII comlang_off, shared official language',
                 'contiguity':'CEPII contig, common land border',
                 'distance_km':'CEPII distw population-weighted national bilateral distance'}.get(c,c.replace('_',' '))
        dictionary.append({'column':c,'meaning':meaning,'unit':unit,'missing_policy':'blank means unavailable; never zero unless genuinely numeric zero'})
    pd.DataFrame(dictionary).to_csv(ROOT/'data_dictionary.csv',index=False)
    audit={'rows':len(d),'historical_numeric_outcomes':int(d.observed_exports_usd.notna().sum()),'observed_outcomes':int(d.data_status.eq('observed').sum()),'illustrative_outcomes':int(d.illustrative_exports_usd.notna().sum()),
           'observed_grid_zeros':int(d.observed_exports_usd.eq(0).sum()),'outcome_status_counts':d.data_status.value_counts().to_dict(),'missing_covariates':{c:int(d[c].isna().sum()) for c in d if c.endswith(('_gdp','_population','_manufacturing','_internet'))},
           'zeros_by_sector':d[d.year<2025].groupby('hs4_code').observed_exports_usd.apply(lambda x:int(x.eq(0).sum())).to_dict(),
           'dataset_sha256':sha(ROOT/'toy_trade_dataset.csv'),'checks':'336 unique IDs; 7 countries; 42 non-self pairs/year; 2 headings/pair/year; 84/year; years 2022–2025'}
    (ROOT/'data/dataset_audit.json').write_text(json.dumps(audit,indent=2),encoding='utf-8')
    print(json.dumps(audit,indent=2),flush=True)
    return d

def validate(d):
    assert len(d)==336 and d.observation_id.tolist()==list(range(1,337))
    assert set(d.year)=={2022,2023,2024,2025}
    assert set(d.exporter)==set(NAMES.values())==set(d.destination)
    assert set(d.hs4_code)==set(CODES)
    assert not d.duplicated(['year','exporter','destination','hs4_code']).any()
    assert not d.exporter.eq(d.destination).any()
    for _,g in d.groupby('year'):
        assert len(g)==84 and len(g[['exporter','destination']].drop_duplicates())==42
        assert g.groupby(['exporter','destination']).size().eq(2).all()
    assert d.loc[d.year==2025,'observed_exports_usd'].isna().all()
    assert d.loc[d.year<2025,'observed_exports_usd'].notna().all()
    return True

if __name__=='__main__':
    ap=argparse.ArgumentParser();ap.add_argument('--raw',action='store_true');args=ap.parse_args()
    if args.raw:extract(Path(os.environ.get('TRADE_RAW_ROOT',r'D:\Trade_Data_Scientist_Gov_Alberta\raw_data_machine_learning')))
    build()
