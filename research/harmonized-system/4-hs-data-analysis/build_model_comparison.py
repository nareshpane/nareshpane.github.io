"""Six-model trade comparison. Raw/reference data are read-only; no downloads.

Run: python research/harmonized-system/4-hs-data-analysis/build_model_comparison.py
Optional: --audit-only, --render-only, --fresh (ignore compact caches).
Use --presentation-only for editorial formatting of the existing page from saved
artifacts, preserving its layout/animation and writing no analytical assets.
Requires numpy, pandas, scipy, scikit-learn, xlrd, matplotlib, statsmodels.
The local reference's vendor directory is an optional READ-ONLY dependency fallback.
No module from the reference analysis itself is imported or executed.
"""
from pathlib import Path
import os
import sys
import json
import hashlib
import argparse
import warnings
import platform
from datetime import datetime, timezone
from html import escape
from format_model_comparison import format_presentation, key_blocks, render_saved_page

RAW_ROOT = Path(os.environ.get('TRADE_RAW_ROOT', r'D:\Trade_Data_Scientist_Gov_Alberta\raw_data_machine_learning'))
QUESTION1_REFERENCE = Path(os.environ.get('TRADE_QUESTION1_REFERENCE', r'D:\Trade_Data_Scientist_Gov_Alberta\question_1'))
TESTING_MODELS_REFERENCE = Path(os.environ.get('TRADE_TESTING_REFERENCE', r'D:\Trade_Data_Scientist_Gov_Alberta\testing_models'))
OUTPUT_DIR = Path(__file__).resolve().parent
PAGE_PATH = OUTPUT_DIR.parent / 'machine-learning-trade-sector-prediction.html'
CACHE = OUTPUT_DIR / 'cache'
sys.dont_write_bytecode = True
os.environ.setdefault('OMP_NUM_THREADS', '4')
os.environ.setdefault('OPENBLAS_NUM_THREADS', '4')
os.environ.setdefault('MKL_NUM_THREADS', '4')
os.environ['MPLCONFIGDIR'] = str(CACHE / 'matplotlib')
# This fallback adds plotting/inference packages, not datasets or model scripts.
vendor = TESTING_MODELS_REFERENCE / 'code/vendor'
if vendor.exists():
    sys.path.append(str(vendor))

import numpy as np
import pandas as pd
import scipy
import sklearn
import statsmodels.api as sm
from scipy.stats import spearmanr
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.pipeline import make_pipeline
from sklearn.linear_model import LinearRegression, ElasticNet
from sklearn.ensemble import RandomForestRegressor, HistGradientBoostingRegressor, HistGradientBoostingClassifier
from sklearn.neural_network import MLPRegressor
from sklearn.metrics import log_loss, brier_score_loss
from sklearn.exceptions import ConvergenceWarning
from threadpoolctl import threadpool_limits

NETWORK = ['CAN', 'CHN', 'GBR', 'JPN', 'KOR', 'MEX', 'USA']
INDICATORS = {'NY.GDP.MKTP.CD': 'gdp', 'SP.POP.TOTL': 'population',
              'NV.IND.MANF.ZS': 'manufacturing', 'IT.NET.USER.ZS': 'internet'}
NUMERIC = ['log_exporter_gdp', 'log_importer_gdp', 'log_exporter_population',
           'log_importer_population', 'exporter_manufacturing', 'importer_manufacturing',
           'exporter_internet', 'importer_internet', 'log_distance', 'common_border',
           'common_language', 'log_external_supply', 'log_external_demand', 'log_world_demand']
FEATURES = NUMERIC + ['HS2']
SEED = 338
CAD_2024 = 1.3698  # Bank of Canada 2024 annual CAD per USD; primary frozen reporting rate.
# The realized 2025 rate is deliberately ONLY introduced in phase B reproduction reporting.
MODEL_NAMES = {1: 'Dynamic linear regression', 2: 'Elastic Net regression',
               3: 'Random Forest regression', 4: 'Multilayer Perceptron',
               5: 'Single-stage gradient boosting', 6: 'Dynamic two-stage boosted partial adjustment'}
BOOST = dict(max_iter=150, max_leaf_nodes=7, learning_rate=0.05, random_state=SEED)
AUDIT = {}
DIAGNOSTICS = {}
FIT_RECORDS = []
VALIDATION = []

def stamp():
    return datetime.now(timezone.utc).isoformat()

def write_json(name, value):
    (OUTPUT_DIR / name).write_text(json.dumps(value, indent=2, ensure_ascii=False, default=str, allow_nan=False), encoding='utf-8')

def protected_manifest():
    """Metadata fingerprint includes every file in all three protected directories."""
    records = []
    for root in [RAW_ROOT, QUESTION1_REFERENCE, TESTING_MODELS_REFERENCE]:
        for p in sorted(root.rglob('*')):
            if p.is_file():
                s = p.stat()
                records.append((str(p), s.st_size, s.st_mtime_ns))
    return hashlib.sha256(repr(records).encode()).hexdigest()

def cached(name, dependencies, builder):
    """Local compact pickle only; source metadata invalidates it. Never trust remote pickles."""
    p = CACHE / (name + '.pkl')
    meta = CACHE / (name + '.json')
    signature = [(str(d), d.stat().st_size, d.stat().st_mtime_ns) for d in dependencies]
    signature = json.loads(json.dumps(signature))
    if not ARGS.fresh and p.exists() and meta.exists() and json.loads(meta.read_text()) == signature:
        print('Compact cache:', name, flush=True)
        return pd.read_pickle(p)
    result = builder()
    pd.to_pickle(result, p)
    meta.write_text(json.dumps(signature), encoding='utf-8')
    return result

# --------------------------------------------------
# Data audit: complete before any model fitting
# --------------------------------------------------
def audit_and_classification():
    global COUNTRIES, COUNTRY_IDS, PRODUCTS, HS4, WDI, GEO, COMMON_GRID
    bp = RAW_ROOT / 'baci'
    COUNTRIES = pd.read_csv(bp / 'country_codes_V202601.csv', dtype=str, keep_default_na=False)
    COUNTRY_IDS=set(COUNTRIES.country_code)
    historical_duplicates=COUNTRIES[COUNTRIES.country_iso3.duplicated(keep=False)].to_dict('records')
    # Source lookup includes obsolete Belgium-Luxembourg/Germany/Sudan records.
    # Reproduce the PDF's current-first ISO3 rule; preserve ALL numeric IDs for auditing.
    COUNTRIES.loc[COUNTRIES.country_iso3=='NAM','country_iso2']='NA'
    COUNTRIES=COUNTRIES.drop_duplicates('country_iso3',keep='first')
    PRODUCTS = pd.read_csv(bp / 'product_codes_HS22_V202601.csv', dtype=str, keep_default_na=False)
    PRODUCTS['HS6'] = PRODUCTS.code.str.zfill(6)
    assert PRODUCTS.HS6.str.fullmatch(r'\d{6}').all()
    assert not PRODUCTS.HS6.duplicated().any()
    HS4 = sorted(PRODUCTS.HS6.str[:4].unique())
    wp = next((RAW_ROOT / 'wdi').glob('API_Download_DS2_EN*.csv'))
    WDI = pd.read_csv(wp, skiprows=4)
    assert set(INDICATORS).issubset(set(WDI['Indicator Code']))
    GEO = pd.read_excel(RAW_ROOT / 'geographic/dist_cepii.xls', sheet_name='dist_cepii')
    for c in ['distw','contig','comlang_off']:
        GEO[c]=pd.to_numeric(GEO[c],errors='coerce')
    assert not GEO.duplicated(['iso_o', 'iso_d']).any()
    net = COUNTRIES[COUNTRIES.country_iso3.isin(NETWORK)]
    assert len(net) == 7
    pairs = net[['country_code', 'country_iso3']].rename(columns={'country_code':'i','country_iso3':'exporter_iso3'}).merge(
        net[['country_code', 'country_iso3']].rename(columns={'country_code':'j','country_iso3':'importer_iso3'}), how='cross')
    pairs = pairs[pairs.i != pairs.j]
    assert len(pairs) == 42
    COMMON_GRID = pairs.merge(pd.DataFrame({'HS4': HS4}), how='cross').reset_index(drop=True)
    AUDIT.update({'baci_revision': 'HS2022', 'baci_vintage': 'V202601',
                  'baci_units': 'thousand current USD (local Readme.txt); multiply v by 1000',
                  'baci_files': [], 'hs6_count': len(PRODUCTS), 'hs4_count': len(HS4),
                  'directed_pairs': len(pairs), 'annual_grid_cells': len(COMMON_GRID),
                  'wdi_years': [c for c in WDI if c.isdigit()], 'wdi_updated': '2026-07-13',
                  'wdi_indicators': INDICATORS, 'wdi_missing': {}, 'geodist_columns': GEO.columns.tolist(),
                  'legal_files': {}, 'statcan_metadata': {}, 'historical_country_duplicates':historical_duplicates,
                  'country_lookup_corrections':'Namibia ISO2 missing in supplied BACI lookup: NAM -> NA. Historic duplicated ISO3 use current-first record as reference code.'})
    for year in [2022, 2023, 2024]:
        p = bp / f'BACI_HS22_Y{year}_V202601.csv'
        assert p.exists()
        AUDIT['baci_files'].append({'file': p.name, 'bytes': p.stat().st_size,
                                   'columns': pd.read_csv(p, nrows=0).columns.tolist()})
        data = annual_wdi(year).set_index('iso3')
        AUDIT['wdi_missing'][str(year)] = {
            'global_missing': {k: int(data[k].isna().sum()) for k in INDICATORS.values()},
            'network_missing': {k: data.loc[NETWORK].index[data.loc[NETWORK,k].isna()].tolist() for k in INDICATORS.values()}}
    for p in sorted((RAW_ROOT / 'legal').glob('*.txt')):
        text = p.read_text(encoding='utf-8-sig', errors='replace')
        AUDIT['legal_files'][p.name] = {'bytes':p.stat().st_size, 'sample':text[:220]}
    for year in [2023, 2024, 2025]:
        folder = RAW_ROOT / f'CIMT-CICM_Dom_Exp_{year}'
        p = folder / f'ODPFN018_{year}12N.csv'
        # Header-only access to 2025 here; absolutely no 2025 observation loading.
        AUDIT['statcan_metadata'][str(year)] = {
            'HS6_file': p.name, 'columns':pd.read_csv(p,nrows=0).columns.tolist(),
            'other_levels':[q.name for q in folder.glob('ODPFN*.csv')],
            'lookup_files':[q.name for q in folder.glob('*.TXT')], 'unit':'current CAD dollars'}
    AUDIT['dynamic_linear_feasibility'] = (
        'An imputation-qualified dynamic specification is estimable: exact 2023 GDP, population and internet exist for all seven; '
        'manufacturing is missing for CAN and USA (2/7), and is median-imputed within 2023 training folds. '
        'The fully observed preferred specification is unavailable. No future-year values are substituted.')
    AUDIT['vintage_limit'] = 'BACI released 2026-01-22 and WDI updated 2026-07-13: retrospective year holdout, not a real-time vintage backtest.'
    write_audit()

def annual_wdi(year):
    s = WDI[WDI['Indicator Code'].isin(INDICATORS)].copy()
    s[str(year)] = pd.to_numeric(s[str(year)], errors='coerce')
    return s.pivot(index='Country Code',columns='Indicator Code',values=str(year)).rename(columns=INDICATORS).reset_index().rename(columns={'Country Code':'iso3'})

def write_audit():
    write_json('data_audit.json', AUDIT)
    text = ['# Local-data audit', '', 'Raw data and both reference directories are read-only. No dataset downloads or ZIP extraction.',
            '2025 outcome access: metadata/header only in Phase A; Alberta observations first loaded after freeze in Phase B.', '',
            '## BACI', f"HS2022 V202601; {AUDIT.get('hs6_count')} HS6 codes; {AUDIT.get('hs4_count')} HS4 headings; 42 directed pairs; {AUDIT.get('annual_grid_cells')} cells/year.",
            AUDIT.get('baci_units',''), 'Files/years and full-file scans are recorded in data_audit.json.', '', '## WDI',
            'Exact columns: 2022, 2023, 2024. GDP NY.GDP.MKTP.CD (current USD); population SP.POP.TOTL (persons); manufacturing NV.IND.MANF.ZS (% GDP); internet IT.NET.USER.ZS (% population).',
            'Missing network values: ' + json.dumps(AUDIT.get('wdi_missing',{})), AUDIT.get('dynamic_linear_feasibility',''), '',
            '## GeoDist and legal', 'GeoDist: ' + ', '.join(AUDIT.get('geodist_columns',[])),
            'Use population-weighted distw in km, contig and comlang_off. Time invariant national geography; not Alberta logistics.',
            'Legal contains eight text captures of tariff annexes, CBP guidance and Yale Section 338 product lists, with 2026 material. No country-pair legal-origin table. Excluded from all primary models; not pre-2025 covariates.', '',
            '## Statistics Canada and reconciliation',
            'Extracted 2023, 2024 and 2025 domestic-export folders contain HS8 (ODPFN016), HS6 (ODPFN018), HS2 (ODPFN020) and dated description/country lookups. Use HS6 source, filter province AB, sum twelve months and all U.S. states, preserve leading zeros, HS4=HS6[:4].',
            'HS4 can be constructed for all three years. No ZIPs are needed. 2025 data scans occur only after predictions freeze.',
            'Country mapping uses BACI numeric IDs -> ISO3 for WDI/GeoDist, ISO2 for StatCan; keep_default_na=False preserves Namibia NA. Unmapped and unscorable flows are explicitly recorded below.',
            'StatCan national/special headings are outside internationally harmonized HS2022; retained as disclosed persistence fallback from 2024, not fitted international products.',
            'HS6 membership, missing-country/year issues, monthly coverage, duplicates and classification differences are in the machine-readable audit.', '',
            'BACI represents petroleum heading 2710 with metadata code 271000; StatCan uses children 271012/271019/271099. These disagree at HS6 but reconcile at HS4=2710. Do not claim exact HS6 harmonization.',
            'StatCan numeric lookup country codes are a separate namespace from BACI numeric country IDs; join the datasets by ISO2/ISO3, not those numeric fields.', '',
            '## Currency and information date', 'BACI v*1000 = USD; WDI GDP current USD; StatCan value current CAD. Dynamic lags divide CAD by the source-year annual rate.',
            'Primary CAD reporting freezes 1.3698 CAD/USD (2024 annual rate). The PDF reproduction applies realized 2025 1.3978 only in Phase B as an ex-post currency diagnostic, never fitting or selection.',
            AUDIT.get('vintage_limit',''), '', '## Derived checks', '```json', json.dumps({k:v for k,v in AUDIT.items() if k in ['annual_scans','alberta_scans','eligibility','fallback','2025_evaluation']},indent=2), '```']
    (OUTPUT_DIR/'data_audit.md').write_text('\n'.join(text),encoding='utf-8')

# --------------------------------------------------
# Common country harmonization, HS4 aggregation and features
# --------------------------------------------------
def prepare_year(year):
    p = RAW_ROOT / f'baci/BACI_HS22_Y{year}_V202601.csv'
    dependencies = [p, RAW_ROOT/'baci/product_codes_HS22_V202601.csv', RAW_ROOT/'baci/country_codes_V202601.csv',
                    next((RAW_ROOT/'wdi').glob('API_Download_DS2_EN*.csv')), RAW_ROOT/'geographic/dist_cepii.xls']
    def build():
        parts = {k:[] for k in ['trade','supply','demand','world']}
        ids = set(COUNTRIES.loc[COUNTRIES.country_iso3.isin(NETWORK),'country_code'])
        seen_ids, seen_hs6, years = set(), set(), set()
        rows = missing_q = 0
        for c in pd.read_csv(p,dtype={'i':str,'j':str,'k':str},chunksize=500000):
            assert set(['t','i','j','k','v','q']).issubset(c.columns)
            years.update(c.t.unique().tolist())
            assert (c.t == year).all()
            c['HS6'] = c.k.str.zfill(6)
            assert c.HS6.str.fullmatch(r'\d{6}').all()
            seen_hs6.update(c.HS6.unique())
            seen_ids.update(c.i.unique());seen_ids.update(c.j.unique())
            c['HS4'] = c.HS6.str[:4]
            c['V'] = pd.to_numeric(c.v,errors='raise') * 1000
            assert np.isfinite(c.V).all() and c.V.ge(0).all()
            inside = c.i.isin(ids) & c.j.isin(ids)
            outside = c[~inside]
            parts['trade'].append(c[inside].groupby(['i','j','HS4'],as_index=False).V.sum())
            parts['supply'].append(outside[outside.i.isin(ids)].groupby(['i','HS4'],as_index=False).V.sum())
            parts['demand'].append(outside.groupby(['j','HS4'],as_index=False).V.sum())
            parts['world'].append(outside.groupby('HS4',as_index=False).V.sum())
            rows += len(c); missing_q += int(c.q.isna().sum())
        assert not (seen_hs6-set(PRODUCTS.HS6))
        assert not (seen_ids-COUNTRY_IDS)
        def combine(kind, keys, name):
            return pd.concat(parts[kind]).groupby(keys,as_index=False).V.sum().rename(columns={'V':name})
        trade=combine('trade',['i','j','HS4'],'V')
        supply=combine('supply',['i','HS4'],'supply')
        demand=combine('demand',['j','HS4'],'demand')
        world=combine('world',['HS4'],'world')
        d=COMMON_GRID.merge(trade,on=['i','j','HS4'],how='left',validate='one_to_one')
        d['V']=d.V.fillna(0)
        d=add_features(d,year,supply,demand,world)
        assert d.distance.gt(0).all()
        info={'rows':rows,'years':sorted(years),'missing_quantity':missing_q,'unknown_country_ids':[],
              'unknown_hs6':[],'active_hs6':len(seen_hs6),'active_hs4':len({s[:4] for s in seen_hs6}),
              'grid_cells':len(d),'positive_cells':int(d.V.gt(0).sum()),'zero_cells':int(d.V.eq(0).sum()),
              'network_feature_missing':{c:int(d[c].isna().sum()) for c in NUMERIC}}
        return (d,supply,demand,world,info)
    result=cached('annual_'+str(year),dependencies,build)
    AUDIT.setdefault('annual_scans',{})[str(year)]=result[4]
    return result[:4]

def add_features(d,year,supply,demand,world):
    w=annual_wdi(year)
    for role in ['exporter','importer']:
        d=d.merge(w.rename(columns={'iso3':role+'_iso3',**{v:role+'_'+v for v in INDICATORS.values()}}),on=role+'_iso3',how='left',validate='many_to_one')
    g=GEO[['iso_o','iso_d','distw','contig','comlang_off']].rename(columns={'iso_o':'exporter_iso3','iso_d':'importer_iso3','distw':'distance','contig':'common_border','comlang_off':'common_language'})
    d=d.merge(g,on=['exporter_iso3','importer_iso3'],how='left',validate='many_to_one')
    d=d.merge(supply,on=['i','HS4'],how='left',validate='many_to_one').merge(demand,on=['j','HS4'],how='left',validate='many_to_one').merge(world,on='HS4',how='left',validate='many_to_one')
    for col in ['supply','demand','world']:
        d[col]=d[col].fillna(0)
        d['log_'+{'supply':'external_supply','demand':'external_demand','world':'world_demand'}[col]]=np.log1p(d[col])
    for col in ['exporter_gdp','importer_gdp','exporter_population','importer_population','distance']:
        d['log_'+col]=np.log(d[col].where(d[col]>0))
    d['HS2']=d.HS4.str[:2]
    return d

def load_alberta(year,phase):
    assert year != 2025 or phase == 'B', '2025 observations forbidden before freeze'
    p=RAW_ROOT/f'CIMT-CICM_Dom_Exp_{year}/ODPFN018_{year}12N.csv'
    def build():
        parts=[];months=set();rows=0; codes=set();keys=set()
        for c in pd.read_csv(p,dtype=str,keep_default_na=False,chunksize=400000):
            period=c.columns[0]
            assert c[period].str.startswith(str(year)).all()
            a=c[c.Province=='AB'].copy()
            months.update(a[period].unique())
            a['HS6']=a.HS6.str.zfill(6)
            assert a.HS6.str.fullmatch(r'\d{6}').all()
            codes.update(a.HS6.unique())
            # Full monthly key includes U.S. state and unit. Detect within/between chunks.
            keycols=[x for x in c.columns if x not in ['Value/Valeur','Quantity/Quantité']]
            assert not a.duplicated(keycols).any()
            hashed=set(pd.util.hash_pandas_object(a[keycols],index=False).tolist())
            assert not keys.intersection(hashed)
            keys.update(hashed)
            a['HS4']=a.HS6.str[:4]
            a['actual_cad']=pd.to_numeric(a['Value/Valeur'],errors='raise').astype('int64')
            assert a.actual_cad.ge(0).all()
            parts.append(a.groupby(['Country/Pays','HS4'],as_index=False).actual_cad.sum())
            rows+=len(a)
        assert len(months)==12
        result=pd.concat(parts).groupby(['Country/Pays','HS4'],as_index=False).actual_cad.sum().rename(columns={'Country/Pays':'iso2'})
        # Independent HS2 provincial source checks monthly/state aggregation.
        q=RAW_ROOT/f'CIMT-CICM_Dom_Exp_{year}/ODPFN020_{year}12N.csv'
        totals=pd.read_csv(q,dtype=str,keep_default_na=False)
        totals=totals[totals.Province=='AB'].copy()
        totals['value']=pd.to_numeric(totals['Value/Valeur'],errors='raise').astype('int64')
        s=totals.groupby('Country/Pays').value.sum()
        t=result.groupby('iso2').actual_cad.sum()
        assert s.sort_index().equals(t.reindex(s.index,fill_value=0).sort_index())
        info={'months':sorted(months),'alberta_rows':rows,'total_cad':int(result.actual_cad.sum()),
              'duplicates':0,'negative_values':0,'hs2_destination_reconciliation':'exact',
              'non_baci_hs6':sorted(codes-set(PRODUCTS.HS6)),
              'non_baci_hs4':sorted(set(result.HS4)-set(HS4)),
              'unmapped_iso2':sorted(set(result.iso2)-set(COUNTRIES.country_iso2))}
        return result,info
    result,info=cached('alberta_'+str(year),[p,RAW_ROOT/f'CIMT-CICM_Dom_Exp_{year}/ODPFN020_{year}12N.csv'],build)
    AUDIT.setdefault('alberta_scans',{})[str(year)]=info
    return result

def scoring_universe(annual2024,lag):
    _,s,d,w=annual2024
    cand=COUNTRIES[COUNTRIES.country_iso3!='CAN'].copy()
    cand=cand.merge(annual_wdi(2024),left_on='country_iso3',right_on='iso3',how='left')
    cand=cand.merge(GEO[GEO.iso_o=='CAN'][['iso_d','distw']],left_on='country_iso3',right_on='iso_d',how='left')
    good=(cand.country_iso2!='') & (cand.gdp>0) & (cand.population>0) & (cand.distw>0)
    AUDIT['eligibility']={'eligible':int(good.sum()),'excluded':[{'iso3':r.country_iso3,'iso2':r.country_iso2,
       'missing_gdp':not(r.gdp>0),'missing_population':not(r.population>0),'missing_distance':not(r.distw>0)} for r in cand[~good].itertuples()]}
    cand=cand[good]
    assert not cand.country_iso2.duplicated().any()
    keys=cand[['country_code','country_iso3','country_iso2']].rename(columns={'country_code':'j','country_iso3':'importer_iso3','country_iso2':'iso2'}).merge(pd.DataFrame({'HS4':HS4}),how='cross')
    keys['exporter_iso3']='CAN'
    keys['i']=COUNTRIES.loc[COUNTRIES.country_iso3=='CAN','country_code'].iloc[0]
    score=add_features(keys,2024,s,d,w).merge(lag,on=['iso2','HS4'],how='left',validate='one_to_one')
    score['lag_usd']=score.actual_cad.fillna(0)/CAD_2024
    score['lag_log']=np.log1p(score.lag_usd)
    outside=lag.merge(keys[['iso2','HS4']],on=['iso2','HS4'],how='left',indicator=True)
    outside=outside[outside._merge=='left_only'][['iso2','HS4','actual_cad']]
    outside['predicted_usd']=outside.actual_cad/CAD_2024
    AUDIT['fallback']={'cells':len(outside),'lag_cad':int(outside.actual_cad.sum()),
       'destinations':sorted(outside.iso2.unique()),'hs4':sorted(outside.HS4.unique()),
       'rule':'Identical constant-USD 2024 persistence for all six models, outside structural eligibility only.'}
    return score,outside

# --------------------------------------------------
# Shared preprocessing, retransformation and evaluation
# --------------------------------------------------
def preparation(dynamic=False):
    numeric=NUMERIC+(['lag_log'] if dynamic else [])
    return ColumnTransformer([
        ('numeric',make_pipeline(SimpleImputer(strategy='median'),StandardScaler()),numeric),
        ('category',OneHotEncoder(handle_unknown='ignore',sparse_output=False,drop='first' if dynamic else None),['HS2'])])

def estimator(model,params=None):
    params=params or {}
    if model==1:return LinearRegression()
    if model==2:return ElasticNet(alpha=params.get('alpha',0.03),l1_ratio=params.get('l1_ratio',0.5),max_iter=5000,tol=1e-4,random_state=SEED)
    if model==3:return RandomForestRegressor(n_estimators=120,max_depth=16,min_samples_leaf=8,max_features=0.7,bootstrap=True,n_jobs=4,random_state=SEED)
    if model==4:return MLPRegressor(hidden_layer_sizes=(32,16),activation='tanh',solver='adam',alpha=0.1,batch_size=512,
               learning_rate_init=0.001,max_iter=180,early_stopping=True,validation_fraction=0.15,n_iter_no_change=15,random_state=SEED)
    if model==5:return HistGradientBoostingRegressor(**BOOST)
    raise ValueError(model)

def fit_single(model,train,params=None,context='final'):
    dynamic=model==1
    feat=FEATURES+(['lag_log'] if dynamic else [])
    pipe=make_pipeline(preparation(dynamic),estimator(model,params))
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter('always',ConvergenceWarning)
        pipe.fit(train[feat],np.log1p(train.V))
    raw=np.expm1(pipe.predict(train[feat])).clip(min=0)
    calibration=float(train.V.sum()/raw.sum())
    e=pipe[-1]
    FIT_RECORDS.append({'model':model,'context':context,'rows':len(train),'calibration':calibration,
          'iterations':int(e.n_iter_) if hasattr(e,'n_iter_') else None,
          'warnings':[str(w.message) for w in caught]})
    return pipe,calibration

def predict_single(model,pipe,c,d):
    feat=FEATURES+(['lag_log'] if model==1 else [])
    return c*np.expm1(pipe.predict(d[feat])).clip(min=0)

def metrics(actual,pred,groups,top_n=10):
    a=np.asarray(actual,dtype=float);p=np.asarray(pred,dtype=float)
    assert np.isfinite(a).all() and np.isfinite(p).all() and (p>=0).all()
    t=pd.DataFrame({'actual':a,'pred':p,'group':np.asarray(groups)}).groupby('group')[['actual','pred']].sum()
    n=min(top_n,len(t))
    corr=float(spearmanr(t.actual,t.pred).statistic)
    return {'cell_mae':float(np.mean(np.abs(p-a))), 'cell_wape':float(np.abs(p-a).sum()/a.sum()),
            'rmsle':float(np.sqrt(np.mean((np.log1p(p)-np.log1p(a))**2))),
            'destination_wape':float((t.pred-t.actual).abs().sum()/t.actual.sum()),
            'rank_correlation':corr if np.isfinite(corr) else None,
            'top_overlap':len(set(t.actual.nlargest(n).index)&set(t.pred.nlargest(n).index)), 'top_n':n}

def record_validation(model,design,exporter,d,pred):
    m=metrics(d.V,pred,d.j,top_n=3)
    VALIDATION.append({'model':model,'design':design,'held_out_exporter':exporter,**m})

# --------------------------------------------------
# Model 6: PDF-defined stages and independently estimated lambda
# --------------------------------------------------
def fit_two_stage(train,context):
    cls=make_pipeline(preparation(),HistGradientBoostingClassifier(**BOOST))
    cls.fit(train[FEATURES],train.V.gt(0).astype(int))
    pos=train[train.V>0]
    reg=make_pipeline(preparation(),HistGradientBoostingRegressor(**BOOST))
    reg.fit(pos[FEATURES],np.log(pos.V))
    c=float(pos.V.sum()/np.exp(reg.predict(pos[FEATURES])).sum())
    FIT_RECORDS.append({'model':6,'context':context,'rows':len(train),'positive_rows':len(pos),
          'calibration':c,'classifier_iterations':int(cls[-1].n_iter_),'regressor_iterations':int(reg[-1].n_iter_)})
    return cls,reg,c

def predict_two_stage(fit,d):
    cls,reg,c=fit
    pi=cls.predict_proba(d[FEATURES])[:,1]
    return pi*c*np.exp(reg.predict(d[FEATURES]))

def oof_two_stage(d,year):
    pred=np.full(len(d),np.nan);probs=np.full(len(d),np.nan)
    for exporter in d.exporter_iso3.unique():
        print('Model 6 OOF',year,exporter,flush=True)
        tr=d[d.exporter_iso3!=exporter];te=d[d.exporter_iso3==exporter]
        assert exporter not in set(tr.exporter_iso3)
        f=fit_two_stage(tr,f'{year} holdout {exporter}')
        pred[te.index]=predict_two_stage(f,te)
        probs[te.index]=f[0].predict_proba(te[FEATURES])[:,1]
    assert np.isfinite(pred).all()
    DIAGNOSTICS[f'model6_{year}_classification']={'exporter_oof_log_loss':float(log_loss(d.V.gt(0),probs)),
           'exporter_oof_brier':float(brier_score_loss(d.V.gt(0),probs))}
    return pred

def dynamic(gap_lag,potential,lam):
    y=np.log1p(np.asarray(gap_lag))
    return np.expm1(y+lam*(np.log1p(np.asarray(potential))-y)).clip(min=0)

def run_model6(first,second):
    d22=first[0];d23=second[0]
    p22=oof_two_stage(d22,2022)
    joined=d22[['i','j','HS4','V']].merge(d23[['i','j','HS4','V']],on=['i','j','HS4'],suffixes=('_22','_23'),validate='one_to_one')
    gap=np.log1p(p22)-np.log1p(joined.V_22)
    change=np.log1p(joined.V_23)-np.log1p(joined.V_22)
    fit=sm.OLS(change,np.asarray(gap).reshape(-1,1)).fit(cov_type='HC3')
    raw=float(fit.params.iloc[0]);lam=float(np.clip(raw,0,1))
    DIAGNOSTICS['model6_lambda']={'raw':raw,'used':lam,'hc3_se':float(fit.bse.iloc[0]),
          'ci':fit.conf_int().iloc[0].tolist(),'restriction_binds':lam!=raw,'uncentered_r2':float(fit.rsquared),
          'gap_change_correlation':float(np.corrcoef(gap,change)[0,1]),'freeze_utc':stamp()}
    print('Lambda independently estimated:',raw,flush=True)
    p23=oof_two_stage(d23,2023)
    pred24=dynamic(d23.V,p23,lam)
    # Freeze next-year prediction BEFORE loading the 2024 trade outcomes.
    pd.DataFrame({'i':d23.i,'j':d23.j,'HS4':d23.HS4,'predicted_2024_usd':pred24}).to_csv(OUTPUT_DIR/'temporal_2024_frozen.csv.gz',index=False,compression={'method':'gzip','mtime':0})
    return lam,pred24

# --------------------------------------------------
# Models 1-5: exporter transfer validation and final fits
# --------------------------------------------------
def tune_elastic(train,context):
    # Small, predeclared grid; every preprocessing fit excludes the held-out exporter.
    rows=[]
    for alpha in [0.01,0.1]:
        for ratio in [0.2,0.8]:
            loss=[]
            for exporter in sorted(train.exporter_iso3.unique()):
                tr=train[train.exporter_iso3!=exporter];te=train[train.exporter_iso3==exporter]
                pipe=make_pipeline(preparation(),estimator(2,{'alpha':alpha,'l1_ratio':ratio}))
                pipe.fit(tr[FEATURES],np.log1p(tr.V))
                loss.append(float(np.mean((pipe.predict(te[FEATURES])-np.log1p(te.V))**2)))
            rows.append({'alpha':alpha,'l1_ratio':ratio,'mean_exporter_log_mse':float(np.mean(loss))})
    best=min(rows,key=lambda x:x['mean_exporter_log_mse'])
    DIAGNOSTICS.setdefault('model2_tuning',{})[context]={'candidates':rows,'chosen':best}
    return best

def run_single(model,train,score):
    oof=np.full(len(train),np.nan)
    for exporter in train.exporter_iso3.unique():
        print('Model',model,'holdout',exporter,flush=True)
        tr=train[train.exporter_iso3!=exporter];te=train[train.exporter_iso3==exporter]
        params=tune_elastic(tr,'outer_'+exporter) if model==2 else None
        pipe,c=fit_single(model,tr,params,context='holdout '+exporter)
        p=predict_single(model,pipe,c,te)
        oof[te.index]=p
        record_validation(model,'2023→2024 exporter transfer' if model==1 else '2024 exporter transfer (nested tuning for EN)',exporter,te,p)
    params=tune_elastic(train,'final') if model==2 else None
    pipe,c=fit_single(model,train,params)
    p=predict_single(model,pipe,c,score)
    DIAGNOSTICS[f'model{model}']={'calibration':c,'in_sample_log_rmse':float(np.sqrt(np.mean((pipe.predict(train[FEATURES+(['lag_log'] if model==1 else [])])-np.log1p(train.V))**2))),
          'aggregate_oof':metrics(train.V,oof,train.exporter_iso3+'_'+train.j,top_n=10)}
    if model==1:
        x=pipe[0].transform(train[FEATURES+['lag_log']]);x=sm.add_constant(x,has_constant='add')
        fit=sm.OLS(np.log1p(train.V),x).fit(cov_type='HC3')
        cluster=sm.OLS(np.log1p(train.V),x).fit(cov_type='cluster',cov_kwds={'groups':train.i+'_'+train.j})
        names=['intercept']+pipe[0].get_feature_names_out().tolist()
        coefficients=pd.DataFrame({'term':names,'coefficient':fit.params,'hc3_se':fit.bse,'dyad_cluster_se':cluster.bse})
        coefficients.to_csv(OUTPUT_DIR/'model_01_coefficients.csv',index=False)
        ix=names.index('numeric__lag_log')
        lag_sd=float(pipe[0].named_transformers_['numeric'][-1].scale_[-1])
        rho=float(fit.params.iloc[ix]/lag_sd)
        residual=np.asarray(fit.resid)
        bins=pd.qcut(fit.fittedvalues,5,duplicates='drop')
        residual_bins=pd.DataFrame({'bin':bins.astype(str),'resid2':residual**2}).groupby('bin').resid2.mean().to_dict()
        singular=np.linalg.svd(x,compute_uv=False)
        DIAGNOSTICS['model1'].update({'rho':rho,'rho_hc3_se':float(fit.bse.iloc[ix]/lag_sd),
             'rho_dyad_cluster_se':float(cluster.bse.iloc[ix]/lag_sd),'matrix_rank':int(np.linalg.matrix_rank(x)),
             'columns':x.shape[1],'condition_number':float(singular[0]/singular[-1]),'residual_variance_bins':residual_bins,
             'gdp_population_corr':float(train.log_exporter_gdp.corr(train.log_exporter_population)),
             'lag_only_oof':None})
        # A lag-only comparison identifies persistence dominance without using 2025.
        lagpred=np.empty(len(train))
        for e in train.exporter_iso3.unique():
            tr=train[train.exporter_iso3!=e];te=train[train.exporter_iso3==e]
            b=LinearRegression().fit(tr[['lag_log']],np.log1p(tr.V))
            raw=np.expm1(b.predict(tr[['lag_log']])).clip(min=0)
            lagpred[te.index]=np.expm1(b.predict(te[['lag_log']])).clip(min=0)*tr.V.sum()/raw.sum()
        DIAGNOSTICS['model1']['lag_only_oof']=metrics(train.V,lagpred,train.exporter_iso3+'_'+train.j)
    if model==2:
        DIAGNOSTICS['model2'].update({'chosen':params,'nonzero_coefficients':int(np.count_nonzero(pipe[-1].coef_)),'coefficient_count':len(pipe[-1].coef_)})
    if model==3:
        # Held-out CAN permutation importance, log target. No Alberta/2025 importance tuning.
        from sklearn.inspection import permutation_importance
        tr=train[train.exporter_iso3!='CAN'];te=train[train.exporter_iso3=='CAN']
        f=make_pipeline(preparation(),estimator(3)).fit(tr[FEATURES],np.log1p(tr.V))
        imp=permutation_importance(f,te[FEATURES],np.log1p(te.V),n_repeats=3,random_state=SEED,scoring='neg_mean_squared_error')
        DIAGNOSTICS['model3']['permutation_importance']={k:float(v) for k,v in zip(FEATURES,imp.importances_mean)}
    if model==4:
        DIAGNOSTICS['model4'].update({'iterations':int(pipe[-1].n_iter_),'best_internal_validation_r2':float(pipe[-1].best_validation_score_),
          'optimizer_loss':float(pipe[-1].loss_),'parameters':int(sum(a.size for a in pipe[-1].coefs_)+sum(a.size for a in pipe[-1].intercepts_)),
          'scoring_log_prediction_min':float(pipe.predict(score[FEATURES]).min()),
          'scoring_log_prediction_max':float(pipe.predict(score[FEATURES]).max()),
          'training_log_target_max':float(np.log1p(train.V).max()),
          'stability_repair':'Initial ReLU 48/24 network extrapolated many orders beyond training trade on pre-2025 small-country features. Fixed tanh 32/16, L2 alpha 0.1; no change of estimator family or common inputs; no 2025 outcome criterion.'})
    if model==5:DIAGNOSTICS['model5']['iterations']=int(pipe[-1].n_iter_)
    return p

def mlp_stability_diagnostic(train,score):
    """Reproduce the initial failure using ONLY 2024 training and scoring inputs.

    This is a functional-form diagnostic, not a 2025-error parameter search.
    The primary MLP's fixed bounded architecture is declared in estimator().
    """
    old=MLPRegressor(hidden_layer_sizes=(48,24),activation='relu',solver='adam',alpha=0.01,
          batch_size=512,learning_rate_init=0.001,max_iter=180,early_stopping=True,
          validation_fraction=0.15,n_iter_no_change=15,random_state=SEED)
    pipe=make_pipeline(preparation(),old)
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter('always',ConvergenceWarning)
        pipe.fit(train[FEATURES],np.log1p(train.V))
    logpred=pipe.predict(score[FEATURES])
    trmax=float(np.log1p(train.V).max())
    worst=score.iloc[int(np.argmax(logpred))]
    record={'information_year':2024,'no_2025_outcomes_used':True,'initial_architecture':'48/24 ReLU, L2=0.01',
        'training_log_target_max':trmax,'scoring_log_prediction_max':float(logpred.max()),
        'orders_of_magnitude_above_training_max':float((logpred.max()-trmax)/np.log(10)),
        'worst_cell':{'iso2':worst.iso2,'HS4':worst.HS4,'importer_gdp':float(worst.importer_gdp),
                     'importer_population':float(worst.importer_population),'importer_internet':float(worst.importer_internet)},
        'training_min_importer_gdp':float(train.importer_gdp.min()),
        'training_min_importer_internet':float(train.importer_internet.min()),
        'warnings':[str(w.message) for w in caught],
        'repair':'Fixed 32/16 tanh network, L2=0.1; same shared features, preprocessing, optimization budget and seed.'}
    write_json('mlp_stability_diagnostic.json',record)
    print('Initial MLP scoring log max / training target max:',record['scoring_log_prediction_max'],trmax,flush=True)

def phase_a():
    first=prepare_year(2022);second=prepare_year(2023)
    # Data audit is written before any fitting, including full 2022/2023 scans.
    write_audit()
    lam,temporal24=run_model6(first,second)
    fourth=prepare_year(2024)
    d24=fourth[0]
    assert np.array_equal(d24[['i','j','HS4']].values,second[0][['i','j','HS4']].values)
    for e in d24.exporter_iso3.unique():
        ix=d24.exporter_iso3==e
        record_validation(6,'2023→2024 temporal, frozen lambda, exporter OOF',e,d24[ix],temporal24[ix])
    persistence=metrics(d24.V,second[0].V,d24.exporter_iso3+'_'+d24.j)
    per=[]
    for e in d24.exporter_iso3.unique():
        ix=d24.exporter_iso3==e;per.append(metrics(d24[ix].V,second[0][ix].V,d24[ix].j,3)['destination_wape'])
    DIAGNOSTICS['temporal_persistence']={**persistence,'mean_exporter_destination_wape':float(np.mean(per))}
    # Earlier Alberta years can be audited/used without touching held-out 2025 outcomes.
    load_alberta(2023,'A')
    lag=load_alberta(2024,'A')
    score,outside=scoring_universe(fourth,lag)
    mlp_stability_diagnostic(d24,score)
    DIAGNOSTICS['scoring_support']={}
    for col in NUMERIC:
        lo=float(d24[col].min());hi=float(d24[col].max())
        DIAGNOSTICS['scoring_support'][col]={'training_min':lo,'training_max':hi,
            'outside_range_fraction':float(((score[col]<lo)|(score[col]>hi)).mean()),
            'missing_fraction':float(score[col].isna().mean())}
    pd.DataFrame({'HS4':HS4}).to_csv(OUTPUT_DIR/'frozen_hs4_universe.csv',index=False)
    score[['iso2','importer_iso3']].drop_duplicates().to_csv(OUTPUT_DIR/'frozen_destination_universe.csv',index=False)
    # Model 1 response is 2024, lag/features are exact-year 2023 (missing manufacturing imputed).
    dyn=second[0].drop(columns='V').copy()
    dyn['lag_log']=np.log1p(second[0].V)
    dyn['V']=d24.V.to_numpy()
    allpred=[]
    for model in range(1,6):
        p=run_single(model,dyn if model==1 else d24,score)
        block=score[['iso2','HS4']].copy();block['predicted_usd']=p;block['scope']='structural grid'
        extra=outside[['iso2','HS4','predicted_usd']].copy();extra['scope']='2024 persistence fallback'
        block=pd.concat([block,extra],ignore_index=True);block['model']=model;allpred.append(block)
    fit=fit_two_stage(d24,'final 2024')
    potential=predict_two_stage(fit,score)
    p=dynamic(score.lag_usd,potential,lam)
    block=score[['iso2','HS4']].copy();block['predicted_usd']=p;block['scope']='structural grid'
    extra=outside[['iso2','HS4','predicted_usd']].copy();extra['scope']='2024 persistence fallback'
    block=pd.concat([block,extra],ignore_index=True);block['model']=6;allpred.append(block)
    frozen=pd.concat(allpred,ignore_index=True)
    assert not frozen.duplicated(['model','iso2','HS4']).any()
    assert np.isfinite(frozen.predicted_usd).all() and frozen.predicted_usd.ge(0).all()
    assert frozen.groupby('model').size().nunique()==1
    # Primary CAD rate, destinations, cells and all rankings are frozen before phase B.
    frozen['predicted_cad']=frozen.predicted_usd*CAD_2024
    dest=frozen.groupby(['model','iso2'],as_index=False).predicted_cad.sum().sort_values(['model','predicted_cad','iso2'],ascending=[True,False,True])
    dest['predicted_rank']=dest.groupby('model').cumcount()+1
    top=frozen.merge(dest[dest.predicted_rank<=10][['model','iso2','predicted_rank']],on=['model','iso2'])
    top=top.sort_values(['model','predicted_rank','predicted_cad','HS4'],ascending=[True,True,False,True])
    top['hs4_rank']=top.groupby(['model','iso2']).cumcount()+1
    top=top[top.hs4_rank<=5]
    frozen.to_csv(OUTPUT_DIR/'frozen_hs4_predictions.csv.gz',index=False,compression={'method':'gzip','mtime':0})
    dest.to_csv(OUTPUT_DIR/'frozen_destination_predictions.csv',index=False)
    top.to_csv(OUTPUT_DIR/'frozen_top5_hs4.csv',index=False)
    write_json('model_diagnostics.json',DIAGNOSTICS)
    pd.DataFrame(FIT_RECORDS).to_csv(OUTPUT_DIR/'fit_records.csv',index=False)
    pd.DataFrame(VALIDATION).to_csv(OUTPUT_DIR/'validation_metrics.csv',index=False)
    paths=['frozen_hs4_predictions.csv.gz','frozen_destination_predictions.csv','frozen_top5_hs4.csv','frozen_destination_universe.csv','frozen_hs4_universe.csv']
    write_json('forecast_freeze.json',{'freeze_utc':stamp(),'phase':'A complete; 2025 Alberta observations not opened',
          'cad_per_usd':CAD_2024,'eligible_destinations':AUDIT['eligibility']['eligible'],'grid_hs4':len(HS4),
          'cells_per_model':int(frozen.groupby('model').size().iloc[0]),'lambda':lam,
          'hashes':{p:hashlib.sha256((OUTPUT_DIR/p).read_bytes()).hexdigest() for p in paths}})
    write_audit()
    return frozen,dest,top

# --------------------------------------------------
# Phase B: external 2025 diagnostic, never fitting or tuning
# --------------------------------------------------
def product_descriptions():
    # Labels only; later CBSA vintage does not contribute explanatory variables or eligibility.
    p=OUTPUT_DIR.parent/'1-harmonized-system-canada/data/hs-t2026-2.json'
    d=json.loads(p.read_text(encoding='utf-8'))
    labels={}
    for s in d['sections']:
        for c in s['chapters']:
            for h in c['headings']:
                labels[h['code']]=(h['description'],'CBSA T2026-2 heading label; code match, later wording vintage')
    for r in PRODUCTS.sort_values('HS6').itertuples():
        labels.setdefault(r.HS6[:4],(r.description,'Representative BACI HS6 child; not official HS4 heading'))
    return labels

def country_names():
    if 'COUNTRIES' in globals():
        names=dict(zip(COUNTRIES.country_iso2,COUNTRIES.country_name))
    else:
        # Publication-only mode reads derived labels; it needs no raw-data joins.
        d=pd.read_csv(OUTPUT_DIR/'destination_predictions.csv',usecols=['iso2','destination'],keep_default_na=False)
        names=dict(zip(d.iso2,d.destination))
    names.update({'US':'United States','KR':'South Korea','GB':'United Kingdom','HK':'Hong Kong','TW':'Taiwan',
                  'XK':'Kosovo','AQ':'Antarctica','AN':'Netherlands Antilles (legacy)','ZX':'Unspecified','ZZ':'High Seas','PC':'Pacific Islands (legacy)'})
    return names

def phase_b(frozen,dest,top):
    freeze=json.loads((OUTPUT_DIR/'forecast_freeze.json').read_text())
    for name,digest in freeze['hashes'].items():assert hashlib.sha256((OUTPUT_DIR/name).read_bytes()).hexdigest()==digest
    opened=stamp()
    actual=load_alberta(2025,'B')
    ad=actual.groupby('iso2',as_index=False).actual_cad.sum().sort_values(['actual_cad','iso2'],ascending=[False,True])
    ad['observed_rank']=np.arange(1,len(ad)+1)
    names=country_names();labels=product_descriptions()
    joined=dest.merge(ad,on='iso2',how='outer')
    # Outer merge cannot add actual-only destinations to frozen prediction/rank eligibility.
    # Evaluate all six on union of actual and frozen markets, predicted zero for new actual-only.
    destinations=[];cells=[];summary=[]
    for model in range(1,7):
        d=dest[dest.model==model].merge(ad,on='iso2',how='outer')
        d['model']=model;d['predicted_cad']=d.predicted_cad.fillna(0);d['actual_cad']=d.actual_cad.fillna(0)
        d['destination']=d.iso2.map(names).fillna(d.iso2)
        d['signed_error_cad']=d.predicted_cad-d.actual_cad;d['absolute_error_cad']=d.signed_error_cad.abs()
        d['relative_error_pct']=100*d.signed_error_cad/d.actual_cad.replace(0,np.nan)
        destinations.append(d)
        c=frozen[frozen.model==model].merge(actual,on=['iso2','HS4'],how='outer')
        c['model']=model;c['actual_cad']=c.actual_cad.fillna(0);c['predicted_cad']=c.predicted_cad.fillna(0)
        m=metrics(c.actual_cad,c.predicted_cad,c.iso2)
        selected=d[d.predicted_rank<=10]
        summary.append({'model':model,'name':MODEL_NAMES[model],'observed_total_cad':float(actual.actual_cad.sum()),
             'predicted_total_cad':float(d.predicted_cad.sum()),**m,
             'mean_absolute_top10_error_cad':float(selected.absolute_error_cad.mean()),
             'median_absolute_top10_error_cad':float(selected.absolute_error_cad.median())})
        # Compact selected product output; complete frozen cells remain downloadable compressed.
        t=top[top.model==model].merge(actual,on=['iso2','HS4'],how='left',validate='one_to_one')
        t['actual_cad']=t.actual_cad.fillna(0);t['destination']=t.iso2.map(names)
        t['description']=t.HS4.map(lambda k:labels.get(k,('National/special heading; code-only label','Not an international HS heading'))[0])
        t['description_source']=t.HS4.map(lambda k:labels.get(k,('','National/special heading'))[1])
        t['signed_error_cad']=t.predicted_cad-t.actual_cad;t['absolute_error_cad']=t.signed_error_cad.abs()
        cells.append(t)
    destinations=pd.concat(destinations,ignore_index=True);products=pd.concat(cells,ignore_index=True);summary=pd.DataFrame(summary)
    destinations.to_csv(OUTPUT_DIR/'destination_predictions.csv',index=False)
    products.to_csv(OUTPUT_DIR/'hs4_predictions.csv',index=False)
    summary.to_csv(OUTPUT_DIR/'model_summary.csv',index=False)
    actual_top=ad.head(10).copy();actual_top['destination']=actual_top.iso2.map(names)
    obsprod=actual.merge(actual_top[['iso2','observed_rank']],on='iso2').sort_values(['observed_rank','actual_cad','HS4'],ascending=[True,False,True])
    obsprod['hs4_rank']=obsprod.groupby('iso2').cumcount()+1;obsprod=obsprod[obsprod.hs4_rank<=5]
    obsprod['destination']=obsprod.iso2.map(names);obsprod['description']=obsprod.HS4.map(lambda k:labels.get(k,('National/special heading; code-only label',''))[0])
    actual_top.to_csv(OUTPUT_DIR/'observed_top10_destinations.csv',index=False)
    obsprod.to_csv(OUTPUT_DIR/'observed_top5_hs4.csv',index=False)
    # PDF's realized 2025 FX is an ex-post presentation diagnostic, not a primary forecast.
    cad_2025_pdf=1.3978
    pdfpred=frozen[frozen.model==6].predicted_usd*cad_2025_pdf
    c=frozen[frozen.model==6][['iso2','HS4']].copy();c['pdf_predicted_cad']=pdfpred.to_numpy()
    c=c.merge(actual,on=['iso2','HS4'],how='outer').fillna({'pdf_predicted_cad':0,'actual_cad':0})
    dm=metrics(c.actual_cad,c.pdf_predicted_cad,c.iso2)
    v=pd.DataFrame(VALIDATION)
    checks=[('lambda',DIAGNOSTICS['model6_lambda']['used'],0.134342),
      ('mean_exporter_temporal_destination_wape',float(v[v.model==6].destination_wape.mean()),0.1574),
      ('pdf_currency_alberta_destination_wape',dm['destination_wape'],0.2201),
      ('pdf_currency_alberta_total_cad',float(pdfpred.sum()),142855000000),
      ('observed_alberta_total_cad',float(actual.actual_cad.sum()),177965000000)]
    check=pd.DataFrame(checks,columns=['checkpoint','derived','reference_rounded']);check['difference']=check.derived-check.reference_rounded
    check.to_csv(OUTPUT_DIR/'model_06_reference_checks.csv',index=False)
    AUDIT['2025_evaluation']={'opened_utc':opened,'forecast_freeze_utc':freeze['freeze_utc'],
       'actual_total_cad':int(actual.actual_cad.sum()),'actual_destinations':len(ad),
       'actual_outside_frozen_destination_cad':int(actual.loc[~actual.iso2.isin(dest.iso2),'actual_cad'].sum()),
       'prediction_files_unchanged':True,'pdf_fx_diagnostic':cad_2025_pdf}
    for name,digest in freeze['hashes'].items():assert hashlib.sha256((OUTPUT_DIR/name).read_bytes()).hexdigest()==digest
    write_audit()
    return summary,destinations,products,actual_top,obsprod

# --------------------------------------------------
# Consistent static chart production: same logic for all six models
# --------------------------------------------------
def make_figures(destinations):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from matplotlib.ticker import FuncFormatter
    destinations=destinations.copy()
    destinations['predicted_rank']=pd.to_numeric(destinations.predicted_rank,errors='coerce')
    plt.rcParams.update({'font.family':'DejaVu Sans','font.size':10,'axes.titlesize':12,
          'axes.labelsize':10,'svg.fonttype':'none','figure.facecolor':'#fffdf9','axes.facecolor':'#fffdf9',
          'axes.spines.top':False,'axes.spines.right':False,'axes.spines.left':False,
          'axes.edgecolor':'#aaa69d','text.color':'#222d32','axes.labelcolor':'#222d32',
          'xtick.color':'#59656b','ytick.color':'#222d32','savefig.facecolor':'#fffdf9'})
    blue='#285c80';gold='#b57a32';teal='#25695f';rust='#a14d37'
    names=country_names()
    # Pre-2025 chart comparison set: Model 6's frozen top ten, plus other models' top ten.
    # This union is prediction-selected, never selected on actual 2025 magnitude.
    frozen=pd.read_csv(OUTPUT_DIR/'frozen_destination_predictions.csv',keep_default_na=False)
    anchor=frozen[frozen.model==6].sort_values('predicted_rank')
    ordered=anchor[anchor.predicted_rank<=10].iso2.tolist()
    extras=sorted(set(frozen[frozen.predicted_rank<=10].iso2)-set(ordered))
    ordered+=extras
    nonus=[x for x in ordered if x!='US']
    common=destinations[destinations.iso2.isin(nonus)]
    maxscatter=max(destinations.actual_cad.max(),destinations.predicted_cad.max())/1e6*1.5
    def finish(fig,name):
        fig.savefig(OUTPUT_DIR/'figures'/name,format='svg',bbox_inches='tight',metadata={'Date':None})
        plt.close(fig)
    for model in range(1,7):
        d=destinations[destinations.model==model].set_index('iso2')
        # Share limits within comparable dynamic/structural families; national-scale
        # potentials would otherwise make small dynamic errors unreadable.
        family=destinations[destinations.model.isin([1,6] if model in [1,6] else [2,3,4,5])]
        common=family[family.iso2.isin(nonus)]
        nonmax=max(common.actual_cad.max(),common.predicted_cad.max())/1e9*1.20
        ustable=family[family.iso2=='US']
        usmax=max(ustable.actual_cad.max(),ustable.predicted_cad.max())/1e9*1.20
        nonerr=max(1,float(family[family.iso2!='US'].signed_error_cad.abs().max()/1e9))*1.15
        userr=max(1,float(ustable.signed_error_cad.abs().max()/1e9))*1.15
        height=max(6.4,0.47*len(nonus)+2.3)
        fig,axs=plt.subplots(2,1,figsize=(10.8,height),gridspec_kw={'height_ratios':[1,max(4,len(nonus)*.50)]},layout='constrained')
        fig.suptitle(f'{model:02d} · Actual and predicted destination exports',fontsize=16,fontweight='bold',x=.02,ha='left')
        for ax,codes,title,lim in [(axs[0],['US'],'United States · separate scale',usmax),(axs[1],nonus,'Other comparison markets · common scale within model family',nonmax)]:
            a=d.reindex(codes).fillna({'actual_cad':0,'predicted_cad':0})
            y=np.arange(len(a));av=a.actual_cad.to_numpy()/1e9;pv=a.predicted_cad.to_numpy()/1e9
            ax.barh(y-.18,av,height=.32,color=blue,label='Observed 2025')
            ax.barh(y+.18,pv,height=.32,color=gold,label='Frozen prediction')
            ax.set_yticks(y,[names.get(c,c) for c in codes]);ax.invert_yaxis();ax.set_xlim(0,lim)
            ax.set_title(title,loc='left',pad=10);ax.set_xlabel('CAD billion · primary 2024 exchange-rate assumption')
            ax.grid(axis='x',alpha=.16);ax.set_axisbelow(True)
            for yy,va,vp in zip(y,av,pv):
                ax.text(va+lim*.012,yy-.18,f'{va:,.2f}',va='center',fontsize=9,color=blue)
                ax.text(vp+lim*.012,yy+.18,f'{vp:,.2f}',va='center',fontsize=9,color=gold)
        fig.legend(*axs[0].get_legend_handles_labels(),loc='upper right',
                   bbox_to_anchor=(.995,.995),frameon=False,ncol=2,fontsize=9)
        finish(fig,f'model_{model:02d}_actual_vs_predicted.svg')
        # Symlog retains zeros explicitly; linear neighbourhood is CAD 1 million.
        fig,ax=plt.subplots(figsize=(8,7.5),layout='constrained')
        fig.get_layout_engine().set(rect=(0,.045,1,.955))
        ax.scatter(d.actual_cad/1e6,d.predicted_cad/1e6,s=24,color=teal,alpha=.62,edgecolors='none')
        ax.set_xscale('symlog',linthresh=1,linscale=1);ax.set_yscale('symlog',linthresh=1,linscale=1)
        ax.plot([0,maxscatter],[0,maxscatter],ls='--',lw=1.3,color='#777',label='Equal actual and predicted')
        ax.set_xlim(0,maxscatter);ax.set_ylim(0,maxscatter)
        ax.set_aspect('equal',adjustable='box')
        ax.set_xlabel('Observed 2025 exports · CAD million');ax.set_ylabel('Frozen prediction · CAD million')
        ax.set_title(f'{model:02d} · All destinations, including zero trade',loc='left',fontsize=16,fontweight='bold',pad=14)
        ax.grid(alpha=.16);ax.set_axisbelow(True)
        tick=[0,1,10,100,1000,10000,100000,1000000]
        ax.set_xticks(tick,[f'{x:,}' for x in tick]);ax.set_yticks(tick,[f'{x:,}' for x in tick])
        ax.set_xlim(0,maxscatter);ax.set_ylim(0,maxscatter)
        ax.xaxis.set_minor_locator(matplotlib.ticker.NullLocator());ax.yaxis.set_minor_locator(matplotlib.ticker.NullLocator())
        selected=d.reindex(d[d.predicted_rank<=5].sort_values('predicted_rank').index)
        for n,(code,r) in enumerate(selected.iterrows()):
            ax.annotate(names.get(code,code),(r.actual_cad/1e6,r.predicted_cad/1e6),xytext=(5,7 if n%2==0 else -13),textcoords='offset points',fontsize=9)
        ax.legend(frameon=False,loc='upper left',fontsize=9)
        fig.text(.11,.005,'Symmetric-log axes; 0–1 million is linear. Zeros remain plotted. Identical axes for all six models.',fontsize=9,color='#59656b')
        finish(fig,f'model_{model:02d}_scatter.svg')
        fig,axs=plt.subplots(3,1,figsize=(10.8,height+1.5),gridspec_kw={'height_ratios':[1,max(4,len(nonus)*.50),1.4]},layout='constrained')
        fig.suptitle(f'{model:02d} · Signed destination errors',fontsize=16,fontweight='bold',x=.02,ha='left')
        for ax,codes,title,lim in [(axs[0],['US'],'United States · separate scale',userr),(axs[1],nonus,'Other comparison markets · common scale within model family',nonerr)]:
            a=d.reindex(codes);err=(a.predicted_cad-a.actual_cad).to_numpy()/1e9
            y=np.arange(len(codes));colors=[teal if e>=0 else rust for e in err]
            ax.hlines(y,0,err,color=colors,lw=2);ax.scatter(err,y,color=colors,s=36,zorder=3)
            ax.axvline(0,color='#777',lw=1);ax.set_yticks(y,[names.get(c,c) for c in codes]);ax.invert_yaxis()
            ax.set_xlim(-lim,lim);ax.set_xlabel('Predicted − observed · CAD billion')
            ax.set_title(title,loc='left',pad=10);ax.grid(axis='x',alpha=.16);ax.set_axisbelow(True)
            for yy,e in zip(y,err):
                ax.annotate(f'{e:+,.2f}',(e,yy),xytext=(6 if e>=0 else -6,0),textcoords='offset points',ha='left' if e>=0 else 'right',va='center',fontsize=9)
        axs[0].text(.01,.94,'← underprediction     overprediction →',transform=axs[0].transAxes,va='top',fontsize=9,color='#59656b')
        remaining=d.loc[~d.index.isin(ordered)].sort_index()
        othererr=remaining.signed_error_cad.to_numpy()/1e9
        ys=np.linspace(-.22,.22,len(remaining))
        axs[2].scatter(othererr,ys,s=19,alpha=.65,color=[teal if e>=0 else rust for e in othererr])
        axs[2].axvline(0,color='#777',lw=1)
        axs[2].set_xscale('symlog',linthresh=.001,linscale=1)
        axs[2].set_xlim(-nonerr,nonerr);axs[2].set_ylim(-.4,.4);axs[2].set_yticks([])
        axs[2].set_title(f'Every other destination · {len(remaining)} dots · zeros retained',loc='left',pad=10)
        axs[2].set_xlabel('Signed error · CAD billion · symmetric-log; linear within ± CAD 1 million')
        axs[2].grid(axis='x',alpha=.16);axs[2].set_axisbelow(True)
        finish(fig,f'model_{model:02d}_errors.svg')

# --------------------------------------------------
# Research notebook publication: static HTML, exactly six closed accordions
# --------------------------------------------------
def html_table(headers,rows,caption,cls=''):
    head=''.join(f'<th scope="col">{escape(str(h))}</th>' for h in headers)
    body=''.join('<tr>'+''.join(f'<td>{v}</td>' for v in r)+'</tr>' for r in rows)
    return f'<div class="table-scroll" role="region" tabindex="0" aria-label="{escape(caption)}"><table class="{cls}"><caption>{escape(caption)}</caption><thead><tr>{head}</tr></thead><tbody>{body}</tbody></table></div>'

def money(x,unit=1e9,sign=False):
    return f'{x/unit:+,.3f}' if sign else f'{x/unit:,.3f}'

def pct(x):return f'{100*x:.2f}%'

def eq(tex):return '<div class="equation" tabindex="0" role="region" aria-label="Equation; scroll horizontally if needed">\\['+tex+'\\]</div>'

def product_tables(products,observed=False):
    result=[]
    group='observed_rank' if observed else 'predicted_rank'
    for rank,d in products.groupby(group,sort=True):
        d=d.sort_values('hs4_rank');name=d.destination.iloc[0]
        rows=[]
        for r in d.itertuples():
            if observed:rows.append([str(int(r.hs4_rank)),f'<code>{r.HS4}</code>',escape(r.description),money(r.actual_cad,1e6)])
            else:rows.append([str(int(r.hs4_rank)),f'<code>{r.HS4}</code>',escape(r.description),money(r.predicted_cad,1e6),money(r.actual_cad,1e6),money(r.absolute_error_cad,1e6)])
        headers=['HS4 rank','HS4','Product heading','Actual CAD m'] if observed else ['HS4 rank','HS4','Product heading','Predicted CAD m','Actual CAD m','Absolute error CAD m']
        result.append(html_table(headers,rows,f'{"Actual" if observed else "Predicted"} destination #{int(rank)} · {name}',cls='products'))
    return ''.join(result)

def model_narrative(model,diag):
    # Each discussion uses the same teaching structure; numbers are derived diagnostics.
    if model==1:
        d=diag['model1']
        intuition='<p>An established Alberta–destination product relationship carries information about buyers, pipelines, contracts and sunk entry costs. Linear regression asks whether last year’s relationship plus observable fundamentals explain the following year in an additive way. It supplies a transparent persistence baseline, rather than treating contemporaneous 2024 trade as a forecast.</p>'
        math=eq(r'y_{ijk,2024}=\alpha+\rho y_{ijk,2023}+\beta^{\mathsf T}X_{ijk,2023}+\gamma_{\mathrm{HS2}(k)}+\varepsilon_{ijk},\qquad y=\log(1+V)')
        math+=eq(r'\widehat\theta=(A^{\mathsf T}A)^{-1}A^{\mathsf T}y')
        math+='<p>The inverse formula assumes full column rank; the code solves least squares numerically and uses a generalized inverse for inference where necessary. Matrix <em>A</em> contains the intercept, lag, numeric features and HS2 indicators. We omit one HS2 category in this model. Conditional-mean correctness requires the unexplained component to have mean zero given these predictors. That is a substantive assumption, not something a small residual can prove.</p>'
        math+=eq(r'\widehat y_{\mathrm{AB},j,k,2025}=\widehat\alpha+\widehat\rho\log(1+V_{\mathrm{AB},j,k,2024})+\widehat\beta^{\mathsf T}X_{\mathrm{CAN\ proxy},j,k,2024}+\widehat\gamma_{\mathrm{HS2}(k)}')
        fit='<p><strong>Imputation-qualified dynamic estimate.</strong> Response: 2024 BACI log1p USD; predictors and lag: 2023. Exact 2023 GDP, population and internet exist for all seven countries. Canada and U.S. manufacturing shares are absent. The fully observed preferred specification is therefore unavailable; we retain the dynamic experiment using explicitly estimated 2023 training-fold medians. Those medians are assumptions, not recovered country observations. Alberta scoring uses the 2024 state and 2024 explanatory information, with missing manufacturing imputed by the fitted 2023 pipeline. No 2024 value is borrowed backward.</p>'
        fit+='<p>Numeric predictors are median-imputed and standardized inside each fit. HS2 is categorical. OLS minimizes squared errors in log1p trade. Dollar predictions use the shared training-only aggregate calibration described above; clipping negative back-transformed values to zero adds another approximation.</p>'
        val='<p>Leave one exporter out of the 2023→2024 transition; fit on the other six and predict that exporter’s 2024 outcomes from its observed 2023 state. This is exporter transfer within the estimation transition, <strong>not an independent later-year validation of Model 1</strong>. Unlike Model 6, there is no second untouched transition available after estimating this specification. A lag-only OLS diagnostic uses the same folds.</p>'
        diagnostics=f'<p>The unstandardized lag coefficient is <strong>ρ = {d["rho"]:.4f}</strong>; HC3 SE {d["rho_hc3_se"]:.4f}; dyad-cluster SE {d["rho_dyad_cluster_se"]:.4f}. A unit increase in log1p lagged trade changes predicted log1p next-year trade by ρ, conditional on other predictors. Matrix rank is {d["matrix_rank"]}/{d["columns"]}, condition number {d["condition_number"]:.2g}; exporter GDP/population correlation is {d["gdp_population_corr"]:.3f}. These are diagnostics of identification and numerical conditioning, not causal estimates.</p>'
        diagnostics+=f'<p>Lag-only exporter-transfer cell WAPE: {pct(d["lag_only_oof"]["cell_wape"])}; full model: {pct(d["aggregate_oof"]["cell_wape"])}. This comparison asks whether the elaborate feature set adds much beyond persistent flows. <a href="4-hs-data-analysis/model_01_coefficients.csv">Coefficients with HC3 and dyad-cluster standard errors</a> are available for inspection.</p>'
        variance=list(d['residual_variance_bins'].values())
        diagnostics+=f'<p>Training residual variance across fitted-log quintiles ranges from {min(variance):.3f} to {max(variance):.3f}; this is a descriptive heteroskedasticity check. The training-only dollar calibration factor is {d["calibration"]:.3f}. Neither residual diagnostics nor aggregate matching validates a conditional economic mean.</p>'
        critique='<p><strong>Predictive limitations.</strong> Linearity imposes one additive log-scale effect across products and destinations. Petroleum’s enormous dollar scale can remain badly predicted despite a small log error. A lag-dominated forecast may miss a new pipeline destination or abrupt policy/price shift. Aggregate calibration cannot repair cell-specific bias.</p><p><strong>Structural and statistical limitations.</strong> The large row count is a large <em>N</em> with very small <em>T</em>: only one fitted transition, not thousands of independent time observations. Shared dyads and products induce correlation. GDP/population and product-market variables overlap; omitted contracts and logistics may correlate with regressors. Heteroskedasticity makes ordinary standard errors unreliable. HC3 does not handle clustering; 42 dyad clusters improve one dimension but do not account for product clustering or generated/calibrated forecasts. A common ρ cannot establish stable long-run dynamics. Log transformation preserves zeros but does not make residuals normal or homoskedastic.</p>'
        teaches='<p>A seemingly successful regression can mostly reflect yesterday’s trade. Its legible coefficients make assumptions inspectable, but neither lag persistence nor robust standard errors validates the economic mechanism. The missing manufacturing data weaken the preferred exact-year specification even when predictions look plausible.</p>'
    elif model==2:
        d=diag['model2'];chosen=d['chosen']
        intuition='<p>GDP, population, manufacturing and connectivity can convey overlapping country information. HS2 indicators add many correlated controls. Elastic Net keeps an additive log-trade structure while limiting how freely correlated predictors can trade off against one another. The question is whether coefficient restraint improves transfer to an unseen exporter.</p>'
        math=eq(r'\min_{\beta}\;\frac{1}{2n}\lVert y-A\beta\rVert_2^2+\alpha\left[r\lVert\beta\rVert_1+\frac{1-r}{2}\lVert\beta\rVert_2^2\right],\quad y=\log(1+V_{2024})')
        math+='<p>The intercept is unpenalized. The L1 term can shrink coefficients to zero; L2 shares shrinkage across correlated predictors. Their combination stabilizes an ill-conditioned additive model without claiming that selected variables are causal determinants. Standardizing numeric predictors makes penalty strength comparable across GDP, percentages and distance; HS2 remains one-hot categorical.</p>'
        fit=f'<p>All {AUDIT["annual_grid_cells"]:,} zero-filled 2024 cells and the common features are used. Median imputation, scaling and one-hot encoding are fitted only on training exporters. A predeclared four-candidate grid uses α ∈ {{0.01, 0.1}} and r ∈ {{0.2, 0.8}}. The final grouped-CV choice is <strong>α = {chosen["alpha"]}, r = {chosen["l1_ratio"]}</strong>, selected by equal-exporter mean log MSE. {d["nonzero_coefficients"]}/{d["coefficient_count"]} fitted coefficients are nonzero. No Alberta outcome influences that choice.</p>'
        val='<p>Outer leave-one-exporter-out evaluates transfer. Within each outer training subset, a second leave-one-exporter-out loop selects α and r, so the outer exporter is excluded from tuning, preprocessing, fitting and dollar calibration. This nested design avoids reporting a tuned-on-the-test-fold score. Only seven exporters means high uncertainty about transfer; a random cell split would mix nearly identical country characteristics into training and validation. These are 2024 cross-sectional tests, not future-year forecasts.</p>'
        diagnostics=f'<p>Final training log RMSE is {d["in_sample_log_rmse"]:.3f}; the training-only dollar calibration factor is {d["calibration"]:.3f}. A low log loss can coexist with distorted destination dollar totals because shrinkage acts on the transformed scale. <a href="4-hs-data-analysis/model_diagnostics.json">Every nested parameter choice and candidate loss</a> is retained.</p>'
        critique='<p><strong>Predictive limitations.</strong> Shrinkage introduces bias and cannot represent a product-specific distance threshold or supply–demand interaction unless it is explicitly engineered. Selecting among four candidates is intentionally modest; seven-exporter scores are noisy.</p><p><strong>Structural and statistical limitations.</strong> Regularization does not solve omitted-variable bias, reverse causality or dependence among cells. Selected coefficients can change when an exporter leaves the sample. One-hot HS2 controls broad sectors, not HS4-specific economic identities. The model maps a contemporaneous 2024 pattern onto Canada-proxy Alberta inputs; it has no equation governing 2024→2025 adjustment.</p>'
        teaches='<p>More stable coefficients can be worth accepting some predictive bias. Yet regularization repairs coefficient instability, not the economic validity of a linear structural-potential model or the absence of a time-transition mechanism.</p>'
    elif model==3:
        d=diag['model3']
        important=sorted(d['permutation_importance'].items(),key=lambda x:-x[1])[:4]
        intuition='<p>A forest can learn that distance matters differently when importer demand is high or exporter supply is weak. A Canada–U.S. border split can interact with a product’s external supply without imposing a single gravity coefficient. Many randomized trees reduce the variability of any one partition.</p>'
        math=eq(r'\widehat f(x)=\frac{1}{B}\sum_{b=1}^{B}T_b(x),\qquad y=\log(1+V_{2024})')
        math+='<p>Each tree receives a bootstrap sample of the training cells and a randomized subset of candidate features at each split. Within a leaf it estimates a local mean log1p flow. Averaging lowers variance when trees make different errors; dependence between trees limits that gain. This is interpolation within partitions, not an economic equation for trade adjustment.</p>'
        fit='<p>Use the full zero-filled 2024 grid, the common feature family and a fresh training-only preprocessing pipeline. The forest has 120 trees, depth at most 16, at least 8 samples per leaf, 70% of transformed features considered per split, bootstrap enabled, seed 338. These manageable settings were declared before 2025 evaluation. It fits squared error in log1p trade; the shared aggregate calibration returns dollar predictions.</p>'
        val='<p>Leave one exporter out of 2024; fit preprocessing, forest and retransformation on the other six. Hyperparameters are fixed rather than selected on the held-out exporter. Bootstrap cells are not independent dyads/products, so the bootstrap is an algorithmic device, not a valid inferential resampling scheme here. This is transfer validation, not temporal validation.</p>'
        diagnostics=f'<p>Final training log RMSE {d["in_sample_log_rmse"]:.3f}; dollar calibration {d["calibration"]:.3f}. The leading raw-feature permutation diagnostics on the held-out Canadian exporter are '+', '.join(f'<code>{escape(k)}</code> ({v:.3f} log-MSE increase)' for k,v in important)+'. Permuting correlated market variables can make unrealistic combinations; these measures are descriptive, not causal importance.</p>'
        critique='<p><strong>Predictive limitations.</strong> Forests predict combinations of training leaf values and extrapolate poorly outside observed national scale, geography and product-market support. Log fitting moderates crude petroleum’s domination, but dollar retransformation can still be driven by a few giant flows. Newly scored destinations are not equivalent to the six importers seen by each training exporter.</p><p><strong>Structural and statistical limitations.</strong> A threshold does not identify an economic mechanism. Correlated cells and predictors can mask which feature actually matters. Conventional impurity importance favors predictors with more split opportunities; we report held-out permutation diagnostics instead. Neither measure establishes causality. Canadian national capability can create implausibly large provincial structural values. There is no one-year dynamics.</p>'
        teaches='<p>Good interpolation can discover useful interactions while leaving economic interpretation weak. A tree’s apparent “distance effect” is a sample partition, not evidence that reducing distance would cause the predicted export gain.</p>'
    elif model==4:
        d=diag['model4']
        intuition='<p>A small neural network can combine market size, connectivity and product-side evidence into nonlinear hidden representations. In this trade dataset, the experiment asks whether learned interactions transfer better than explicit linear coefficients or tree partitions—not whether adding depth automatically helps.</p>'
        math=eq(r'h_1=\tanh(W_1x+b_1),\quad h_2=\tanh(W_2h_1+b_2),\quad \widehat y=W_3h_2+b_3')
        math+='<p>The two hidden layers contain 32 and 16 units. Each hidden activation is bounded between −1 and 1; the output estimates log1p 2024 trade. Backpropagation uses the chain rule to pass each log-scale prediction error through the layers; Adam updates weights using adaptive moving estimates of gradients. L2 weight decay discourages large weights. The finite network’s practical fit depends on optimization, regularization and data support; universal approximation is not a performance guarantee.</p>'
        fit='<p>Common 2024 features, training-fold median imputation and standardization, HS2 one-hot indicators; no numeric HS-code ordering. Scaling is critical: otherwise percentage-valued connectivity and huge economic/product scales create uneven optimization geometry. Adam uses learning rate 0.001, batches of 512, L2 α = 0.1, maximum 180 epochs, 15% training-only internal validation and patience 15, seed 338. Random cell early stopping stays within outer training exporters but does not itself enforce exporter separation.</p><p><strong>Documented stability repair:</strong> the initial 48/24 ReLU network produced finite but absurdly large transformed predictions when pre-2025 scoring GDP and connectivity fell far outside the seven-country training range. ReLU can extend a fitted slope indefinitely; exponentiation magnified that extrapolation. We simplified to fixed 32/16 bounded tanh layers and stronger L2 regularization, retaining the MLP family and identical common inputs. The repair was justified by historical feature support and the frozen transformed predictions, not by minimizing Alberta’s observed 2025 error. All final models were then refitted and frozen again. Bounded hidden activations still do not impose an Alberta export-capacity constraint.</p>'
        val='<p>Outer leave-one-exporter-out tests 2024 transfer. The internal stopping subset never includes that outer exporter. Architecture and optimization settings are fixed before evaluation. Only one seed is used: the page reports optimizer dependence rather than claiming seed-robust dominance. The 2025 comparison remains an external diagnostic.</p>'
        diagnostics=f'<p>The final network has {d["parameters"]:,} parameters, completed {d["iterations"]} epochs, and its best random-cell internal validation R² is {d["best_internal_validation_r2"]:.3f}. That internal score is not the exporter-held-out score below. Training log RMSE {d["in_sample_log_rmse"]:.3f}, aggregate calibration {d["calibration"]:.3f}. Final scoring log predictions range from {d["scoring_log_prediction_min"]:.3f} to {d["scoring_log_prediction_max"]:.3f}; largest training log target {d["training_log_target_max"]:.3f}. Convergence/early-stopping records for all folds are retained in <a href="4-hs-data-analysis/fit_records.csv">fit records</a>.</p>'
        critique='<p><strong>Predictive limitations.</strong> Standardization, learning rate, initialization and early stopping alter the fitted surface. The network may overfit correlated cells or produce extreme extrapolations, amplified by exponentiation. Medium-sized economic tables do not give a neural network the repeated spatial structure that makes deep image models successful; strong trees may fit thresholds more efficiently.</p><p><strong>Structural and statistical limitations.</strong> Hidden activations do not identify economic mechanisms. The same exporter variables repeat across thousands of products; nominal sample size exaggerates independent socioeconomic information. Optimization finds a local solution, and no causal, capacity or equilibrium constraints are imposed. The model has no temporal equation and national proxies remain imperfect.</p>'
        teaches='<p>A different mathematical architecture does not supply missing provincial information. A network’s flexibility can improve or worsen predictions while making the reasons harder to inspect; scaling and optimization are part of the model, not incidental housekeeping.</p>'
    elif model==5:
        d=diag['model5']
        intuition='<p>Single-stage boosting sequentially corrects mistakes in the predicted log1p trade surface. It can capture a supply–demand interaction or nonlinear distance response with small trees. This section isolates the flexible structural estimator before adding Model 6’s extensive/intensive separation and time adjustment.</p>'
        math=eq(r'F_M(x)=F_0(x)+\eta\sum_{m=1}^{M}h_m(x),\qquad r_{n,m}=-\left.\frac{\partial L(y_n,F(x_n))}{\partial F(x_n)}\right|_{F=F_{m-1}}')
        math+='<p>For squared log-scale error, pseudo-residuals are the current prediction errors. Later trees correct patterns left by earlier ones. Histogram boosting bins continuous values internally and uses gradients and Hessians for efficient splits and leaf updates. A forest averages largely independent randomized trees; boosting builds a dependent sequence of corrective trees.</p>'
        fit=f'<p>One HistGradientBoostingRegressor fits <strong>log(1 + V)</strong> on every zero-filled 2024 cell. There is no positive-trade classifier, positive-only regressor, λ, or Alberta lag blending. Settings match the structural Model 6 tree budget: max_iter 150, max_leaf_nodes 7, learning_rate 0.05, seed 338. Automatic early stopping retains scikit-learn defaults; actual final iterations: {d["iterations"]}. The preprocessing and dollar calibration are training-only.</p>'
        val='<p>Leave one exporter out of 2024. Tree settings are predeclared. Automatic stopping uses a random cell subset of the six training exporters; this can look optimistic when country/product patterns repeat, although the outer test exporter remains excluded. The internal experiment is cross-sectional transfer.</p>'
        diagnostics=f'<p>Final training log RMSE {d["in_sample_log_rmse"]:.3f}; training-only dollar calibration {d["calibration"]:.3f}. The same tree size and shrinkage do not make this estimator equivalent to Model 6: fitting log1p over zeros and positives defines a different loss and structural signal.</p>'
        critique='<p><strong>Predictive limitations.</strong> Learning rate, leaf count and stopping jointly constrain complexity; small corrective trees can still fit repeated cross-sectional patterns. Zero trade and positive flow magnitudes compete within one squared-error response mechanism. Poor outside-network support and large petroleum dollar errors remain possible.</p><p><strong>Structural and statistical limitations.</strong> Its output is <strong>2025 model-implied trade potential based on 2024 information</strong>, not a fully identified one-year forecast. No mechanism says how quickly Alberta could move toward that signal. Canadian supply and GDP represent national capability. Boosting’s flexibility does not distinguish feasible opportunity from statistical resemblance. This unresolved structural-to-dynamic step motivates Model 6.</p>'
        teaches='<p>Improving the shape of a cross-sectional prediction does not supply a clock. A flexible structural estimator can produce credible-looking rankings while having no identified rule that turns 2024 fundamentals into next-year realized exports.</p>'
    else:
        d=diag['model6_lambda']
        intuition='<p>Trade has an entry question and a size question. Model 6 first estimates whether a bilateral HS4 flow is positive and its size conditional on being positive. It then moves only partway from Alberta’s observed 2024 relationship toward that structural signal. The observed lag preserves information about contracts and infrastructure that GDP, distance and external supply cannot fully recover.</p>'
        math=eq(r'D_{ijkt}=\mathbf1\{V_{ijkt}>0\},\quad \widehat\pi(X)=P(D=1\mid X),\quad L_D=-\sum_n[d_n\log\pi_n+(1-d_n)\log(1-\pi_n)]')
        math+='<p><strong>Extensive margin:</strong> HistGradientBoostingClassifier on the entire grid, binary log-loss. Its probability concerns positive current-year trade conditional on the features. It is not a destination-rank probability, a commercial-success probability or feasibility certification.</p>'
        math+=eq(r'z=\log V\quad(D=1),\qquad \widehat m(X)\approx E[\log V\mid D=1,X]')
        math+='<p><strong>Intensive margin:</strong> HistGradientBoostingRegressor on positive flows only. This log(V) target differs from the log(1 + V) used in the dynamic equation and all single-stage models. Separate fitted preprocessing pipelines respect the different training subsets.</p>'
        math+=eq(r'c_{\mathrm{train}}=\frac{\sum_{n:D_n=1}V_n}{\sum_{n:D_n=1}\exp[\widehat m(X_n)]},\quad \widehat V^+(X)=c_{\mathrm{train}}\exp[\widehat m(X)],\quad V^*(X)=\widehat\pi(X)\widehat V^+(X)')
        math+='<p>Exponentiating a conditional log mean is not an arithmetic conditional mean. Aggregate calibration matches positive <em>training</em> dollars; it is not an exact conditional smearing correction. Multiplying by π gives the structural expected-trade signal V*. It is not yet Alberta’s next-year forecast.</p>'
        math+=eq(r'p_t=\log(1+V^*_t),\quad y_t=\log(1+V_t),\quad \widehat y_{t+1}=y_t+\lambda(p_t-y_t)')
        math+=eq(r'\widehat V_{t+1}=\exp\{\log(1+V_t)+\lambda[\log(1+V^*_t)-\log(1+V_t)]\}-1')
        math+=eq(r'1+\widehat V_{t+1}=(1+V_t)^{1-\lambda}(1+V^*_t)^\lambda')
        math+='<p>This is geometric adjustment in gross trade levels, not a dollar-weighted average. λ = 0 leaves the USD state unchanged; λ = 1 closes the full log gap in one year; intermediate λ preserves persistence while moving partway. The final inverse-log dynamic plug-in remains distinct from an unbiased conditional-mean forecast.</p>'
        new_cell=float(np.expm1(d['used']*np.log1p(10_000_000)))
        math+=f'<p>For illustration, a zero lag and a USD 10 million structural signal imply only about <strong>USD {new_cell:.2f}</strong> after one year at the fitted λ. Log-space adjustment therefore attenuates new cells sharply; it does not identify whether entry is commercially feasible. The “+1” makes near-zero behaviour sensitive to monetary units, so the implementation keeps USD dollars consistently rather than switching to thousands midway.</p>'
        fit='<p>Both stages reproduce the supplied PDF and supporting script: max_iter 150, max_leaf_nodes 7, learning_rate 0.05, random_state 338, default automatic early stopping. Each stage uses numeric median imputation and standardization plus HS2 one-hot encoding. Training-only positive-dollar calibration is estimated anew in every fold. Legal variables and Alberta 2025 outcomes are excluded.</p>'
        fit+=eq(r'\Delta y_{2023}=\lambda(p^{\mathrm{exporter\ OOF}}_{2022}-y_{2022})+\varepsilon,\qquad \widehat\lambda=\frac{\sum_n g_n\Delta y_n}{\sum_n g_n^2},\quad g=p^{\mathrm{OOF}}_{2022}-y_{2022}')
        fit+='<p>Estimate λ without an intercept using 2022 exporter-out-of-fold potential and actual 2022/2023. The theoretical [0,1] restriction follows the source’s clipping rule; it is never selected using later errors. Changing the structural estimator changes the gap, so an older estimator’s λ cannot be reused.</p>'
        val='<ol class="timeline"><li><strong>2022→2023:</strong> exclude each exporter from both stages, preprocessing and calibration; estimate λ from its OOF gap and actual log change.</li><li><strong>2023→2024:</strong> freeze λ, independently repeat 2023 exporter OOF fitting, combine potential with actual 2023, save predictions, then join actual 2024 solely for validation.</li><li><strong>2024→Alberta 2025:</strong> fit final 2024 stages, score the frozen universe with Canadian proxies, combine with Alberta’s actual 2024 lag.</li><li><strong>2025:</strong> open Alberta outcomes only after all six models, destination ranks and HS4 ranks are frozen. This is an external diagnostic.</li></ol>'
        cls=diag['model6_2023_classification']
        diagnostics=f'<p>Independently derived λ = <strong>{d["used"]:.6f}</strong>; HC3 SE {d["hc3_se"]:.6f}; 95% interval [{d["ci"][0]:.6f}, {d["ci"][1]:.6f}]. Restriction binds: {"yes" if d["restriction_binds"] else "no"}. Gap/change correlation {d["gap_change_correlation"]:.4f}; uncentered adjustment R² {d["uncentered_r2"]:.4f}. 2023 extensive-margin OOF log-loss {cls["exporter_oof_log_loss"]:.4f}, Brier score {cls["exporter_oof_brier"]:.4f}. A statistically positive adjustment rate is not proof of superior dollar forecasting.</p>'
        persistence=diag['temporal_persistence']
        diagnostics+=f'<p><strong>Persistence challenge:</strong> on the same 2023→2024 temporal test, constant-USD lag persistence has mean exporter destination WAPE {pct(persistence["mean_exporter_destination_wape"])}. The temporal table below reports Model 6’s result. A more elaborate structure can lose to a simple persistent state. This benchmark is diagnostic, not a seventh principal model.</p>'
        critique='<p><strong>Predictive limitations.</strong> A common λ may shrink high-potential new cells sharply while reducing an established large flow. Sector-specific price shocks, new infrastructure and tariffs are absent. Probability estimation and positive-value regression can both err; their product feeds a nonlinear dollar adjustment. Eligible destinations greatly exceed the seven-country training network.</p><p><strong>Structural and statistical limitations.</strong> Separating margins and retaining lagged trade gives a clearer economic story, but λ is estimated from one transition and tested on one subsequent transition. It may differ by sector/destination. HC3 does not fully account for dyad/product clustering, generated structural potential or retransformation uncertainty; its interval is narrower than fully integrated uncertainty would justify. Canadian GDP, population, manufacturing, internet, geography and external supply remain imperfect provincial proxies. The signal is neither an identified equilibrium nor evidence of commercial feasibility.</p>'
        teaches='<p>Adding a time mechanism changes the meaning of the output, not merely its error score. More defensible persistence logic and margin separation can coexist with worse predictions than simpler alternatives. The temporal benchmark and external provincial diagnostic ask different questions and must both remain visible.</p>'
    return (f'<h3>Intuition and relevance to Alberta</h3>{intuition}<h3>Target, notation and mathematics</h3>{math}'
       f'<h3>Data, preprocessing and fitting</h3>{fit}<h3>Validation logic</h3>{val}'
       f'<h3>Diagnostics and interpretation</h3>{diagnostics}<h3>Predictive and structural critique</h3>{critique}',teaches)

def render_page():
    global AUDIT
    # The published page's representative key blocks were added after the initial
    # builder. Retain them rather than losing the Dataset Size placement on rebuild.
    keys = key_blocks(PAGE_PATH.read_text(encoding='utf-8'))
    assert len(keys) == 6, 'Expected the existing six representative key blocks'
    AUDIT=json.loads((OUTPUT_DIR/'data_audit.json').read_text(encoding='utf-8'))
    diag=json.loads((OUTPUT_DIR/'model_diagnostics.json').read_text(encoding='utf-8'))
    summ=pd.read_csv(OUTPUT_DIR/'model_summary.csv')
    dest=pd.read_csv(OUTPUT_DIR/'destination_predictions.csv',dtype={'iso2':str},keep_default_na=False)
    prod=pd.read_csv(OUTPUT_DIR/'hs4_predictions.csv',dtype={'HS4':str,'iso2':str},keep_default_na=False)
    observed=pd.read_csv(OUTPUT_DIR/'observed_top10_destinations.csv',dtype={'iso2':str},keep_default_na=False)
    obsprod=pd.read_csv(OUTPUT_DIR/'observed_top5_hs4.csv',dtype={'HS4':str,'iso2':str},keep_default_na=False)
    valid=pd.read_csv(OUTPUT_DIR/'validation_metrics.csv')
    fits=pd.read_csv(OUTPUT_DIR/'fit_records.csv')
    refs=pd.read_csv(OUTPUT_DIR/'model_06_reference_checks.csv')
    overview=[]
    ideas={1:'Additive one-year persistence',2:'Regularized additive structure',3:'Average randomized tree partitions',4:'Learn nonlinear hidden representations',5:'Sequential single-response correction',6:'Trade margins + geometric adjustment'}
    weaknesses={1:'One transition; missing manufacturing; lag dominance',2:'Linearity and omitted-variable bias remain',3:'Weak extrapolation; no dynamics',4:'Optimizer/preprocessing dependence; no dynamics',5:'Structural potential lacks adjustment clock',6:'One λ transition; generated-regressor uncertainty'}
    for r in summ.itertuples():
        m=int(r.model)
        years='2023 predictors → 2024 response' if m==1 else '2022–2024, ordered phases' if m==6 else '2024 only'
        design='Transition + exporter transfer' if m==1 else 'Nested exporter CV' if m==2 else 'Separate temporal test' if m==6 else 'Exporter-held-out 2024'
        overview.append([f'<a href="#model-{m:02d}">{m:02d} · {escape(MODEL_NAMES[m])}</a>',ideas[m],years,'Yes' if m in [1,6] else 'No',
               'Yes (two margins)' if m==6 else 'Yes (log1p)', 'Yes' if m>=3 else 'No',design,money(r.predicted_total_cad),pct(r.destination_wape),f'{r.top_overlap}/10',weaknesses[m]])
    comparison=html_table(['Model','Main idea','Training years','Lag?','Zeros?','Nonlinear?','Validation','2025 CAD bn','2025 destination WAPE','Top-ten overlap','Main weakness'],overview,'Six approaches · model order is conceptual, not an error ranking','comparison')
    observed_rows=[[str(int(r.observed_rank)),escape(r.destination),money(r.actual_cad)] for r in observed.itertuples()]
    observation_table=html_table(['Actual rank','Destination','2025 domestic exports CAD bn'],observed_rows,'Observed Statistics Canada Alberta exports · ranked by actual 2025 value')
    sections=[]
    for r in summ.itertuples():
        m=int(r.model);narrative,teaches=model_narrative(m,diag)
        dd=dest[(dest.model==m)&(pd.to_numeric(dest.predicted_rank,errors='coerce')<=10)].copy()
        dd['predicted_rank']=pd.to_numeric(dd.predicted_rank);dd=dd.sort_values('predicted_rank')
        rows=[]
        for q in dd.itertuples():
            relative=f'{float(q.relative_error_pct):+.1f}%' if q.relative_error_pct!='' else 'Undefined (actual = 0)'
            rows.append([str(int(q.predicted_rank)),escape(q.destination),money(q.predicted_cad),money(q.actual_cad),money(q.signed_error_cad,sign=True),money(q.absolute_error_cad),relative])
        table=html_table(['Predicted rank','Destination','Predicted CAD bn','Actual CAD bn','Signed error CAD bn','Absolute error CAD bn','Relative error'],rows,f'Model {m:02d} · frozen predicted top ten; actual values joined afterward')
        vv=valid[valid.model==m]
        vrows=[[escape(q.held_out_exporter),pct(q.cell_wape),money(q.cell_mae,1e6),f'{q.rmsle:.3f}',pct(q.destination_wape),f'{q.rank_correlation:.3f}',f'{q.top_overlap}/3'] for q in vv.itertuples()]
        vtable=html_table(['Held-out exporter','Cell WAPE','Cell MAE USD m','RMSLE','Destination WAPE','Spearman','Top-three overlap'],vrows,'2023→2024 temporal test' if m==6 else '2023→2024 transition exporter transfer' if m==1 else '2024 cross-sectional exporter transfer')
        vtable+=f'<p class="small">Equal-exporter mean destination WAPE: <strong>{pct(vv.destination_wape.mean())}</strong>. Only six destinations exist per training exporter, so the internal overlap diagnostic uses top three; top ten would be uninformative. Cell metrics use USD; external diagnostics use CAD. Internal and external experiments are not interchangeable.</p>'
        fr=fits[(fits.model==m)&(fits.context.str.startswith('final'))]
        if m==6:
            f=fr.iloc[0];vtable+=f'<p class="small">Final classifier iterations: {int(f.classifier_iterations)}; positive-value regressor iterations: {int(f.regressor_iterations)}; positive-dollar calibration: {f.calibration:.4f}.</p>'
        metrics_html=f'<dl class="result-strip"><div><dt>Primary prediction</dt><dd>CAD {money(r.predicted_total_cad)} bn</dd></div><div><dt>Destination WAPE</dt><dd>{pct(r.destination_wape)}</dd></div><div><dt>Spearman</dt><dd>{r.rank_correlation:.3f}</dd></div><div><dt>Top ten overlap</dt><dd>{r.top_overlap}/10</dd></div></dl>'
        external=f'<p>Observed total: <strong>CAD {money(r.observed_total_cad)} bn</strong>. External HS4-cell MAE: CAD {money(r.cell_mae,1e6)} m; cell WAPE: {pct(r.cell_wape)}; RMSLE: {r.rmsle:.3f}. Mean absolute error over the <em>frozen predicted</em> top ten: CAD {money(r.mean_absolute_top10_error_cad)} bn; median: CAD {money(r.median_absolute_top10_error_cad)} bn. Destination WAPE aggregates products before measuring error; cell WAPE can be higher because product errors cancel in totals.</p>'
        scoring='<p>Alberta is a separate scoring entity. Canadian 2024 GDP, population, manufacturing share (imputed where absent), internet share, Canada-to-destination distance/border/language and external Canadian HS4 supply are <strong>Canadian proxies</strong>; they are not Alberta measurements. Destination demand and outside-network demand use 2024 BACI. Only Models 1 and 6 use Alberta’s actual 2024 destination–HS4 lag. All six use the same pre-2025 structural grid and identical disclosed persistence fallback. No 2025 destination or product outcome selected these lists.</p>'
        terminology='<p class="notice">2025 Alberta outcomes below are an external diagnostic. This model’s output is '+('an imputation-qualified one-year prediction' if m==1 else 'a one-year partial-adjustment prediction' if m==6 else 'model-implied trade potential based on 2024 information')+'. It is not a certificate of commercial opportunity.</p>'
        charts=[]
        for suffix,title,alt,cap in [
          ('actual_vs_predicted','Actual versus predicted exports','Grouped horizontal bars compare observed and frozen predicted Alberta exports, with a separate United States scale.',
           'United States has its own panel. Markets and order are the same prediction-selected comparison set across all models. Axis limits are shared within dynamic Models 1/6 and within structural Models 2–5; bar labels are CAD billion.'),
          ('scatter','How values align across all destinations','Scatter of observed 2025 Alberta exports against frozen model predictions, including zero trade, with an equality reference line.',
           'Every evaluated destination is plotted. Symmetric-log axes retain zeros and are linear below CAD 1 million; identical scales across models. Labels identify the frozen predicted top five.'),
          ('errors','Where the model over- and underpredicts','Signed lollipop chart of predicted minus observed destination exports. Negative values indicate underprediction.',
           'Named markets use separate U.S. and non-U.S. panels. Limits are shared within dynamic Models 1/6 and within structural Models 2–5. The final strip plots every remaining evaluation destination, with a symmetric-log axis preserving zero errors; country identities are in the CSV.')]:
            charts.append(f'<figure><h4>{title}</h4><a href="4-hs-data-analysis/figures/model_{m:02d}_{suffix}.svg"><img src="4-hs-data-analysis/figures/model_{m:02d}_{suffix}.svg" alt="{alt}" loading="lazy"></a><figcaption>{cap} Predictions assume 1.3698 CAD/USD; actual values are nominal observed CAD. On narrow screens, scroll the figure horizontally or open its SVG.</figcaption></figure>')
        sections.append(f'<details class="model" id="model-{m:02d}"><summary><span class="model-number">{m:02d}</span><span>{escape(MODEL_NAMES[m])}</span><span class="model-tag">{"Dynamic" if m in [1,6] else "Structural"}</span></summary><div class="model-body">{keys[m]}{terminology}{narrative}<h3>Internal validation results</h3>{vtable}<h3>Alberta scoring assumptions</h3>{scoring}<h3>Frozen 2025 predictions and external comparison</h3>{metrics_html}{external}{table}{"".join(charts)}<h3>Top-five predicted HS4 products in each predicted destination</h3><p class="small">Destination and within-destination HS4 ranks were frozen in Phase A. Values are CAD million. Labels use exact-code CBSA T2026-2 heading text where available; later wording vintage is disclosed, and no tariff rule is inferred. BACI child fallbacks are labelled in the downloadable CSV. Actual 2025 product values were joined only afterward.</p>{product_tables(prod[prod.model==m])}<div class="teaches"><h3>What this model teaches us</h3>{teaches}</div></div></details>')
    refrows=[]
    for q in refs.itertuples():
        if 'wape' in q.checkpoint:derived=pct(q.derived);reference=pct(q.reference_rounded)
        elif 'cad' in q.checkpoint:derived='CAD '+money(q.derived)+' bn';reference='CAD '+money(q.reference_rounded)+' bn'
        else:derived=f'{q.derived:.6f}';reference=f'{q.reference_rounded:.6f}'
        refrows.append([escape(q.checkpoint.replace('_',' ')),derived,reference])
    reference_check=html_table(['Checkpoint','Independently derived','PDF rounded check'],refrows,'Model 6 reproduction · ex-post PDF currency convention, separate from primary results')
    nav='<nav class="collection-nav" aria-label="Collection navigation"><a href="harmonized-system-index.html">Collection Home</a><a href="harmonized-system-canada.html">Page 1</a><a href="section-338-hs4-hs6-exposure-canada.html">Page 2</a><a href="alberta-trade-by-hs.html">Page 3</a><a href="machine-learning-trade-sector-prediction.html" aria-current="page">Page 4</a><a href="../../research.html">Main Research Page</a></nav>'
    feature_rows=[['Economic size','log exporter/importer GDP','WDI NY.GDP.MKTP.CD · current USD'],
      ['Population','log exporter/importer population','WDI SP.POP.TOTL · persons'],
      ['Economic structure','exporter/importer manufacturing share','WDI NV.IND.MANF.ZS · % GDP'],
      ['Connectivity','exporter/importer internet use','WDI IT.NET.USER.ZS · % population'],
      ['Geography','log distance; border; official language','CEPII distw (km), contig, comlang_off'],
      ['Product supply','log1p external exporter HS4 supply','BACI outside seven-country internal network'],
      ['Product demand','log1p external importer HS4 demand','BACI outside seven-country internal network'],
      ['Product market','log1p outside-network/world HS4 demand','All BACI flows outside internal network'],
      ['Product identity','HS2 one-hot indicators from HS4','Categorical; no numeric HS-code magnitude']]
    feature_table=html_table(['Feature family','Primary inputs','Definition/source'],feature_rows,'Common explanatory information · Models 1–5 follow Model 6’s core economic family')
    obs_total=summ.observed_total_cad.iloc[0]
    fallback=AUDIT['fallback'];eligible=AUDIT['eligibility']['eligible']
    support=diag['scoring_support']
    support_gdp=100*support['log_importer_gdp']['outside_range_fraction']
    support_internet=100*support['importer_internet']['outside_range_fraction']
    intro=f'''<header class="hero"><div class="eyebrow">Study 04 / 04 · Reproducible research · 2025 external diagnostic</div><h1>Machine Learning for<br>Trade-Sector Prediction</h1><p class="lead">Six ways to model Alberta’s exports—and six reasons to question the result.</p><p class="hero-intro">Prediction error is evidence about one task. It is not proof that a model’s economic assumptions, statistical inference or commercial interpretation are sound.</p></header>
<section class="card introduction" aria-labelledby="question"><h2 id="question">Alberta’s export growth and diversification potential</h2><p>The Assistant Deputy Minister’s Office asks for Alberta’s top ten domestic-export destinations in 2025, the top five HS4 products in each, and a machine-learning outlook using 2024 bilateral HS4 merchandise flows among Canada, the United States, Mexico, China, Japan, South Korea and the United Kingdom. For prediction, Alberta is treated as a separate entity; comparable Canadian variables serve as explicit proxies when provincial variables are unavailable.</p><p>HS4 is the international four-digit product-heading level. HS2 through HS6 are internationally harmonized; Canadian HS8/HS10 extensions need not match another country’s national codes. This notebook preserves that assignment and compares six estimators at the bilateral product level before aggregating and ranking destinations.</p><div class="thesis"><p><strong>Better prediction ≠ stronger economic assumptions.</strong> A model with lower error can rely on implausible national-to-provincial scale transfer. A model with clearer margin separation or time ordering can still predict worse.</p></div><p><strong>Structural potential</strong> is the value implied by a contemporaneous feature relationship. A <strong>one-year prediction</strong> also needs a time-transition rule, implemented in Models 1 and 6. Neither is a proven <strong>commercial opportunity</strong>: none establishes causal effects, customers, profitability, tariff advantage, infrastructure capacity, demand certainty or guaranteed sales.</p></section>
<section class="card" id="data"><div class="eyebrow">Audit before estimation</div><h2>One data backbone, different mathematical assumptions</h2><p>All data were read from supplied local copies. BACI <strong>HS2022 V202601</strong> has 2022, 2023 and 2024 flows, measured in thousand current USD; <code>v × 1,000</code> is aggregated from six-character HS6 to four-character HS4. The product metadata yields <strong>{AUDIT['hs4_count']:,} headings</strong>. Seven exporters × six foreign partners = 42 directed pairs and <strong>{AUDIT['annual_grid_cells']:,} cells per year</strong>; valid absent flows are zero, never deleted to simplify logs.</p><p>WDI supplies exact-year 2022–2024 GDP, population, manufacturing and internet columns. GDP, population and internet are complete for the seven countries in 2023. Manufacturing is missing for Canada in 2023/2024 and the U.S. in 2022–2024. Each fit estimates medians from its own training subset and year. <strong>Imputation is a modelling assumption, not an observed manufacturing value.</strong> Model 1’s dynamic implementation is qualified accordingly. GeoDist is time invariant national geography.</p>{feature_table}<h3>External aggregates and leakage</h3>'''
    intro+=eq(r'\mathcal O_t=\{(a,b,k):\neg(a\in\mathcal N\ \land\ b\in\mathcal N)\},\quad S_{ik,t}=\sum_{(i,b,k)\in\mathcal O_t}V_{ibk,t},\quad Q_{jk,t}=\sum_{(a,j,k)\in\mathcal O_t}V_{ajk,t},\quad W_{k,t}=\sum_{(a,b,k)\in\mathcal O_t}V_{abk,t}')
    intro+=f'''<p>Here 𝒩 is the seven-country training network. Supply, demand and outside-network world demand exclude <em>every</em> internal network flow, so a training target cannot appear in its own product aggregate. Importer demand for a new outside-network destination includes its 2024 flows, including national Canadian flows; these are historical features, not Alberta 2025 targets, but they make out-of-network feature construction imperfectly comparable. We retain the PDF’s exclusion logic rather than redesigning Model 6.</p><p>The legal directory contains tariff-annex and guidance text captures, including 2026 material—not a clean pre-2025 country-pair legal-origin dataset. It is excluded from all primary models. Country identifiers join BACI numeric codes to ISO3 (WDI/GeoDist) and ISO2 (StatCan); obsolete duplicate records and Namibia’s missing ISO2 are recorded in the audit.</p><h3>Alberta is separate; many inputs are Canadian</h3><p>Alberta-specific information is its observed domestic destination–HS4 trade in 2023/2024; Models 1 and 6 use the 2024 lag for final scoring. Exporter GDP, population, manufacturing share, internet use, Canada-to-destination geography and external Canadian supply are national proxies. Calling Canadian GDP “Alberta GDP” would conceal a serious scale mismatch. Proxy features can suggest national manufacturing capacity Alberta does not possess; only the dynamic models preserve the province’s observed trade state.</p><h3>Eligibility, scope and currency</h3><p>Usable 2024 GDP, population, ISO identifiers and Canada-to-destination distance yield <strong>{eligible} eligible foreign destinations</strong>, derived rather than assumed. Every eligible destination is crossed with the fixed HS2022 headings. Known 2024 cells outside that grid ({fallback['cells']:,} cells; CAD {money(fallback['lag_cad'],1e6)} m) receive identical constant-USD persistence in all six models. This disclosed fallback retains national/special headings and unscorable markets without training on them. No observed 2025 market or product determines eligibility. Actual-only 2025 cells receive zero prediction in evaluation, rather than being silently dropped.</p><p><strong>Transfer is a major extrapolation problem:</strong> {support_gdp:.1f}% of eligible destinations have 2024 GDP outside the training-network range; {support_internet:.1f}% have internet use outside that range (missing values are recorded separately). Exporter-held-out validation among seven large economies does not test these smaller destination profiles. Identical feature definitions do not guarantee comparable feature support.</p><p>Statistics Canada’s domestic-export HS6 extracts are filtered to Alberta province of origin, summed over all twelve months and U.S. states, then grouped to HS4 and destination. Annual destination totals reconcile exactly with the separate HS2 source. Domestic exports cover goods grown, extracted, manufactured or materially transformed in Canada; Alberta identifies provincial origin rather than the customs crossing location. Their values are current CAD dollars, not BACI USD. <a href="https://www150.statcan.gc.ca/n1/pub/71-607-x/2021004/concepts-eng.htm">Statistics Canada?s concepts and valuation definitions</a> confirm this scope and currency. These provincial domestic exports differ conceptually from reconciled international merchandise flows.</p><p><strong>Primary CAD forecasts assume the known 2024 annual exchange rate, 1.3698 CAD/USD.</strong> Alberta 2024 lags divide by that rate; model fitting/dynamics use USD. Model 6’s PDF uses the realized 2025 rate 1.3978. That future rate appears only in a separately labelled ex-post reproduction diagnostic below, never in the primary frozen forecasts or fitting. Realized 2025 currency conversion is not a pre-2025 prediction. Annual rates: <a href="https://www.bankofcanada.ca/rates/exchange/annual-average-exchange-rates/">Bank of Canada</a>.</p><h3>A retrospective holdout, not a real-time vintage claim</h3><p>BACI V202601 was released in January 2026; this WDI copy was updated July 2026. Exact predictor <em>years</em> precede the evaluation year, but revisions and release delays mean this is a retrospective experiment. It does not reconstruct what an analyst could actually have known on January 1, 2025.</p><div class="holdout"><strong>Two enforced phases</strong><p><b>A — fit and freeze:</b> select hyperparameters inside permitted historical information, fit, score and save all bilateral predictions, destination ranks and top-five product ranks. Record timestamps and SHA-256 hashes.</p><p><b>B — external diagnostic:</b> only then read Alberta 2025 outcomes, aggregate observed results and measure disagreement. Verify prediction files remain unchanged. 2025 never selects features, λ, calibration, eligibility or rankings.</p></div><p class="small"><a href="4-hs-data-analysis/data_audit.md">Compact data audit</a> · <a href="4-hs-data-analysis/data_audit.json">Full audit and reconciliations</a> · <a href="4-hs-data-analysis/forecast_freeze.json">Freeze manifest</a> · <a href="4-hs-data-analysis/frozen_destination_universe.csv">Frozen destinations</a></p></section>
<section class="card" id="comparison"><div class="eyebrow">Compare the experiment, then the error</div><h2>Six-model overview</h2><p><a href="#observed">Actual top-ten destinations and their products</a> are reported separately below the model sections. Rows follow the progression from persistence and additive structure to flexible structural estimators and explicit partial adjustment. They are not ordered by WAPE and there is no “winner.” All 2025 figures are external diagnostics of frozen predictions.</p>{comparison}<p class="small">CAD bn = billions of Canadian dollars under the primary exchange-rate assumption. Models 2–5 estimate structural potential using 2024 information; Models 1 and 6 have explicit time mechanisms. Validation designs differ; do not read their internal scores as identical experiments.</p><h3>How to read the errors</h3>'''
    intro+=eq(r'\mathrm{MAE}=\frac1n\sum_n|\widehat V_n-V_n|,\quad \mathrm{WAPE}=\frac{\sum_n|\widehat V_n-V_n|}{\sum_n V_n},\quad \mathrm{RMSLE}=\sqrt{\frac1n\sum_n[\log(1+\widehat V_n)-\log(1+V_n)]^2}')
    intro+='''<p>Destination metrics first sum HS4 predictions. Spearman compares destination ranks; top-ten overlap counts shared destinations. A high rank correlation can coexist with badly understated dollar exports. WAPE is not bounded by 100%: forecasts far above the observed scale can legitimately produce larger percentages. Signed error is predicted minus actual; relative error divides by actual and is undefined at zero. Top-ten mean/median errors use the frozen predicted ten, not an outcome-selected list.</p><h3>Retransformation is part of the model</h3>'''
    intro+=eq(r'\widehat V_n^{(0)}=\max\{\exp[\widehat f(X_n)]-1,0\},\qquad c_{\mathrm{train}}=\frac{\sum_{n\in\mathrm{train}}V_n}{\sum_{n\in\mathrm{train}}\widehat V_n^{(0)}},\qquad \widehat V=c_{\mathrm{train}}\widehat V^{(0)}')
    intro+='''<p>Models 1–5 use this same aggregate training-only dollar calibration, separately estimated in every fold. It corrects the training aggregate, not each conditional mean; heteroskedastic log errors can still distort dollar predictions. It never rescales to Alberta’s 2025 total. Model 6 uses its PDF’s positive-only exponential calibration, then probability weighting and dynamic adjustment.</p></section>'''
    observed_us=float(observed.loc[observed.iso2=='US','actual_cad'].iloc[0])
    observed_crude=float(obsprod.loc[(obsprod.iso2=='US')&(obsprod.HS4=='2709'),'actual_cad'].iloc[0])
    observations=f'''<section class="card" id="observed"><div class="eyebrow">Observed · evaluation only</div><h2>Actual Alberta domestic exports in 2025</h2><p>Statistics Canada’s observed total is <strong>CAD {money(obs_total)} billion</strong>. The United States accounts for {100*observed_us/obs_total:.2f}% of the total; U.S.-bound crude petroleum (HS4 2709) alone accounts for {100*observed_crude/obs_total:.2f}%. This concentration explains both the separate U.S. chart panel and why dollar errors can be dominated by petroleum. Destinations include separately reported territorial markets such as Hong Kong. These rankings describe actual 2025 outcomes. They were constructed in Phase B and did not feed model fitting, scoring eligibility or predicted rankings.</p>{observation_table}<h3>Actual top-five HS4 products within each actual top-ten destination</h3><p class="small">CAD million; official heading text where exact code matches the local CBSA hierarchy. Later label vintage supplies wording only.</p>{product_tables(obsprod,True)}</section>'''
    refs_html=f'''<section class="card" id="reproduction"><h2>Model 6 reproduction checks</h2><p>The structural stages, λ estimation, temporal validation, preprocessing, product exclusions and default early stopping follow the local <code>machine_learning_model.pdf</code> and its supporting script. Checkpoints were computed independently; reference numbers were never assigned to fitted outputs.</p>{reference_check}<p>The PDF-currency total and WAPE apply 1.3978 CAD/USD to the already-frozen USD forecast. Primary results use 1.3698; the difference is a disclosed reporting-rate assumption, not a fit change. Small differences from the rounded PDF checkpoints can reflect rounding and software versions; material deviations require investigation. Full precision is saved in <a href="4-hs-data-analysis/model_06_reference_checks.csv">reference checks</a>.</p></section>
<section class="card" id="reproducibility"><h2>Reproduce and inspect</h2><p>The shared builder audits files, constructs features once, fits all models, freezes predictions, evaluates and publishes the notebook. Compact local caches avoid rereading large BACI files. Raw files are never copied into this repository. Cache files are ignored by git; run <code>--fresh</code> to rebuild them.</p><pre><code>python research/harmonized-system/4-hs-data-analysis/build_model_comparison.py
python -m http.server 8000</code></pre><p class="small">Override local roots with <code>TRADE_RAW_ROOT</code>, <code>TRADE_QUESTION1_REFERENCE</code> and <code>TRADE_TESTING_REFERENCE</code>. Analysis dependencies are documented in the script; the published page has no Python or npm runtime dependency.</p><div class="downloads"><a href="4-hs-data-analysis/file_inventory.md">Files and reproduction notes</a><a href="4-hs-data-analysis/build_model_comparison.py">Analysis and page builder</a><a href="4-hs-data-analysis/model_summary.csv">Six-model summary</a><a href="4-hs-data-analysis/destination_predictions.csv">All destination comparisons</a><a href="4-hs-data-analysis/hs4_predictions.csv">300 predicted product comparisons</a><a href="4-hs-data-analysis/frozen_hs4_predictions.csv.gz">Complete frozen HS4 predictions (gzip CSV)</a><a href="4-hs-data-analysis/validation_metrics.csv">Internal validation by exporter</a><a href="4-hs-data-analysis/fit_records.csv">Fit/calibration/iteration records</a><a href="4-hs-data-analysis/model_diagnostics.json">Model diagnostics</a><a href="4-hs-data-analysis/quality_checks.json">Quality checks and software versions</a></div></section>
<section class="card references" id="references"><h2>References and Data Sources</h2><p>Local copies supply the analysis. The links document their sources and estimators; no large dataset was downloaded for this task.</p><ul>
<li><a href="https://www.cepii.fr/CEPII/en/bdd_modele/bdd_modele_item.asp?id=37">CEPII BACI</a> — HS2022 V202601, annual bilateral trade. Local readme confirms thousand-USD units; Gaulier and Zignago (2010), CEPII Working Paper 2010-23.</li>
<li><a href="https://www.cepii.fr/CEPII/en/bdd_modele/presentation.asp?id=6">CEPII GeoDist</a> — population-weighted distance, contiguity and official language.</li>
<li><a href="https://databank.worldbank.org/source/world-development-indicators">World Bank World Development Indicators</a> — four exact indicator codes listed above; local 2022–2024 extract.</li>
<li><a href="https://open.canada.ca/data/dataset/2909a648-5753-4924-878a-b069392d9cde">Statistics Canada merchandise trade data</a> and <a href="https://www150.statcan.gc.ca/n1/pub/71-607-x/71-607-x2021004-eng.htm">Trade Data Explorer</a> — local provincial domestic-export files for 2023, 2024 and 2025.</li>
<li><a href="https://www.bankofcanada.ca/rates/exchange/annual-average-exchange-rates/">Bank of Canada annual exchange rates</a> — 2024 primary reporting assumption; 2025 ex-post PDF reproduction convention.</li>
<li><a href="https://www.cbsa-asfc.gc.ca/trade-commerce/tariff-tarif/2026/html/tblmod-2-eng.html">CBSA T2026-2 Customs Tariff hierarchy</a> — local exact-code HS4 labels only, not predictor or tariff assumptions.</li>
<li><a href="https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.LinearRegression.html">scikit-learn LinearRegression</a> and <a href="https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.ElasticNet.html">ElasticNet</a>; <a href="https://doi.org/10.1111/j.1467-9868.2005.00503.x">Zou and Hastie (2005), Regularization and Variable Selection via the Elastic Net</a>.</li>
<li><a href="https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.RandomForestRegressor.html">scikit-learn RandomForestRegressor</a>; <a href="https://doi.org/10.1023/A:1010933404324">Breiman (2001), Random Forests</a>.</li>
<li><a href="https://scikit-learn.org/stable/modules/generated/sklearn.neural_network.MLPRegressor.html">scikit-learn MLPRegressor</a> and <a href="https://scikit-learn.org/stable/modules/neural_networks_supervised.html">supervised neural network guide</a>.</li>
<li><a href="https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.HistGradientBoostingRegressor.html">HistGradientBoostingRegressor</a>, <a href="https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.HistGradientBoostingClassifier.html">HistGradientBoostingClassifier</a>, <a href="https://scikit-learn.org/stable/modules/ensemble.html#histogram-based-gradient-boosting">histogram boosting guide</a>; <a href="https://doi.org/10.1214/aos/1013203451">Friedman (2001), Greedy Function Approximation: A Gradient Boosting Machine</a>.</li>
<li><a href="https://scikit-learn.org/stable/modules/cross_validation.html#cross-validation-iterators-for-grouped-data">Grouped cross-validation</a> and <a href="https://scikit-learn.org/stable/modules/compose.html">preprocessing pipelines</a>; <a href="https://www.statsmodels.org/stable/generated/statsmodels.regression.linear_model.OLS.html">statsmodels OLS</a> for HC3 and dyad-cluster diagnostics.</li>
<li>Supplied methodological source: <code>testing_models/machine_learning_model.pdf</code>, 20 pages; supporting read-only <code>code/test_boosted_partial_adjustment_model.py</code>. Neither is redistributed or modified.</li>
</ul></section>'''
    synthesis=f'''<section class="card" id="interpretation"><h2>What the comparison tells us</h2><p><strong>More features need not add useful forecast information.</strong> Model 1?s full dynamic specification has exporter-transfer cell WAPE {pct(diag['model1']['aggregate_oof']['cell_wape'])}, compared with {pct(diag['model1']['lag_only_oof']['cell_wape'])} for its lag-only diagnostic. Observable fundamentals and HS2 controls add interpretation, but the single transition and dollar retransformation do not guarantee improved transfer.</p><p><strong>A good internal fit can coexist with weak validation.</strong> The bounded MLP?s random-cell stopping R? is {diag['model4']['best_internal_validation_r2']:.3f}, while its equal-exporter 2024 held-out destination WAPE is {pct(valid[valid.model==4].destination_wape.mean())}. Repeated national characteristics make random cells an easier test than an excluded exporter. Neither test certifies transfer to small outside-network destinations or Alberta?s provincial production mix.</p><p><strong>A clearer economic structure need not beat persistence.</strong> Model 6 separates entry and value and estimates a chronological adjustment mechanism. Yet its temporal mean exporter WAPE is {pct(valid[valid.model==6].destination_wape.mean())}, versus {pct(diag['temporal_persistence']['mean_exporter_destination_wape'])} for a constant-USD lag benchmark on the same test. That result remains visible instead of being tuned away.</p><p>Models 2?5 have no provincial capacity or adding-up constraint; national Canadian proxies can imply aggregate values far above Alberta?s observed scale. Their errors are evidence about those assumptions and transfer conditions, not simply evidence that a particular algorithm is inferior. All reported export estimates are point predictions; coefficient standard errors do not supply integrated forecast uncertainty or prove a feasible business opportunity.</p></section>'''
    html='''<!DOCTYPE html><html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width, initial-scale=1"><meta name="description" content="Six trade models, audited local data, frozen Alberta 2025 predictions and a critical comparison of predictive performance and economic assumptions."><title>Machine Learning for Trade-Sector Prediction | Naresh Neupane</title><link rel="stylesheet" href="4-hs-data-analysis/css/style.css"><link rel="stylesheet" href="4-hs-data-analysis/css/comparison.css"><script>window.MathJax={tex:{inlineMath:[['\\\\(','\\\\)']]},options:{enableMenu:false},startup:{typeset:true}};</script><script defer src="https://cdn.jsdelivr.net/npm/mathjax@3.2.2/es5/tex-chtml.js"></script><script defer src="4-hs-data-analysis/js/comparison.js"></script></head><body id="top"><a class="skip" href="#main">Skip to content</a><div class="page">'''+nav+'<main id="main">'+intro+'<div class="model-controls"><p>Explore the mathematical approaches</p><button type="button" id="expand-all">Expand all six</button><button type="button" id="collapse-all">Collapse all</button></div>'+''.join(sections)+synthesis+observations+refs_html+'</main><footer><a class="back-top" href="#top">Back to top ↑</a><p>Naresh Neupane · Independent research &amp; educational notes</p>'+nav+'</footer></div></body></html>'
    import re
    html=re.sub(r'(?=</?(?:section|details|summary|header|footer|nav|main|figure|table|thead|tbody|tr|caption|h[1-4]|div|p|ul|ol|li|dl|pre)(?:\s|>))', '\n', html)
    PAGE_PATH.write_text(format_presentation(html)+'\n',encoding='utf-8')

def main():
    if ARGS.presentation_only:
        render_saved_page()
        return
    for p in [OUTPUT_DIR,CACHE,OUTPUT_DIR/'figures']:p.mkdir(parents=True,exist_ok=True)
    before=protected_manifest()
    if ARGS.render_only:
        make_figures(pd.read_csv(OUTPUT_DIR/'destination_predictions.csv',dtype={'iso2':str},keep_default_na=False))
        render_page()
        write_audit()
        verify_publication()
        assert before==protected_manifest(),'Publication changed protected source/reference metadata'
        return
    audit_and_classification()
    if ARGS.audit_only:
        for year in [2022,2023,2024]:prepare_year(year)
        for year in [2023,2024]:load_alberta(year,'A')
        write_audit()
        assert before==protected_manifest()
        return
    with threadpool_limits(limits=4):
        frozen,dest,top=phase_a()
    result=phase_b(frozen,dest,top)
    make_figures(result[1])
    render_page()
    assert before==protected_manifest(),'Protected source/reference file metadata changed'
    html_checks=verify_publication()
    write_json('quality_checks.json',{'protected_directories_unchanged':True,'completed_utc':stamp(),
          'python':platform.python_version(),'numpy':np.__version__,'pandas':pd.__version__,
          'scipy':scipy.__version__,'sklearn':sklearn.__version__,
          'all_six_models':sorted(result[0].model.tolist()),'predictions_frozen_before_2025':True,**html_checks})
    print(result[0].to_string(index=False),flush=True)

def verify_publication():
    from html.parser import HTMLParser
    from urllib.parse import urlparse,unquote
    class PageParser(HTMLParser):
        def __init__(self):super().__init__();self.details=[];self.refs=[];self.figures=[];self.ids=[]
        def handle_starttag(self,tag,attrs):
            a=dict(attrs)
            if tag=='details':self.details.append(a)
            if 'id' in a:self.ids.append(a['id'])
            for k in ['src','href']:
                if k in a:self.refs.append(a[k])
            if tag=='img':self.figures.append(a)
    parser=PageParser();parser.feed(PAGE_PATH.read_text(encoding='utf-8'))
    assert len(parser.details)==6 and all('open' not in d for d in parser.details)
    assert len(set(parser.ids))==len(parser.ids)
    assert len(parser.figures)==18 and all(x.get('alt') for x in parser.figures)
    missing=[]
    for ref in parser.refs:
        parsed=urlparse(ref)
        if parsed.scheme or not parsed.path:continue
        p=PAGE_PATH.parent/unquote(parsed.path)
        # quality_checks is written immediately after this verifier.
        if p.name!='quality_checks.json' and not p.is_file():missing.append(str(p))
    assert not missing,missing
    summary=pd.read_csv(OUTPUT_DIR/'model_summary.csv')
    assert set(summary.model)==set(range(1,7))
    d=pd.read_csv(OUTPUT_DIR/'destination_predictions.csv',dtype={'iso2':str},keep_default_na=False)
    p=pd.read_csv(OUTPUT_DIR/'hs4_predictions.csv',dtype={'HS4':str,'iso2':str},keep_default_na=False)
    assert len(p)==300 and p.groupby('model').size().eq(50).all()
    assert p.HS4.str.fullmatch(r'\d{4}').all()
    assert p.HS4.str.startswith('0').any()
    assert all(d[d.model==m].predicted_cad.sum()==summary.loc[summary.model==m,'predicted_total_cad'].iloc[0] or
               np.isclose(d[d.model==m].predicted_cad.sum(),summary.loc[summary.model==m,'predicted_total_cad'].iloc[0],rtol=1e-12) for m in range(1,7))
    for model in range(1,7):
        ranks=pd.to_numeric(d[d.model==model].predicted_rank,errors='coerce')
        assert set(ranks[ranks<=10])==set(range(1,11))
    freeze=json.loads((OUTPUT_DIR/'forecast_freeze.json').read_text())
    audit=json.loads((OUTPUT_DIR/'data_audit.json').read_text(encoding='utf-8'))
    assert freeze['freeze_utc']<audit['2025_evaluation']['opened_utc']
    for name,digest in freeze['hashes'].items():assert hashlib.sha256((OUTPUT_DIR/name).read_bytes()).hexdigest()==digest
    return {'html_details':6,'initial_open_details':0,'figures':18,'all_relative_assets_exist':True,
       'hs4_leading_zeros_preserved':True,'product_comparisons':300,'frozen_files_unchanged':True}

if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--presentation-only', action='store_true')
    parser.add_argument('--audit-only',action='store_true');parser.add_argument('--render-only',action='store_true');parser.add_argument('--fresh',action='store_true')
    ARGS=parser.parse_args()
    main()
