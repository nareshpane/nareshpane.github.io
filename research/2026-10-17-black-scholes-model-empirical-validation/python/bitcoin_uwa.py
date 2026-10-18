"""Actual UWA adapter: preserves sources, documents verified fields, reuses the tested workflow.
Run from any directory. All persistent outputs stay under project bitcoin_data.
"""
import json,argparse,re
from pathlib import Path
import numpy as np
import pandas as pd
from bitcoin_data import file_sha256
from bitcoin_reproduce import PROJECT,DATA_ROOT,run

EXPECTED={
 'instrument_name.csv':'a39733081781987aed135b619681064d8c62cc918e97679705ebe65fcc95ffb2',
 'Deribit_w_IV.csv':'47af56f73630b54a025b554088c03771404af15bbe9091dddee7d1c644c903b2',
 'Price_5m.csv':'694cba508d41f773af0c07d5090543436cf7cc09b45714345a097ff6a6a3572f'}

def source_inventory(raw,processed,x,catalog):
    def describe(name,frame,contract):
        fields=[c for c in ['timestamp','time_trade','time_expire','time_create','date'] if c in frame]
        return dict(bytes=(raw/name).stat().st_size,sha256=EXPECTED[name],encoding="utf-8",rows=len(frame),columns=list(frame.columns),
            missing_percent={c:float(frame[c].isna().mean()*100) for c in frame},types={c:[str(frame[c].dtype)] for c in frame},
            min={c:str(frame[c].min()) for c in fields},max={c:str(frame[c].max()) for c in fields},
            contracts=int(frame[contract].nunique()),dates=int(frame.date.nunique()) if 'date' in frame else None,
            sides={str(k):int(v) for k,v in frame.type.value_counts().items()})
    inventory={name:describe(name,f,k) for name,f,k in [('instrument_name.csv',catalog,'name'),('Deribit_w_IV.csv',x,'instrument_name')]}
    missing=None;types={};n=0;maxgap=0.;last=None;bad_cadence=0;first=latest=None;dayset=set()
    for c in pd.read_csv(raw/'Price_5m.csv',chunksize=200000,na_values=['.']):
        t=pd.to_datetime(c.time_m);n+=len(c);missing=c.isna().sum() if missing is None else missing+c.isna().sum()
        for key in c:types.setdefault(key,set()).add(str(c[key].dtype))
        if last is not None:bad_cadence+=int((t.iloc[0]-last).total_seconds()!=60)
        bad_cadence+=int((t.diff().dropna().dt.total_seconds()!=60).sum());last=t.iloc[-1]
        first=c.time_m.iloc[0] if first is None else first;latest=c.time_m.iloc[-1];dayset.update(t.dt.strftime('%Y-%m-%d'))
        mean=c.filter(regex='^price_(?!avg)').mean(axis=1);maxgap=max(maxgap,float(abs(mean-c.price_avg).max()))
    if bad_cadence or maxgap>1e-7:raise ValueError('Actual history definition/cadence changed')
    inventory['Price_5m.csv']=dict(bytes=(raw/'Price_5m.csv').stat().st_size,sha256=EXPECTED['Price_5m.csv'],encoding='utf-8',rows=n,columns=list(missing.index),
        missing_percent={k:float(v/n*100) for k,v in missing.items()},types={k:sorted(v) for k,v in types.items()},
        min={'time_m':first},max={'time_m':latest},dates=len(dayset),contracts=None,sides={})
    (processed/'initial-inventory.json').write_text(json.dumps(inventory,indent=2),encoding='utf8')
    (processed/'history-average-audit.json').write_text(json.dumps(dict(rows=n,max_absolute_difference=maxgap,non_minute_gaps=bad_cadence),indent=2))
    h=pd.read_csv(raw/'Price_5m.csv',usecols=['time_m','price_avg']);h['clock']=pd.to_datetime(h.time_m,utc=True)
    samples=x[['timestamp','index_price']].copy();samples['at']=pd.to_datetime(samples.timestamp,unit='ms',utc=True)
    samples=samples.groupby(samples['at'].dt.strftime('%Y-%m')).apply(lambda g:g.sort_values('timestamp').iloc[::max(1,len(g)//200)],include_groups=False).reset_index(drop=True).sort_values('at')
    diagnostics=[]
    for offset in [-8,0,8,12]:
        z=h[['clock','price_avg']].copy();z.clock=z.clock-pd.Timedelta(hours=offset)
        m=pd.merge_asof(samples,z,left_on='at',right_on='clock',direction='backward',tolerance=pd.Timedelta(minutes=5))
        gap=abs(np.log(m.index_price/m.price_avg));diagnostics.append(dict(assumed_clock_utc_offset_hours=offset,n=int(gap.notna().sum()),median_abs_log_gap=float(gap.median()),p95_abs_log_gap=float(gap.quantile(.95))))
    (processed/'clock-diagnostic.json').write_text(json.dumps(diagnostics,indent=2),encoding='utf8')


def prepare():
    raw=DATA_ROOT/'raw';processed=DATA_ROOT/'processed';private=DATA_ROOT/'private'
    for name,digest in EXPECTED.items():
        if file_sha256(raw/name)!=digest:raise ValueError('Source changed; review before using adapter: '+name)
    x=pd.read_csv(raw/'Deribit_w_IV.csv');catalog=pd.read_csv(raw/'instrument_name.csv')
    source_inventory(raw,processed,x,catalog)
    clock=pd.to_datetime(x.timestamp,unit='ms',utc=True);text=pd.to_datetime(x.time_trade,utc=True)
    assert ((clock-text).dt.total_seconds().between(0,.999001)).all()
    assert (clock.dt.strftime('%Y-%m-%d')==x.date).all()
    assert np.allclose(x.price*x.index_price,x.price_USD,rtol=0,atol=1e-7)
    names=x.instrument_name.str.rsplit('-',n=2)
    assert (names.str[1].astype(float)==x.strike).all()
    assert (x.instrument_name.str[-1].map({'C':'call','P':'put'})==x.type).all()
    expiry=pd.to_datetime(x.time_expire,utc=True)
    assert (expiry.dt.hour==8).all() and (expiry.dt.minute==0).all()
    assert not catalog.name.duplicated().any()
    indexed=catalog.set_index('name')
    assert set(x.instrument_name)<=set(indexed.index)
    assert (x.time_expire.to_numpy()==indexed.loc[x.instrument_name,'time_expire'].to_numpy()).all()
    assert (x.groupby('date').irate.nunique()==1).all()
    # irate is an exogenous annual fractional T-bill proxy supplied by the researchers.
    # Source availability statement identifies Yahoo ^IRX, NOT FRED DGS3MO.
    rates=x[['date','irate']].drop_duplicates().sort_values('date')
    rates['available_at']=(pd.to_datetime(rates.date,utc=True)+pd.Timedelta(days=1)).astype(str)
    rates['annual_percent']=rates.irate*100
    rates.to_csv(processed/'uwa_prior_rates.csv',index=False)
    audit=dict(raw_n=len(x),raw_dates=int(clock.dt.date.nunique()),raw_contracts=int(x.instrument_name.nunique()),
        calls=int((x.type=='call').sum()),puts=int((x.type=='put').sum()),
        earliest=str(clock.min()),latest=str(clock.max()),exact_duplicate_rows=int(x.duplicated().sum()),
        duplicate_business_keys=int(x.duplicated(['timestamp','instrument_name','price','amount','direction']).sum()),
        transaction_id='Absent; only exact source rows deduplicated, no independent trade ID invented',
        premium_evidence='Original BTC price and independent index_price present; price_USD equals their product on every source row. Paper identifies Deribit trade API provenance; iv_Tbill is separate and not an input.',
        conversion_max_abs_gap=float(abs(x.price*x.index_price-x.price_USD).max()),
        vendor_iv_missing=int(x.iv_Tbill.isna().sum()),
        tt_max_abs_year_gap=float(abs((expiry-clock).dt.total_seconds()/31536000-x.tt).max()),
        timestamp_text_subsecond_gap=[float((clock-text).dt.total_seconds().min()),float((clock-text).dt.total_seconds().max())],
        rate_source='Dataset irate; source paper Yahoo ^IRX; daily values used only from next UTC midnight. End-of-day availability is an assumption, not verified release vintages.',
        rate_dates=len(rates),yield_decimal_range=[float(rates.irate.min()),float(rates.irate.max())],
        cutoff='2019-01-01T00:00:00Z',cutoff_reason='Whole calendar-year boundary chosen before examining pricing errors; train earlier years, evaluate 2019 onward.',
        price_history_clock='UTC interpretation; timezone not encoded in CSV. Cross-checked against epoch-anchored trade index; see clock-diagnostic.json. Sensitivity tests alternative completed-day boundaries.',
        catalog_note='Some unused 2016 catalog entries expire at 15:00; all actually traded sample expirations are 08:00 UTC. No blanket catalog expiry override.')
    (processed/'uwa-audit.json').write_text(json.dumps(audit,indent=2),encoding='utf8')
    spec=dict(verified=True,evidence_reference='Dataset source time_expire/identifier reconciliation; Deribit 2019 European cash-settlement explanation and 2020 BTC intrinsic conversion. Prices per one BTC notional, not multiplied by trade amount.',
        exercise_style='European',settlement_currency='BTC',multiplier_btc=1,premium_currency='BTC',premium_per_contract=False,
        effective_start='2016-11-01T00:00:00Z',effective_end='2020-03-01T00:00:00Z',expiry_hour_utc=8)
    manifest=dict(data_class='market_observations',schema_evidence='Three manually downloaded official UWA CSVs: exact SHA-256 and source-column checks in uwa-audit.json',
        conversion_basis_evidence='All source price_USD = price * co-recorded index_price within 1e-7 USD; use this recorded index for valuation, not a different exchange spot.',
        rate_availability_evidence='Academic irate annual decimal ^IRX proxy shifted to next UTC midnight; semiannual-equivalent continuous approximation, not a bootstrapped zero curve. Independent FRED request failed WinError10061; not relabelled FRED.',
        carry_q=0,carry_evidence='Explicit zero-yield USD spot-BSM benchmark, no measured BTC funding/forward basis; not a Bitcoin dividend assumption.',
        training_cutoff_utc=audit['cutoff'],max_underlying_lag='5min',specifications=spec,
        options=dict(file='Deribit_w_IV.csv',timestamp='timestamp',timestamp_unit='ms',contract='instrument_name',premium='price',quantity='amount',embedded_spot='index_price',
            provider_iv='iv_Tbill',provider_iv_unit='percent',price_category='executed_trade',price_evidence=audit['premium_evidence'],exact_duplicate_fields=list(x.columns),dtypes={'timestamp':'float64'}),
        history=dict(file='Price_5m.csv',timestamp='time_m',timezone='UTC',price='price_avg',availability_delay_seconds=60,price_definition='Arithmetic mean of available named exchange BTC/USD series; one-minute cadence despite filename. Treat minute as available at its end for matching; UTC daily close becomes available next midnight. Separate from Deribit index.',evidence_reference='Actual headers, full one-minute cadence/average audit; paper identifies Bitcoincharts intraday provenance.'),
        rates=dict(file='uwa_prior_rates.csv',directory='processed',timestamp='available_at',available_at='available_at',observation_date='date',yield_='annual_percent'))
    manifest['rates']['yield']=manifest['rates'].pop('yield_')
    for key in ['options','history','rates']:
        base=processed if key=='rates' else raw;manifest[key]['sha256']=file_sha256(base/manifest[key]['file'])
    (processed/'uwa-manifest.json').write_text(json.dumps(manifest,indent=2),encoding='utf8')
    rights=json.loads((private/'rights-review.json').read_text(encoding='utf8'))
    return manifest,rights

if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--run-id',default='uwa-validated-20261017');args=parser.parse_args()
    if not re.fullmatch(r'[A-Za-z0-9_-]+',args.run_id):parser.error('Use a simple run identifier, no path separators')
    manifest,rights=prepare()
    result=run(manifest,rights,DATA_ROOT/'raw',DATA_ROOT/'private'/args.run_id,DATA_ROOT/'processed'/args.run_id)
    from bitcoin_diagnostics import enrich
    result=enrich(args.run_id)
    print(json.dumps(dict(counts=result['counts'],fit=result['fit'],metrics=result['metrics'],paired=result['paired']),indent=2))
