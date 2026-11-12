"""Normalized-data analysis; UWA field mapping is supplied by bitcoin_uwa.py.

Source columns, price semantics and historical specifications must be verified
before an adapter is used on market data. No exchange-mark or IV-derived target.
"""
import numpy as np
import pandas as pd
from scipy.optimize import minimize_scalar
from bitcoin_data import parse_instrument,utc_times
from bitcoin_pricing import premium_usd,benchmark,market_iv,rate_continuous

MODELS=('H30','H60','B')

def normalize_chunk(chunk:pd.DataFrame,mapping:dict,spec:dict,source:str,offset:int=0)->pd.DataFrame:
    if spec.get('verified') is not True or not spec.get('evidence_reference') or spec.get('exercise_style')!='European' or spec.get('settlement_currency')!='BTC':
        raise ValueError('Verified historical European inverse-BTC specifications required')
    for key in ['multiplier_btc','premium_currency','premium_per_contract','effective_start','effective_end','expiry_hour_utc']:
        if key not in spec:raise ValueError('Missing verified specification: '+key)
    if mapping.get('price_category') not in ['executed_trade','bid_ask_midpoint'] or not mapping.get('price_evidence'):
        raise ValueError('Independent observed premium and its semantics must be documented')
    required=['timestamp','contract']+(['premium'] if mapping['price_category']=='executed_trade' else ['bid','ask'])
    if any(mapping.get(k) not in chunk for k in required):raise ValueError('Source column mapping does not match this CSV')
    f=pd.DataFrame(index=chunk.index)
    f['timestamp']=utc_times(chunk[mapping['timestamp']],source_timezone=mapping.get('timezone'),unit=mapping.get('timestamp_unit'))
    f['contract']=chunk[mapping['contract']].astype('string');cache={}
    for symbol in f.contract.dropna().unique():
        try:cache[symbol]=parse_instrument(symbol,expiry_hour_utc=int(spec['expiry_hour_utc']))
        except ValueError:cache[symbol]=None
    f['strike']=f.contract.map(lambda x:cache[x].strike if cache.get(x) else np.nan)
    f['kind']=f.contract.map(lambda x:cache[x].kind if cache.get(x) else None)
    f['expiration']=pd.to_datetime(f.contract.map(lambda x:cache[x].expiration if cache.get(x) else pd.NaT),utc=True)
    f['symbol_valid']=f.strike.notna()
    f['spec_valid']=f.timestamp.between(pd.Timestamp(spec['effective_start']),pd.Timestamp(spec['effective_end']))
    if mapping['price_category']=='executed_trade':f['original_premium']=pd.to_numeric(chunk[mapping['premium']],errors='coerce')
    else:
        bid=pd.to_numeric(chunk[mapping['bid']],errors='coerce');ask=pd.to_numeric(chunk[mapping['ask']],errors='coerce')
        f['original_premium']=((bid+ask)/2).where((bid>=0)&(ask>=bid))
    if mapping.get('quotes_synchronized') is True and mapping.get('bid') in chunk and mapping.get('ask') in chunk:
        f['original_bid']=pd.to_numeric(chunk[mapping['bid']],errors='coerce');f['original_ask']=pd.to_numeric(chunk[mapping['ask']],errors='coerce')
    if mapping.get('quantity') in chunk:f['quantity']=pd.to_numeric(chunk[mapping['quantity']],errors='coerce')
    if mapping.get('provider_iv') in chunk:
        f['provider_iv_raw']=pd.to_numeric(chunk[mapping['provider_iv']],errors='coerce')
        if mapping.get('provider_iv_unit') in ['decimal','percent']:
            f['provider_iv']=f.provider_iv_raw/(100 if mapping['provider_iv_unit']=='percent' else 1)
    f['source_record']=np.arange(offset+1,offset+len(f)+1);f['source_file']=source
    f['record_key']=source+':'+f.source_record.astype(str)
    if mapping.get('trade_id') in chunk:f['trade_id']=chunk[mapping['trade_id']].astype('string')
    f['price_category']=mapping['price_category'];f['currency']=spec['premium_currency']
    if mapping.get('embedded_spot') in chunk:
        f['source_spot']=pd.to_numeric(chunk[mapping['embedded_spot']],errors='coerce')
    return f

def match_rates(frame:pd.DataFrame,rates:pd.DataFrame,max_age='7D')->pd.DataFrame:
    """Availability time is required, not inferred from observation date."""
    r=rates.copy().sort_values('available_at')
    if not isinstance(r.available_at.dtype,pd.DatetimeTZDtype) or r.available_at.isna().any() or r.available_at.duplicated().any():raise ValueError('Verified unique rate availability timestamps required')
    r=r.dropna(subset=['annual_percent'])
    r['rate']=rate_continuous(r.annual_percent)
    return pd.merge_asof(frame.sort_values('timestamp'),r,left_on='timestamp',right_on='available_at',direction='backward',
        allow_exact_matches=True,tolerance=pd.Timedelta(max_age),suffixes=('','_rate'))

def quality_filters(frame:pd.DataFrame,*,money=(.8,1.2),days=(7,90))->tuple[pd.DataFrame,list]:
    f=frame.copy();f['T']=(f.expiration-f.timestamp).dt.total_seconds()/(365*86400);f['m']=f.strike/f.spot
    conditions=[('valid_symbol_and_spec',f.symbol_valid&f.spec_valid),('valid_prior_underlying',np.isfinite(f.spot)&(f.spot>0)&f.underlying_at.notna()&(f.underlying_at<=f.timestamp)),
        ('positive_observed_premium',np.isfinite(f.original_premium)&(f.original_premium>0)),('positive_strike',np.isfinite(f.strike)&(f.strike>0)),
        ('maturity',f['T'].between(days[0]/365,days[1]/365)),('moneyness',f.m.between(*money)),
        ('independent_volatility',np.isfinite(f.H30)&np.isfinite(f.H60)&(f.H30>=0)&(f.H60>=0)),('known_rate',np.isfinite(f.rate))]
    if 'quantity' in f and 'price_category' in f:
        conditions.insert(1,('positive_executed_quantity',(f.price_category!='executed_trade')|(np.isfinite(f.quantity)&(f.quantity>0))))
    counts=[]
    for label,mask in conditions:
        before=len(f);f=f[mask.reindex(f.index)].copy();counts.append(dict(stage=label,before=before,removed=before-len(f),remaining=len(f)))
    return f,counts

def chronological_split(frame:pd.DataFrame,cutoff)->tuple[pd.DataFrame,pd.DataFrame]:
    cutoff=pd.Timestamp(cutoff)
    if cutoff.tzinfo is None:raise ValueError('UTC-aware cutoff required')
    if frame.record_key.duplicated().any():raise ValueError('Duplicate provenance identifiers')
    train=frame[frame.timestamp<cutoff].copy();evaluation=frame[frame.timestamp>=cutoff].copy()
    if train.empty or evaluation.empty:raise ValueError('Nonempty chronological train and evaluation periods required')
    if set(train.record_key)&set(evaluation.record_key) or train.timestamp.max()>=evaluation.timestamp.min():raise AssertionError('Temporal leakage')
    if 'trade_id' in frame and set(train.trade_id.dropna())&set(evaluation.trade_id.dropna()):raise ValueError('Duplicate trade ID crosses split')
    return train,evaluation

def calibrate(train:pd.DataFrame,*,carry:float=0.,min_observations:int=20,bounds=(.01,4.))->dict:
    if len(train)<min_observations or train.contract.nunique()<3:raise ValueError('Insufficient training observations or contracts')
    fit=minimize_scalar(lambda sigma:np.mean((benchmark(train.spot,train.strike,train['T'],train.rate,sigma,carry,train.kind)['usd']-train.observed_usd)**2),
        bounds=bounds,method='bounded',options={'xatol':1e-9})
    if not fit.success:raise ValueError('Calibration failed')
    return dict(sigma=float(fit.x),training_n=len(train),training_contracts=train.contract.nunique(),training_dates=train.timestamp.dt.date.nunique(),
        training_mse=float(fit.fun),bounds=list(bounds),near_bound=bool(min(fit.x-bounds[0],bounds[1]-fit.x)<1e-5))

def evaluate(frame:pd.DataFrame,fitted_sigma:float,*,carry:float=0.)->pd.DataFrame:
    f=frame.copy()
    f['sigma_H30']=f.H30;f['sigma_H60']=f.H60;f['sigma_B']=fitted_sigma
    for name,sigma in [('H30',f.sigma_H30),('H60',f.sigma_H60),('B',fitted_sigma)]:
        values=benchmark(f.spot,f.strike,f['T'],f.rate,sigma,carry,f.kind)
        f[name]=values['usd'];f[name+'_btc']=values['btc'];f[name+'_error']=f[name]-f.observed_usd
    # A shared complete-case sample, never per-model silent drops.
    if not np.isfinite(f[list(MODELS)]).all().all():raise ValueError('Nonfinite model values; establish and disclose a common sample before comparing')
    f['independent_iv']=market_iv(f.observed_usd,f.spot,f.strike,f['T'],f.rate,carry,f.kind)
    f['term']=pd.cut(f['T']*365,[6.999999,21,45,90],include_lowest=True)
    f['money']=pd.cut(f.m,[.799999,.9,.97,1.03,1.1,1.200001],include_lowest=True)
    return f

def metrics(frame:pd.DataFrame,model:str,*,relative_floor_usd:float=5.)->dict:
    if not len(frame):raise ValueError('No observations; metrics are unavailable, not zero')
    e=frame[model]-frame.observed_usd;a=abs(e);eligible=frame.observed_usd>=relative_floor_usd
    out=dict(n=len(frame),dates=frame.timestamp.dt.date.nunique(),contracts=frame.contract.nunique(),calls=int((frame.kind=='call').sum()),puts=int((frame.kind=='put').sum()),
        mae=float(a.mean()),median_absolute=float(a.median()),rmse=float(np.sqrt(np.mean(e**2))),bias=float(e.mean()),median_signed=float(e.median()),
        spot_bps=float(np.mean(a/frame.spot)*10000),mae_btc=float(np.mean(a/frame.spot)),relative_price_n=int(eligible.sum()),relative_price_floor_usd=relative_floor_usd,
        relative_price_error=float((a[eligible]/frame.loc[eligible,'observed_usd']).mean()) if eligible.any() else None,
        within_5_usd=float(np.mean(a<=5)),within_1_spot_bp=float(np.mean(a/frame.spot<=.0001)))
    if {'bid_usd','ask_usd'}<=set(frame):
        ok=frame.bid_usd.notna()&frame.ask_usd.notna()&(frame.bid_usd>=0)&(frame.ask_usd>=frame.bid_usd)
        out.update(spread_n=int(ok.sum()),within_spread=float(np.mean((frame.loc[ok,model]>=frame.loc[ok,'bid_usd'])&(frame.loc[ok,model]<=frame.loc[ok,'ask_usd']))) if ok.any() else None)
    return out

def paired(frame:pd.DataFrame,historical:str='H30')->dict:
    delta=abs(frame[historical]-frame.observed_usd)-abs(frame.B-frame.observed_usd)
    base=metrics(frame,historical)['mae'];candidate=metrics(frame,'B')['mae']
    per_contract=frame.assign(delta=delta).groupby('contract').delta.mean()
    return dict(n=len(frame),mean=float(delta.mean()),median=float(delta.median()),improved_observations_percent=float(np.mean(delta>1e-10)*100),
        improved_contracts_percent=float(np.mean(per_contract>1e-10)*100),aggregate_mae_reduction_percent=100*(base-candidate)/base if base>0 else None,
        direction='Historical absolute error minus calibrated absolute error; positive favors calibration')

def date_cluster(frame:pd.DataFrame,*,replicates:int=1000,seed:int=20261017,min_dates:int=20)->dict:
    if replicates<2:raise ValueError('At least two replicates')
    f=frame.assign(day=frame.timestamp.dt.strftime('%Y-%m-%d'));days=sorted(f.day.unique())
    if len(days)<min_dates:return dict(status='insufficient_distinct_dates',dates=len(days),minimum=min_dates,replicates=0)
    grouped=f.groupby('day');counts=grouped.size().reindex(days).to_numpy();totals={m:grouped.apply(lambda g:float(abs(g[m]-g.observed_usd).sum()),include_groups=False).reindex(days).to_numpy() for m in MODELS}
    rng=np.random.default_rng(seed);values={m:[] for m in MODELS};gains=[]
    for _ in range(replicates):
        ix=rng.integers(0,len(days),len(days));denom=counts[ix].sum()
        for m in MODELS:values[m].append(float(totals[m][ix].sum()/denom))
        gains.append(values['H30'][-1]-values['B'][-1])
    return dict(status='conditional_date_cluster_percentile',dates=len(days),replicates=replicates,seed=seed,
        mae_intervals={m:np.quantile(v,[.025,.975]).tolist() for m,v in values.items()},paired_mean_interval=np.quantile(gains,[.025,.975]).tolist(),
        limitations='Whole UTC dates resampled; trade-weighted errors. Fixed trained sigma, no refitting. Does not account for cross-date serial dependence; enough dates is not proof of independent clusters.')
