"""Post-pricing checks and dependence-aware sensitivity; no refitting to evaluation prices."""
import json,argparse,re
import numpy as np
import pandas as pd
from bitcoin_reproduce import DATA_ROOT
from bitcoin_analysis import MODELS,metrics,paired,evaluate
from bitcoin_pricing import benchmark
from bitcoin_volatility import daily_estimators
from bitcoin_data import match_underlying

def block_uncertainty(e,replicates=1000,seed=20261017,block_days=7):
    if replicates<2 or block_days<1:raise ValueError('Invalid resampling controls')
    f=e.copy();f['date']=f.timestamp.dt.floor('D');days=pd.date_range(f.date.min(),f.date.max(),tz='UTC',freq='D')
    if f.date.nunique()<20:return dict(status='insufficient_dates')
    counts=f.groupby('date').size().reindex(days,fill_value=0).to_numpy()
    totals={m:abs(f[m]-f.observed_usd).groupby(f.date).sum().reindex(days,fill_value=0).to_numpy() for m in MODELS}
    rng=np.random.default_rng(seed);values={m:[] for m in MODELS};paired_values=[]
    for _ in range(replicates):
        starts=rng.integers(0,len(days),int(np.ceil(len(days)/block_days)))
        ix=((starts[:,None]+np.arange(block_days))%len(days)).ravel()[:len(days)];n=counts[ix].sum()
        for m in MODELS:values[m].append(float(totals[m][ix].sum()/n))
        paired_values.append(values['H30'][-1]-values['B'][-1])
    return dict(status='circular_moving_calendar_day_blocks',dates=int(f.date.nunique()),calendar_days=len(days),block_days=block_days,replicates=replicates,seed=seed,
        mae_intervals={m:np.quantile(values[m],[.025,.975]).tolist() for m in MODELS},paired_mean_interval=np.quantile(paired_values,[.025,.975]).tolist(),
        limitations='Seven-calendar-day blocks retain local dependence; circular boundaries and fixed trained sigma are approximations. Does not establish regime generality or account for arbitrary long-range dependence.')

def enrich(run_id="uwa-validated-20261017"):
    if not re.fullmatch(r"[A-Za-z0-9_-]+",run_id):raise ValueError("Invalid run identifier")
    private=DATA_ROOT/'private'/run_id;processed=DATA_ROOT/'processed'/run_id
    path=private/'bitcoin-results.json';report=json.loads(path.read_text())
    e=pd.read_csv(private/'evaluation-private.csv',parse_dates=['timestamp','expiration']);e.timestamp=pd.to_datetime(e.timestamp,utc=True,format="mixed")
    e=e.sort_values('timestamp').reset_index(drop=True)
    data=pd.read_csv(processed/'analytical-observations.csv.gz',parse_dates=['timestamp','expiration']);data.timestamp=pd.to_datetime(data.timestamp,utc=True,format="mixed")
    hraw=pd.read_csv(DATA_ROOT/'raw/Price_5m.csv',usecols=['time_m','price_avg'])
    h=pd.DataFrame({'timestamp':pd.to_datetime(hraw.time_m,utc=True),'spot':hraw.price_avg})
    completed=h.copy();completed.timestamp=completed.timestamp+pd.Timedelta(minutes=1)
    def completed_matches(frame):
        m=match_underlying(frame[['timestamp','record_key']],completed,max_lag='5min').set_index('record_key')
        if m.spot.isna().any():raise ValueError('New completed-minute match fails; re-establish common sample')
        for source,target in [('spot','matched_history_spot'),('underlying_at','underlying_at'),('lag_seconds','lag_seconds')]:
            frame[target]=frame.record_key.map(m[source])
        frame['history_source_at']=frame.underlying_at-pd.Timedelta(minutes=1)
        return frame
    e=completed_matches(e);data=completed_matches(data)
    # Only matching metadata/proxy changes; core index-price, rate and volatility inputs are unchanged.
    e.to_csv(private/'evaluation-private.csv',index=False)
    data.to_csv(processed/'analytical-observations.csv.gz',index=False,compression='gzip')
    report['matching_lag_seconds']=e.lag_seconds.quantile([0,.5,.95,1]).to_dict()
    report['underlying_basis']['median_absolute_relative_gap']=float(abs(e.spot/e.matched_history_spot-1).median())
    report['underlying_basis']['p95_absolute_relative_gap']=float(abs(e.spot/e.matched_history_spot-1).quantile(.95))
    report['minute_availability']='History minute labelled t is conservatively matched only at t+60 seconds. Daily 23:59 close becomes available next midnight. Co-recorded option index remains valuation spot.'
    report['block_uncertainty']=block_uncertainty(e)
    for m in ['H30','H60']:data['sigma_'+m]=data[m];data[m]=benchmark(data.spot,data.strike,data['T'],data.rate,data['sigma_'+m],0,data.kind)['usd']
    report['historical_all_eligible']={m:metrics(data,m) for m in ['H30','H60']}
    report['zero_quantity_audit']={'excluded_source_records':1,'rule':'amount > 0 for executed records; not a bid/ask size or trading-volume filter','first_run_note':'Initial private run retained one zero-amount source row; retained as audit baseline. Validated run excludes it before pricing.'}
    report['training_evaluation_record_overlap']=len(set(data.loc[data.timestamp<pd.Timestamp(report['cutoff_utc']),'record_key'])&set(e.record_key))
    # Demonstrate IV-missing records remain eligible for independent-volatility pricing.
    report['vendor_iv_missing_in_evaluation']=int(e.provider_iv.isna().sum())
    report['rate_input_range']=[float(e.rate.min()),float(e.rate.max())]
    # Return-window boundary sensitivity, still using only completed historical days.
    for shift in [-8,8]:
        alt=h.copy();alt.timestamp=alt.timestamp+pd.Timedelta(hours=shift);v=daily_estimators(alt);v.available_at=v.available_at-pd.Timedelta(hours=shift)
        matched=pd.merge_asof(e[['timestamp']].sort_values('timestamp'),v[['available_at','H30']],left_on='timestamp',right_on='available_at',direction='backward',tolerance=pd.Timedelta(days=1))
        e['boundary']=benchmark(e.spot,e.strike,e['T'],e.rate,matched.H30.to_numpy(),0,e.kind)['usd']
        report['sensitivities']['completed_day_boundary_shift_'+str(shift)+'h']=metrics(e,'boundary')
    e['proxy_spot']=benchmark(e.matched_history_spot,e.strike,e['T'],e.rate,e.sigma_H30,0,e.kind)['usd']
    report['sensitivities']['history_proxy_spot_H30']=metrics(e,'proxy_spot')
    path.write_text(json.dumps(report,indent=2,allow_nan=False),encoding='utf8')
    return report

if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument("--run-id",default="uwa-validated-20261017");args=parser.parse_args()
    r=enrich(args.run_id);print(json.dumps(dict(blocks=r['block_uncertainty'],all_eligible=r['historical_all_eligible'],sensitivities=r['sensitivities']),indent=2))
