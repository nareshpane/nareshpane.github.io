"""Export attributed, publication-reviewed aggregates and exactly twenty limited examples.
The original datasets and detailed analytical rows never enter public assets.
"""
import json,argparse,re
from pathlib import Path
import numpy as np
import pandas as pd
from bitcoin_reproduce import DATA_ROOT,PROJECT
from bitcoin_data import require_rights
from bitcoin_analysis import metrics,paired,MODELS

def clean(value):
    if isinstance(value,dict):return {str(k):clean(v) for k,v in value.items()}
    if isinstance(value,(list,tuple)):return [clean(v) for v in value]
    if isinstance(value,np.generic):value=value.item()
    if isinstance(value,float) and not np.isfinite(value):return None
    return value

def publish(run_id="uwa-validated-20261017"):
    if not re.fullmatch(r"[A-Za-z0-9_-]+",run_id):raise ValueError("Invalid run identifier")
    require_rights(json.loads((DATA_ROOT/'private/rights-review.json').read_text()),publication=True)
    private=DATA_ROOT/'private'/run_id;processed=DATA_ROOT/'processed'/run_id;asset=PROJECT
    report=json.loads((private/'bitcoin-results.json').read_text())
    audit=json.loads((DATA_ROOT/'processed/uwa-audit.json').read_text())
    inv=json.loads((DATA_ROOT/'processed/initial-inventory.json').read_text())
    e=pd.read_csv(private/'evaluation-private.csv',parse_dates=['timestamp','expiration'])
    e['timestamp']=pd.to_datetime(e.timestamp,utc=True,format="mixed");e['expiration']=pd.to_datetime(e.expiration,utc=True,format="mixed")
    vol=pd.read_csv(processed/'historical-volatility.csv.gz');vol.available_at=pd.to_datetime(vol.available_at,utc=True)
    output=dict(status='computed',citation='Hoang & Baur (2020), The University of Western Australia, doi:10.26182/5eabbcb4fbf28; original Deribit observations. Independent BSM analysis.',
        license='Official attachment CC BY label; exact version unspecified',audit=audit,inventory=inv,
        quality_flow=report['quality_flow'],counts=report['counts'],coverage=report['coverage'],evaluation_coverage=report['evaluation_coverage'],fit=report['fit'],metrics=report['metrics'],
        paired=report['paired'],breakdowns=report['breakdowns'],uncertainty=report['uncertainty'],sensitivities=report['sensitivities'],iv=report['iv'],underlying_basis=report.get('underlying_basis'),
        block_uncertainty=report['block_uncertainty'],historical_all_eligible=report['historical_all_eligible'],shared_contract_ids_across_time=report['shared_contract_ids_across_time'],rate_input_range=report['rate_input_range'],vendor_iv_missing_in_evaluation=report['vendor_iv_missing_in_evaluation'],
        sample_columns={'timestamp':'timestamp: Unix milliseconds -> UTC','contract':'instrument_name','kind':'type / identifier suffix','strike':'strike (USD)','expiration':'time_expire (UTC)',
            'spot':'index_price: co-recorded BTC/USD reference','observed_btc':'price (BTC per one BTC notional)','observed_usd':'price * index_price (equals source price_USD)',
            'H30':'Derived BSM USD with prior H30','H60':'Derived BSM USD with prior H60','B':'Derived BSM USD with earlier fitted sigma','H30_error':'Derived H30 minus observed USD'},
        groups={})
    # All source rows contribute to their groups; no arbitrary evaluation subsample.
    for side in ['all','call','put']:
      for term in ['all']+sorted(e.term.unique().tolist()):
        g=e[(e.kind==side) if side!='all' else np.ones(len(e),bool)]
        if term!='all':g=g[g.term==term]
        if g.empty:continue
        key=side+'|'+term;d=dict(n=len(g),dates=int(g.timestamp.dt.date.nunique()),metrics={m:metrics(g,m) for m in MODELS},paired={h:paired(g,h) for h in ['H30','H60']},scatter={},residual={},smile=[],money=[],maturity=[],side=[])
        # Observed-premium quantile bins; dots are conditional means, not individual trades.
        q=pd.qcut(g.observed_usd,30,duplicates='drop')
        for m in MODELS:
          points=[]
          for _,v in g.groupby(q,observed=True):
            if len(v)>=20:points.append(dict(market=float(v.observed_usd.mean()),model=float(v[m].mean()),n=len(v),bias=float((v[m]-v.observed_usd).mean())))
          d['scatter'][m]=points
        errs=np.concatenate([(g[m]-g.observed_usd).to_numpy() for m in MODELS]);edges=np.linspace(errs.min()-.01,errs.max()+.01,61)
        for m in MODELS:
          counts,_=np.histogram(g[m]-g.observed_usd,edges);d['residual'][m]=[dict(lo=float(lo),hi=float(hi),n=int(n)) for lo,hi,n in zip(edges[:-1],edges[1:],counts)]
        for dim,target in [('money','money'),('term','maturity'),('kind','side')]:
          for label,v in g.groupby(dim,observed=True):
            if len(v)>=20:d[target].append(dict(label=str(label),n=len(v),models={m:metrics(v,m) for m in MODELS}))
        for (label,kind),v in g[g.independent_iv.notna()].groupby(['money','kind'],observed=True):
          if len(v)>=20:d['smile'].append(dict(label=str(label),side=kind,n=len(v),m=float(v.m.median()),iv=float(v.independent_iv.median()),H30=float(v.sigma_H30.median()),H60=float(v.sigma_H60.median())))
        output['groups'][key]=d
    # The source ends intraday: do not label its last partial day as a completed close.
    last_available=pd.to_datetime(inv['Price_5m.csv']['max']['time_m'],utc=True)+pd.Timedelta(minutes=1)
    vol=vol[vol.available_at<=last_available]
    output['daily']=[dict(date=str(t.date()-pd.Timedelta(days=1)),price=float(s),H30=float(h30) if pd.notna(h30) else None,H60=float(h60) if pd.notna(h60) else None) for t,s,h30,h60 in zip(vol.available_at,vol.daily_close,vol.H30,vol.H60)]
    output['heatmap']=[]
    for (term,money),g in e.groupby(['term','money'],observed=True):
      if len(g)>=20:output['heatmap'].append(dict(term=str(term),money=str(money),n=len(g),models={m:metrics(g,m) for m in MODELS}))
    # Equal-side chronological quantiles, twenty permitted illustrative observations.
    examples=[]
    fields=['timestamp','contract','kind','strike','expiration','spot','observed_btc','observed_usd','H30','H60','B','H30_error']
    for side in ['call','put']:
      g=e[e.kind==side].sort_values(['timestamp','contract']);ix=np.linspace(0,len(g)-1,10,dtype=int)
      for row in g.iloc[ix][fields].to_dict('records'):
        row['timestamp']=str(row['timestamp']);row['expiration']=str(row['expiration']);examples.append(row)
    output['examples']=examples
    output=clean(output);target=asset/'data'
    (target/'bitcoin-results.json').write_text(json.dumps(output,separators=(',',':'),allow_nan=False),encoding='utf8')
    (target/'bitcoin-results.js').write_text('window.BITCOIN_RESULTS = '+json.dumps(output,separators=(',',':'),allow_nan=False)+';',encoding='utf8')
    print('Published',len(examples),'examples and aggregate groups for',len(e),'common held-out transactions')

if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--run-id',default='uwa-validated-20261017');args=parser.parse_args();publish(args.run_id)
