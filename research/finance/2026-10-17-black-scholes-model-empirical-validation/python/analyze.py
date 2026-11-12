"""Read user-downloaded inputs; export aggregates only. No raw quotes are written."""
import argparse, json, hashlib
from pathlib import Path
import numpy as np
import pandas as pd
from scipy.optimize import minimize_scalar
from pricing import price, implied

def metrics(d, pred):
    e=np.asarray(d[pred]-d.mid); a=np.abs(e)
    return dict(n=len(d),dates=d.quote_date.nunique(),mae=float(a.mean()),median=float(np.median(a)),rmse=float(np.sqrt(np.mean(e*e))),bias=float(e.mean()),
        spot_bps=float(np.mean(a/d.S)*10000),spread_units=float(np.mean(a/(d.ask-d.bid))),within=float(np.mean((d[pred]>=d.bid)&(d[pred]<=d.ask))))

def quote_mask(d, threshold=.10):
    return np.isfinite(d.bid)&np.isfinite(d.ask)&(d.bid>0)&(d.ask>d.bid)&(d.mid>=5)&((d.ask-d.bid)/d.mid<=threshold)&(d.bid_size>=1)&(d.ask_size>=1)

def records(d): return json.loads(d.to_json(orient='records',double_precision=6))

def run(args):
    root=Path(args.source); manifest=json.loads((root/'day_by_date/manifest.json').read_text())
    sp=pd.read_csv(args.spx); sp.columns=sp.columns.str.lower(); sp['date']=pd.to_datetime(sp.date).dt.strftime('%Y-%m-%d'); sp=sp.drop_duplicates('date').set_index('date').sort_index()
    close=sp.spx.astype(float); returns=np.log(close/close.shift(1))
    vol={n:returns.rolling(n).std(ddof=1).shift(1)*np.sqrt(252) for n in [20,30]}
    fred=pd.read_csv(args.fred); fred.columns=['date','yield']; fred['yield']=pd.to_numeric(fred['yield'],errors='coerce'); fred=fred.set_index('date')['yield']
    frames=[]; counts={}; discrepancies=[]; rate_diff=[]; total=0; rawspx=0
    cols=['contract','underlying','expiration','type','strike','style','quote_date','bid','ask','bid_size','ask_size','underlying_close','settlement_time']
    for file in sorted((root/'day_by_date').glob('*_options.csv')):
        d=pd.read_csv(file,usecols=cols); total+=len(d)
        day=file.name[:10]; reference=float(close.loc[day]); sx=d[d.underlying=='SPX']; rawspx+=int(abs(sx.underlying_close.iloc[0]-reference)>.005)
        d=d[(d.underlying=='SPXW')&(d.style=='E')&(d.settlement_time=='PM')].copy()
        if len(d) and abs(d.underlying_close.iloc[0]-reference)>.005: discrepancies.append(dict(date=day,difference=float(d.underlying_close.iloc[0]-reference)))
        d['S']=reference; d['hv20']=vol[20].loc[day]; d['hv30']=vol[30].loc[day]
        d['days']=(pd.to_datetime(d.expiration)-pd.Timestamp(day)).dt.days; d['T']=d.days/365; d['m']=d.strike/d.S
        d['mid']=(d.bid+d.ask)/2
        curve=manifest['rates'][day]
        # README tenor order: 1m,3m,1y,2y,5y,10y,30y. Flat below 1m.
        tenors=np.array([1/12,.25,1,2,5,10,30]); valid=np.array([x is not None for x in curve]); ys=np.array([np.nan if x is None else x for x in curve])/100
        y=np.interp(d['T'],tenors[valid],ys[valid]); d['r']=2*np.log1p(y/2) # bond-equivalent approximation
        if day in fred.index and pd.notna(fred.loc[day]): rate_diff.append(float(curve[1]-fred.loc[day]))
        frames.append(d)
    d=pd.concat(frames,ignore_index=True); counts['All sample rows']=total; counts['SPXW / E / PM']=len(d)
    d=d[d.days.between(7,90)]; counts['7–90 calendar days']=len(d)
    d=d[d.m.between(.9,1.1)]; counts['Strike / spot 0.90–1.10']=len(d)
    # Preserve broad pre-quality population for prespecified sensitivity reruns.
    base=d.copy()
    for name,mask in [('Finite bid and ask',lambda g:np.isfinite(g.bid)&np.isfinite(g.ask)),('Both quote sides positive',lambda g:(g.bid>0)&(g.ask>0)),('Ask strictly above bid',lambda g:g.ask>g.bid),('Both displayed sizes ≥ 1',lambda g:(g.bid_size>=1)&(g.ask_size>=1)),('Midpoint ≥ 5 points',lambda g:g.mid>=5),('Relative spread ≤ 10%',lambda g:(g.ask-g.bid)/g.mid<=.10)]:
        d=d[mask(d)]; counts[name]=len(d)
    d=d[np.isfinite(d.hv20)&np.isfinite(d.hv30)]; counts['Prior 20 / 30 return windows']=len(d)
    output=[]; calibration=[]; parity=[]; skipped={'insufficient_pairs':0,'carry_bounds':0,'no_evaluation':0}
    # Strike partition is stable across dates; both sides of a strike share a partition.
    d['train']=(np.rint(d.strike/5).astype(int)%2==0)
    assert not d.duplicated(['contract','quote_date']).any(), 'Duplicate contract-day observations'
    assert not set(d[d.train].contract)&set(d[~d.train].contract), 'Contract leakage'
    counts['Odd-strike evaluation partition']=int((~d.train).sum())
    for (day,expiry),g in d.groupby(['quote_date','expiration']):
        tr=g[g.train & g.m.between(.97,1.03)]
        pairs=tr.pivot_table(index='strike',columns='type',values='mid',aggfunc='first').dropna()
        if not {'call','put'}.issubset(pairs.columns) or len(pairs)<3 or len(tr)<6:
            skipped['insufficient_pairs']+=1; continue
        S,T,r=g[['S','T','r']].iloc[0]; forwards=pairs.index+np.exp(r*T)*(pairs['call']-pairs['put'])
        F=float(np.median(forwards)); q=float(r-np.log(F/S)/T)
        if not (-.1<q<.15):
            skipped['carry_bounds']+=1; continue
        fit=minimize_scalar(lambda v:np.mean((price(tr.S,tr.strike,tr['T'],tr.r,v,q,tr.type)-tr.mid)**2),bounds=(.03,1.5),method='bounded',options={'xatol':1e-9})
        ev=g[~g.train].copy()
        if not len(ev):
            skipped['no_evaluation']+=1; continue
        ev['q']=q; ev['fitvol']=fit.x
        for name,sig,div in [('A',ev.hv20,q),('A30',ev.hv30,q),('B',fit.x,q),('qminus',ev.hv20,q-.01),('qplus',ev.hv20,q+.01),('qfixed',ev.hv20,.015)]:
            ev[name]=price(ev.S,ev.strike,ev['T'],ev.r,sig,div,ev.type)
        ev['iv']=implied(ev.mid,ev.S,ev.strike,ev['T'],ev.r,q,ev.type)
        ep=ev.pivot_table(index='strike',columns='type',values='mid',aggfunc='first').dropna()
        if {'call','put'}.issubset(ep.columns):
            parity.extend((ep['call']-ep['put']-(S*np.exp(-q*T)-ep.index*np.exp(-r*T))).tolist())
        calibration.append(dict(date=day,expiry=expiry,q=q,vol=fit.x,ntrain=len(tr),npairs=len(pairs)))
        output.append(ev)
    e=pd.concat(output,ignore_index=True); counts['Held-out quotes with valid training group']=len(e)
    e['moneyness']=pd.cut(e.m,[.9,.95,.98,1.02,1.05,1.100001],labels=['0.90–0.95','0.95–0.98','0.98–1.02','1.02–1.05','1.05–1.10'],include_lowest=True)
    e['maturity']=pd.cut(e.days,[6,21,45,90],labels=['7–21','22–45','46–90'])
    # Optional publication supplement: same evaluation frame; never rewrites validated results.
    if getattr(args,'supplement_output',None):
        from revision_outputs import publish
        publish(e,root,Path(args.supplement_output))
        return
    result=dict(status='computed',source_period=['2022-07-01','2022-12-30'],coverage=[e.quote_date.min(),e.quote_date.max()],counts=counts,
        contracts=e.contract.nunique(),dates=e.quote_date.nunique(),metrics={k:metrics(e,k) for k in ['A','A30','B','qminus','qplus','qfixed']},
        provenance=dict(zip_sha256=hashlib.sha256(Path(args.archive).read_bytes()).hexdigest(),manifest_sha256=hashlib.sha256((root/'day_by_date/manifest.json').read_bytes()).hexdigest(),
            spx_disagreements=rawspx,spxw_discrepancies=discrepancies,fred_comparisons=len(rate_diff),fred_max_abs_difference=max(map(abs,rate_diff)),rates_carried_forward=manifest['rates_carried_forward']),
        calibration=dict(groups=len(calibration),training_quotes=sum(x['ntrain'] for x in calibration),skipped=skipped,q_quantiles=np.quantile([x['q'] for x in calibration],[0,.25,.5,.75,1]).tolist()),
        parity=dict(n=len(parity),mae=float(np.mean(np.abs(parity))),median=float(np.median(parity))))
    result['breakdowns']={}
    for dim in ['type','moneyness','maturity','quote_date']:
        result['breakdowns'][dim]=[{ 'group':str(key),**{k:metrics(g,k) for k in ['A','B']}} for key,g in e.groupby(dim,observed=True)]
    result['heatmap']=[dict(m=str(m),t=str(t),**{k:metrics(g,k) for k in ['A','B']}) for (m,t),g in e.groupby(['moneyness','maturity'],observed=True)]
    # Publish grouped price comparisons, never a downloadable quote panel.
    e['pricebin']=(e.mid//25).astype(int)*25
    result['scatter']=[dict(bin=int(b),n=len(g),mid=float(g.mid.mean()),A=float(g.A.mean()),B=float(g.B.mean())) for b,g in e.groupby('pricebin') if len(g)>=30]
    result['histogram']={k:dict(zip(['counts','edges'],[x.tolist() for x in np.histogram(e[k]-e.mid,bins=np.arange(-150,155,5))])) for k in ['A','B']}
    result['smile']=[dict(m=str(m),t=str(t),n=int(g.iv.notna().sum()),iv=float(g.iv.median())) for (m,t),g in e.groupby(['moneyness','maturity'],observed=True) if g.iv.notna().sum()>=30]
    result['iv_invalid']=int(e.iv.isna().sum())
    result['volatility']=[dict(date=day,hv20=float(vol[20].loc[day]),hv30=float(vol[30].loc[day])) for day in sorted(e.quote_date.unique())]
    rng=np.random.default_rng(20261017); daily=e.groupby('quote_date').apply(lambda g:pd.Series({'n':len(g),'a':abs(g.A-g.mid).sum(),'b':abs(g.B-g.mid).sum()}),include_groups=False)
    boots=[]
    for _ in range(1000):
        sample=daily.iloc[rng.integers(0,len(daily),len(daily))].sum(); boots.append([sample.a/sample.n,sample.b/sample.n])
    result['bootstrap']=dict(replicates=1000,seed=20261017,A=np.quantile(np.array(boots)[:,0],[.025,.975]).tolist(),B=np.quantile(np.array(boots)[:,1],[.025,.975]).tolist())
    result['stratified_intervals']=[]
    for (kind,term),g in e.groupby(['type','maturity'],observed=True):
        day=g.groupby('quote_date').apply(lambda z:pd.Series({'n':len(z),'a':abs(z.A-z.mid).sum(),'b':abs(z.B-z.mid).sum()}),include_groups=False)
        boot=[]
        for _ in range(1000):
            s=day.iloc[rng.integers(0,len(day),len(day))].sum();boot.append([s.a/s.n,s.b/s.n])
        result['stratified_intervals'].append(dict(kind=str(kind),term=str(term),n=len(g),dates=len(day),A=metrics(g,'A')['mae'],B=metrics(g,'B')['mae'],A_ci=np.quantile(np.array(boot)[:,0],[.025,.975]).tolist(),B_ci=np.quantile(np.array(boot)[:,1],[.025,.975]).tolist()))
    result['quality_sensitivity']=[dict(threshold=x,n=int(quote_mask(base,x).sum()),evaluated=int(((e.ask-e.bid)/e.mid<=x).sum()),A=metrics(e[(e.ask-e.bid)/e.mid<=x],'A'),B=metrics(e[(e.ask-e.bid)/e.mid<=x],'B')) for x in [.02,.05,.10]]
    out=Path(args.output); out.mkdir(parents=True,exist_ok=True)
    (out/'results.json').write_text(json.dumps(result,ensure_ascii=False,indent=2,allow_nan=False),encoding='utf-8')
    (out/'results.js').write_text('window.EMPIRICAL = '+json.dumps(result,ensure_ascii=False,allow_nan=False)+';',encoding='utf-8')
    print(json.dumps({k:result[k] for k in ['coverage','counts','metrics','bootstrap','provenance']},indent=2))

if __name__=='__main__':
    p=argparse.ArgumentParser(); [p.add_argument('--'+a,required=True) for a in ['source','spx','fred','archive','output']]; p.add_argument('--supplement-output'); run(p.parse_args())
