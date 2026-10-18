"""Exactly 20 explanatory rows plus aggregate figures; uses unchanged evaluation frame."""
import csv,hashlib,json
from pathlib import Path
import numpy as np

# Source order and meanings inspected against the supplied 34-field README.
FIELDS=[
('contract','Contract symbol','OSI identifier; root, expiration, side and strike.'),
('underlying','Underlying code','Source identifier, including SPXW root alias.'),
('expiration','Expiration date','Contractual expiration, YYYY-MM-DD.'),
('type','Option type','Call or put.'),('strike','Strike price (pt)','Exercise index level.'),
('style','Exercise style','E European; A American.'),('quote_date','Quote date','Trading date represented by file.'),
('bid','Best bid (pt)','End-of-day standing bid, index points.'),('bid_size','Bid size (contracts)','Displayed size at bid.'),
('ask','Best ask (pt)','End-of-day standing ask, index points.'),('ask_size','Ask size (contracts)','Displayed size at ask.'),
('quote_time','Quote update (UTC)','Individual quote timestamp; unavailable in 2022.'),
('volume','Daily volume (contracts)','Includes multi-leg executions; not a count of standalone trades.'),
('open_interest','Open interest (contracts)','Outstanding contracts reflecting prior business-day close.'),
('open','First trade (pt)','First trade of that session, when available.'),('high','Session high (pt)','Highest session trade.'),
('low','Session low (pt)','Lowest session trade.'),('close','Last session trade (pt)','Trade price, not the quote midpoint.'),
('trade_vwap','Trade VWAP (pt)','Volume-weighted trade price.'),('transactions','Trade count','Number of trades.'),
('multileg_volume','Multi-leg volume','Contracts attributed to multi-leg trades; may exceed total volume.'),
('active_minutes','Active minutes','One-minute windows containing a trade.'),
('last_trade_date','Last observed trade date','Most recent trading date observed within the archive up to this file.'),
('underlying_close','Source underlying level (pt)','Source underlying_close; SPXW may differ from official SPX close. Not silently harmonized in this column.'),
('settlement_time','Settlement time','AM or PM; separate from exercise style.'),
('iv_bid','Vendor bid IV (decimal)','Vendor inversion of bid; 0.25 means 25% annual volatility.'),
('iv_ask','Vendor ask IV (decimal)','Vendor inversion of ask.'),('iv','Vendor midpoint IV (decimal)','Vendor inversion of midpoint; not used to test its own price.'),
('iv_flag','Vendor solve status','0 parity forward; 1 fallback; 2–7 no midpoint solution; see provider documentation.'),
('delta','Vendor delta','Price change per one index point, vendor calculation.'),
('gamma','Vendor gamma','Delta change per one index point.'),('theta','Vendor theta (pt/year)','Annual time decay; divide by 365 for per-day convention.'),
('vega','Vendor vega (pt/unit vol)','Per 1.00 volatility; divide by 100 for one volatility percentage point.'),
('rho','Vendor rho (pt/unit rate)','Per 1.00 rate; divide by 100 for one rate percentage point.')]
DERIVED=[('mid','Our midpoint (pt)','(bid + ask) / 2.'),('S','Harmonized SPX level (pt)','Independent Cboe close used in pricing.'),('m','Our strike / spot','K / harmonized SPX close.'),('days','Our maturity (days)','Calendar expiration minus quote date.'),('A','Our Test A value (pt)','Prior 20-return volatility, training carry.'),('B','Our Test B value (pt)','Training-fitted common volatility, held-out contract.')]

def publish(e,source,out):
    selected=[];used=set()
    for di,day in enumerate(['2022-07-01','2022-09-19','2022-11-16','2022-12-30']):
        for i,(m,days) in enumerate(zip([.92,.965,1.,1.035,1.08],[14,35,65,14,35])):
            kind='call' if (i+di)%2==0 else 'put'
            g=e[(e.quote_date==day)&(e.type==kind)&~e.contract.isin(used)].copy()
            g['distance']=abs(g.m-m)/.02+abs(g.days-days)/30
            row=g.sort_values(['distance','contract']).iloc[0];used.add(row.contract);selected.append(row)
    by_key={(r.quote_date,r.contract):r for r in selected};examples=[];hashes={}
    for day in sorted({r.quote_date for r in selected}):
        file=source/'day_by_date'/f'{day}_options.csv';hashes[day]=hashlib.sha256(file.read_bytes()).hexdigest()
        with file.open(newline='',encoding='utf-8') as stream:
            reader=csv.DictReader(stream);assert reader.fieldnames==[f[0] for f in FIELDS]
            for line,row in enumerate(reader,2):
                key=(day,row['contract'])
                if key in by_key:
                    ev=by_key[key];examples.append(dict(id=f'O{len(examples)+1:02}',source=row,derived={k:float(ev[k]) for k,_,_ in DERIVED},source_line=line))
    assert len(examples)==20
    spread=e.ask-e.bid; edges=np.array([0,.25,.5,1,2,3,5,10,20,50,100,np.inf]);hist=np.histogram(spread,bins=edges)[0]
    grouped=[]
    for (kind,term,m),g in e.groupby(['type','maturity','moneyness'],observed=True):
        grouped.append(dict(kind=str(kind),term=str(term),m=str(m),n=len(g),x=float(g.m.mean()),mid=float(g.mid.mean()),A=float(g.A.mean()),B=float(g.B.mean()),biasA=float((g.A-g.mid).mean()),biasB=float((g.B-g.mid).mean())))
    vol=[]
    for day,g in e.groupby('quote_date'):
        atm=g[g.m.between(.98,1.02)]
        vol.append(dict(date=day,n=len(g),n_atm_iv=int(atm.iv.notna().sum()),hv20=float(g.hv20.iloc[0]),hv30=float(g.hv30.iloc[0]),iv=float(atm.iv.median()),fit=float(g.fitvol.median())))
    result=dict(n=len(e),dates=e.quote_date.nunique(),coverage=[e.quote_date.min(),e.quote_date.max()],
        spread_histogram=[dict(label=f'{a:g}–{b:g}' if np.isfinite(b) else f'≥{a:g}',n=int(n),lo=float(a)) for a,b,n in zip(edges[:-1],edges[1:],hist)],
        grouped_prices=grouped,volatility=vol)
    sample=dict(purpose='Exactly 20 explanatory contract-day examples, not a market-data substitute.',dates=sorted(hashes),source_hashes=hashes,
        columns=[dict(field=k,label=l,definition=d,origin='source-computed' if i>=25 else 'source') for i,(k,l,d) in enumerate(FIELDS)]+[dict(field=k,label=l,definition=d,origin='research-derived') for k,l,d in DERIVED],rows=examples)
    out.mkdir(parents=True,exist_ok=True)
    # Only 20 original rows, no larger raw panel in any script or auxiliary export.
    for name,data,globalname in [('illustrative-observations',sample,'OPTION_EXAMPLES'),('supplement',result,'SUPPLEMENT')]:
        (out/(name+'.js')).write_text('window.'+globalname+' = '+json.dumps(data,ensure_ascii=False,allow_nan=False)+';\n',encoding='utf-8')
    print(f'Exported exactly {len(examples)} source examples and new aggregates for {len(e)} unchanged evaluation rows.')
