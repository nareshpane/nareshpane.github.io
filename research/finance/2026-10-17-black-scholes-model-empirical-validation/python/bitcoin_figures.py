"""Private aggregate SVG reports. No public-output path or website integration.

The UWA adapter supplies actual verified records; synthetic fixtures remain labelled.
"""
from html import escape
from pathlib import Path
import numpy as np
import pandas as pd
COLORS={'H30':'#205492','H60':'#087f80','B':'#b47c28'}

def svg_lines(title,xlabel,ylabel,series,*,square=False,zero=True):
    finite=[(a,b) for s in series for a,b in s['points'] if np.isfinite(a) and np.isfinite(b)]
    if not finite:return '<p>No valid values for this figure; none invented.</p>'
    values=np.array(finite);xmin,xmax=values[:,0].min(),values[:,0].max();ymin,ymax=values[:,1].min(),values[:,1].max()
    if square:xmin=ymin=0.;xmax=ymax=max(xmax,ymax)*1.04
    elif zero:ymin=min(0,ymin)
    dx=max(xmax-xmin,1e-5);dy=max(ymax-ymin,1e-5);xmax=xmin+dx;ymax=ymin+dy
    w,h=(660,660) if square else (900,430);width,height=(470,470) if square else (720,270)
    x=lambda a:85+(a-xmin)/(xmax-xmin)*width;y=lambda b:85+height-(b-ymin)/(ymax-ymin)*height
    text=lambda a,b,s,extra='':f'<text x="{a:.3f}" y="{b:.3f}" {extra}>{escape(str(s))}</text>'
    svg=[f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {w} {h}" role="img" aria-label="{escape(title)}"><title>{escape(title)}</title>']
    for v in np.linspace(xmin,xmax,5):svg.extend([f'<path d="M{x(v)} 85v{height}" stroke="#e3e6da"/>',text(x(v),height+111,f'{v:.2f}','text-anchor="middle"')])
    for v in np.linspace(ymin,ymax,5):svg.extend([f'<path d="M85 {y(v)}h{width}" stroke="#e3e6da"/>',text(73,y(v)+4,f'{v:.2f}','text-anchor="end"')])
    if square:svg.append(f'<path d="M85 {85+height}L{85+width} 85" stroke="#88968d" stroke-dasharray="5 4"/>')
    for i,s in enumerate(series):
        svg.extend([f'<rect x="{85+i*170}" y="33" width="15" height="5" fill="{s["color"]}"/>',text(105+i*170,40,s['name'])])
        if s.get('dots'):
            for a,b in s['points']:
                if np.isfinite(a) and np.isfinite(b):svg.append(f'<circle cx="{x(a)}" cy="{y(b)}" r="3.5" fill="{s["color"]}" tabindex="0"><title>{escape(s["name"])} x={a:.6g}; y={b:.6g}</title></circle>')
        else:
            path=[];drawing=False
            for a,b in s['points']:
                if np.isfinite(a) and np.isfinite(b):path.append(('L' if drawing else 'M')+f'{x(a):.3f} {y(b):.3f}');drawing=True
                else:drawing=False
            svg.append('<path d="'+' '.join(path)+f'" stroke="{s["color"]}" stroke-width="2" fill="none"/>')
    svg.extend([text(85+width/2,height+151,xlabel,'text-anchor="middle"'),f'<text transform="translate(18 {85+height/2}) rotate(-90)" text-anchor="middle">{escape(ylabel)}</text>','</svg>'])
    return ''.join(svg)

def render_private(output:Path,evaluation:pd.DataFrame,history:pd.DataFrame,estimates:pd.DataFrame,result:dict)->None:
    evaluation=evaluation.copy()
    if not isinstance(evaluation.term.dtype,pd.CategoricalDtype):evaluation['term']=pd.cut(evaluation['T']*365,[6.999999,21,45,90],include_lowest=True)
    if not isinstance(evaluation.money.dtype,pd.CategoricalDtype):evaluation['money']=pd.cut(evaluation.m,[.799999,.9,.97,1.03,1.1,1.200001],include_lowest=True)
    out=Path(output);panels=[];fixture=result.get('data_class')=='synthetic_test_fixture'
    banner='SYNTHETIC TEST FIXTURE — NOT MARKET EVIDENCE' if fixture else 'PRIVATE REPRODUCIBILITY REPORT — WEBSITE USES REVIEWED AGGREGATES'
    def add(name,title,svg,caption):
        (out/(name+'.svg')).write_text(svg,encoding='utf8');panels.append('<section><h2>'+escape(title)+'</h2>'+svg+'<p>'+escape(caption)+'</p></section>')
    estimates=estimates[estimates.available_at<=history.timestamp.max()+pd.Timedelta(minutes=1)]
    n=len(evaluation);x=estimates.available_at.map(pd.Timestamp.toordinal)
    add('B3-history','Underlying price history',svg_lines('Actual supplied daily closes','UTC calendar-day ordinal','USD per BTC',[dict(name='Daily close',color='#b47c28',points=list(zip(x,estimates.daily_close)))],zero=False),'Only supplied observations; missing daily closes break the curve. Ordinal dates are calendar days; coverage is stated in the audit.')
    add('B4-volatility','Independent historical volatility',svg_lines('H30 and H60','UTC calendar-day ordinal','Annualized sigma (%)',[dict(name=k,color=COLORS[k],points=list(zip(x,estimates[k]*100))) for k in ['H30','H60']]),'Sample standard deviation of 30/60 consecutive daily log returns × √365. Availability is next UTC midnight; no forward filling.')
    # Binned means use all evaluation rows, not a convenient raw-observation subset.
    bins=np.linspace(0,max(float(evaluation.observed_usd.max()),1)*1.000001,31)
    grouped=evaluation.assign(price_bin=pd.cut(evaluation.observed_usd,bins,include_lowest=True)).groupby('price_bin',observed=True)
    points=grouped[['observed_usd','H30','H60','B']].mean()
    add('B5-prices','Observed versus theoretical prices',svg_lines('Binned price means, common evaluation set','Observed premium (USD equivalent / BTC notional)','Model value (same USD units)',[dict(name=k,color=COLORS[k],points=list(zip(points.observed_usd,points[k])),dots=True) for k in COLORS],square=True),f'n={n} evaluation transactions pooled in 30 price bins. Each point is a group mean, not a fabricated transaction. All models share the same rows; the line is equality.')
    residuals=[evaluation[k]-evaluation.observed_usd for k in COLORS];edges=np.linspace(min(v.min() for v in residuals)-.01,max(v.max() for v in residuals)+.01,41);series=[]
    for k,values in zip(COLORS,residuals):
        counts,_=np.histogram(values,edges);p=[]
        for a,b,count in zip(edges[:-1],edges[1:],counts):p.extend([(a,int(count)),(b,int(count))])
        series.append(dict(name=k,color=COLORS[k],points=p))
    add('B6-residuals','Signed residual distributions',svg_lines('Identical error bins','Model minus observed (USD equivalent)','Transactions per bin',series),f'n={n} per model, common bins and axes; no independent-observation inference is implied.')
    cells=[];terms=list(evaluation.term.cat.categories);money=list(evaluation.money.cat.categories)
    for (term,m),g in evaluation.groupby(['term','money'],observed=True):cells.append((terms.index(term),money.index(m),len(g),float(abs(g.H30-g.observed_usd).mean())))
    maximum=max((v for _,_,_,v in cells),default=1) or 1
    svg=['<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 900 350" role="img" aria-label="H30 error by maturity and moneyness with counts">']
    for i,label in enumerate(terms):svg.append(f'<text x="10" y="{120+i*60}">{escape(str(label))} days</text>')
    for j,label in enumerate(money):svg.append(f'<text x="{200+j*130}" y="55" font-size="11">{escape(str(label))}</text>')
    for i,j,count,value in cells:svg.append(f'<rect x="{190+j*130}" y="{90+i*60}" width="123" height="52" rx="5" fill="#c47664" fill-opacity="{.12+.65*value/maximum}"/><text x="{198+j*130}" y="{111+i*60}">${value:.2f}</text><text x="{198+j*130}" y="{130+i*60}">n={count}</text>')
    svg.append('</svg>')
    add('B8-heatmap','H30 MAE by maturity and K/S',''.join(svg),'Discrete measured groups, USD equivalent, no interpolation. Empty cells have no result. All principal model breakdowns are in the private JSON.')
    valid=evaluation[evaluation.independent_iv.notna()];series=[]
    for side,color in [('call','#205492'),('put','#087f80')]:
        grouped=valid[valid.kind==side].groupby('money',observed=True)[['m','independent_iv']].median()
        series.append(dict(name=side,color=color,points=list(zip(grouped.m,grouped.independent_iv*100)),dots=True))
    add('B7-IV','Descriptive implied-volatility diagnostics',svg_lines('Pooled medians by K/S and side','K/S','Independent inverted IV (%)',series),f'{len(valid)} valid inversions; maturity/date mixtures limit interpretation. This is in-sample price inversion, not independent validation or an Experiment A input.')
    rows=''.join('<tr><th>'+k+'</th>'+''.join(f'<td>{result["metrics"][k][stat]:.4f}</td>' for stat in ['mae','median_absolute','rmse','bias','spot_bps'])+'</tr>' for k in COLORS)
    page='<!doctype html><html lang="en"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>Private Bitcoin benchmark report</title><style>body{font:16px/1.65 system-ui;background:#fff7ea;color:#29434b;max-width:1100px;padding:25px;margin:auto}section{background:#fffdf5;border:1px solid #a5bea4;border-radius:12px;padding:20px;margin:20px 0}svg{display:block;width:100%;font:12px system-ui;fill:#29434b}table{width:100%;border-collapse:collapse}td,th{padding:12px;border-bottom:1px solid #ccd4c3}p{max-width:100ch}</style><h1>Bitcoin option benchmark</h1><strong>'+banner+'</strong><p>Coverage: '+escape(' to '.join(result['evaluation_coverage']))+f'. Common evaluation observations: {n}. </p><table><tr><th>Model</th><th>MAE USD</th><th>Median USD</th><th>RMSE USD</th><th>Bias USD</th><th>Spot bp</th></tr>'+rows+'</table>'+''.join(panels)+'<p>'+escape(result['limitations'])+'</p><p>Complete filters, inventories, groups, paired differences, assumptions and resampling metadata: bitcoin-results.json. No website release is automatic.</p></html>'
    (out/'private-report.html').write_text(page,encoding='utf8')
