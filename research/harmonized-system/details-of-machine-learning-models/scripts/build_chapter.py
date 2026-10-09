"""Render the chapter and compact publication figures from saved numeric outputs."""
from pathlib import Path
import sys, os, json, re, math
from html import escape
os.environ['MPLCONFIGDIR']=str(Path(__file__).resolve().parents[1]/'results/matplotlib-cache')
sys.dont_write_bytecode=True
vendor=Path(os.environ.get('TRADE_TESTING_REFERENCE',r'D:\Trade_Data_Scientist_Gov_Alberta\testing_models'))/'code/vendor'
if vendor.exists():sys.path.append(str(vendor))
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from chapter_content import CONTENT
from presentation_explanations import transformations,metric_notation
from display_numbers import ordinary as f, informative as sig, probability, percentage
ROOT=Path(__file__).resolve().parents[1]
BASE='details-of-machine-learning-models/'
PAGE=ROOT.parent/'details-of-machine-learning-models.html'
sys.path.append(str(ROOT.parent/'shared'))
from collection_navigation import render_top_nav
TITLE='Inside Six Machine Learning Models: Mathematics, Data, and Predictions'

def load(name):return json.loads((ROOT/'results'/name).read_text(encoding='utf-8'))
ROUNDING_NOTE='Numerical values are displayed to one decimal place; calculations use the original, unrounded values at full available precision.'
def number_cell(value,header,allow_small=True,decimals=1):
    """Only format display cells; never return numbers to any model calculation."""
    if value is None or isinstance(value,(float,np.floating)) and not np.isfinite(value):return '<td>N/A</td>',False
    original=str(value)
    match=re.fullmatch(r'([-+]?\d[\d,]*(?:\.\d+)?(?:[eE][-+]?\d+)?)\s*(%)?',original)
    if not match:return '<td>'+original+'</td>',False
    x=float(match.group(1).replace(',',''));percent=bool(match.group(2))
    identity=bool(re.search(r'\b(?:ID|IDs|Year|HS[246]|Iteration|Tree|Node|Layer|Unit|Border|N|Count)\b',header,re.I)) and not bool(re.search(r'amount|input|error|mean|bias|value|probability|weight|slope',header,re.I))
    if identity and x.is_integer():
        display=original if re.search(r'\bHS[246]\b',header,re.I) and re.fullmatch(r'\d+',original) else str(int(x))
        rounded=False
    else:
        display=f(x,decimals)+( '%' if percent else '');rounded=True
        # Small learned slopes and probabilities must not masquerade as exact zero/one.
        sensitive=allow_small and bool(re.search(r'learned slope|^slope$|weight to|probability|fraction|Brier|log loss|accuracy',header,re.I))
        if sensitive and 0<abs(x)<.05:display=f'{x:.1e}'+('%' if percent else '')
        elif sensitive and 'probab' in header.lower() and .95<x<1:
            display=probability(x)
    return '<td class="numeric">'+display+'</td>',rounded
def eq(s):
    # Numeric display strings become conventional TeX scientific notation.
    s=re.sub(r'([-+]?\d+\.\d)e([-+]?\d+)',lambda m:m[1]+r'\times10^{'+str(int(m[2]))+'}',s)
    s=re.sub(r'(?<=\d),(?=\d{3}(?:[,\.\D]|$))',r'{,}',s)
    return '<div class="equation" tabindex="0" role="region" aria-label="Mathematical equation; scroll horizontally">\\['+s+'\\]</div>'
def p(s,cls=''):return '<p'+(' class="'+cls+'"' if cls else '')+'>'+s+'</p>'
def table(headers,rows,caption='',cls='',attrs='',rounding_note=None):
    body=[];rounded=False
    for row in rows:
        cells=[]
        for header,value in zip(headers,row):
            if str(row[0]) in ['N','top_overlap_3']:header+=' Count'
            if caption=='Observation 263: complete input record':
                if str(row[0]) in ['observation_id','year','macro_year']:header='ID'
                elif str(row[0]) in ['hs4_code','HS2']:header=str(row[0]).upper().replace('_CODE','')
                elif str(row[0]) in ['contiguity','common_language']:header='Border'
            decimals=2 if caption=='Exact first-tree decision path' and header in ['This input','Learned threshold'] else 1
            cell,changed=number_cell(value,header,allow_small='rounded-exact' not in cls,decimals=decimals);cells.append(cell);rounded|=changed
        body.append('<tr>'+''.join(cells)+'</tr>')
    kind='text-table' if 'text-table' in cls else 'mathematical-table' if caption in ['Every term in the fitted linear predictor','Compute the first hidden neuron','Exact first-tree decision path'] else 'numerical-table'
    if kind=='text-table' and len(headers)>=6:cls+=' mixed-table'
    if caption=='Observation 263: complete input record':cls+=' record-table'
    result='<div class="table-scroll '+kind+' '+cls+'" tabindex="0" role="region" aria-label="'+escape(caption or 'Scrollable table',quote=True)+'"><table'+attrs+'>'+('<caption>'+caption+'</caption>' if caption else '')+'<thead><tr>'+''.join('<th scope="col">'+h+'</th>' for h in headers)+'</tr></thead><tbody>'+''.join(body)+'</tbody></table></div>'
    if rounded:result+=p(rounding_note or ROUNDING_NOTE,'rounding-caption')
    return result
def card(title,body,id):return '<section class="card" id="'+id+'"><h2>'+title+'</h2>'+body+'</section>'
def nav(top=False):
    if top:return render_top_nav(PAGE.name)
    return '<nav class="collection-nav" aria-label="Collection navigation"><a href="harmonized-system-index.html">Collection Home</a><a href="canada-and-provinces-trade-by-hs.html">Canada &amp; Provinces</a><a href="alberta-trade-by-hs.html">Alberta’s Export Atlas</a><a href="machine-learning-trade-sector-prediction.html">Full Six-Model Study</a><a href="../../research.html">Main Research Page</a></nav>'
def math_wrap(s):return re.sub(r'\\\[(.*?)\\\]',lambda m:eq(m.group(1)),s,flags=re.S)

BOARD_MATH=[
 r'\begin{gathered}\widehat y_{t+1}=a+\rho y_t\\+\beta^{\mathsf T}x_t+\gamma_{\mathrm{HS2}}\end{gathered}',
 r'\begin{gathered}J=\frac{\mathrm{SSE}}{2n}+\alpha r\|b\|_1\\+\frac{\alpha(1-r)}2\|b\|_2^2\end{gathered}',
 r'F(z)=\frac1{120}\sum_{b=1}^{120}T_b(z)',
 r'\begin{gathered}16\to8\ (\tanh)\to4\ (\tanh)\to1\\F(z)=h_2w_3+b_3\end{gathered}',
 r'\begin{gathered}F_m=F_{m-1}+0.05h_m\\150\ \text{correction trees}\end{gathered}',
 r'\begin{gathered}\widehat V^*=\widehat\pi c_+e^{\widehat f}\\\widehat y_{t+1}=y_t+\lambda(p_t-y_t)\end{gathered}'
]
def classroom(k,name):
    labels=['Next-year log trade','Elastic Net objective','Average of 120 trees','Two tanh hidden layers','Sequential corrections','Potential and adjustment']
    return f'<figure class="classroom-wrap"><div class="classroom-scene"><img class="classroom" src="{BASE}images/algorithm-{k:02d}.png" width="1672" height="941" loading="lazy" alt="The same elderly female professor and two male students at wooden desks in a warm classroom, discussing {escape(name,quote=True)}."><div class="blackboard-overlay" data-algorithm="{k}"><span class="board-label">{labels[k-1]}</span><div class="board-math">\\({BOARD_MATH[k-1]}\\)</div></div></div><figcaption>Generated classroom scene with verified SVG mathematics; the full derivation follows.</figcaption></figure>'

def figures(preds,metrics):
    dest=ROOT/'results/figures';dest.mkdir(exist_ok=True)
    plt.rcParams.update({'font.family':'DejaVu Sans','font.size':9,'axes.facecolor':'#fffdf8','figure.facecolor':'#fffdf8','axes.spines.top':False,'axes.spines.right':False,'svg.fonttype':'none'})
    for k in range(1,7):
        d=preds[(preds.model==k)&(preds.split=='validation_2024_observed')].copy()
        a=d.observed_exports_usd.to_numpy()/1e9;b=d.predicted_exports_usd.to_numpy()/1e9;e=b-a
        fig,ax=plt.subplots(figsize=(5.6,3.3),layout='constrained');ax.scatter(a,b,s=19,color='#be5b3c',alpha=.7)
        lim=max(a.max(),b.max());ax.plot([0,lim],[0,lim],color='#487967',linestyle='--',linewidth=1)
        ax.set(xlabel='Observed exports (USD billions)',ylabel='Predicted exports (USD billions)',title='2024 observed temporal validation · 84 cells');ax.grid(alpha=.15)
        fig.savefig(dest/f'model-{k:02d}-scatter.svg');plt.close(fig)
        ix=[0,1,2,10,11];ss=d.iloc[ix];pos=np.arange(5)
        fig,ax=plt.subplots(figsize=(5.6,3.3),layout='constrained');ax.bar(pos-.18,ss.observed_exports_usd/1e9,.36,label='Observed',color='#658b70');ax.bar(pos+.18,ss.predicted_exports_usd/1e9,.36,label='Predicted',color='#d38257')
        ax.set_xticks(pos,['ID '+str(v) for v in ss.observation_id]);ax.set(ylabel='Exports (USD billions)',title='Five fixed validation cells · same IDs in each model');ax.legend(frameon=False);ax.grid(axis='y',alpha=.15)
        fig.savefig(dest/f'model-{k:02d}-bars.svg');plt.close(fig)
        fig,ax=plt.subplots(figsize=(5.6,3.3),layout='constrained');ax.hist(e,bins=12,color='#d38257',edgecolor='#fffdf8');ax.axvline(0,color='#476955',linewidth=1)
        ax.set(xlabel='Signed error: prediction − observed (USD billions)',ylabel='Number of cells',title='2024 residual distribution');ax.grid(axis='y',alpha=.15)
        fig.savefig(dest/f'model-{k:02d}-residuals.svg');plt.close(fig)
        vals=[metrics[k-1]['by_sector_validation'][hs]['WAPE'] for hs in ['1001','8703']]
        fig,ax=plt.subplots(figsize=(5.6,3.3),layout='constrained');ax.bar(['1001 · Wheat','8703 · Cars'],vals,color=['#d38257','#8c9e69'])
        ax.set(ylabel='Cell WAPE (%)',title='2024 error by sector · 42 cells each');ax.grid(axis='y',alpha=.15)
        fig.savefig(dest/f'model-{k:02d}-sector.svg');plt.close(fig)
    for kind,label,scale in [('WAPE','Cell WAPE (%)',1),('RMSE','RMSE (USD billions)',1e9)]:
        fig,ax=plt.subplots(figsize=(8,3.5),layout='constrained');pos=np.arange(6)
        ax.bar(pos-.18,[m['validation'][kind]/scale for m in metrics],.36,label='2024 observed validation',color='#739176')
        ax.bar(pos+.18,[m['scenario'][kind]/scale for m in metrics],.36,label='2025 illustrative scenario',color='#d38257')
        ax.set_xticks(pos,['1 · Dynamic OLS','2 · Elastic Net','3 · Forest','4 · MLP','5 · Boost','6 · Two-stage'],rotation=12,ha='right')
        ax.set(ylabel=label,title='Common 84-cell populations; no observed 2025 test');ax.legend(frameon=False,fontsize=8);ax.grid(axis='y',alpha=.15)
        fig.savefig(dest/f'comparison-{kind.lower()}.svg');plt.close(fig)

def worked(t,record):
    k=t['model'];pid=t['observation_id']
    s=p(f'Follow <strong>observation {pid}</strong>: Canada → United States, HS4 1001, 2025. Its inputs come from observed 2024 cell {pid-84}. The 2025 observed outcome is unavailable; the following computation traces the fitted prediction, evaluated against the scenario separately.')
    s+=p('Numerical substitutions below are approximate displays. Predictions use the saved, unrounded inputs and parameters; adding or exponentiating the displayed rounded entries need not reproduce the reported dollars. Symbolic exponents and model settings retain their exact meaning.','small')
    if k in [1,2]:
        rows=[[escape(name.replace('numeric__','').replace('category__','')),sig(x),sig(b),sig(v)] for name,x,b,v in zip(t['feature_names'],t['standardized_features'],t['coefficients'],t['contributions'])]
        s+=table(['Feature','Standardized/encoded input','Learned slope','Input × slope'],rows,'Every term in the fitted linear predictor','compact')
        s+=eq(r'\widehat y=a+\sum_j z_jb_j\simeq '+sig(t['intercept'])+r'+('+sig(sum(t['contributions']))+r')\simeq '+sig(t['log_prediction']))
        s+=p(f'The intercept plus all {len(rows)} contributions yields the log prediction. The full coefficient and preprocessing vectors are in the worked-example JSON; coefficients on standardized logs are not directly raw elasticities.')
    elif k==3:
        tr=t['first_tree'];rows=[[v['node'],escape(v['feature']),v['value'],v['threshold'],escape(v['branch'])] for v in tr['path']]
        s+=table(['Node','Feature','This input','Learned threshold','Taken branch'],rows,'Exact first-tree decision path','compact',rounding_note='Decision-path inputs and thresholds use two decimal places so the branch comparisons remain distinguishable; calculations use the original, unrounded values at full available precision.')
        s+=p(f'Tree 1 reaches leaf {tr["leaf_node"]}, whose learned log1p mean is {sig(tr["leaf_log_prediction"])} from {f(tr["bootstrap_weighted_samples"],0)} weighted bootstrap draws. This is one tree, not the final forest prediction.')
        s+=table(['Tree','Log prediction'],[[i+1,sig(v)] for i,v in enumerate(t['tree_predictions'])],'All 120 tree outputs','compact')
        s+=eq(r'F(z)=\frac{1}{120}\sum_{b=1}^{120}T_b(z)\simeq '+sig(t['log_prediction']))
    elif k==4:
        layer=t['layers'][0];row=[[escape(n),sig(x),sig(w[0]),sig(x*w[0])] for n,x,w in zip(t['feature_names'],layer['inputs'],layer['weights'])]
        s+=table(['Feature','Input','Weight to hidden unit 1','Product'],row,'Compute the first hidden neuron','compact')
        z=layer['preactivation'][0];s+=eq(r'a_{1,1}=b_{1,1}+\sum_j z_jW_{1,j1}\simeq '+sig(layer['biases'][0])+r'+\sum_j z_jW_{1,j1}\simeq '+sig(z)+r',\quad h_{1,1}=\tanh(a_{1,1})\simeq '+sig(layer['activation'][0]))
        allrows=[]
        for i,l in enumerate(t['layers']):
            for j,(a,h,b) in enumerate(zip(l['preactivation'],l['activation'],l['biases'])):allrows.append([i+1,j+1,sig(b),sig(a),sig(h),'tanh' if i<2 else 'linear'])
        s+=table(['Layer','Unit','Bias','Weighted sum + bias','Output','Activation'],allrows,'All hidden units and the final output','compact')
        s+=p(f'The forward pass reconstructs the saved log prediction, displayed as approximately {sig(t["log_prediction"])}. There are {t["parameter_count"]} trainable scalars; this is a large count relative to 84 repeated trade cells. Adam reached 2000 iterations with a convergence warning. The output is a reproducible finite-budget fit, not a claimed optimum.')
    elif k==5:
        row=[[0,sig(t['baseline']),'Initial mean log1p target']]+[[i+1,sig(v),sig(t['updates'][i])] for i,v in enumerate(t['staged_predictions'])]
        s+=table(['Iteration','Cumulative log prediction','Added eta × leaf correction'],row,'All 150 fitted updates; the first five show the mechanism','compact')
        s+=eq(r'F_0\simeq '+sig(t['baseline'])+r',\quad F_1=F_0+\eta h_1(z)\simeq F_0+('+sig(t['updates'][0])+r')\simeq '+sig(t['staged_predictions'][0]))
        s+=p('Each saved increment already includes the learning rate .05. These corrections are fitted to all training residuals, not solely this observation. The final cumulative prediction is '+sig(t['log_prediction'])+'.')
    else:
        s+=eq(r'\widehat g\simeq '+sig(t['classifier_logit'])+r',\quad\widehat\pi=\frac1{1+e^{-\widehat g}}\simeq '+probability(t['probability']))
        s+=eq(r'\widehat f\simeq '+sig(t['positive_log_prediction'])+r',\quad\widehat V_+=c_+e^{\widehat f},\quad c_+\simeq '+sig(t['calibration'])+r',\quad\widehat V_+\simeq '+sig(t['positive_amount'])+r'\text{ USD}')
        s+=eq(r'\widehat V^*=\widehat\pi\widehat V_+\simeq ('+probability(t['probability'])+r')\times '+sig(t['positive_amount'])+r'\simeq '+sig(t['potential']))
        s+=eq(r'\widehat\lambda\simeq\frac{'+sig(t['lambda_numerator'])+'}{'+sig(t['lambda_denominator'])+r'}\simeq '+sig(t['lambda']))
        s+=p('The unconstrained estimate is '+sig(t['lambda_raw'])+'; compare with [0,1] to see whether the restriction binds. All 84 origin/target IDs and their gap products are saved in <a href="'+BASE+'results/lambda_estimation.csv">lambda_estimation.csv</a>.')
        s+=eq(r'\widehat y_{2025}\simeq\log(1+'+sig(t['lag'])+')+'+sig(t['lambda'])+r'\,('+sig(t['gap_log'])+r')\simeq '+sig(np.log1p(t['prediction'])))
        s+=eq(r'\widehat V_{2025}=\exp(\widehat y_{2025})-1\simeq '+sig(t['prediction']))
        s+=table(['Iteration','Classifier probability','Positive regressor log amount'],[[i+1,sig(a),sig(b)] for i,(a,b) in enumerate(zip(t['classifier_staged_probabilities'],t['positive_regressor_staged_log_predictions']))],'Both structural boosting stages: every fitted iteration','compact')
        s+=p('The classifier starts from training positive prevalence; the positive regressor starts from mean log positive dollars. These two sequences use different target losses and separately fitted preprocessing. No probability threshold is imposed before multiplication.')
        return s
    s+=eq(r'\widehat V=c\max\{\exp(\widehat y)-1,0\},\quad c\simeq '+sig(t['calibration'])+r',\quad\widehat V\simeq '+sig(t['prediction'])+r'\text{ USD}')
    s+=p('The calibration multiplier is learned solely from the model’s final training sample. It reconciles the training total; it does not reconcile the holdout total or individual observations. Source JSON retains full available floating-point precision. Independent reconstruction checks use a relative tolerance of 10⁻⁶ (with a small absolute tolerance near zero).','small')
    return s

def prediction_table(k,d):
    d=d[d.model==k];d=d[d.split.isin(['validation_2024_observed','forecast_2025_unobserved'])]
    body='pred-body-'+str(k)
    s=f'<div class="prediction-controls" data-body="{body}"><label>Search evaluation predictions by country, HS4 or ID<input type="search" aria-label="Search model {k} predictions" placeholder="Canada, 1001, or observation ID"></label><div class="controls"><label>Evaluation year<select aria-label="Model {k} evaluation year"><option value="">Both evaluation years</option><option value="2024">2024 · observed validation</option><option value="2025">2025 · unobserved scenario</option></select></label><button type="button">Clear prediction filters</button><span class="prediction-count status" role="status"></span></div></div>'
    headers=['Observation ID','Year','Exporter','Destination','HS4','Observed (USD)','Predicted (USD)','Error (USD)','Absolute error (USD)','Signed % error','Illustrative value (USD)','Scenario error (USD)']
    s+='<div class="table-scroll prediction-table numerical-table" tabindex="0" role="region" aria-label="All model '+str(k)+' evaluation predictions"><table><caption>84 observed validation predictions + 84 unobserved 2025 predictions</caption><thead><tr>'+''.join('<th scope="col">'+h+'</th>' for h in headers)+'</tr></thead><tbody id="'+body+'">'
    for r in d.itertuples():
        vals=[r.observation_id,r.year,r.exporter,r.destination,r.hs4_code,f(r.observed_exports_usd),f(r.predicted_exports_usd),f(r.signed_error_usd),f(r.absolute_error_usd),f(r.percentage_error)+'%' if np.isfinite(r.percentage_error) else 'N/A',f(r.illustrative_exports_usd),f(r.scenario_error_usd)]
        search=' '.join(str(v) for v in vals[:5]).lower()
        s+='<tr data-year="'+str(r.year)+'" data-search="'+escape(search,quote=True)+'">'+''.join('<td'+(' class="numeric"' if j==0 or j>=5 else '')+'>'+escape(str(v))+'</td>' for j,v in enumerate(vals))+'</tr>'
    return s+'</tbody></table></div>'+p(ROUNDING_NOTE,'rounding-caption')+p('Signed error = predicted − observed: positive means overprediction. Individual percentage error is N/A when observed trade is zero or unavailable. A scenario error is predicted − illustrative value, not an observed prediction error. Final training fitted values are downloadable with every eligible row in predictions.csv.','small')

def metric_demo(k,d,m):
    v=d[(d.model==k)&(d.split=='validation_2024_observed')].head(3)
    a=v.observed_exports_usd.to_numpy();b=v.predicted_exports_usd.to_numpy();e=b-a;n=3
    mse=sum(e*e)/n;mae=sum(abs(e))/n;wape=100*sum(abs(e))/sum(a);rmsle=np.sqrt(np.mean((np.log1p(b)-np.log1p(a))**2))
    s=p('For a hand-sized calculation use validation observations '+', '.join(str(v) for v in v.observation_id)+'. These are actual BACI outcomes, unlike the 2025 scenario. Evaluate errors in USD; the following formulas use millions of USD only to make the squares readable.')
    s+=table(['ID','Observed (USD)','Predicted (USD)','Error (USD)','Absolute error (USD)'],[[r.observation_id,f(r.observed_exports_usd),f(r.predicted_exports_usd),f(r.signed_error_usd),f(r.absolute_error_usd)] for r in v.itertuples()],'Three observed cells for arithmetic','compact')
    parts='+'.join('('+sig(x/1e6)+')^2' for x in e)
    s+=p('The arithmetic substitutions are rounded approximations; these example scores are calculated from the unrounded saved predictions.','small')
    s+=eq(r'\mathrm{MSE}_{3}\simeq\frac{'+parts+r'}{3}\simeq '+sig(mse/1e12)+r'\;(\text{million USD})^2')
    s+=eq(r'\mathrm{RMSE}_{3}\simeq\sqrt{'+sig(mse/1e12)+r'}\simeq '+sig(np.sqrt(mse)/1e6)+r'\;\text{million USD}')
    s+=eq(r'\mathrm{MAE}_{3}\simeq\frac{'+sig(sum(abs(e))/1e6)+r'}3\simeq '+sig(mae/1e6)+r'\;\text{million USD}')
    s+=eq(r'\mathrm{WAPE}_{3}\simeq100\frac{'+sig(sum(abs(e)))+'}{'+sig(sum(a))+r'}\simeq '+sig(wape)+r'\%')
    logdiff=np.log1p(b)-np.log1p(a)
    s+=eq(r'\mathrm{RMSLE}_{3}\simeq\sqrt{\frac{'+ '+'.join('('+sig(v)+')^2' for v in logdiff)+r'}3}\simeq '+sig(rmsle))
    grouped=v.groupby('destination')[['observed_exports_usd','predicted_exports_usd']].sum()
    s+=eq(r'\mathrm{Destination\ WAPE}_{3}\simeq100\frac{'+sig(abs(grouped.predicted_exports_usd-grouped.observed_exports_usd).sum())+'}{'+sig(sum(a))+r'}\simeq '+sig(100*abs(grouped.predicted_exports_usd-grouped.observed_exports_usd).sum()/sum(a))+r'\%')
    s+=p('The first two example cells share a destination. Sum their signed errors before taking the destination absolute error; this allows cancellation. Cell WAPE takes absolute values first. Full metrics below use 84 rows, not these three.')
    denominator=sum((a-a.mean())**2)
    s+=eq(r'R^2_3\simeq1-\frac{'+sig(sum(e*e))+'}{'+sig(denominator)+r'}\simeq '+sig(1-sum(e*e)/denominator))
    ranks=grouped.rank(method='average');ra=ranks.observed_exports_usd.to_numpy();rp=ranks.predicted_exports_usd.to_numpy()
    rc=float(np.corrcoef(ra,rp)[0,1]) if np.std(ra)>0 and np.std(rp)>0 else None
    s+=table(['Destination','Observed sum (USD)','Predicted sum (USD)','Observed rank','Predicted rank'],[[dest,f(r.observed_exports_usd),f(r.predicted_exports_usd),f(ranks.loc[dest,'observed_exports_usd'],1),f(ranks.loc[dest,'predicted_exports_usd'],1)] for dest,r in grouped.iterrows()],'Small rank-correlation calculation','compact')
    overlap=len(set(grouped.observed_exports_usd.nlargest(min(3,len(grouped))).index)&set(grouped.predicted_exports_usd.nlargest(min(3,len(grouped))).index))
    selected=grouped.predicted_exports_usd.nlargest(min(3,len(grouped))).index;rankerr=abs(grouped.loc[selected,'predicted_exports_usd']-grouped.loc[selected,'observed_exports_usd'])
    s+=p(f'For these {len(grouped)} destinations, correlate the two rank columns: Spearman = {f(rc)}. The top-min(3,N) intersection contains {overlap} destinations. On destinations selected by predicted rank, mean absolute aggregate error is USD {f(rankerr.mean())}, and median is USD {f(rankerr.median())}. This tiny two-destination calculation illustrates the definitions; full top-three metrics below rank seven destinations.')
    metric_keys=['N','MSE','RMSE','MAE','WAPE','Destination WAPE','RMSLE','R2','rank_correlation','top_overlap_3','mean_absolute_top3_error','median_absolute_top3_error']
    rows=[]
    for key in metric_keys:
        unit='USD²' if key=='MSE' else 'USD' if key in ['RMSE','MAE','mean_absolute_top3_error','median_absolute_top3_error'] else '%' if 'WAPE' in key else 'count' if key in ['N','top_overlap_3'] else 'unitless'
        rows.append([key,unit,m['training'].get(key),m['validation'].get(key),0 if key=='N' else 'N/A',m['scenario'].get(key)])
    s+=table(['Metric','Unit','Final training fit','2024 observed validation','2025 observed holdout','2025 illustrative scenario'],rows,'Keep the four evaluation activities separate','compact')
    if k==6:
        s+=p('The training column measures 2024 structural potential, whereas validation and scenario columns measure the next-year dynamic forecast. They are different tasks; comparing these columns as an overfitting gap would be invalid. The observed 2025 holdout has zero available outcomes; its N is 0, and its error metrics are undefined.')
        cls=m['classification'];rows=[]
        for yr,x in cls.items():rows.append([yr,x['N'],x['positive_N'],x['log_loss'],x['Brier'],f(100*x['accuracy'])+'%'])
        s+=table(['OOF year','N','Positive N','Binary log loss','Brier score','Accuracy at .5 (%)'],rows,'Structural classifier diagnostics: same-year exporter OOF','compact')
        s+=eq(r'\operatorname{LogLoss}=-\frac1n\sum_i[d_i\log\pi_i+(1-d_i)\log(1-\pi_i)],\quad \operatorname{Brier}=\frac1n\sum_i(\pi_i-d_i)^2')
        oo=pd.read_csv(ROOT/'results/structural_oof_2023.csv');row=oo.iloc[0];raw=pd.read_csv(ROOT/'toy_trade_dataset.csv',dtype={'hs4_code':str});z=int(raw.loc[raw.observation_id==row.observation_id,'observed_exports_usd'].iloc[0]>0);pr=row.oof_probability
        s+=p(f'For observed cell {int(row.observation_id)}, d={z} and OOF pi≈{probability(pr)}. Its log-loss contribution is approximately {sig(-z*np.log(pr)-(1-z)*np.log(1-pr))}; its Brier contribution (pi−{z})² is approximately {sig((pr-z)**2)}; the .5 classification is {int(pr>=.5)}. The reported scores average all 84 cells.')
        bins=cls['2023']['probability_bins']
        s+=table(['Probability bin','N','Mean predicted probability','Observed positive fraction'],[[r['bin'],r['N'],r['mean_probability'] if r['mean_probability'] is not None else 'N/A',r['positive_fraction'] if r['positive_fraction'] is not None else 'N/A'] for r in bins],'OOF 2023 calibration check','compact')
        s+=p('Accuracy depends on a threshold and class prevalence; it does not replace probability quality. Log loss punishes confident wrong predictions; Brier measures squared probability error. Comparing bin means with positive fractions checks calibration, but tiny bins provide weak evidence.')
    return s

def interpretation(k,d,m,fitrecords):
    v=d[(d.model==k)&(d.split=='validation_2024_observed')].copy();worst=v.nlargest(3,'absolute_error_usd');zero=v[v.observed_exports_usd==0];mean=float(v.signed_error_usd.mean())
    s=p(f'Temporal validation has Cell WAPE {f(m["validation"]["WAPE"])}% and RMSE USD {f(m["validation"]["RMSE"])}. Mean signed error is USD {f(mean)}: this is an aggregate bias diagnostic, not a test of unbiasedness. The largest three absolute errors are shown below; every other case remains in the scrollable table.')
    s+=table(['ID','Country pair','HS4','Observed (USD)','Predicted (USD)','Absolute error (USD)'],[[r.observation_id,r.exporter+' → '+r.destination,r.hs4_code,f(r.observed_exports_usd),f(r.predicted_exports_usd),f(r.absolute_error_usd)] for r in worst.itertuples()],'Largest observed validation errors','compact')
    s+=p(f'There are {len(zero)} zero outcomes in 2024 validation. Their mean predicted flow is USD {f(zero.predicted_exports_usd.mean())}; individual percentage errors are undefined, but absolute errors and WAPE over the nonzero-total population remain meaningful. Sector WAPE is {f(m["by_sector_validation"]["1001"]["WAPE"])}% for wheat and {f(m["by_sector_validation"]["8703"]["WAPE"])}% for cars.')
    explanations={1:'The 84-row early transition provides only one origin year, with highly related GDP/population/product features. A log-linear rule can extrapolate dramatically after exponentiation. Large validation errors are evidence against stable small-sample forecasting; the better final-fit scenario performance uses an additional transition and is not proof that the early failure vanished prospectively.',2:'The penalty shrinks unstable slopes, but one additive log equation must span tiny wheat flows and very large car flows. The chosen tuning criterion is log MSE; dollar losses place much greater weight on large cells. Training-total calibration can amplify some predictions while reducing aggregate mismatch on its own training sample.',3:'Averages reduce individual-tree volatility, but minleaf8 mixes dissimilar cells in this small panel. Log-space leaf means and scalar dollar correction can understate large flows and assign positive trade to zero cells. The forest cannot extrapolate a new trend outside its fitted leaves.',4:'The optimizer warning means the budget stopped before convergence. The network has more trainable parameters than independent trade cells, and tanh can saturate. These outputs demonstrate a reproducible architecture and forward pass; they do not establish that the network has found an adequate optimum or a stable economic pattern.',5:'Sequential corrections capture nonlinear structure but a leaf must contain at least 20 training rows. Two very different sectors and many correlated dyads make this a coarse partition. Destination WAPE can look much better than Cell WAPE because opposite cell errors cancel after aggregation.',6:'The estimated lambda is small, so the trade lag dominates the forecast. Both genuine 2024 validation and the artificial 2025 scenario are close to persistence. The scenario was explicitly generated by perturbing 2024 trade; that design favors lag-based forecasts and cannot establish universal superiority of this composite model.'}
    s+=p(explanations[k])
    s+=p('The final training score and temporal-validation score use different fitted samples and sometimes different tasks. A low training error with worse temporal error suggests fragility; it does not isolate overfitting from time shifts or coverage differences. One observed validation year and one artificial scenario cannot rank algorithms for future trade in general.')
    return s

def dataset_section(d):
    headers={'observation_id':'Observation ID','year':'Year','exporter':'Exporter','destination':'Destination','hs4_code':'HS4','hs4_description':'Official HS4 description','observed_exports_usd':'Observed exports (USD)','illustrative_exports_usd':'Illustrative exports (USD)','exporter_gdp':'Exporter GDP (USD)','destination_gdp':'Destination GDP (USD)','exporter_population':'Exporter population (persons)','destination_population':'Destination population (persons)','distance_km':'Distance (km)','common_language':'Common official language (0/1)','contiguity':'Common border (0/1)','data_status':'Outcome status','trade_status':'Trade value status','data_source':'Source & vintage','macro_year':'Input year'}
    order=['observation_id','year','exporter','destination','hs4_code','hs4_description','observed_exports_usd','illustrative_exports_usd','exporter_gdp','destination_gdp','exporter_population','destination_population','distance_km','common_language','contiguity','exporter_manufacturing','destination_manufacturing','exporter_internet','destination_internet','external_supply_usd','external_demand_usd','world_external_demand_usd','macro_year','data_status','trade_status','covariate_status','scenario_procedure','exporter_iso3','destination_iso3']
    opts=lambda values:'<option value="">All</option>'+''.join('<option value="'+escape(str(v),quote=True)+'">'+escape(str(v))+'</option>' for v in values)
    s=p('Canada → United States and United States → Canada are different directed cells. No country trades with itself in this table. Every pair has both headings in every year. Balanced structure does not imply equal trade. Filtering changes what is displayed; it never deletes rows from the underlying 336-row dataset.')
    s+='<div class="filter-grid"><label>Search country name or HS4<input id="dataset-search" type="search" placeholder="Canada 1001" aria-describedby="dataset-search-help"></label><label>Year<select id="filter-year">'+opts([2022,2023,2024,2025])+'</select></label><label>Exporter<select id="filter-exporter">'+opts(sorted(d.exporter.unique()))+'</select></label><label>Destination<select id="filter-destination">'+opts(sorted(d.destination.unique()))+'</select></label><label>HS4 sector<select id="filter-sector">'+opts(['1001','8703'])+'</select></label></div>'
    s+=p('Search terms combine with all four filters. Try “Canada 1001”; all supplied words must match the country names, product code or heading description.','small').replace('<p class="small">','<p class="small" id="dataset-search-help">',1)
    s+='<div class="controls"><button id="clear-filters" type="button">Clear Filters</button><a class="download" href="'+BASE+'toy_trade_dataset.csv" download>Download complete CSV</a><a href="'+BASE+'data_dictionary.csv">Data dictionary</a><span id="dataset-count" class="status" role="status" aria-live="polite">336 displayed / 336 total observations</span></div>'
    s+='<div class="table-scroll dataset" tabindex="0" role="region" aria-label="Complete 336-observation dataset; scroll vertically and horizontally"><table><caption>All 336 rows · money in current USD · N/A is unavailable, never silently zero</caption><thead><tr>'+''.join('<th scope="col">'+headers.get(c,c.replace('_',' ').title()+(' (%)' if c.endswith(('_manufacturing','_internet')) else ' (USD)' if c.endswith('_usd') else ''))+'</th>' for c in order)+'</tr></thead><tbody id="dataset-body">'
    for r in d.to_dict('records'):
        s+='<tr data-year="'+str(r['year'])+'" data-exporter="'+escape(r['exporter'],quote=True)+'" data-destination="'+escape(r['destination'],quote=True)+'" data-sector="'+r['hs4_code']+'" data-search="'+escape(' '.join(str(r[c]) for c in ['exporter','destination','hs4_code','hs4_description']).lower(),quote=True)+'">'
        for c in order:
            value=r[c]
            if c in ['observed_exports_usd','illustrative_exports_usd'] or c.endswith(('_gdp','_usd')):value=f(value,0)
            elif c.endswith('_population'):value=f(value,0)
            elif c=='distance_km':value=f(value,0)
            elif c.endswith(('_manufacturing','_internet')):value=f(value)+'%' if pd.notna(value) else 'N/A'
            if c=='hs4_description':
                s+='<td class="description"><span class="description-ellipsis" title="'+escape(str(value),quote=True)+'">'+escape(str(value))+'</span></td>'
            elif c in ['covariate_status','scenario_procedure','data_source']:
                s+='<td class="metadata"><span class="metadata-text">'+escape(str(value))+'</span></td>'
            else:
                numeric=c.endswith(('_gdp','_usd','_population','_manufacturing','_internet')) or c in ['observation_id','distance_km','common_language','contiguity']
                s+='<td'+(' class="numeric"' if numeric else '')+'>'+escape(str(value))+'</td>'
        s+='</tr>'
    s+='</tbody></table></div>'
    s+=p('Quantities are displayed to zero decimal places and percentage-valued features to one decimal place; calculations and the downloadable dataset retain the original, unrounded values.','rounding-caption')
    labels=d[['hs4_code','hs4_description']].drop_duplicates()
    s+=table(['HS4','Official heading description','Teaching role'],[[r.hs4_code,escape(r.hs4_description),'45 grid zeros over 2022–2024' if r.hs4_code=='1001' else 'Positive in all 126 historical cells'] for r in labels.itertuples()],'Two different trade sectors','text-table')
    s+=p('BACI HS2022 V202601 reconciles exporter and importer reporting; its values are source estimates of historical trade, not forecasts. Sum HS6 v × 1,000 to HS4 USD. Of the 252 historical outcomes, 207 are source-supported positive values and 45 are explicitly imputed grid zeros. A valid grid cell absent from BACI is assigned zero under the original project’s convention: <code>data_status=imputed</code>, <code>trade_status=baci_grid_zero</code>. Absence does not prove a customs-reported physical zero. Validation includes these disclosed zero assumptions. WDI GDP is current USD, population is persons, manufacturing and internet are percentages. Manufacturing gaps stay blank here and are median-imputed inside each fit. See <a href="'+BASE+'data_sources.md">provenance and limitations</a>.')
    return card('01 / Explore the Complete Teaching Dataset',s,'dataset')

def inputs_section(d,traces,preds):
    sample=d.iloc[0];one=d[d.observation_id==263].iloc[0];t=traces[1]
    s=p(f'Observation 1 is Canada → China, wheat (1001), in 2022: USD {f(sample.observed_exports_usd)}. Its exporter GDP is USD {f(sample.exporter_gdp)} and distance is {f(sample.distance_km)} km. The dependent variable is exports; explanatory variables describe economic size, geography and product supply/demand. Which year they describe depends on the model.')
    s+=p('A scalar is one number. A vector is an ordered list. A matrix is a rectangular collection of numbers. A parameter is learned during fitting (such as a slope or leaf value); a hyperparameter is a setting specified or selected before the final fit (such as tree count). A prediction is the fitted rule applied to a new input. A categorical code labels a group and must not be treated as a numerical quantity.')
    families=[['Economic size','log exporter GDP; log destination GDP','<span title="World Development Indicators: NY.GDP.MKTP.CD">World Bank, GDP in USD</span>'],['Population','log exporter population; log destination population','<span title="World Development Indicators: SP.POP.TOTL">World Bank, total population (persons)</span>'],['Structure & connectivity','Manufacturing and internet, each endpoint','<span title="World Development Indicators: NV.IND.MANF.ZS and IT.NET.USER.ZS">World Bank, manufacturing as % of GDP; internet users as % of population</span>'],['Geography','log distance; contiguity; official common language','<span title="CEPII GeoDist: distw, contig, comlang_off">CEPII, population-weighted bilateral distance (km), shared border and shared official language (0/1)</span>'],['Product flows','log1p external supply/demand/world demand','CEPII BACI, HS4 export/import totals in USD; exclude all flows inside the seven-country network'],['Category','HS2 = first two digits of HS4','HS product chapters: 10 (wheat), 87 (cars)'],['Dynamic state','log1p prior trade','CEPII BACI, prior-year bilateral HS4 exports in USD; only Models 1 and 6']]
    s+=table(['Feature family','Actual inputs','Definition/source'],families,'Exact source feature family','text-table')
    s+=transformations(d,t,one,eq,p,sig,f)
    s+=eq(r'x_i=[\log GDP_e,\log GDP_d,\log Pop_e,\log Pop_d,Manuf_e,Manuf_d,Internet_e,Internet_d,\log dist,border,language,\log(1+S),\log(1+Q),\log(1+W)]')
    s+=eq(r'\mathcal O_t=\{(a,b,k):\neg(a\in\mathcal N\land b\in\mathcal N)\},\quad S_{ek,t}=\sum_{(e,b,k)\in\mathcal O_t}V_{ebk,t},\quad Q_{dk,t}=\sum_{(a,d,k)\in\mathcal O_t}V_{adk,t},\quad W_{k,t}=\sum_{(a,b,k)\in\mathcal O_t}V_{abk,t}')
    s+=p('The symbol Σ means add over all members of the indicated set; ∈ means “belongs to.” The excluded internal network contains Germany in this chapter, rather than South Korea in the reference. This matters to the computed aggregates. They are not sums from the displayed 336 rows: the builder scans the full BACI source for the two products. Excluding every internal flow prevents a cell target from appearing inside its own predictors.')
    record=load('follow_observation.json');rows=[]
    for j,name in enumerate(t['feature_names']):
        label=name.replace('numeric__','').replace('category__','')
        value=record.get(label) if j<14 else t['standardized_features'][j]
        if value is None:value=t['imputation_medians'][j]
        rows.append([label,value,t['standardized_features'][j]])
    s+=table(['Input','Raw / transformed value','Final Elastic Net Standardized Input'],rows,"Example of Transformation: Observation 263's Complete Transformed Feature Vector",'compact rounded-exact',rounding_note='Numerical values are displayed to one decimal place for readability. All model calculations use the original, unrounded values, retaining full available numerical precision.')
    s+=p('The middle column contains logged values where a log is specified, original percentages or binary flags where no log is specified, and the learned median where manufacturing is missing. The right column is the exact fitted input coordinate displayed with rounding. Small nonzero values may display as 0.0 here; the stored values, not these printed entries, enter prediction. In other tables, small learned slopes and sensitive probabilities retain scientific or additional precision when needed.','small')
    excerpt=load('matrix_excerpt.json')['rows']
    s+=p('These four historical observations show selected columns of the numeric features x before scaling, alongside their border flag and chapter label. Each row belongs to one real trade cell. This is a small excerpt, not the complete fitted matrix: the remaining numeric columns must be added, the numeric branch standardized, and HS2 expanded into its two categorical columns to obtain the final 16-column Elastic Net input.')
    s+=table(['ID','log GDP exporter','log GDP destination','log population exporter','log population destination','log distance','Border','HS2'],[[r['observation_id']]+[r[c] for c in ['log_exporter_gdp','log_destination_gdp','log_exporter_population','log_destination_population','log_distance']]+[r['contiguity'],r['HS2']] for r in excerpt],'Four real rows: a portion of the matrix before scaling','compact rounded-exact',rounding_note='Numerical values are displayed to one decimal place. The actual calculations use unrounded values at their full available numerical precision.')
    s+=eq(r'X\in\mathbb R^{252\times16},\quad y\in\mathbb R^{252},\quad b\in\mathbb R^{16}\quad\text{(Models 2–5 final fits)}')
    s+=p('ℝ means real numbers. Each of 252 rows is a historical trade observation; 16 columns are the 14 scaled numeric inputs plus two HS2 indicators. The intercept is stored separately. Elastic Net has 16 slope parameters; trees and the MLP have their own parameter structures rather than this coefficient vector. Dynamic OLS uses 15 numeric columns including the lag, one dummy, and an intercept: A is 168 × 17, theta has 17 entries. Model 6 has an 84 × 16 classifier matrix and a smaller positive-only matrix; lambda separately uses 84 paired gaps. A matrix dimension counts stored columns, not necessarily independent columns.')
    s+=p('No explicit GDP×distance or other product interaction column is supplied in the reference. Forest and boosting trees induce conditional threshold interactions; neural layers produce nonlinear mixtures. Do not add arbitrary interaction terms to the linear model and call it the same source specification.')
    origin=d[d.observation_id==179].iloc[0];lag=t1=traces[0]
    s+='<h4>A model-specific lag, separate from the Elastic Net vector</h4>'
    s+=eq(r'y_{i,t}=\log(1+V_{i,t}),\qquad z_{i,\mathrm{lag}}=\frac{y_{i,t}-\mu_{\mathrm{lag}}}{s_{\mathrm{lag}}}\quad\text{(Model 1 only)}')
    s+=p('A lag is the earlier outcome of the same exporter–destination–HS4 cell. It is not the previous row of a sorted file. For the cell represented by observation 263, row 179 supplies actual 2024 wheat exports of USD '+f(origin.observed_exports_usd)+'. Model 1 appends its standardized log1p lag to its feature vector. Model 6 uses the unstandardized log1p state in its final adjustment equation, separately from the two structural tree feature vectors. Elastic Net has no lag column.')
    s+=eq(r'y_{i,2024}\simeq\log(1+'+sig(origin.observed_exports_usd)+r')\simeq '+sig(record['lag_log'])+r',\quad z_{i,\mathrm{lag}}\simeq\frac{'+sig(record['lag_log'])+'-'+sig(t1['scaling_means'][14])+'}{'+sig(t1['scaling_sd'][14])+r'}\simeq '+sig(t1['standardized_features'][14]))
    s+=p('The lag uses 2024 outcomes to score 2025; it never uses the unavailable 2025 outcome. Lag construction pairs matching cells across years, reducing Model 1’s eligible final targets to 168 transitions. Its intercept and one dropped chapter indicator give a 168 × 17 design including the intercept; Model 6 uses 84 paired historical gaps to estimate its adjustment speed.')
    s+='<h3>Chronology: tune, validate, refit, then score</h3>'
    s+=table(['Activity','Inputs and targets','Rows / IDs'],[['Observed development','2022–2024 BACI outcomes','252 / IDs 1–252'],['Elastic selection','Fit 2022 target; 2022 X predicts 2023','84 fitting + 84 tuning / IDs 1–84 → 85–168'],['Static 2024 validation','Fit 2022/23 outcomes; score 2023 X against 2024','168 fitting + 84 validation / IDs 169–252'],['Dynamic OLS validation','Fit 2022 X+lag → 2023; score 2023 → 2024','84 fitting + 84 validation'],['Dynamic OLS final','Two transitions; origin years 2022 and 2023','168 target rows / IDs 85–252'],['Model 6 lambda','2022 exporter-OOF potential versus 2023 change','84 gap/change pairs'],['Model 6 validation','2023 exporter-OOF potential + frozen lambda → 2024','7 folds; each 72 fitting, 12 OOF rows; 84 validation'],['Final static Models 2–5','Pool observed 2022–2024','252 fitting rows'],['Final Model 6 stages','2024 all cells / positive subset','84 / positive N shown in Model 6'],['2025 prediction','Only 2024 covariates and lag when required','84 forecast IDs 253–336; 0 observed outcomes']],'Time-aware counts; no 2025 tuning','text-table')
    s+=p('Internal validation here is the genuinely observed 2024 year, with all forecast-origin inputs from 2023. Final 2025 testing would be a separate unseen year, but its actual targets are missing. Static methods were fitted to contemporaneous relations; reusing prior-year covariates makes a forecast proxy, not an estimated transition model. Final refitting can use 2024 after validation is recorded; that changes the fitted sample, so final training scores cannot be read as the training score of the earlier validation fit.')
    s+='<h3>Errors, units and aggregation</h3>'
    s+=eq(r'e_i=\widehat V_i-V_i,\quad MSE=\frac1n\sum_i e_i^2,\quad RMSE=\sqrt{MSE},\quad MAE=\frac1n\sum_i|e_i|,\quad WAPE=100\frac{\sum_i|e_i|}{\sum_i V_i}')
    s+=eq(r'RMSLE=\sqrt{\frac1n\sum_i[\log(1+\widehat V_i)-\log(1+V_i)]^2},\quad Destination\ WAPE=100\frac{\sum_d|\sum_{i:dest(i)=d}e_i|}{\sum_i V_i}')
    s+=eq(r'R^2=1-\frac{\sum_i e_i^2}{\sum_i(V_i-\overline V)^2},\qquad Percentage\ error_i=100\frac{e_i}{V_i}\quad(V_i>0)')
    s+=metric_notation(d,preds,eq,p,table,sig,f)
    s+=p('If actual variance is zero, R² is N/A; R² can be negative out of sample. RMSLE measures proportional mismatch in log1p space and is unitless. This chapter pools destinations over all exporters; original validation reports destination metrics separately for each held-out exporter. Always compare identical populations and aggregation rules.')
    s+=p('Spearman correlation is the ordinary correlation of average rank numbers, not dollar values. The top-three overlap is the size of the intersection between observed and predicted top-three destination sets. Mean and median absolute top-three destination errors use the destinations chosen by the predicted ranking. The original final study uses top ten; there are only seven destinations here, so this chapter explicitly uses three. MSE/RMSE/R² are added teaching metrics; Cell MAE, Cell WAPE, RMSLE, Destination WAPE, rank correlation, overlap and ranked-market errors preserve the source definitions. Classification diagnostics apply only to Model 6.')
    s+=p('There are 84 recurring trade cells, not 336 statistically independent experiments. Country-pair and product shocks recur across years; GDP repeats across many rows. The matrix can store many rows yet have few independent economic signals. Treating all rows as independent would overstate precision.')
    return card('02 / How the Observations Become Model Inputs',s,'inputs')

def build():
    d=pd.read_csv(ROOT/'toy_trade_dataset.csv',dtype={'hs4_code':str});preds=pd.read_csv(ROOT/'results/predictions.csv',dtype={'hs4_code':str});metrics=load('metrics.json');traces=load('worked_examples.json');fitrecords=load('fit_records.json')
    # Presentation revision: reuse existing plots and numerical artifacts unchanged.
    head='<!DOCTYPE html><html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><meta name="description" content="A reproducible chapter deriving six actual trade models, with 336 visible rows, observed 2024 validation and honestly labelled illustrative 2025 scenarios."><title>'+TITLE+'</title><link rel="stylesheet" href="'+BASE+'css/chapter.css"><script>window.MathJax={tex:{inlineMath:[["\\\\(","\\\\)"]],displayMath:[["\\\\[","\\\\]"]],tags:"ams"},svg:{fontCache:"local"},options:{enableMenu:false}};</script><script defer src="'+BASE+'js/vendor/tex-svg.js"></script><script defer src="'+BASE+'js/chapter.js"></script><link rel="stylesheet" href="shared/collection-navigation.css"></head><body id="top"><a class="skip" href="#main">Skip to content</a><div class="page">'+nav(True)+'<main id="main">'
    provenance='<aside class="page-provenance" aria-label="Page provenance"><div class="provenance-item"><span class="provenance-icon" aria-hidden="true">📅</span><span><strong class="provenance-label">Created:</strong> October 8, 2026</span></div><div class="provenance-item"><span class="provenance-icon" aria-hidden="true">🧰</span><span><strong class="provenance-label">Harness:</strong> Codex on Windows</span></div><div class="provenance-item"><span class="provenance-icon"><svg viewBox="0 0 24 24" aria-hidden="true"><path d="M3 2h8l3 3v7H3z" fill="#ece2f9" stroke="#8054a2" stroke-width="1.3"/><path d="M10 2v4h4M6 8h5M6 10h4" fill="none" stroke="#8054a2"/><path d="M8 12v4h10M8 16v5M18 16v-3" fill="none" stroke="#509da5" stroke-width="1.4"/><rect x="5" y="19" width="6" height="3" rx="1" fill="#66b3b3"/><path d="m18 9 1-1 2 1 1 2-1 2-2 1-2-1-1-2 1-2z" fill="#e6b35c" stroke="#ba8132"/><circle cx="19" cy="11" r="1.3" fill="#fff9f0"/></svg></span><span><strong class="provenance-label">Model:</strong> Unspecified</span></div><div class="provenance-item"><span class="provenance-icon" aria-hidden="true">🧠</span><span><strong class="provenance-label">Reasoning:</strong> Unspecified</span></div></aside>'
    hero='<header class="hero"><div><div class="eyebrow">Local educational draft · Created 8 October 2026 · Python, scikit-learn, MathJax &amp; imagegen</div><h1>'+TITLE+'</h1><p class="lead">A reproducible 336-observation trade experiment using seven countries, two HS4 sectors, and four years.</p></div>'+provenance+'</header>'
    notice='<div class="notice"><strong>2025 is an illustrative scenario, not observed trade.</strong>'+p('The supplied BACI archive ends in 2024. This table contains 252 historical BACI-derived outcomes and 84 explicitly illustrative 2025 scenario outcomes in a separate column. The observed 2025 column is blank. The strongest empirical comparison available here is observed 2024 temporal validation; actual 2025 holdout metrics and an observed 2025 follow-through cannot be completed. Alberta 2025 domestic exports in CAD do not substitute for national bilateral BACI flows.')+'</div>'
    stats='<div class="stats"><div><strong>336</strong><span>visible rows · 84 recurring cells</span></div><div><strong>7 × 6</strong><span>directed non-self country pairs</span></div><div><strong>252</strong><span>historical outcomes · 2022–2024</span></div><div><strong>84</strong><span>2025 scenarios · 0 observed tests</span></div></div>'
    toc='<nav class="toc" aria-label="Chapter contents"><a href="#dataset">01 · Complete dataset</a><a href="#inputs">02 · Model inputs</a>'+''.join(f'<a href="#algorithm-{k:02d}">0{k+2} · Model {k}</a>' for k in range(1,7))+'<a href="#comparison">09 · Compare</a><a href="#follow">10 · Follow one cell</a><a href="#limits">11 · What we can learn</a></nav>'
    s=head+hero+notice+stats+toc+dataset_section(d)+inputs_section(d,traces,preds)
    s+='<div class="controls"><button id="expand-all" type="button">Expand All</button><button id="collapse-all" type="button">Collapse All</button><span class="small">Six independent chapters · open more than one at a time</span></div>'
    for k,(c,m,t) in enumerate(zip(CONTENT,metrics,traces),1):
        s+=f'<details class="model" id="algorithm-{k:02d}"><summary><span>0{k+2} / Algorithm {k}: {escape(m["name"])}</span><span class="expand-label">+ Expand</span></summary><div class="model-body">'
        s+=classroom(k,m['name'])
        s+='<h3>A / Historical origin and motivation</h3>'+p(c['origin'])+'<h3>B / The idea before equations</h3>'+p(c['intuition'])+'<h3>C / Define the mathematical objects</h3>'
        s+=table(['Symbol','Type and mathematical meaning','Meaning in this experiment'],[[escape(v) for v in row] for row in c['notation']],'Model-specific notation','text-table')
        s+='<h3>D / Build and derive the estimator</h3>'+math_wrap(c['math'])+'<div class="note"><strong>Assumptions and scope</strong>'+p(c['assumptions'])+'</div>'
        if k<6:
            s+=p('The source’s dollar-total calibration is shared by Models 1–5. After fitting in log1p units, compute raw nonnegative dollar fitted values r_i on training rows only. Choose c so the rescaled fitted total equals the observed training total:')+eq(r'r_i=\max(e^{F(z_i)}-1,0),\quad c=\frac{\sum_{i\in train}V_i}{\sum_{i\in train}r_i},\quad \sum_{train}c r_i=\sum_{train}V_i')
        s+='<h3>E / Numerically follow a fitted prediction</h3>'+worked(t,load('follow_observation.json'))
        s+='<h3>F / Use the complete sample without leakage</h3>'
        s+=p(f'The final model uses {m["training_N"]} training rows; its separate temporal-validation fit uses {m["validation_training_N"]} rows and predicts 84 observed 2024 outcomes (IDs 169–252). Every model predicts the same 84 2025 IDs (253–336) from 2024 inputs. There are zero observed 2025 targets. No random train/test mixing is used for the temporal comparison.')
        if k==6:s+=p(f'Final 2024 classification uses 84 rows, positive-size regression uses {m["positive_training_N"]}. Lambda uses 84 paired 2022→2023 changes. Each exporter OOF fold trains structural stages on 72 cells and scores the excluded exporter’s 12 cells; positive-regression counts vary by fold. Full counts and explicit training IDs are in fit_records.json. The trained classification probability describes same-year positivity given X, and is used in structural potential rather than as a direct next-year event forecast.')
        elif k==1:s+=p('Final fitting pairs IDs 1–84 with 85–168 and IDs 85–168 with 169–252. The origin row supplies every feature and lag; the target row supplies only next-year exports. This consumes two transitions and 168 eligible targets, not 252 lagged targets. Validation only pairs the first transition for fitting. No pre-2022 lag is fabricated.')
        else:s+=p('Final fitting uses IDs 1–252 as contemporaneous targets. The earlier validation fit uses IDs 1–168; every validation input is copied from the matching 2023 cell (IDs 85–168), while the actual target is its 2024 counterpart (IDs 169–252). One-hot encoding and numeric medians/scales are learned inside each fit, then held fixed for scoring.')
        if k==2:s+=p('Elastic candidates and the selected historical criterion are saved in elastic_tuning.json. The 2024 validation outcome does not select the penalty, and the 2025 scenario never selects any setting.')
        final=[r for r in fitrecords if r['model']==k and r['context']=='final']
        if final:
            r=final[0];s+=p(f'The stored design including an intercept has {r["design_columns_with_intercept"]} columns and rank {r["matrix_rank"]}; its numerical condition number is {sig(r["condition_number"])}. Column count is not independent information. These diagnostics concern the inputs, not a proof that a nonlinear estimator is identified.')
            s+='<details><summary>Inspect the exact final hyperparameters and optimizer warnings</summary><pre>'+escape(json.dumps({'settings':r['settings'],'warnings':r['warnings']},indent=2))+'</pre></details>'
        else:s+='<details><summary>Inspect the exact structural tree settings</summary><pre>'+escape(json.dumps({'max_iter':150,'max_leaf_nodes':7,'learning_rate':.05,'min_samples_leaf':20,'max_bins':255,'l2_regularization':0,'early_stopping':False,'seed':338},indent=2))+'</pre></details>'
        s+='<h3>G / Predictions, observed outcomes and errors</h3>'+prediction_table(k,preds)
        s+='<div class="charts">'
        for typ,cap in [('scatter','84 observed validation cells; dashed line is perfect prediction.'),('bars','A fixed five-cell subset, including both products; identical IDs across models.'),('residuals','Signed USD errors on genuinely observed 2024 outcomes.'),('sector','Cell WAPE by heading; same 42 directed pairs per heading.')]:s+=f'<figure><img src="{BASE}results/figures/model-{k:02d}-{typ}.svg" alt="Model {k}: {escape(cap,quote=True)}" loading="lazy"><figcaption>{cap}</figcaption></figure>'
        difference=c['difference']
        if k==6:difference=difference.replace('the original 0.134342','the original estimate (approximately '+f(.134342)+')')
        s+='</div><h3>H / Calculate the evaluation metrics</h3>'+metric_demo(k,preds,m)+'<h3>I / Read the results with the mathematics</h3>'+interpretation(k,preds,m,fitrecords)+'<h3>J / How this teaching example differs from the full research model</h3>'+p(difference)
        s+=p('This is mathematical teaching and independent small-sample re-estimation. It does not reproduce the original Alberta result, infer a commercial opportunity, or estimate tariffs. Two headings, a different network, revised data vintages and one observed validation year offer much less support. No statistical prediction interval is claimed. Forecasting one year, filling a missing historical cell and estimating national-to-provincial potential are different tasks.')+'</div></details>'
    s+=comparison_section(metrics,preds)
    s+=follow_section(d,preds,traces)
    s+=card('11 / What Can We Really Learn From 336 Observations?',p('The small table is excellent for understanding mechanics: every input, zero and error is inspectable. It is a weak basis for broad predictive claims. There are only 84 recurring cells and four years; exporter attributes repeat, destination attributes repeat, and common product or dyad shocks link outcomes. More stored rows do not necessarily mean more independent evidence.')+p('High-dimensional features consume information. GDP, population, internet and manufacturing correlate; product supply and demand reflect shared sector conditions. The linear design can be poorly conditioned; the 177-parameter network can be unstable; tree leaves have few genuinely different relationships. Regularization and leaf-size limits trade bias against variance rather than guaranteeing reliable extrapolation.')+p('Leakage can make results deceptively good. Never use a cell’s target inside its product aggregates, use 2025 realized macro data to claim a January forecast, build a lag from the target year, estimate imputation/scaling on held-out rows, or tune against the final scenario. This pipeline excludes all internal-network flows from aggregates, fits preprocessing only on permitted rows, and keeps 2025 outcomes absent from every fitting target.')+p('A fitted value reuses a row’s outcome to estimate parameters. A forecast applies a rule to inputs from before the target year. A contemporaneous imputation estimates missing trade with same-year predictors; it is not a time-transition forecast. Models 1 and 6 have explicit transitions; Models 2–5 here use past-year predictors as forecast proxies for a structural relation.')+p('Observed 2024 validation supplies some real evidence. The 2025 scenario supplies none about actual 2025 forecasting success: its values are generated from the 2024 lag and favor persistent models by construction. Keep true 2025 outcomes unseen during tuning if they are later added. A new evaluation must freeze the already selected pipeline before using those outcomes. Revised-source vintages also limit claims of real-time feasibility.')+p('The original full research includes many headings, an exporter-transfer problem, Alberta-specific domestic-export lags, national proxy covariates and CAD reporting. Those differences can dominate algorithm choice. This chapter teaches what each rule computes and why outputs differ; it cannot establish universal winners or a causal trade mechanism.'),'limits')
    s+=card('Reproduce, inspect and extend the draft',p('The published page uses static HTML, lightweight local JavaScript, local SVG plots, six generated PNG illustrations and a locally vendored MathJax 3.2.2 SVG renderer. It needs no npm build, Python runtime in the browser, backend, paid API or external CDN connection. Python is needed only to rebuild the analysis. Figures are generated by Matplotlib from saved outputs.')+'<pre><code>python research/harmonized-system/details-of-machine-learning-models/dataset_build.py\npython research/harmonized-system/details-of-machine-learning-models/scripts/analyze_models.py\npython research/harmonized-system/details-of-machine-learning-models/scripts/build_chapter.py\npython research/harmonized-system/details-of-machine-learning-models/scripts/verify_project.py\npython -m http.server 8000</code></pre>'+p('Add <code>--raw</code> to dataset_build.py to rescan documented source files at <code>TRADE_RAW_ROOT</code>. The portable default verifies and uses the included source-derived 252-row subset, retaining raw-file SHA-256 provenance. Neither mode edits original files.')+'<div class="controls">'+''.join('<a class="download" href="'+BASE+path+'">'+name+'</a>' for path,name in [('README.md','Workflow & limitations'),('data_sources.md','Data provenance'),('model_inventory.md','Six-model source inventory'),('toy_trade_dataset.csv','336-row CSV'),('data_dictionary.csv','Data dictionary'),('results/predictions.csv','All fitted/evaluation predictions'),('results/metrics.json','Full metrics'),('results/worked_examples.json','Exact numerical traces'),('results/fit_records.json','Training IDs & settings'),('results/quality_checks.json','Checks & package versions'),('data/source_manifest.json','Source hashes'),('results/model_comparison.csv','Comparison CSV')])+'</div>','reproduce')
    s+='</main><footer>'+p('Naresh Neupane · Independent research &amp; educational notes · Local draft')+nav()+'<a href="#top">Back to top ↑</a></footer></div></body></html>'
    PAGE.write_text(s,encoding='utf-8')
    print('Chapter built:',PAGE,'bytes',PAGE.stat().st_size)

def comparison_section(metrics,preds):
    s=p('All six algorithms score the same 84 cells. No model silently drops a difficult zero. The observed 2025 evaluation population has N=0; predictions are available for 84 cells, but actual holdout error is unavailable. Use the separate observed-validation and scenario tables to avoid mistaking generated outcomes for empirical evidence.')
    s+=table(['Model','Final training N','Validation N','2025 prediction N','Observed 2025 N','Observed holdout MAE','RMSE','WAPE'],[[m['name'],m['training_N'],84,84,0,'N/A','N/A','N/A'] for m in metrics],'2025 observed holdout: outcomes unavailable','text-table')
    for split,title in [('validation','Observed 2024 chronological validation'),('scenario','2025 illustrative scenario comparison—not an observed test')]:
        rows=[[m['name'],m['training_N'],84,84,f(m[split]['MAE']),f(m[split]['RMSE']),f(m[split]['WAPE'])+'%',f(m[split]['Destination WAPE'])+'%',f(m[split]['RMSLE'])] for m in metrics]
        s+=table(['Model','Final training N','Validation N','2025 prediction N','MAE (USD)','RMSE (USD)','Cell WAPE','Destination WAPE','RMSLE'],rows,title)
    s+=p('Validation fit N differs from final fit N: Model 1 uses 84 in its earlier validation fit and 168 finally; Models 2–5 use 168 and 252. Model 6’s temporal prediction uses 2023 exporter OOF stages and the 84-pair lambda learned earlier; its final structural fit uses 84 2024 cells. Its positive-stage N is '+str(metrics[5]['positive_training_N'])+'. All validation errors have the same 84 observed 2024 targets; all scenario errors have the same 84 generated 2025 values.')
    s+='<div class="charts">'+''.join(f'<figure><img src="{BASE}results/figures/comparison-{kind}.svg" alt="Comparison of six models: observed 2024 validation and illustrative 2025 {kind.upper()}; actual 2025 error unavailable."><figcaption>Green: observed 2024 validation. Peach: generated 2025 scenario. Actual 2025 test metrics are unavailable.</figcaption></figure>' for kind in ['wape','rmse'])+'</div>'
    approaches=[['1 · Dynamic OLS','One linear log plane plus observed trade lag','Stable additive effects and persistence; unstable collinear slopes'],['2 · Elastic Net','Same plane without lag, penalized slopes','Shrink correlated effects; penalty adds bias'],['3 · Forest','Average independent bootstrap tree fits','Threshold interactions; averages regions, little trend extrapolation'],['4 · MLP','Learned nonlinear tanh layers','Optimization and architecture sensitive; finite-budget warning'],['5 · Boosting','Sequential small residual trees','Greedy corrections; leaf size and binning regulate flexibility'],['6 · Two-stage dynamic','Probability × positive amount, then lag blend','Separate margins; common estimated adjustment speed']]
    s+=table(['Algorithm','What is learned','Assumption / consequence'],approaches,'Why identical raw data can yield different predictions','text-table')
    v=preds[preds.split=='validation_2024_observed'].pivot(index='observation_id',columns='model',values='predicted_exports_usd');span=v.max(axis=1)-v.min(axis=1);ids=span.nlargest(4).index
    rows=[]
    for oid in ids:
        r=preds[(preds.model==1)&(preds.observation_id==oid)&(preds.split=='validation_2024_observed')].iloc[0]
        rows.append([oid,r.exporter+' → '+r.destination,r.hs4_code,f(r.observed_exports_usd)]+[f(v.loc[oid,k]) for k in range(1,7)])
    s+=table(['ID','Pair','HS4','Observed (USD)']+['Model '+str(k)+' (USD)' for k in range(1,7)],rows,'Four validation cells with largest prediction spread—not selected for accuracy')
    s+=p('These differences arise from lag dependence, shrinkage, threshold partitions, nonlinear activations, boosting corrections and dollar calibration. Trees infer interactions that the additive linear rules cannot; the dynamic methods retain historical trade directly. Errors in transformed units become asymmetric dollar errors after exponentiation.')
    s+=p('The small estimated lambda makes Model 6 close to a persistence baseline. The scenario was constructed by modestly perturbing 2024 outcomes; it naturally rewards persistence. Its lower scenario WAPE is not independent confirmation of superior structural modelling. The large OLS validation error and MLP warning illustrate fragility, not definitive family rankings. Small differences between flexible estimators may reflect sample variation, serial dependence or settings rather than repeatable advantage.')
    return card('09 / Comparing All Six Algorithms',s,'comparison')

def follow_section(d,preds,traces):
    one=d[d.observation_id==263].iloc[0];origin=d[d.observation_id==179].iloc[0]
    s=p(f'<strong>Observation 263: Canada → United States, HS4 1001 (wheat and meslin), 2025.</strong> Its prior-year counterpart is observed cell 179, with USD {f(origin.observed_exports_usd)} of exports. The 2025 observed outcome is unavailable; the illustrative value is USD {f(one.illustrative_exports_usd)}. This cannot satisfy an observed-2025 case study, so both missing observed errors and explicitly labelled scenario errors are shown.')
    rows=[]
    for col in d:
        value=one[col];value=f(value) if isinstance(value,(float,np.floating)) else str(value)
        rows.append([col,escape(value)])
    s+=table(['Input field','Entire stored record'],rows,'Observation 263: complete input record','text-table compact')
    steps=['a + sum of standardized coefficient products; lag included','a + penalized coefficient products; no lag','Mean of 120 log-space tree outputs','16 → 8 tanh → 4 tanh → 1 linear output','Initial log mean + 150 shrinkage corrections','pi × calibrated positive amount; then lambda gap adjustment']
    rows=[]
    for t,step in zip(traces,steps):
        pred=t['prediction'];rows.append([metrics_name(t['model']), '14 economic inputs + HS2'+(' + trade lag' if t['model'] in [1,6] else ''),step,f(pred),'N/A','N/A','N/A',f(pred-one.illustrative_exports_usd),f(abs(pred-one.illustrative_exports_usd))])
    s+=table(['Algorithm','Key inputs','Main prediction operation','Predicted (USD)','Observed (USD)','Observed signed error','Observed absolute error','Scenario signed error (USD)','Scenario absolute error (USD)'],rows,'One 2025 record through all six rules','text-table')
    s+=p('The raw outcome definition is identical, but the mathematical operations are not. Even numeric scaling differs because models use different training subsets. OLS and the composite use the 2024 trade state; the others do not. The forest averages leaves before the dollar inverse, boosting adds corrections before the inverse, and the MLP applies nonlinear hidden transformations. Model 6 uses log dollars on its positive subset and log1p dollars for dynamics.')
    # A real observed counterpart provides an empirical six-model follow-through too.
    rows=[]
    for k in range(1,7):
        r=preds[(preds.model==k)&(preds.observation_id==179)&(preds.split=='validation_2024_observed')].iloc[0]
        rows.append([metrics_name(k),f(r.observed_exports_usd),f(r.predicted_exports_usd),f(r.signed_error_usd),f(r.absolute_error_usd)])
    s+='<h3>The genuinely observed counterpart: cell 179 in 2024</h3>'+p('These predictions use 2023 information, before the 2024 outcomes. They were generated by the earlier validation fits, not the final fits used in the 2025 worked example. Showing both years keeps the empirical comparison separate from the scenario.')
    s+=table(['Algorithm','Observed 2024 (USD)','Predicted 2024 (USD)','Signed error (USD)','Absolute error (USD)'],rows,'Observed Canada → United States wheat validation cell 179')
    return card('10 / Follow One Trade Observation Through All Six Models',s,'follow')
def metrics_name(k):return ['Dynamic linear regression','Elastic Net regression','Random Forest regression','Multilayer Perceptron','Single-stage gradient boosting','Dynamic two-stage boosted partial adjustment'][k-1]

if __name__=='__main__':build()
