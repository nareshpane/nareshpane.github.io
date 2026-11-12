"""Educational re-estimation; original page, code, models and datasets read-only.

Temporal validation is separated from final fits and scenario evaluation.
Outputs contain exact numeric traces, not coefficients transcribed by hand.
"""
from pathlib import Path
import sys, os, json, platform, warnings, hashlib
os.environ.setdefault('OMP_NUM_THREADS','2')
os.environ.setdefault('OPENBLAS_NUM_THREADS','2')
sys.dont_write_bytecode=True
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
import numpy as np
import pandas as pd
import scipy, sklearn
from scipy.stats import spearmanr
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler,OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.pipeline import make_pipeline
from sklearn.linear_model import LinearRegression,ElasticNet
from sklearn.ensemble import RandomForestRegressor,HistGradientBoostingRegressor,HistGradientBoostingClassifier
from sklearn.neural_network import MLPRegressor
from sklearn.metrics import log_loss,brier_score_loss
from threadpoolctl import threadpool_limits
from dataset_build import validate,sha

NAMES=['Dynamic linear regression','Elastic Net regression','Random Forest regression','Multilayer Perceptron','Single-stage gradient boosting','Dynamic two-stage boosted partial adjustment']
NUM=['log_exporter_gdp','log_destination_gdp','log_exporter_population','log_destination_population','exporter_manufacturing','destination_manufacturing','exporter_internet','destination_internet','log_distance','contiguity','common_language','log_external_supply','log_external_demand','log_world_demand']
FEATURES=NUM+['HS2'];SEED=338
BOOST=dict(max_iter=150,max_leaf_nodes=7,learning_rate=.05,random_state=SEED,early_stopping=False)
FIT_RECORDS=[];CHECKS=[]

def clean(x):
    if isinstance(x,dict):return {str(k):clean(v) for k,v in x.items()}
    if isinstance(x,(list,tuple,np.ndarray)):return [clean(v) for v in x]
    if isinstance(x,(np.integer,)):return int(x)
    if isinstance(x,(np.floating,float)):return float(x) if np.isfinite(x) else None
    if isinstance(x,np.bool_):return bool(x)
    return x
def write(name,x):
    (ROOT/'results'/name).write_text(json.dumps(clean(x),indent=2,ensure_ascii=False,allow_nan=False),encoding='utf-8')
def records(d):return clean(d.to_dict('records'))
def features(d):
    d=d.copy()
    for col in ['exporter_gdp','destination_gdp','exporter_population','destination_population']:d['log_'+col]=np.log(d[col])
    d['log_distance']=np.log(d.distance_km)
    for src,out in [('external_supply_usd','log_external_supply'),('external_demand_usd','log_external_demand'),('world_external_demand_usd','log_world_demand')]:d[out]=np.log1p(d[src])
    d['HS2']=d.hs4_code.str[:2]
    return d
def prep(dynamic=False):
    return ColumnTransformer([('numeric',make_pipeline(SimpleImputer(strategy='median'),StandardScaler()),NUM+(['lag_log'] if dynamic else [])),('category',OneHotEncoder(handle_unknown='ignore',sparse_output=False,drop='first' if dynamic else None),['HS2'])])
def estimator(k,params=None):
    params=params or {'alpha':.01,'l1_ratio':.2}
    if k==1:return LinearRegression()
    if k==2:return ElasticNet(**params,max_iter=5000,tol=1e-4,random_state=SEED)
    if k==3:return RandomForestRegressor(n_estimators=120,max_depth=16,min_samples_leaf=8,max_features=.7,bootstrap=True,n_jobs=1,random_state=SEED)
    if k==4:return MLPRegressor(hidden_layer_sizes=(8,4),activation='tanh',solver='adam',alpha=.1,batch_size=64,learning_rate_init=.001,max_iter=2000,early_stopping=False,n_iter_no_change=2000,tol=0,random_state=SEED)
    return HistGradientBoostingRegressor(**BOOST)
def fit(k,d,params=None,context='final'):
    cols=FEATURES+(['lag_log'] if k==1 else [])
    p=make_pipeline(prep(k==1),estimator(k,params))
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter('always');p.fit(d[cols],np.log1p(d.observed_exports_usd))
    raw=np.maximum(np.expm1(p.predict(d[cols])),0)
    c=float(d.observed_exports_usd.sum()/raw.sum())
    design=np.column_stack([np.ones(len(d)),p[0].transform(d[cols])])
    FIT_RECORDS.append({'model':k,'context':context,'training_ids':d.observation_id.tolist(),'rows':len(d),'calibration':c,'settings':p[-1].get_params(),'warnings':[str(w.message) for w in caught],
       'design_columns_with_intercept':design.shape[1],'matrix_rank':int(np.linalg.matrix_rank(design)),'condition_number':float(np.linalg.cond(design))})
    return p,c
def predict(k,f,d):
    p,c=f;return c*np.maximum(np.expm1(p.predict(d[FEATURES+(['lag_log'] if k==1 else [])])),0)
def twofit(d,context):
    cls=make_pipeline(prep(),HistGradientBoostingClassifier(**BOOST))
    cls.fit(d[FEATURES],d.observed_exports_usd.gt(0).astype(int))
    pos=d[d.observed_exports_usd>0]
    reg=make_pipeline(prep(),HistGradientBoostingRegressor(**BOOST)).fit(pos[FEATURES],np.log(pos.observed_exports_usd))
    c=float(pos.observed_exports_usd.sum()/np.exp(reg.predict(pos[FEATURES])).sum())
    FIT_RECORDS.append({'model':6,'context':context,'rows':len(d),'positive_rows':len(pos),'training_ids':d.observation_id.tolist(),'positive_ids':pos.observation_id.tolist(),'calibration':c,'settings':BOOST,'warnings':[]})
    return cls,reg,c
def twopredict(f,d):
    cls,reg,c=f;prob=cls.predict_proba(d[FEATURES])[:,1];ell=reg.predict(d[FEATURES]);return prob*c*np.exp(ell),prob,ell
def oof(d):
    pot=np.zeros(len(d));prob=np.zeros(len(d))
    for ex in sorted(d.exporter.unique()):
        ix=d.exporter.eq(ex).to_numpy();f=twofit(d[~ix],f'OOF {int(d.year.iloc[0])} exclude {ex}')
        pot[ix],prob[ix],_=twopredict(f,d[ix])
    return pot,prob
def dynamic(lag,pot,lam):return np.maximum(np.expm1(np.log1p(lag)+lam*(np.log1p(pot)-np.log1p(lag))),0)
def transition(origin,target):
    # Stable keys are checked, not positionally assumed from source-file order.
    cols=['exporter','destination','hs4_code']
    assert origin[cols].reset_index(drop=True).equals(target[cols].reset_index(drop=True))
    d=origin.copy();d['lag_log']=np.log1p(origin.observed_exports_usd.to_numpy())
    d['input_observation_id']=origin.observation_id.to_numpy()
    d['observation_id']=target.observation_id.to_numpy();d['year']=target.year.to_numpy()
    d['observed_exports_usd']=target.observed_exports_usd.to_numpy()
    return d
def metrics(a,p,d):
    a=np.asarray(a,float);p=np.asarray(p,float);e=p-a
    if not np.isfinite(a).all():return {'N':0,'reason':'No observed 2025 bilateral outcomes in archive'}
    den=a.sum();dest=pd.DataFrame({'a':a,'p':p,'dest':d.destination.to_numpy()}).groupby('dest')[['a','p']].sum()
    rank=spearmanr(dest.a,dest.p).statistic
    z=np.log1p(p)-np.log1p(a)
    return clean({'N':len(a),'MSE':np.mean(e**2),'RMSE':np.sqrt(np.mean(e**2)),'MAE':np.mean(abs(e)),
      'WAPE':100*sum(abs(e))/den if den else None,'Destination WAPE':100*sum(abs(dest.p-dest.a))/den if den else None,
      'RMSLE':np.sqrt(np.mean(z*z)),'R2':1-sum(e*e)/sum((a-a.mean())**2) if np.var(a)>0 else None,
      'rank_correlation':rank,'top_overlap_3':len(set(dest.a.nlargest(3).index)&set(dest.p.nlargest(3).index)),
      'mean_absolute_top3_error':float(abs(dest.p-dest.a).loc[dest.p.nlargest(3).index].mean()),
      'median_absolute_top3_error':float(abs(dest.p-dest.a).loc[dest.p.nlargest(3).index].median())})
def block(k,split,d,p):
    r=d[['observation_id','year','exporter','destination','hs4_code','observed_exports_usd','illustrative_exports_usd']].copy()
    r['model']=k;r['split']=split;r['predicted_exports_usd']=p
    r['signed_error_usd']=p-r.observed_exports_usd;r['absolute_error_usd']=abs(r.signed_error_usd)
    r['percentage_error']=100*r.signed_error_usd/r.observed_exports_usd.replace(0,np.nan)
    r['scenario_error_usd']=p-r.illustrative_exports_usd
    return r
def treepath(tree,x,names):
    t=tree.tree_;node=0;path=[]
    while t.children_left[node]!=t.children_right[node]:
        j=t.feature[node];left=x[j]<=t.threshold[node]
        path.append({'node':node,'feature':names[j],'value':x[j],'threshold':t.threshold[node],'branch':'left <=' if left else 'right >'})
        node=t.children_left[node] if left else t.children_right[node]
    return {'path':path,'leaf_node':node,'leaf_log_prediction':float(t.value[node,0,0]),'bootstrap_weighted_samples':float(t.weighted_n_node_samples[node])}
def trace(k,f,one,p):
    pipe,c=f;cols=FEATURES+(['lag_log'] if k==1 else []);x=pipe[0].transform(one[cols])[0];names=pipe[0].get_feature_names_out().tolist();e=pipe[-1]
    result={'model':k,'observation_id':int(one.observation_id.iloc[0]),'feature_names':names,'standardized_features':x.tolist(),
      'imputation_medians':pipe[0].named_transformers_['numeric'][0].statistics_.tolist(),
      'scaling_means':pipe[0].named_transformers_['numeric'][1].mean_.tolist(),'scaling_sd':pipe[0].named_transformers_['numeric'][1].scale_.tolist(),
      'log_prediction':float(pipe.predict(one[cols])[0]),'calibration':c,'prediction':float(p),'settings':e.get_params()}
    if k in [1,2]:
        b=e.coef_;value=float(e.intercept_+np.dot(x,b));assert np.isclose(value,result['log_prediction'],atol=1e-10)
        result.update({'intercept':e.intercept_,'coefficients':b,'contributions':x*b,'reconstructed_log_prediction':value})
    if k==3:
        trees=[float(t.predict([x])[0]) for t in e.estimators_];result.update({'first_tree':treepath(e.estimators_[0],x,names),'tree_predictions':trees})
        assert np.isclose(np.mean(trees),result['log_prediction'])
    if k==4:
        a=x.copy();layers=[]
        for i,(w,b) in enumerate(zip(e.coefs_,e.intercepts_)):
            z=a@w+b;out=np.tanh(z) if i<len(e.coefs_)-1 else z
            layers.append({'inputs':a,'weights':w,'biases':b,'preactivation':z,'activation':out});a=out
        assert np.isclose(float(a[0]),result['log_prediction'],atol=1e-10)
        result['layers']=layers;result['loss_curve']=e.loss_curve_;result['parameter_count']=sum(w.size for w in e.coefs_)+sum(b.size for b in e.intercepts_)
    if k==5:
        staged=[float(v[0]) for v in e.staged_predict([x])];baseline=float(e._baseline_prediction[0,0])
        result.update({'baseline':baseline,'staged_predictions':staged,'updates':np.diff([baseline]+staged)})
        assert np.isclose(staged[-1],result['log_prediction'])
    assert np.isclose(c*max(np.expm1(result['log_prediction']),0),p,rtol=1e-10)
    CHECKS.append(f'Model {k}: independent prediction arithmetic reconstructed')
    return clean(result)

def run():
    (ROOT/'results').mkdir(exist_ok=True)
    d=pd.read_csv(ROOT/'toy_trade_dataset.csv',dtype={'hs4_code':str});validate(d)
    d=features(d);years={y:d[d.year==y].reset_index(drop=True) for y in [2022,2023,2024,2025]}
    a,b,c,q=[years[y] for y in [2022,2023,2024,2025]]
    # All 2025 predictor rows deliberately have macro_year=2024. Never log their targets.
    assert q.macro_year.eq(2024).all() and q.observed_exports_usd.isna().all()
    dyntrain=transition(a,b);dynvalid=transition(b,c);dynfinal=pd.concat([dyntrain,dynvalid],ignore_index=True)
    q['lag_log']=np.log1p(c.observed_exports_usd.to_numpy())
    dev=pd.concat([a,b],ignore_index=True);full=pd.concat([a,b,c],ignore_index=True)
    # Predeclared Elastic Net candidates tune using only the 2022→2023 transition.
    tune=[]
    for alpha in [.01,.1]:
        for ratio in [.2,.8]:
            f=fit(2,a,{'alpha':alpha,'l1_ratio':ratio},'2022 fit for 2023 tuning')
            logpred=f[0].predict(a[FEATURES]);loss=np.mean((logpred-np.log1p(b.observed_exports_usd))**2)
            tune.append({'alpha':alpha,'l1_ratio':ratio,'next_year_log_MSE':float(loss)})
    chosen=min(tune,key=lambda t:t['next_year_log_MSE']);params={x:chosen[x] for x in ['alpha','l1_ratio']}
    allblocks=[];summaries=[];traces=[]
    chosen_ix=int(q.index[(q.exporter=='Canada')&(q.destination=='United States')&(q.hs4_code=='1001')][0]);one=q.iloc[[chosen_ix]].copy()
    for k in range(1,6):
        print('Fit model',k,flush=True)
        tr=dyntrain if k==1 else dev;va=dynvalid if k==1 else b.copy()
        # Static 2024 validation uses 2023 X, not full-year realized 2024 X.
        if k!=1:va['observation_id']=c.observation_id.to_numpy();va['year']=2024;va['observed_exports_usd']=c.observed_exports_usd.to_numpy()
        vf=fit(k,tr,params if k==2 else None,'pre-2024 temporal validation');vp=predict(k,vf,va)
        ft=dynfinal if k==1 else full;f=fit(k,ft,params if k==2 else None);p=predict(k,f,q);tp=predict(k,f,ft)
        allblocks.extend([block(k,'training_final',ft,tp),block(k,'validation_2024_observed',va,vp),block(k,'forecast_2025_unobserved',q,p)])
        summaries.append({'model':k,'name':NAMES[k-1],'training_N':len(ft),'validation_training_N':len(tr),'validation_N':84,'holdout_predictions_N':84,'observed_holdout_N':0,
          'training':metrics(ft.observed_exports_usd,tp,ft),'validation':metrics(c.observed_exports_usd,vp,c),
          'holdout_observed':metrics(q.observed_exports_usd,p,q),'scenario':metrics(q.illustrative_exports_usd,p,q),
          'by_sector_validation':{hs:metrics(c[c.hs4_code==hs].observed_exports_usd,vp[c.hs4_code.eq(hs)],c[c.hs4_code==hs]) for hs in ['1001','8703']}})
        traces.append(trace(k,f,one,p[chosen_ix]))
    print('Fit model 6 OOF and dynamic stages',flush=True)
    pa,proba=oof(a);gap=np.log1p(pa)-np.log1p(a.observed_exports_usd.to_numpy());change=np.log1p(b.observed_exports_usd.to_numpy())-np.log1p(a.observed_exports_usd.to_numpy())
    raw=float(gap@change/(gap@gap));lam=float(np.clip(raw,0,1))
    pb,probb=oof(b);vp=dynamic(b.observed_exports_usd.to_numpy(),pb,lam)
    f=twofit(c,'final structural 2024');potential,prob,ell=twopredict(f,q);p=dynamic(c.observed_exports_usd.to_numpy(),potential,lam)
    trainpot,_,_=twopredict(f,c)
    allblocks.extend([block(6,'training_structural_2024',c,trainpot),block(6,'validation_2024_observed',c,vp),block(6,'forecast_2025_unobserved',q,p)])
    diag={}
    for yr,source,pr in [(2022,a,proba),(2023,b,probb)]:
        z=source.observed_exports_usd.gt(0).astype(int)
        diag[str(yr)]={'N':84,'positive_N':int(z.sum()),'log_loss':log_loss(z,pr,labels=[0,1]),'Brier':brier_score_loss(z,pr),'accuracy':float(np.mean((pr>=.5)==z)),
          'probability_bins':[{'bin':f'{lo:.1f}–{lo+.2:.1f}','N':int(((pr>=lo)&(pr<=lo+.2 if lo==.8 else pr<lo+.2)).sum()),
          'mean_probability':float(pr[(pr>=lo)&(pr<=lo+.2 if lo==.8 else pr<lo+.2)].mean()) if ((pr>=lo)&(pr<=lo+.2 if lo==.8 else pr<lo+.2)).any() else None,
          'positive_fraction':float(z.to_numpy()[(pr>=lo)&(pr<=lo+.2 if lo==.8 else pr<lo+.2)].mean()) if ((pr>=lo)&(pr<=lo+.2 if lo==.8 else pr<lo+.2)).any() else None} for lo in [0,.2,.4,.6,.8]]}
    summaries.append({'model':6,'name':NAMES[5],'training_N':84,'positive_training_N':int(c.observed_exports_usd.gt(0).sum()),'validation_training_N':84,'lambda_training_N':84,'validation_N':84,'holdout_predictions_N':84,'observed_holdout_N':0,
      'training':metrics(c.observed_exports_usd,trainpot,c),'validation':metrics(c.observed_exports_usd,vp,c),'holdout_observed':metrics(q.observed_exports_usd,p,q),'scenario':metrics(q.illustrative_exports_usd,p,q),
      'by_sector_validation':{hs:metrics(c[c.hs4_code==hs].observed_exports_usd,vp[c.hs4_code.eq(hs)],c[c.hs4_code==hs]) for hs in ['1001','8703']},'classification':diag})
    lag=float(c.observed_exports_usd.iloc[chosen_ix]);pred=float(p[chosen_ix]);pi=float(prob[chosen_ix]);lf=float(ell[chosen_ix]);cc=f[2]
    reconstructed=float(np.expm1(np.log1p(lag)+lam*(np.log1p(pi*cc*np.exp(lf))-np.log1p(lag))))
    assert np.isclose(reconstructed,pred,rtol=1e-10);CHECKS.append('Model 6: all three stages independently reconstructed')
    traces.append({'model':6,'observation_id':int(one.observation_id.iloc[0]),'probability':pi,'positive_log_prediction':lf,'calibration':cc,'positive_amount':float(cc*np.exp(lf)),
      'potential':float(potential[chosen_ix]),'lag':lag,'lambda_raw':raw,'lambda':lam,'lambda_numerator':float(gap@change),'lambda_denominator':float(gap@gap),'prediction':pred,
      'gap_log':float(np.log1p(potential[chosen_ix])-np.log1p(lag)),
      'classifier_logit':float(np.log(pi/(1-pi))),
      'classifier_staged_probabilities':[float(v[0,1]) for v in f[0][-1].staged_predict_proba(f[0][0].transform(one[FEATURES]))],
      'positive_regressor_staged_log_predictions':[float(v[0]) for v in f[1][-1].staged_predict(f[1][0].transform(one[FEATURES]))]})
    pd.DataFrame({'origin_id':a.observation_id,'target_id':b.observation_id,'oof_potential':pa,'oof_probability':proba,'lag_log':np.log1p(a.observed_exports_usd),'gap':gap,'change':change,'gap_times_change':gap*change,'gap_squared':gap*gap}).to_csv(ROOT/'results/lambda_estimation.csv',index=False)
    pd.DataFrame({'observation_id':b.observation_id,'oof_potential':pb,'oof_probability':probb}).to_csv(ROOT/'results/structural_oof_2023.csv',index=False)
    out=pd.concat(allblocks,ignore_index=True);out.to_csv(ROOT/'results/predictions.csv',index=False,float_format='%.17g')
    # Freeze every forecast before the scenario comparison is exported; scenario never fitted.
    freeze=out[out.year==2025][['model','observation_id','predicted_exports_usd']]
    freeze.to_csv(ROOT/'results/frozen_2025_predictions.csv',index=False,float_format='%.17g')
    write('forecast_manifest.json',{'sha256':sha(ROOT/'results/frozen_2025_predictions.csv'),'origin_covariate_year':2024,'observed_holdout_N':0,'scenario_used_in_fitting':False,'seed':SEED})
    write('predictions.json',records(out));write('metrics.json',summaries);write('worked_examples.json',traces);write('fit_records.json',FIT_RECORDS)
    write('elastic_tuning.json',{'design':'Fit 2022 targets, predict 2023 using 2022 X; no 2024/2025 target in selection','candidates':tune,'chosen':chosen})
    # Concrete, manageable feature matrix excerpt; full vectors downloadable.
    write('matrix_excerpt.json',{'rows':records(full.head(4)[['observation_id']+NUM[:4]+['log_distance','contiguity','HS2']]),'static_dimension':[252,16],'dynamic_dimension_with_intercept':[168,17]})
    write('follow_observation.json',clean(one.iloc[0].to_dict()))
    write('quality_checks.json',{'checks':CHECKS+['No 2025 outcome in fit IDs','All six forecast exactly same 84 IDs','All forecast-origin covariates are 2024','Grid structure validated'],
      'versions':{'python':platform.python_version(),'numpy':np.__version__,'pandas':pd.__version__,'scipy':scipy.__version__,'scikit_learn':sklearn.__version__},
      'dataset_sha256':sha(ROOT/'toy_trade_dataset.csv'),'seed':SEED})
    assert all(r['training_ids'] and max(r['training_ids'])<=252 for r in FIT_RECORDS)
    assert freeze.groupby('model').size().eq(84).all() and freeze.groupby('model').observation_id.apply(set).map(lambda s:s==set(range(253,337))).all()
    pd.DataFrame([{'model':r['name'],'training_N':r['training_N'],'validation_N':84,'holdout_prediction_N':84,'observed_holdout_N':0,**{'validation_'+k:v for k,v in r['validation'].items()},**{'scenario_'+k:v for k,v in r['scenario'].items()}} for r in summaries]).to_csv(ROOT/'results/model_comparison.csv',index=False)
    print(pd.DataFrame([{'model':r['model'],'validation_WAPE':r['validation']['WAPE'],'scenario_WAPE':r['scenario']['WAPE']} for r in summaries]).to_string(index=False),flush=True)

if __name__=='__main__':
    with threadpool_limits(limits=2):run()
