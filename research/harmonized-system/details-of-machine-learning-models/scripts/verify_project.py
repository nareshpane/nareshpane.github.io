"""Independent saved-output audit: grid, metrics, numerical traces, static links."""
from pathlib import Path
from html.parser import HTMLParser
import sys,json,hashlib,math
import numpy as np
import pandas as pd
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from dataset_build import validate,sha
PAGE=ROOT.parent/'details-of-machine-learning-models.html'

class Inspect(HTMLParser):
    def __init__(self):super().__init__();self.refs=[];self.ids=[];self.tables={};self.current=None;self.models=0;self.model_open=0
    def handle_starttag(self,tag,attrs):
        a=dict(attrs)
        if 'id' in a:self.ids.append(a['id'])
        if tag in ['img','script','link','a']:
            ref=a.get('src',a.get('href',''))
            if ref:self.refs.append(ref)
        if tag=='details' and a.get('class')=='model':self.models+=1;self.model_open+=int('open' in a)
        if tag=='tbody':self.current=a.get('id');self.tables.setdefault(self.current,0)
        if tag=='tr' and self.current:self.tables[self.current]+=1
    def handle_endtag(self,tag):
        if tag=='tbody':self.current=None

def run():
    tests=[]
    d=pd.read_csv(ROOT/'toy_trade_dataset.csv',dtype={'hs4_code':str});validate(d);tests.append('Dataset grid, IDs, no duplicates/self-trade, exact countries/years/headings')
    original=pd.read_csv(ROOT/'data/source_observed_subset.csv',dtype={'hs4_code':str})
    assert np.array_equal(original.observed_exports_usd,d.loc[d.year<2025,'observed_exports_usd'])
    manifest=json.loads((ROOT/'data/source_manifest.json').read_text());assert manifest['subset_sha256']==sha(ROOT/'data/source_observed_subset.csv')
    assert d.loc[d.year==2025,'macro_year'].eq(2024).all();assert d.loc[d.year==2025,'observed_exports_usd'].isna().all()
    assert d.loc[d.year==2025,'data_status'].eq('illustrative').all();tests.append('Historical values match hashed extract; 2025 targets unavailable and inputs historical')
    # Reconstruct the artificial procedure independently from its declared row order.
    old=d[d.year==2024];sc=d[d.year==2025];rng=np.random.default_rng(338)
    factor=np.exp(np.where(old.hs4_code=='1001',-.08,.06)+rng.normal(0,.10,84))
    np.testing.assert_allclose(sc.illustrative_exports_usd,old.observed_exports_usd.to_numpy()*factor,rtol=1e-14)
    tests.append('Scenario seed/formula independently reproduced; never stored as observed')
    pred=pd.read_csv(ROOT/'results/predictions.csv',dtype={'hs4_code':str});metrics=json.loads((ROOT/'results/metrics.json').read_text());traces=json.loads((ROOT/'results/worked_examples.json').read_text())
    assert np.isfinite(pred.predicted_exports_usd).all() and pred.predicted_exports_usd.ge(0).all()
    for k in range(1,7):
        forecast=pred[(pred.model==k)&(pred.year==2025)];assert set(forecast.observation_id)==set(range(253,337))
        assert forecast.observed_exports_usd.isna().all() and forecast.signed_error_usd.isna().all() and forecast.percentage_error.isna().all()
        for split,key,outcome in [('validation_2024_observed','validation','observed_exports_usd'),('forecast_2025_unobserved','scenario','illustrative_exports_usd')]:
            g=pred[(pred.model==k)&(pred.split==split)];a=g[outcome].to_numpy();b=g.predicted_exports_usd.to_numpy();e=b-a;m=metrics[k-1][key]
            assert len(g)==84
            np.testing.assert_allclose([m['MSE'],m['RMSE'],m['MAE'],m['WAPE'],m['RMSLE']], [np.dot(e,e)/84,np.linalg.norm(e)/math.sqrt(84),abs(e).mean(),100*abs(e).sum()/a.sum(),np.linalg.norm(np.log1p(b)-np.log1p(a))/math.sqrt(84)],rtol=1e-12)
            totals=pd.DataFrame({'a':a,'p':b,'dest':g.destination}).groupby('dest')[['a','p']].sum()
            np.testing.assert_allclose(m['Destination WAPE'],100*abs(totals.p-totals.a).sum()/a.sum(),rtol=1e-12)
            assert m['Destination WAPE']<=m['WAPE']+1e-9
        assert metrics[k-1]['holdout_observed']['N']==0
        t=traces[k-1];pvalue=t['prediction']
        # Check independent model operations and displayed 12-significant-digit arithmetic.
        rounded=lambda x:float(format(x,'.12g'))
        if k<6:
            if k in [1,2]:
                z=np.array(t['standardized_features']);b=np.array(t['coefficients']);value=t['intercept']+z@b
                rounded_log=rounded(t['intercept'])+sum(rounded(x)*rounded(y) for x,y in zip(z,b))
            elif k==3:value=np.mean(t['tree_predictions']);rounded_log=np.mean([rounded(v) for v in t['tree_predictions']])
            elif k==4:
                a=np.array(t['standardized_features'])
                for i,l in enumerate(t['layers']):
                    a=a@np.array(l['weights'])+np.array(l['biases']);a=np.tanh(a) if i<2 else a
                value=float(a[0]);rounded_log=rounded(t['log_prediction'])
            else:value=t['baseline']+sum(t['updates']);rounded_log=rounded(t['log_prediction'])
            assert math.isclose(value,t['log_prediction'],rel_tol=1e-11,abs_tol=1e-10)
            actual=t['calibration']*max(math.expm1(value),0)
            shown=rounded(t['calibration'])*max(math.expm1(rounded_log),0)
        else:
            pi=1/(1+math.exp(-t['classifier_logit']));pot=pi*t['calibration']*math.exp(t['positive_log_prediction'])
            lam=np.clip(t['lambda_numerator']/t['lambda_denominator'],0,1)
            actual=math.expm1(math.log1p(t['lag'])+lam*(math.log1p(pot)-math.log1p(t['lag'])))
            shown=math.expm1(rounded(math.log1p(t['prediction'])))
        assert math.isclose(actual,pvalue,rel_tol=1e-10,abs_tol=1e-6)
        assert math.isclose(shown,pvalue,rel_tol=1e-6,abs_tol=.01)
        tests.append(f'Model {k}: metrics, common coverage, inverse units and exact/rounded worked arithmetic independently checked')
    for r in json.loads((ROOT/'results/fit_records.json').read_text()):assert max(r['training_ids'])<=252
    tests.append('All training targets precede 2025; all explicit fit IDs audited')
    freeze=json.loads((ROOT/'results/forecast_manifest.json').read_text());assert freeze['sha256']==sha(ROOT/'results/frozen_2025_predictions.csv')
    html=PAGE.read_text(encoding='utf-8');parser=Inspect();parser.feed(html)
    assert parser.models==6 and parser.model_open==0
    assert parser.tables['dataset-body']==336
    assert all(parser.tables[f'pred-body-{k}']==168 for k in range(1,7))
    assert len(parser.ids)==len(set(parser.ids))
    for ref in parser.refs:
        if ref.startswith('#'):assert ref[1:] in parser.ids
        elif not ref.startswith(('http:','https:','mailto:')):assert (PAGE.parent/ref.split('#')[0]).is_file(),ref
    tests.append('336 static table rows, 6 collapsed model sections, 168 evaluation rows each, unique HTML IDs and all relative paths')
    index=(ROOT.parent/'harmonized-system-index.html').read_text(encoding='utf-8');assert 'href="details-of-machine-learning-models.html"' in index
    tests.append('Companion landing link verified')
    report={'status':'PASS','checks':tests,'relative_asset_links_checked':len(parser.refs),'dataset_sha256':sha(ROOT/'toy_trade_dataset.csv'),'forecast_sha256':freeze['sha256']}
    (ROOT/'results/static_verification.json').write_text(json.dumps(report,indent=2),encoding='utf-8')
    print(json.dumps(report,indent=2))
if __name__=='__main__':run()
