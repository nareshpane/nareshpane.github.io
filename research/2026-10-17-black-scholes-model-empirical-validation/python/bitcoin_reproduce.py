"""Rights-gated workflow. The UWA adapter records actual fields and empirical execution separately from fixtures.

No inferred schema or automatic permission. A human-reviewed manifest is required.
All results remain private; no command here publishes or edits the research page.
"""
import argparse,json,platform,subprocess,tempfile
from pathlib import Path
import numpy as np
import pandas as pd
import scipy
from bitcoin_data import require_rights,download_csv,inspect_csv,file_sha256,utc_times,match_underlying
from bitcoin_volatility import match_volatility,daily_estimators
from bitcoin_pricing import premium_usd,benchmark
from bitcoin_analysis import normalize_chunk,match_rates,quality_filters,chronological_split,calibrate,evaluate,metrics,paired,date_cluster,MODELS

PROJECT=Path(__file__).resolve().parents[1]
DATA_ROOT=PROJECT/'bitcoin_data'

def storage(path:Path,role:str,*,synthetic_fixture:bool=False)->Path:
    """Persistent Bitcoin inputs/results stay in the specified ignored tree.

    Only labelled software-test fixtures may use short-lived OS temporary paths.
    Ignore does not provide HTTP access control; never deploy bitcoin_data.
    """
    if role not in ['raw','processed','private']:raise ValueError('Unknown data-storage role')
    p=Path(path).resolve();allowed=(DATA_ROOT/role).resolve()
    if p!=allowed and allowed not in p.parents:
        temp=Path(tempfile.gettempdir()).resolve()
        if not (synthetic_fixture and temp in p.parents):raise ValueError('Bitcoin data must stay inside project bitcoin_data/'+role)
        return p
    result=subprocess.run(['git','check-ignore','--quiet',str(p/'guard.txt')],capture_output=True,cwd=PROJECT)
    if result.returncode!=0:raise ValueError('Project bitcoin_data must be Git-ignored')
    return p

def read_history(path:Path,mapping:dict)->pd.DataFrame:
    if not mapping.get('evidence_reference') or not mapping.get('price_definition'):raise ValueError('Underlying provenance/definition required')
    frames=[]
    for chunk in pd.read_csv(path,usecols=[mapping['timestamp'],mapping['price']],chunksize=100000):
        h=pd.DataFrame({'timestamp':utc_times(chunk[mapping['timestamp']],source_timezone=mapping.get('timezone'),unit=mapping.get('timestamp_unit')),
                        'spot':pd.to_numeric(chunk[mapping['price']],errors='coerce')})
        # Do not silently impute a missing price. Missing bars become gaps.
        frames.append(h[h.timestamp.notna()&np.isfinite(h.spot)&(h.spot>0)])
    if not frames:raise ValueError('Empty underlying input')
    return pd.concat(frames,ignore_index=True).sort_values('timestamp')

def run(manifest:dict,rights:dict,raw_dir:Path,private_dir:Path,processed_dir:Path|None=None)->dict:
    require_rights(rights)
    if manifest.get('data_class') not in ['market_observations','synthetic_test_fixture']:raise ValueError('Explicit market/fixture data classification required')
    if not manifest.get('schema_evidence') or not manifest.get('conversion_basis_evidence') or not manifest.get('rate_availability_evidence'):
        raise ValueError('Verified schema, conversion and rate availability evidence are required')
    fixture=manifest['data_class']=='synthetic_test_fixture'
    raw=storage(raw_dir,'raw',synthetic_fixture=fixture);out=storage(private_dir,'private',synthetic_fixture=fixture)
    processed=storage(processed_dir or (out.parent/'processed' if fixture else DATA_ROOT/'processed'),'processed',synthetic_fixture=fixture)
    if (out/'bitcoin-results.json').exists():raise FileExistsError('Preserve existing report; choose a new private output directory')
    if (processed/'analytical-observations.csv.gz').exists():raise FileExistsError('Preserve existing processed dataset; choose a new processed run directory')
    input_processed=processed if fixture else storage(DATA_ROOT/'processed','processed')
    directories={key:(input_processed if manifest[key].get('directory')=='processed' else raw) for key in ['options','history','rates']}
    paths={key:directories[key]/manifest[key]['file'] for key in directories}
    for key,path in paths.items():
        if path.resolve().parent!=directories[key].resolve():raise ValueError('Manifest filenames must stay inside their selected data directory')
        if file_sha256(path)!=manifest[key]['sha256']:raise ValueError('Source checksum mismatch: '+key)
    inventories={}
    for key,path in paths.items():
        m=manifest[key]
        print('Inventory:',key,path.name,flush=True)
        inventories[key]=inspect_csv(path,rights,timestamp_field=m.get('timestamp',m.get('available_at')),contract_field=m.get('contract'),
            id_field=m.get('trade_id'),source_timezone=m.get('timezone'),unit=m.get('timestamp_unit'))
    history=read_history(paths['history'],manifest['history']);estimates=daily_estimators(history)
    matching_history=history.copy()
    matching_history.timestamp=matching_history.timestamp+pd.Timedelta(seconds=manifest['history'].get('availability_delay_seconds',0))
    rates_raw=pd.read_csv(paths['rates']);rmap=manifest['rates']
    if rmap.get('observation_date') not in rates_raw:raise ValueError('Actual rate observation-date column required as well as availability time')
    rates=pd.DataFrame({'available_at':utc_times(rates_raw[rmap['available_at']],source_timezone=rmap.get('timezone'),unit=rmap.get('timestamp_unit')),
                        'annual_percent':pd.to_numeric(rates_raw[rmap['yield']],errors='coerce'),
                        'rate_observation_date':pd.to_datetime(rates_raw[rmap['observation_date']],errors='coerce').dt.strftime('%Y-%m-%d')})
    if (rates.annual_percent.notna()&rates.rate_observation_date.isna()).any():raise ValueError('Known yield with invalid observation date')
    mapping=manifest['options'];spec=manifest['specifications'];carry=float(manifest['carry_q'])
    if not manifest.get('carry_evidence'):raise ValueError('Independent carry assumption must be explicit')
    retained=[];counts={};seen_ids=set();seen_rows=set();offset=0
    def count(key,n):counts[key]=counts.get(key,0)+int(n)
    dtypes={mapping[k]:'string' for k in ['contract','trade_id'] if mapping.get(k)}
    dtypes.update(mapping.get('dtypes',{}))
    for chunk in pd.read_csv(paths['options'],chunksize=100000,dtype=dtypes):
        print('Normalize source rows',offset+1,'to',offset+len(chunk),flush=True)
        count('raw',len(chunk));f=normalize_chunk(chunk,mapping,spec,paths['options'].name,offset);offset+=len(chunk)
        f=f[f.timestamp.notna()];count('valid_timestamp',len(f))
        if mapping.get('exact_duplicate_fields'):
            hashes=pd.util.hash_pandas_object(chunk[mapping['exact_duplicate_fields']],index=False).reindex(f.index)
            keep=[]
            for value in hashes:
                keep.append(value not in seen_rows);seen_rows.add(value)
            f=f.loc[keep];count('unique_exact_source_rows',len(f))
        if 'trade_id' in f:
            keep=[]
            for value in f.trade_id:
                valid=pd.notna(value) and str(value) not in seen_ids
                keep.append(valid)
                if pd.notna(value):seen_ids.add(str(value))
            f=f.loc[keep]
        count('unique_known_trade_id' if mapping.get('trade_id') else 'trade_id_unavailable_not_deduplicated',len(f))
        if f.empty:continue
        f=match_underlying(f,matching_history,max_lag=manifest.get('max_underlying_lag','5min'))
        if 'source_spot' in f:
            f['matched_history_spot']=f.spot
            f['spot']=f.source_spot
        f=match_volatility(f,history,estimates=estimates);f=match_rates(f,rates)
        f,stages=quality_filters(f)
        for stage in stages:count(stage['stage'],stage['remaining'])
        if f.empty:continue
        f['observed_usd']=premium_usd(f.original_premium,f.spot,spec['premium_currency'],multiplier=spec['multiplier_btc'],per_contract=spec['premium_per_contract'])
        f['observed_btc']=f.observed_usd/f.spot
        for source,target in [('original_bid','bid_usd'),('original_ask','ask_usd')]:
            if source in f:f[target]=f[source]*(f.spot if spec['premium_currency']=='BTC' else 1)/(spec['multiplier_btc'] if spec['premium_per_contract'] else 1)
        retained.append(f)
    if not retained:raise ValueError('No valid observations; inspect private inventory and quality rules, not fabricated metrics')
    data=pd.concat(retained,ignore_index=True)
    train,test=chronological_split(data,manifest['training_cutoff_utc']);fit=calibrate(train,carry=carry)
    print('Training/evaluation',len(train),len(test),'fitted sigma',fit['sigma'],flush=True)
    ev=evaluate(test,fit['sigma'],carry=carry)
    previous=0;flow=[]
    for stage,remaining in counts.items():
        flow.append(dict(stage=stage,remaining=remaining,removed=previous-remaining if stage!='raw' else 0));previous=remaining
    report=dict(status='computed_privately_from_reviewed_inputs',data_class=manifest['data_class'],publication_automatically_performed=False,
        empirical_computation_ran=manifest['data_class']=='market_observations',inventories=inventories,
        source_evidence=manifest['schema_evidence'],source_files={k:dict(file=p.name,bytes=p.stat().st_size,sha256=file_sha256(p)) for k,p in paths.items()},
        software=dict(python=platform.python_version(),numpy=np.__version__,pandas=pd.__version__,scipy=scipy.__version__),
        counts=counts,quality_flow=flow,
        coverage=[str(data.timestamp.min()),str(data.timestamp.max())],evaluation_coverage=[str(ev.timestamp.min()),str(ev.timestamp.max())],
        fit=fit,cutoff_utc=manifest['training_cutoff_utc'],carry_q=carry,contract_specifications=spec,
        shared_contract_ids_across_time=int(len(set(train.contract)&set(test.contract))),
        underlying_definition=manifest['history']['price_definition'],conversion_basis_evidence=manifest['conversion_basis_evidence'],
        rate_availability_evidence=manifest['rate_availability_evidence'],matching_lag_seconds=ev.lag_seconds.quantile([0,.5,.95,1]).to_dict(),
        metrics={m:metrics(ev,m) for m in MODELS},paired={h:paired(ev,h) for h in ['H30','H60']},breakdowns={},
        uncertainty=date_cluster(ev),iv=dict(valid=int(ev.independent_iv.notna().sum()),invalid=int(ev.independent_iv.isna().sum()),
            meaning='Descriptive inversion of observed USD-equivalent premiums, not independent validation or an Experiment A input'),
        limitations='Spot BSM benchmark; conversion, index timing, financing, collateral, fees, jumps and model misspecification remain separate issues. Date clusters may be serially dependent.')
    if 'matched_history_spot' in ev:
        report['underlying_basis']={'valuation':'Co-recorded source index_price; its individual update time is not supplied',
            'volatility':'Separate backward-matched historical price proxy',
            'median_absolute_relative_gap':float(abs(ev.spot/ev.matched_history_spot-1).median()),
            'p95_absolute_relative_gap':float(abs(ev.spot/ev.matched_history_spot-1).quantile(.95))}
    for dim in ['kind','term','money']:
        report['breakdowns'][dim]=[dict(group=str(k),models={m:metrics(g,m) for m in MODELS},paired=paired(g)) for k,g in ev.groupby(dim,observed=True)]
    report['breakdowns']['date']=[dict(group=str(k),models={m:metrics(g,m) for m in MODELS}) for k,g in ev.groupby(ev.timestamp.dt.date)]
    report['sensitivities']={}
    valid_iv=ev.independent_iv.notna()
    report['iv']['comparison']={'n':int(valid_iv.sum()),'median_iv':float(ev.loc[valid_iv,'independent_iv'].median()) if valid_iv.any() else None,
        'median_H30':float(ev.sigma_H30.median()),'median_H60':float(ev.sigma_H60.median()),'fitted_sigma':fit['sigma']}
    if 'provider_iv' in ev:
        both=valid_iv&ev.provider_iv.notna()
        report['iv']['provider_comparison']=dict(n=int(both.sum()),mean_absolute_gap=float(abs(ev.loc[both,'independent_iv']-ev.loc[both,'provider_iv']).mean()) if both.any() else None,
            caveat='Only explicitly mapped units; vendor forward, financing and timing conventions may differ from spot benchmark')
    else:report['iv']['provider_comparison']={'status':'not_compared; absent field or unverified units'}
    for name,mask in [('moneyness_090_110',ev.m.between(.9,1.1)),('maturity_14_60',(ev['T']*365).between(14,60))]:
        sub=ev[mask];report['sensitivities'][name]={m:metrics(sub,m) for m in MODELS} if len(sub) else {'status':'no_observations'}
    for label,rate in [('zero_rate',0.),('rate_minus_50bp',ev.rate-.005),('rate_plus_50bp',ev.rate+.005)]:
        ev[label]=benchmark(ev.spot,ev.strike,ev['T'],rate,ev.sigma_H30,carry,ev.kind)['usd'];report['sensitivities'][label]=metrics(ev,label)
    out.mkdir(parents=True,exist_ok=True)
    processed.mkdir(parents=True,exist_ok=True)
    data.to_csv(processed/'analytical-observations.csv.gz',index=False,compression='gzip')
    estimates.to_csv(processed/'historical-volatility.csv.gz',index=False,compression='gzip')
    (processed/'source-inventory.json').write_text(json.dumps(inventories,indent=2),encoding='utf8')
    ev.to_csv(out/'evaluation-private.csv',index=False)
    (out/'bitcoin-results.json').write_text(json.dumps(report,indent=2,allow_nan=False),encoding='utf8')
    from bitcoin_figures import render_private
    render_private(out,ev,history,estimates,report)
    return report

def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('action',choices=['download','inventory','run']);p.add_argument('--rights',required=True)
    p.add_argument('--raw-dir',type=Path,default=DATA_ROOT/'raw');p.add_argument('--private-dir',type=Path,default=DATA_ROOT/'private')
    p.add_argument('--processed-dir',type=Path,default=DATA_ROOT/'processed');p.add_argument('--manifest',type=Path)
    p.add_argument('--url');p.add_argument('--filename',choices=['instrument_name.csv','Deribit_w_IV.csv','Price_5m.csv']);p.add_argument('--contract-field')
    a=p.parse_args();rights=json.loads(Path(a.rights).read_text(encoding='utf8'));require_rights(rights)
    raw=storage(a.raw_dir,'raw')
    processed=storage(a.processed_dir,'processed')
    if a.action=='download':
        if not a.url or not a.filename:p.error('Download needs exact official attachment URL and filename')
        if a.filename!='instrument_name.csv':
            first=json.loads((processed/'instrument_name.inventory.json').read_text(encoding='utf8'))
            if not first.get('contracts') or first.get('unparsed') or first['sha256']!=file_sha256(raw/'instrument_name.csv'):raise ValueError('Validate the actual small file and contract field first')
        receipt=download_csv(a.url,raw/a.filename,rights)
        print(json.dumps(receipt,indent=2))
        if a.filename=='instrument_name.csv':
            inventory=inspect_csv(raw/a.filename,rights,contract_field=a.contract_field)
            processed.mkdir(parents=True,exist_ok=True)
            (processed/'instrument_name.inventory.json').write_text(json.dumps(inventory,indent=2),encoding='utf8')
            print('Small file inventory written; verify actual header/contract field before larger downloads.')
    elif a.action=='inventory':
        if not a.filename:p.error('Inventory requires --filename')
        inventory=inspect_csv(raw/a.filename,rights,contract_field=a.contract_field)
        processed.mkdir(parents=True,exist_ok=True)
        target=processed/(a.filename.removesuffix('.csv')+'.inventory.json');target.write_text(json.dumps(inventory,indent=2),encoding='utf8');print(json.dumps(inventory,indent=2))
    else:
        if not a.manifest or not a.private_dir:p.error('Run needs reviewed --manifest and --private-dir')
        result=run(json.loads(a.manifest.read_text(encoding='utf8')),rights,raw,a.private_dir,a.processed_dir)
        print('Private evaluation complete:',result['metrics']['H30']['n'],'common held-out observations. Public release is a separate gate.')

if __name__=='__main__':main()
