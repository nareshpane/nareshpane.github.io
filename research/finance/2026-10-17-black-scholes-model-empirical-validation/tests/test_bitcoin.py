"""Synthetic software fixtures only: no Bitcoin market data or findings."""
import io,json,math,sys,os,shutil,uuid,unittest
from contextlib import contextmanager
from pathlib import Path
from unittest.mock import patch
from urllib.error import HTTPError
import numpy as np
import pandas as pd
ASSET=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ASSET/'python'))
from pricing import greeks
from bitcoin_pricing import benchmark,inverse_payoff,premium_usd,rate_continuous,maturity,market_iv,forward_price
from bitcoin_data import parse_instrument,utc_times,match_underlying,require_rights,download_csv,inspect_csv,file_sha256,official_url
from bitcoin_volatility import daily_estimators,match_volatility,realized_five_minute
from bitcoin_analysis import normalize_chunk,match_rates,quality_filters,chronological_split,calibrate,evaluate,metrics,paired,date_cluster
from bitcoin_reproduce import run,storage,PROJECT,DATA_ROOT

@contextmanager
def fixture_directory():
    # Windows tempfile's restrictive ACL can exclude the managed runtime token.
    # Default-mode directories in the approved external task area avoid that ACL.
    parent=Path(os.environ.get('TEMP','/tmp'))/'bsm-bitcoin-framework-20261017'/'fixtures'
    path=parent/uuid.uuid4().hex
    path.mkdir(parents=True)
    try:yield str(path)
    finally:
        if path.resolve().parent!=parent.resolve():raise AssertionError('Fixture cleanup boundary')
        shutil.rmtree(path,ignore_errors=True)

RIGHTS={'processing':{'authorized':True,'evidence_reference':'SYNTHETIC FIXTURE ONLY; no market permission claimed','reviewed_by':'unit test','reviewed_on':'2026-10-17'}}

class BitcoinMathTests(unittest.TestCase):
    def test_independent_call_reference(self):self.assertAlmostEqual(float(benchmark(100,100,1,.05,.2)['usd']),10.450583572185565,places=10)
    def test_independent_put_reference(self):self.assertAlmostEqual(float(benchmark(100,100,1,.05,.2,kind='put')['usd']),5.573526022256971,places=10)
    def test_parity_and_bounds(self):
        for s in [70,100,140]:
            c=float(benchmark(s,100,.7,.03,.6,.01)['usd']);p=float(benchmark(s,100,.7,.03,.6,.01,'put')['usd'])
            self.assertAlmostEqual(c-p,s*math.exp(-.01*.7)-100*math.exp(-.03*.7),places=10)
            self.assertTrue(max(s*math.exp(-.01*.7)-100*math.exp(-.03*.7),0)<=c<=s*math.exp(-.01*.7))
            self.assertTrue(0<=p<=100*math.exp(-.03*.7))
    def test_greeks_finite_difference(self):
        args=[100,105,.5,.03,.5];h=.001
        delta=(benchmark(args[0]+h,*args[1:])['usd']-benchmark(args[0]-h,*args[1:])['usd'])/(2*h)
        self.assertAlmostEqual(float(delta),greeks(*args)['delta'],places=7)
        gamma=(benchmark(args[0]+h,*args[1:])['usd']-2*benchmark(*args)['usd']+benchmark(args[0]-h,*args[1:])['usd'])/h**2
        self.assertAlmostEqual(float(gamma),greeks(*args)['gamma'],places=6)
    def test_units_and_multiplier(self):
        self.assertEqual(float(premium_usd(.04,10000,'BTC',multiplier=1)),400)
        self.assertEqual(float(premium_usd(.08,10000,'BTC',multiplier=2,per_contract=True)),400)
        self.assertEqual(float(premium_usd(800,10000,'USD',multiplier=2,per_contract=True)),400)
        model=benchmark(10000,10000,.2,.02,.7);self.assertAlmostEqual(float(model['btc']*10000),float(model['usd']))
    def test_terminal_btc_payoff(self):
        self.assertAlmostEqual(float(inverse_payoff(12000,10000)),1/6)
        self.assertAlmostEqual(float(inverse_payoff(8000,10000,'put')),1/4)
        self.assertEqual(float(inverse_payoff(8000,10000)),0)
        self.assertAlmostEqual(float(inverse_payoff(12000,10000,multiplier=2)),1/3)
    def test_terminal_conversion_not_initial_spot(self):
        payoff=inverse_payoff(12000,10000);self.assertAlmostEqual(float(payoff*12000),2000)
        self.assertNotAlmostEqual(float(payoff*10000),2000)
    def test_invalid_kind_units_and_inputs(self):
        for fn in [lambda:benchmark(100,100,1,.02,.3,kind='future'),lambda:premium_usd(1,100,'EUR',multiplier=1),lambda:inverse_payoff(0,100),lambda:premium_usd(-1,100,'BTC',multiplier=1)]:
            with self.assertRaises(ValueError):fn()
    def test_iv_inversion_calls_and_puts(self):
        for kind in ['call','put']:
            target=benchmark(100,105,.4,.02,.75,kind=kind)['usd']
            self.assertAlmostEqual(float(market_iv(target,100,105,.4,.02,kind=kind)),.75,places=9)
        self.assertTrue(np.isnan(market_iv(200,100,100,1,.02)))
    def test_zero_vol_and_expiry(self):
        self.assertAlmostEqual(float(benchmark(100,90,0,.02,.4)['usd']),10)
        self.assertAlmostEqual(float(benchmark(100,90,1,.02,0)['usd']),100-90*math.exp(-.02))
    def test_forward_spot_consistency(self):
        forward=100*math.exp((.03-.01)*.7)
        self.assertAlmostEqual(float(forward_price(forward,105,.7,.03,.4)),float(benchmark(100,105,.7,.03,.4,.01)['usd']))
    def test_rate_approximation_and_missing(self):
        self.assertAlmostEqual(float(rate_continuous(2)),2*math.log(1.01),places=12)
        with self.assertRaises(ValueError):rate_continuous(np.nan)

class BitcoinDataTests(unittest.TestCase):
    def test_identifier_and_strike(self):
        x=parse_instrument('BTC-27DEC19-5500-C');self.assertEqual(x.strike,5500);self.assertEqual(x.kind,'call');self.assertEqual(str(x.expiration),'2019-12-27 08:00:00+00:00')
        self.assertEqual(parse_instrument('BTC-5JUN20-9500.5-P').kind,'put')
    def test_invalid_identifiers(self):
        for value in ['ETH-27DEC19-5500-C','BTC-PERPETUAL','BTC-31FEB20-100-C','BTC-27DEC19-0-C']:
            with self.assertRaises(ValueError):parse_instrument(value)
    def test_timezone_and_epoch_unit(self):
        a=utc_times(['2020-01-01T10:00:00+02:00']);self.assertEqual(a.iloc[0],pd.Timestamp('2020-01-01T08:00:00Z'))
        self.assertEqual(utc_times([0],unit='ms').iloc[0],pd.Timestamp('1970-01-01T00:00:00Z'))
        numeric_strings=utc_times(['0','missing'],unit='ms');self.assertEqual(numeric_strings.iloc[0],pd.Timestamp('1970-01-01T00:00:00Z'));self.assertTrue(pd.isna(numeric_strings.iloc[1]))
        with self.assertRaises(ValueError):utc_times([1577836800])
        with self.assertRaises(ValueError):utc_times(['2020-01-01 08:00:00'])
        self.assertEqual(utc_times(['2020-01-01 08:00:00'],source_timezone='UTC').iloc[0],a.iloc[0])
    def test_maturity_actual_seconds(self):
        self.assertAlmostEqual(maturity('2020-01-01T08:00Z','2020-01-02T20:00Z'),1.5/365)
        with self.assertRaises(ValueError):maturity('2020-01-01','2020-02-01')
    def test_underlying_backward_only_and_max_lag(self):
        h=pd.DataFrame({'timestamp':pd.to_datetime(['2020-01-01T00:00Z','2020-01-01T00:05Z'],utc=True),'spot':[100.,200.]})
        t=pd.DataFrame({'timestamp':pd.to_datetime(['2020-01-01T00:04Z','2020-01-01T00:11Z'],utc=True)})
        result=match_underlying(t,h);self.assertEqual(result.spot.iloc[0],100);self.assertEqual(result.lag_seconds.iloc[0],240);self.assertTrue(np.isnan(result.spot.iloc[1]))
    def test_rate_availability_and_missing_not_zero(self):
        rates=pd.DataFrame({'available_at':pd.to_datetime(['2020-01-01T20:00Z','2020-01-02T20:00Z'],utc=True),'annual_percent':[2.,np.nan]})
        trades=pd.DataFrame({'timestamp':pd.to_datetime(['2020-01-01T19:00Z','2020-01-02T21:00Z'],utc=True)})
        r=match_rates(trades,rates);self.assertTrue(np.isnan(r.rate.iloc[0]));self.assertAlmostEqual(r.rate.iloc[1],float(rate_continuous(2)))
    def test_rights_gate_before_any_network_or_folder(self):
        with fixture_directory() as temp:
            p=Path(temp)/'not-created/instrument_name.csv'
            with patch('bitcoin_data.build_opener') as mock:
                with self.assertRaises(PermissionError):download_csv('https://research-repository.uwa.edu.au/file.csv',p,{})
                mock.assert_not_called();self.assertFalse(p.parent.exists())
        with self.assertRaises(PermissionError):require_rights(RIGHTS,publication=True)
    def test_host_and_redirect_review(self):
        for url in ['http://research-repository.uwa.edu.au/x','https://www.deribit.com/api/x','https://research-repository.uwa.edu.au.evil.example/x']:
            with self.assertRaises(ValueError):official_url(url)
    def test_streamed_fixture_download_and_inventory(self):
        payload=b'contract,trade_id\nBTC-27MAR20-10000-C,one\nBTC-27MAR20-10000-P,two\n'
        class Response(io.BytesIO):
            status=200;url='https://research-repository.uwa.edu.au/fixture.csv';headers={'Content-Type':'text/csv','Content-Length':str(len(payload))}
        class Opener:
            def open(self,*a,**k):return Response(payload)
        with fixture_directory() as temp,patch('bitcoin_data.build_opener',return_value=Opener()):
            p=Path(temp)/'instrument_name.csv';receipt=download_csv(Response.url,p,RIGHTS)
            self.assertEqual(receipt['bytes'],len(payload));self.assertEqual(receipt['sha256'],file_sha256(p));self.assertEqual(receipt['status'],200)
            inv=inspect_csv(p,RIGHTS,contract_field='contract',id_field='trade_id');self.assertEqual(inv['rows'],2);self.assertEqual(inv['calls'],1);self.assertEqual(inv['puts'],1);self.assertEqual(inv['duplicate_ids'],0)
    def test_html_response_is_not_dataset(self):
        class Response(io.BytesIO):
            status=200;url='https://research-repository.uwa.edu.au/login';headers={'Content-Type':'text/html'}
        class Opener:
            def open(self,*a,**k):return Response(b'<html>Login</html>')
        with fixture_directory() as temp,patch('bitcoin_data.build_opener',return_value=Opener()):
            with self.assertRaises(ValueError):download_csv(Response.url,Path(temp)/'instrument_name.csv',RIGHTS)
    def test_manual_html_save_and_missing_fields_are_rejected(self):
        with fixture_directory() as temp:
            p=Path(temp)/'instrument_name.csv';p.write_text('<!DOCTYPE html><html>Denied</html>',encoding='utf8')
            with self.assertRaisesRegex(ValueError,'HTML'):inspect_csv(p,RIGHTS)
            p.write_text('contract\nBTC-27MAR20-10000-C\n',encoding='utf8')
            with self.assertRaisesRegex(ValueError,'absent'):inspect_csv(p,RIGHTS,contract_field='guessed_name')
            inventory=inspect_csv(p,RIGHTS,contract_field='contract')
            self.assertEqual(inventory['encoding'],'utf-8')
            p.write_text('contract\nBTC-27MAR20-10000-C\n',encoding='utf-8-sig')
            self.assertEqual(inspect_csv(p,RIGHTS,contract_field='contract')['encoding'],'utf-8-sig')
    def test_access_denial_is_not_retried(self):
        with fixture_directory() as temp,patch('bitcoin_data.build_opener') as factory:
            factory.return_value.open.side_effect=HTTPError('https://research-repository.uwa.edu.au/x',403,'Denied',{},None)
            with self.assertRaises(HTTPError):download_csv('https://research-repository.uwa.edu.au/x',Path(temp)/'instrument_name.csv',RIGHTS)
            self.assertEqual(factory.return_value.open.call_count,1)
    def test_one_transient_retry(self):
        payload=b'contract\nBTC-27MAR20-10000-C\n'
        class Response(io.BytesIO):
            status=200;url='https://research-repository.uwa.edu.au/x';headers={'Content-Type':'text/csv'}
        with fixture_directory() as temp,patch('bitcoin_data.build_opener') as factory,patch('bitcoin_data.time.sleep'):
            factory.return_value.open.side_effect=[HTTPError(Response.url,503,'Transient',{},None),Response(payload)]
            receipt=download_csv(Response.url,Path(temp)/'instrument_name.csv',RIGHTS)
            self.assertEqual(factory.return_value.open.call_count,2);self.assertEqual(receipt['bytes'],len(payload))
    def test_no_unreviewed_public_output_path(self):
        with self.assertRaises(ValueError):storage(ASSET/'data','private')
    def test_project_local_data_roles_and_ignore(self):
        self.assertEqual(PROJECT,ASSET)
        for role in ['raw','processed','private']:
            self.assertEqual(storage(DATA_ROOT/role,role),DATA_ROOT/role)
        with self.assertRaises(ValueError):storage(DATA_ROOT/'raw','private')
        with self.assertRaises(ValueError):storage(DATA_ROOT/'raw'/'..'/'..'/'escape','raw')
        with self.assertRaises(ValueError):storage(DATA_ROOT/'other','other')
    def test_external_storage_requires_temporary_synthetic_fixture(self):
        with fixture_directory() as temp:
            with self.assertRaises(ValueError):storage(Path(temp)/'raw','raw')
            self.assertEqual(storage(Path(temp)/'raw','raw',synthetic_fixture=True),Path(temp)/'raw')
    def test_attribution_and_access_status_are_explicit(self):
        html=(PROJECT.parent/'2026-10-17-black-scholes-model-empirical-validation.html').read_text(encoding='utf8')
        self.assertIn('Public Academic Dataset and Attribution',html)
        self.assertIn('Data Sources, Academic References, and Reproducibility',html)
        self.assertIn('178,236',html)
        self.assertIn('license version',html)
        self.assertIn('Market validation executed on independently recorded premiums.',html)
        self.assertIn('bitcoin_data/processed',html)

class BitcoinVolatilityTests(unittest.TestCase):
    def history(self):
        dates=pd.date_range('2019-10-01',periods=100,tz='UTC')+pd.Timedelta(hours=12)
        return pd.DataFrame({'timestamp':dates,'spot':10000*np.exp(.001*np.arange(100)+.04*np.sin(np.arange(100)/4))})
    def test_daily_sample_std_and_calendar_annualization(self):
        h=self.history();d=daily_estimators(h);returns=np.diff(np.log(h.spot.to_numpy()))
        for w in [30,60]:self.assertAlmostEqual(d.iloc[-1]['H'+str(w)],np.std(returns[-w:],ddof=1)*np.sqrt(365),places=12)
    def test_current_day_price_does_not_enter_prior_volatility(self):
        h=self.history();trade=pd.DataFrame({'timestamp':[h.timestamp.iloc[80]+pd.Timedelta(hours=1)]})
        a=match_volatility(trade,h);changed=h.copy();changed.loc[80:,'spot']*=3;b=match_volatility(trade,changed)
        self.assertEqual(a.H30.iloc[0],b.H30.iloc[0]);self.assertEqual(a.H60.iloc[0],b.H60.iloc[0])
    def test_missing_day_breaks_window(self):
        h=self.history().drop(index=85);d=daily_estimators(h);self.assertTrue(np.isnan(d.H30.iloc[-1]))
    def test_minimum_history(self):
        d=daily_estimators(self.history().iloc[:30]);self.assertTrue(d.H30.isna().all());self.assertTrue(d.H60.isna().all())
    def test_five_minute_quadratic_variation_units(self):
        dates=pd.date_range('2020-01-01',periods=289,freq='5min',tz='UTC');ret=.0001
        h=pd.DataFrame({'timestamp':dates,'spot':100*np.exp(ret*np.arange(289))})
        rv=realized_five_minute(h,1);self.assertAlmostEqual(rv.RV1.iloc[-1],np.sqrt(365*288*ret**2),places=10)
        self.assertTrue(realized_five_minute(h.drop(index=100),1).RV1.isna().all())

class BitcoinStatTests(unittest.TestCase):
    def fixture(self):
        return pd.DataFrame({'timestamp':pd.to_datetime(['2020-01-01T12:00Z','2020-01-02T12:00Z','2020-01-03T12:00Z'],utc=True),
            'contract':['a','a','b'],'record_key':['1','2','3'],'kind':['call','call','put'],'spot':[100.]*3,
            'observed_usd':[10.,10.,10.],'H30':[12.,14.,10.],'H60':[11.,13.,11.],'B':[11.,11.,10.]})
    def test_metrics_sign_units_and_tolerance(self):
        d=self.fixture();m=metrics(d,'H30');self.assertEqual(m['n'],3);self.assertEqual(m['mae'],2);self.assertAlmostEqual(m['rmse'],np.sqrt(20/3));self.assertEqual(m['bias'],2);self.assertEqual(m['spot_bps'],200);self.assertNotIn('within_spread',m)
    def test_paired_estimand_and_contract_aggregation(self):
        d=self.fixture();p=paired(d);self.assertAlmostEqual(p['mean'],4/3);self.assertAlmostEqual(p['aggregate_mae_reduction_percent'],200/3)
        self.assertAlmostEqual(p['improved_observations_percent'],200/3);self.assertEqual(p['improved_contracts_percent'],50)
    def test_temporal_split_disjoint_and_duplicates(self):
        d=self.fixture();a,b=chronological_split(d,'2020-01-02T00:00Z');self.assertEqual(len(a),1);self.assertEqual(len(b),2);self.assertFalse(set(a.record_key)&set(b.record_key))
        d.loc[1,'record_key']='1'
        with self.assertRaises(ValueError):chronological_split(d,'2020-01-02T00:00Z')
    def test_cluster_gate_and_reproducibility(self):
        d=self.fixture();self.assertEqual(date_cluster(d)['replicates'],0)
        a=date_cluster(d,replicates=100,min_dates=2,seed=42);b=date_cluster(d,replicates=100,min_dates=2,seed=42);self.assertEqual(a,b);self.assertEqual(a['dates'],3)
    def test_marks_and_unverified_exercise_rejected(self):
        with self.assertRaises(ValueError):normalize_chunk(pd.DataFrame(),{}, {},'fixture.csv')
        spec=dict(verified=True,evidence_reference='fixture',exercise_style='European',settlement_currency='BTC',multiplier_btc=1,premium_currency='BTC',premium_per_contract=False,effective_start='2019-01-01T00:00Z',effective_end='2021-01-01T00:00Z',expiry_hour_utc=8)
        with self.assertRaises(ValueError):normalize_chunk(pd.DataFrame(),{'price_category':'mark','price_evidence':'model mark'},spec,'fixture.csv')
    def test_empty_metrics_not_zero(self):
        with self.assertRaises(ValueError):metrics(self.fixture().iloc[:0],'H30')
    def quality_fixture(self):
        f=self.fixture().assign(strike=100.,symbol_valid=True,spec_valid=True,original_premium=10.,rate=.02,H30=.3,H60=.4)
        f['expiration']=f.timestamp+pd.Timedelta(days=30);f['underlying_at']=f.timestamp-pd.Timedelta(seconds=30)
        return f
    def test_quality_filter_edge_cases_and_flow(self):
        f=self.quality_fixture();valid,stages=quality_filters(f);self.assertEqual(len(valid),len(f))
        for column,bad in [('original_premium',0),('spot',np.nan),('strike',200),('H30',np.nan),('rate',np.nan),('spec_valid',False)]:
            changed=f.copy();changed.loc[0,column]=bad;valid,stages=quality_filters(changed)
            self.assertEqual(len(valid),len(f)-1,column);self.assertEqual(sum(x['removed'] for x in stages),1)
        changed=f.copy();changed.loc[0,'underlying_at']=changed.loc[0,'timestamp']+pd.Timedelta(seconds=1)
        self.assertEqual(len(quality_filters(changed)[0]),len(f)-1)
        changed=f.copy();changed.loc[0,'expiration']=changed.loc[0,'timestamp']
        self.assertEqual(len(quality_filters(changed)[0]),len(f)-1)
    def test_large_premium_not_removed_based_on_error(self):
        f=self.quality_fixture();f.loc[0,'original_premium']=100000
        self.assertEqual(len(quality_filters(f)[0]),len(f))
    def test_cheap_premium_relative_error_denominator(self):
        f=self.fixture();f['observed_usd']=.01;m=metrics(f,'H30')
        self.assertEqual(m['relative_price_n'],0);self.assertIsNone(m['relative_price_error']);self.assertEqual(m['n'],len(f))
    def test_optional_spread_coverage_requires_real_columns(self):
        f=self.fixture();f['bid_usd']=[11,13,np.nan];f['ask_usd']=[13,15,np.nan];m=metrics(f,'H30')
        self.assertEqual(m['spread_n'],2);self.assertEqual(m['within_spread'],1)

class BitcoinWorkflowTests(unittest.TestCase):
    def test_full_workflow_with_labelled_synthetic_data_only(self):
        with fixture_directory() as tmp:
            root=Path(tmp);raw=root/'raw';raw.mkdir();out=root/'private'
            dates=pd.date_range('2019-10-01',periods=170,tz='UTC')+pd.Timedelta(hours=12)
            spots=10000*np.exp(.0001*np.arange(170)+.025*np.sin(np.arange(170)/8))
            history=pd.DataFrame({'clock':dates.astype(str),'close':spots});history.to_csv(raw/'fixture_history.csv',index=False)
            rates=pd.DataFrame({'known_at':(dates-pd.Timedelta(hours=12)).astype(str),'observed_date':(dates-pd.Timedelta(days=1)).strftime('%Y-%m-%d'),'yield':2.});rates.to_csv(raw/'fixture_rates.csv',index=False)
            rows=[];counter=0
            for timestamp,spot in zip(dates,spots):
                if not pd.Timestamp('2020-01-15T00:00Z')<=timestamp<pd.Timestamp('2020-02-25T00:00Z'):continue
                for strike in [9000,10000,11000]:
                    for kind in ['call','put']:
                        # Independently coded erf formula for synthetic fixture premiums.
                        T=(pd.Timestamp('2020-03-27T08:00Z')-timestamp).total_seconds()/(365*86400);r=2*math.log(1.01);sigma=.4
                        d1=(math.log(spot/strike)+(r+.5*sigma*sigma)*T)/(sigma*math.sqrt(T));d2=d1-sigma*math.sqrt(T);z=1 if kind=='call' else -1
                        N=lambda x:.5*(1+math.erf(x/math.sqrt(2)))
                        usd=z*(spot*N(z*d1)-strike*math.exp(-r*T)*N(z*d2));counter+=1
                        rows.append(dict(epoch=timestamp.timestamp(),symbol=f'BTC-27MAR20-{strike}-{kind[0].upper()}',premium=usd/spot,id=str(counter),vendor_iv=2.5))
            options=pd.DataFrame(rows+[rows[0]]);options.to_csv(raw/'fixture_options.csv',index=False)
            spec=dict(verified=True,evidence_reference='SYNTHETIC TEST FIXTURE ONLY',exercise_style='European',settlement_currency='BTC',multiplier_btc=1,premium_currency='BTC',premium_per_contract=False,effective_start='2019-01-01T00:00Z',effective_end='2021-01-01T00:00Z',expiry_hour_utc=8)
            manifest=dict(data_class='synthetic_test_fixture',schema_evidence='Fixture, not UWA schema',conversion_basis_evidence='Fixture exact spot',rate_availability_evidence='Fixture known-at times',carry_q=0,carry_evidence='Fixture assumption',training_cutoff_utc='2020-02-01T00:00Z',specifications=spec,
                options=dict(file='fixture_options.csv',timestamp='epoch',timestamp_unit='s',contract='symbol',premium='premium',trade_id='id',provider_iv='vendor_iv',price_category='executed_trade',price_evidence='Independently calculated SYNTHETIC fixture premiums'),
                history=dict(file='fixture_history.csv',timestamp='clock',price='close',price_definition='Synthetic noon USD/BTC samples, not actual five-minute history',evidence_reference='Fixture generator'),
                rates=dict(file='fixture_rates.csv',available_at='known_at',observation_date='observed_date',yield_='yield'))
            manifest['rates']['yield']=manifest['rates'].pop('yield_')
            for key in ['options','history','rates']:manifest[key]['sha256']=file_sha256(raw/manifest[key]['file'])
            result=run(manifest,RIGHTS,raw,out)
            self.assertFalse(result['empirical_computation_ran']);self.assertAlmostEqual(result['fit']['sigma'],.4,places=6)
            counts={m['n'] for m in result['metrics'].values()};self.assertEqual(len(counts),1)
            self.assertEqual(result['inventories']['options']['rows'],len(rows)+1);self.assertEqual(result['inventories']['options']['duplicate_ids'],1)
            self.assertEqual(result['counts']['unique_known_trade_id'],len(rows));self.assertLess(result['metrics']['B']['mae'],.001)
            self.assertIn('SYNTHETIC TEST FIXTURE',(out/'private-report.html').read_text(encoding='utf8'))
            processed=root/'processed'
            saved=pd.read_csv(processed/'analytical-observations.csv.gz')
            self.assertEqual(len(saved),sum(result['counts'][key] for key in ['known_rate']))
            self.assertEqual(set(saved.record_key),set(pd.DataFrame(rows).index.map(lambda i:'fixture_options.csv:'+str(i+1))))
            self.assertTrue((processed/'historical-volatility.csv.gz').exists())
            self.assertTrue((processed/'source-inventory.json').exists())
            for groups in result['breakdowns'].values():
                for model in ['H30','H60','B']:self.assertEqual(sum(g['models'][model]['n'] for g in groups),result['metrics'][model]['n'])

if __name__=='__main__':unittest.main()
