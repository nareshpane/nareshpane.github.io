"""Actual UWA source/output reconciliation. Skips only when local originals/results are absent."""
import json,hashlib,math,sys,unittest
from pathlib import Path
import numpy as np
import pandas as pd
A=Path(__file__).resolve().parents[1];ROOT=A;sys.path.insert(0,str(A/'python'))
from bitcoin_uwa import EXPECTED
from bitcoin_pricing import benchmark
from bitcoin_diagnostics import block_uncertainty
from bitcoin_data import match_underlying

class UWAEmpiricalTests(unittest.TestCase):
 @classmethod
 def setUpClass(cls):
    base=ROOT/'bitcoin_data';private=base/'private/uwa-validated-20261017';processed=base/'processed/uwa-validated-20261017'
    if not (private/'bitcoin-results.json').exists():raise unittest.SkipTest('Local UWA market computation unavailable; not replaced with synthetic findings')
    cls.raw=base/'raw';cls.r=json.loads((private/'bitcoin-results.json').read_text());cls.d=json.loads((A/'data/bitcoin-results.json').read_text())
    cls.e=pd.read_csv(private/'evaluation-private.csv');cls.e.timestamp=pd.to_datetime(cls.e.timestamp,utc=True,format='mixed')
    cls.data=pd.read_csv(processed/'analytical-observations.csv.gz');cls.data.timestamp=pd.to_datetime(cls.data.timestamp,utc=True,format='mixed')
 def test_source_hashes_and_actual_inventory(self):
    for name,digest in EXPECTED.items():
      h=hashlib.sha256()
      with (self.raw/name).open('rb') as f:
       for b in iter(lambda:f.read(1024*1024),b''):h.update(b)
      self.assertEqual(h.hexdigest(),digest);self.assertEqual(self.d['inventory'][name]['sha256'],digest)
    self.assertEqual(self.d['inventory']['Deribit_w_IV.csv']['rows'],761067)
    self.assertEqual(self.d['inventory']['instrument_name.csv']['rows'],6724)
    self.assertEqual(self.d['inventory']['Price_5m.csv']['rows'],2697313)
 def test_filters_counts_and_positive_actual_quantities(self):
    r=self.r;self.assertEqual(r['counts']['unique_exact_source_rows'],761067-14124)
    self.assertEqual(r['counts']['positive_executed_quantity'],746942)
    self.assertEqual(len(self.data),250453);self.assertTrue((self.data.quantity>0).all())
    remaining=[v['remaining'] for v in r['quality_flow']];self.assertTrue(all(a>=b for a,b in zip(remaining,remaining[1:])))
    self.assertEqual(sum(v['removed'] for v in r['quality_flow']),761067-250453)
 def test_exact_common_holdout_and_chronological_separation(self):
    cutoff=pd.Timestamp('2019-01-01T00:00Z');train=self.data[self.data.timestamp<cutoff];test=self.data[self.data.timestamp>=cutoff]
    self.assertEqual(len(train),72217);self.assertEqual(len(test),178236)
    self.assertEqual(set(test.record_key),set(self.e.record_key));self.assertFalse(set(train.record_key)&set(test.record_key))
    self.assertFalse(self.e.record_key.duplicated().any());self.assertEqual(self.r['training_evaluation_record_overlap'],0)
    self.assertEqual({v['n'] for v in self.r['metrics'].values()},{178236})
    self.assertTrue(np.isfinite(self.e[['H30','H60','B']]).all().all())
 def test_observed_units_and_independent_price_formula(self):
    e=self.e;self.assertTrue(np.allclose(e.observed_usd,e.original_premium*e.spot,rtol=0,atol=1e-8))
    for row in e.iloc[np.linspace(0,len(e)-1,20,dtype=int)].itertuples():
      sigma=row.sigma_H30;d1=(math.log(row.spot/row.strike)+(row.rate+sigma*sigma/2)*row.T)/(sigma*math.sqrt(row.T));d2=d1-sigma*math.sqrt(row.T)
      N=lambda x:.5*(1+math.erf(x/math.sqrt(2)));z=1 if row.kind=='call' else -1
      independent=z*(row.spot*N(z*d1)-row.strike*math.exp(-row.rate*row.T)*N(z*d2))
      self.assertAlmostEqual(independent,row.H30,places=8);self.assertAlmostEqual(row.H30_error,row.H30-row.observed_usd,places=8)
 def test_completed_minute_and_rate_availability(self):
    e=self.e;known=pd.to_datetime(e.underlying_at,utc=True,format='mixed');start=pd.to_datetime(e.history_source_at,utc=True,format='mixed')
    self.assertTrue(((known-start).dt.total_seconds()==60).all());self.assertTrue((known<=e.timestamp).all())
    self.assertTrue(((e.timestamp-known).dt.total_seconds()<=300).all())
    rate_at=pd.to_datetime(e.available_at_rate,utc=True,format='mixed');obs=pd.to_datetime(e.rate_observation_date,utc=True)
    self.assertTrue((rate_at<=e.timestamp).all());self.assertTrue((obs<e.timestamp.dt.floor('D')).all())
    self.assertTrue(np.allclose(e.rate,2*np.log1p(e.annual_percent/200)))
 def test_independent_prior_returns_on_actual_prices(self):
    h=pd.read_csv(self.raw/'Price_5m.csv',usecols=['time_m','price_avg']);h['day']=pd.to_datetime(h.time_m,utc=True).dt.floor('D')
    closes=h.groupby('day').price_avg.last()
    for row in self.e.iloc[np.linspace(0,len(self.e)-1,12,dtype=int)].itertuples():
      available=closes[closes.index<row.timestamp.floor('D')]
      for w in [30,60]:
       logs=np.diff(np.log(available.iloc[-w-1:].to_numpy()));value=np.std(logs,ddof=1)*math.sqrt(365)
       self.assertAlmostEqual(value,getattr(row,'sigma_H'+str(w)),places=10)
 def test_all_metrics_independently_reconcile(self):
    for name,v in self.r['metrics'].items():
      err=self.e[name].to_numpy()-self.e.observed_usd.to_numpy()
      self.assertAlmostEqual(abs(err).mean(),v['mae'],places=9);self.assertAlmostEqual(np.sqrt(np.mean(err**2)),v['rmse'],places=9)
      self.assertAlmostEqual(err.mean(),v['bias'],places=9);self.assertAlmostEqual(np.mean(abs(err)/self.e.spot)*10000,v['spot_bps'],places=9)
      self.assertNotIn('within_spread',v)
    delta=abs(self.e.H30-self.e.observed_usd)-abs(self.e.B-self.e.observed_usd)
    self.assertAlmostEqual(delta.mean(),self.r['paired']['H30']['mean'],places=9)
 def test_groups_histograms_and_public_points_reconcile(self):
    d=self.d
    for dim,groups in d['breakdowns'].items():self.assertEqual(sum(g['models']['H30']['n'] for g in groups),178236)
    for g in d['groups'].values():
      for name in ['H30','H60','B']:
       self.assertEqual(sum(x['n'] for x in g['residual'][name]),g['n']);self.assertEqual(sum(x['n'] for x in g['scatter'][name]),g['n'])
    self.assertEqual(sum(v['n'] for v in d['heatmap']),178236)
    self.assertEqual(d['metrics'],self.r['metrics'])
 def test_twenty_examples_match_original_rows(self):
    rows=self.d['examples'];self.assertEqual(len(rows),20);self.assertEqual(sum(r['kind']=='call' for r in rows),10)
    keys={(pd.Timestamp(r['timestamp']).value//1000000,r['contract'],round(r['observed_btc'],12)):r for r in rows};found=set()
    for chunk in pd.read_csv(self.raw/'Deribit_w_IV.csv',chunksize=100000):
      z=chunk[chunk.instrument_name.isin([r['contract'] for r in rows])]
      for row in z.itertuples():
       key=(int(row.timestamp),row.instrument_name,round(row.price,12))
       if key in keys:
        sample=keys[key];found.add(key);self.assertAlmostEqual(sample['spot'],row.index_price);self.assertAlmostEqual(sample['observed_usd'],row.price_USD)
    self.assertEqual(found,set(keys))
 def test_vendor_iv_missing_does_not_remove_pricing_targets(self):
    self.assertEqual(int(self.e.provider_iv.isna().sum()),260)
    self.assertTrue(np.isfinite(self.e.loc[self.e.provider_iv.isna(),['H30','H60','B']]).all().all())
    self.assertEqual(self.r['iv']['valid']+self.r['iv']['invalid'],len(self.e))
 def test_day_block_bootstrap_and_no_partial_close_publication(self):
    x=block_uncertainty(self.e,replicates=30,seed=7);y=block_uncertainty(self.e,replicates=30,seed=7)
    self.assertEqual(x,y);self.assertEqual(x['block_days'],7)
    self.assertEqual(self.d['daily'][-1]['date'],'2020-02-16')
    self.assertEqual(self.r['block_uncertainty']['replicates'],1000)
 def test_prior_minute_boundary_fixture(self):
    # This fixture verifies timing software, not empirical market accuracy.
    h=pd.DataFrame({'timestamp':pd.to_datetime(['2019-01-01T00:01Z','2019-01-01T00:02Z']), 'spot':[100,110]})
    t=pd.DataFrame({'timestamp':pd.to_datetime(['2019-01-01T00:01:30Z'])});m=match_underlying(t,h)
    self.assertEqual(m.spot.iloc[0],100)

if __name__=='__main__':unittest.main()
