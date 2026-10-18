"""Audit limited source examples and new summaries independently of page rendering."""
import csv,hashlib,json,math,os,re,sys,unittest
from pathlib import Path
import numpy as np
ROOT=Path(__file__).resolve().parents[2];ASSET=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ASSET/'python'))
from pricing import price
from revision_outputs import FIELDS,DERIVED

def bundle(name,prefix):
    text=(ASSET/'data'/name).read_text(encoding='utf8');return json.loads(text.removeprefix('window.'+prefix+' = ').strip().removesuffix(';'))

class RevisionTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.x=bundle('illustrative-observations.js','OPTION_EXAMPLES');cls.s=bundle('supplement.js','SUPPLEMENT');cls.e=json.loads((ASSET/'data/results.json').read_text(encoding='utf8'))
    def test_exactly_twenty_complete_examples(self):
        x=self.x;self.assertEqual(len(x['rows']),20);self.assertEqual(len(x['columns']),40)
        self.assertEqual(len({r['id'] for r in x['rows']}),20)
        self.assertEqual(len({(r['source']['contract'],r['source']['quote_date']) for r in x['rows']}),20)
        self.assertEqual(len(x['dates']),4)
        for r in x['rows']:
            self.assertEqual(list(r['source']),[f[0] for f in FIELDS]);self.assertEqual(len(r['source']),34)
            self.assertEqual(r['source']['quote_time'],'');self.assertEqual(r['source']['style'],'E');self.assertEqual(r['source']['settlement_time'],'PM')
            self.assertEqual(r['source']['underlying'],'SPXW');self.assertEqual(list(r['derived']),[f[0] for f in DERIVED])
    def test_friendly_headings_and_origins(self):
        c=self.x['columns'];self.assertEqual([x['field'] for x in c[:34]],[f[0] for f in FIELDS])
        self.assertTrue(all(x['origin']=='source-computed' for x in c[25:34]));self.assertTrue(all(x['origin']=='research-derived' for x in c[34:]))
        for x in c:self.assertTrue(x['label']);self.assertTrue(x['definition'])
    def test_rows_match_external_original_exactly(self):
        source=Path(os.environ.get('TEMP','/tmp'))/'bsm-research-20261017/source/day_by_date'
        if not source.exists():self.skipTest('Original input outside repository is unavailable; download it to verify source rows')
        for day in self.x['dates']:
            f=source/f'{day}_options.csv';self.assertEqual(hashlib.sha256(f.read_bytes()).hexdigest(),self.x['source_hashes'][day])
            selected={r['source_line']:r for r in self.x['rows'] if r['source']['quote_date']==day}
            with f.open(newline='',encoding='utf8') as stream:
                reader=csv.DictReader(stream)
                for line,row in enumerate(reader,2):
                    if line in selected:self.assertEqual(row,selected[line]['source'])
    def test_derived_examples(self):
        for r in self.x['rows']:
            src=r['source'];d=r['derived']
            self.assertAlmostEqual(d['mid'],(float(src['bid'])+float(src['ask']))/2)
            self.assertAlmostEqual(d['m'],float(src['strike'])/d['S']);self.assertTrue(7<=d['days']<=90)
            self.assertTrue(.9<=d['m']<=1.1);self.assertEqual(round(float(src['strike'])/5)%2,1)
    def test_new_aggregates_reconcile(self):
        s=self.s;e=self.e;n=e['metrics']['A']['n'];self.assertEqual(s['n'],n)
        self.assertEqual(sum(d['n'] for d in s['grouped_prices']),n);self.assertEqual(sum(d['n'] for d in s['spread_histogram']),n)
        for key in ['A','B']:
            bias=sum(d['n']*d['bias'+key] for d in s['grouped_prices'])/n
            self.assertAlmostEqual(bias,e['metrics'][key]['bias'],places=9)
            for d in s['grouped_prices']:self.assertAlmostEqual(d[key]-d['mid'],d['bias'+key],places=9)
        self.assertEqual(sum(d['n'] for d in s['volatility']),n)
        for v,old in zip(s['volatility'],e['volatility']):
            self.assertEqual(v['date'],old['date']);self.assertEqual(v['hv20'],old['hv20']);self.assertTrue(0<v['iv']<5)
    def test_original_results_hashes(self):
        provenance=json.loads((ASSET/'data/revision-provenance.json').read_text())
        for name,digest in provenance['preserved_results_sha256'].items():self.assertEqual(hashlib.sha256((ASSET/name).read_bytes()).hexdigest(),digest)
    def test_worked_call_example(self):
        expected=float(price(100,100,1,.05,.2,.02));self.assertAlmostEqual(expected,9.227005508154036,places=10)
        html=(ROOT/'2026-10-17-black-scholes-model-empirical-validation.html').read_text(encoding='utf8')
        self.assertIn(f'{expected:.8f}',html);self.assertIn('d_2=0.0500',html);self.assertIn('=0.2500',html)
    def test_narrow_publication(self):
        self.assertEqual(set(p.name for p in (ASSET/'data').iterdir()),{'results.json','results.js','illustrative-observations.js','supplement.js','revision-provenance.json','bitcoin-results.json','bitcoin-results.js'})
        # Published assets contain no raw data files.
        # Every other publication directory remains free of raw datasets.
        local_only=[ASSET/'bitcoin_data']
        self.assertFalse(any(p.suffix.lower() in {'.csv','.zip','.parquet'} for p in ASSET.rglob('*') if p.is_file() and not any(d in p.parents for d in local_only)))
        import subprocess
        for directory in local_only:
            if directory.exists():
                for f in directory.rglob('*'):
                    if f.is_file():
                        self.assertEqual(subprocess.run(['git','check-ignore','--quiet',str(f)],capture_output=True).returncode,0)
                        self.assertFalse(subprocess.check_output(['git','ls-files','--',str(f)],text=True).strip())
        self.assertEqual(len(self.x['rows']),20)

if __name__=='__main__':unittest.main()
