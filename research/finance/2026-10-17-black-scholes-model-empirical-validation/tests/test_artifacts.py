import json, unittest, math
from pathlib import Path
from html.parser import HTMLParser

ROOT=Path(__file__).resolve().parents[2]
ASSETS=Path(__file__).resolve().parents[1]
class PageParser(HTMLParser):
    def __init__(self):super().__init__();self.ids=[];self.urls=[];self.sections=[]
    def handle_starttag(self,tag,attrs):
        a=dict(attrs)
        if 'id' in a:self.ids.append(a['id'])
        if tag=='section' and 'id' in a:self.sections.append(a['id'])
        for k in ['src','href']:
            if k in a:self.urls.append(a[k])

class PublishedArtifactTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):cls.d=json.loads((ASSETS/'data/results.json').read_text(encoding='utf-8'))
    def test_counts_and_partitions(self):
        d=self.d;n=d['metrics']['A']['n'];self.assertEqual(n,d['metrics']['B']['n']);self.assertEqual(d['dates'],127)
        counts=list(d['counts'].values());self.assertTrue(all(a>=b for a,b in zip(counts,counts[1:])))
        for dim,groups in d['breakdowns'].items():
            for key in ['A','B']:
                self.assertEqual(sum(g[key]['n'] for g in groups),n)
                self.assertAlmostEqual(sum(g[key]['n']*g[key]['mae'] for g in groups)/n,d['metrics'][key]['mae'],places=9)
        self.assertEqual(sum(g['A']['n'] for g in d['heatmap']),n)
        self.assertEqual(sum(g['n'] for g in d['stratified_intervals']),n)
    def test_metrics_and_bootstrap(self):
        d=self.d
        for m in d['metrics'].values():
            self.assertGreaterEqual(m['rmse'],m['mae']);self.assertGreaterEqual(m['mae'],abs(m['bias']));self.assertTrue(0<=m['within']<=1)
        for k in ['A','B']:
            self.assertTrue(d['bootstrap'][k][0]<d['metrics'][k]['mae']<d['bootstrap'][k][1])
            self.assertLessEqual(sum(d['histogram'][k]['counts']),d['metrics'][k]['n'])
        self.assertEqual(d['bootstrap']['replicates'],1000);self.assertEqual(d['bootstrap']['seed'],20261017)
    def test_safe_aggregates(self):
        d=self.d
        for row in d['scatter']:self.assertGreaterEqual(row['n'],30)
        for row in d['smile']:self.assertGreaterEqual(row['n'],30);self.assertTrue(0<row['iv']<5)
        self.assertNotIn('contract',d);self.assertNotIn('raw',d)
        js=(ASSETS/'data/results.js').read_text(encoding='utf-8')
        self.assertEqual(json.loads(js.removeprefix('window.EMPIRICAL = ').removesuffix(';')),d)
    def test_links_and_ids(self):
        p=PageParser();p.feed((ROOT/'2026-10-17-black-scholes-model-empirical-validation.html').read_text(encoding='utf-8'))
        self.assertEqual(len(p.ids),len(set(p.ids)))
        for url in p.urls:
            if url.startswith(('https:','http:','mailto:')):continue
            if url.startswith('#'):self.assertIn(url[1:],p.ids)
            else:self.assertTrue((ROOT/url.split('#')[0]).exists(),url)
        self.assertEqual(p.sections,[f'c{i:02}' for i in range(1,10)])
    def test_harmonization(self):
        p=self.d['provenance'];self.assertEqual(p['spx_disagreements'],0);self.assertEqual(len(p['spxw_discrepancies']),5)
        self.assertEqual(p['fred_comparisons'],125);self.assertEqual(p['fred_max_abs_difference'],0)

if __name__=='__main__':unittest.main()
