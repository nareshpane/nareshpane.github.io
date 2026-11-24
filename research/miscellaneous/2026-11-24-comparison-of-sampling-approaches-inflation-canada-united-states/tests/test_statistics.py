"""Public data and finite-population invariants."""
import csv
import hashlib
import json
import math
import sys
import unittest
from pathlib import Path
from random import Random

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'scripts'))
from prepare_data import pct, lagged, PROCESSED, RELEASE
from simulate import economy, draw, DESIGNS

class StatisticalChecks(unittest.TestCase):
    def test_inflation_and_missing_calendar_months(self):
        levels = [100,105,None,111,112]
        self.assertEqual(lagged(levels,1)[2:4], [None,None])
        self.assertAlmostEqual(lagged(levels,1)[1],5)
        self.assertAlmostEqual(lagged(levels,1)[4],100*(112/111-1))
        self.assertEqual(pct(100,100),0)

    def test_rebasing_and_chaining(self):
        self.assertAlmostEqual(pct(176,162),pct(100*176/140,100*162/140))
        weights = [.22,.31,.47]
        r = 1.08
        self.assertAlmostEqual(sum(w*r for w in weights),r)
        self.assertAlmostEqual(math.exp(sum(w*math.log(r) for w in weights)),r)
        self.assertAlmostEqual(100*1.02*.99,100.98)
        old_link,new_link = 151,108
        self.assertEqual(old_link*(new_link/new_link),old_link)

    def test_published_october_arithmetic(self):
        report = json.loads((PROCESSED/'release-audit.json').read_text(encoding='utf-8'))
        reproduced=report['reproduced']
        for published,computed in [('CA_yoy','ca_annual'),('CA_monthly','ca_monthly'),
                                    ('CA_sa_monthly','ca_monthly_sa'),('US_sa_monthly','us_monthly_sa')]:
            self.assertEqual(round(reproduced[computed],1),RELEASE[published])
        self.assertIsNone(reproduced['us_annual'])
        self.assertIsNone(reproduced['annual_gap_pp'])
        with (PROCESSED/'inflation-rates.csv').open(encoding='utf-8') as file:
            data={r['month']:r for r in csv.DictReader(file)}
        self.assertEqual(data['2025-10']['us_index'],'')
        self.assertEqual(data['2025-11']['us_monthly'],'')
        self.assertEqual(data['2025-11']['us_monthly_sa'],'')

    def test_basket_shares(self):
        dataset=json.loads((PROCESSED/'cpi.json').read_text(encoding='utf-8'))
        for country,groups in dataset['weights']['major'].items():
            for year, shares in dataset['weights'][country].items():
                with self.subTest(country=country,year=year):
                    self.assertAlmostEqual(sum(shares[k] for k in groups),100,delta=.08)

    def test_monte_carlo_identity_and_reproducible_draws(self):
        outfile=ROOT/'data/simulated/summary.json'
        data=json.loads(outfile.read_text(encoding='utf-8'))
        self.assertEqual(len(data['rows']),63)
        for r in data['rows']:
            self.assertAlmostEqual(r['mse'],r['variance']+r['bias']**2)
            self.assertAlmostEqual(r['rmse']**2,r['mse'])
            self.assertGreaterEqual(r['coverage'],0)
            self.assertLessEqual(r['coverage'],1)
        outlets,_,truth=economy(data['seed'],'Baseline')
        groups=[[r for r in outlets if r['region']==i] for i in range(8)]
        self.assertAlmostEqual(sum(r['share'] for r in outlets),1)
        self.assertAlmostEqual(sum(r['share']*r['true'] for r in outlets),truth)
        for design in DESIGNS:
            self.assertEqual(draw(outlets,groups,Random(4),design,56),draw(outlets,groups,Random(4),design,56))
        self.assertAlmostEqual(draw(outlets,groups,Random(4),'SRS',len(outlets))[0],truth)

    def test_source_hashes(self):
        retrievals=json.loads((ROOT/'data/metadata/retrievals.json').read_text(encoding='utf-8'))
        for row in retrievals:
            if row['status']=='retrieved' or row.get('local_status'):
                data=(ROOT/row['path']).read_bytes()
                self.assertEqual(len(data),row['bytes'],row['path'])
                self.assertEqual(hashlib.sha256(data).hexdigest(),row['sha256'],row['path'])

    def test_bls_release_crosswalk_and_response_rates(self):
        with (PROCESSED/'bls-series-check.csv').open(encoding='utf-8') as file:
            rows=list(csv.DictReader(file))
        self.assertEqual(len(rows),11)
        gasoline=next(row for row in rows if row['key']=='gasoline')
        self.assertEqual(round(pct(float(gasoline['october_2026']),float(gasoline['october_2025'])),1),37.4)
        with (PROCESSED/'bls-response-rates.csv').open(encoding='utf-8') as file:
            responses={(row['month'],row['domain']):row for row in csv.DictReader(file)}
        self.assertAlmostEqual(float(responses['2026-10','C&S outlets']['observed_pct']),70.9)
        gap=responses.get(('2025-10','C&S outlets'),{})
        self.assertFalse(gap.get('observed_pct'))

if __name__=='__main__':
    unittest.main()

