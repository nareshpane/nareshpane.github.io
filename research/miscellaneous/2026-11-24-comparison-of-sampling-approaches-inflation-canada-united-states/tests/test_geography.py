"""Reconcile the geographic CPI snapshots, area partitions and published examples."""
import csv
import json
import unittest
from pathlib import Path

OUT = Path(__file__).resolve().parents[1] / 'data/processed/geography'

def rows(filename):
    with (OUT / filename).open(encoding='utf-8', newline='') as file:
        return list(csv.DictReader(file))

class GeographicChecks(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.data = json.loads((OUT / 'geography.json').read_text(encoding='utf-8'))

    def test_classifications_and_nonoverlapping_partitions(self):
        d = self.data
        self.assertEqual(len(d['maps']['CA']), 13)
        self.assertEqual(sum(r['strata'] for r in d['maps']['CA'].values()), 19)
        self.assertEqual({k for k,r in d['maps']['CA'].items() if r['capital']}, {'YT','NT','NU'})
        self.assertEqual(len(d['maps']['US']), 51)
        self.assertEqual(len(d['psus']), 75)
        self.assertEqual(sum(p['code'].startswith('S') for p in d['psus']), 23)
        self.assertEqual(len({p['basic'] for p in d['psus']}), 32)
        self.assertTrue(all(p['point'] and p['paths'] and not p['missing'] for p in d['psus']))
        cross_state = next(p for p in d['psus'] if p['code'] == 'S12A')
        self.assertEqual(set(cross_state['states']), {'NJ','NY','PA'})
        self.assertEqual(len(cross_state['paths']), 25)
        by_area = {r['area_code']:r['basic_areas'].split(', ') for r in rows('us_cpi_area_aggregation.csv')}
        for codes in (d['areas']['US'], d['divisions']):
            basic = [b for code in codes for b in by_area[code]]
            self.assertCountEqual(basic, by_area['0000'])
        self.assertEqual(sum(d['population'].values()), 289606345)

    def test_rate_arithmetic_and_identifiers(self):
        d = self.data
        for country, filename, count in [('CA','canada_geographical_cpi.csv',79),
                                          ('US','us_regional_cpi.csv',126)]:
            records = rows(filename)
            unique = {(r['series_id'],r['month']) for r in records}
            self.assertEqual(len(records), len(unique))
            self.assertEqual(len(records), len(d['dates'])*count)
            id_set = {r['series_id'] for r in records}
            if country == 'US':
                self.assertEqual(id_set,{r['series_id'] for r in rows('us_series_catalogue.csv')})
            for r in records:
                self.assertLessEqual(r['month'], '2026-10')
                self.assertEqual(r['adjustment'], 'Not seasonally adjusted')
                for key, base in [('monthly','previous_month_index'),('annual','previous_year_index')]:
                    with self.subTest(country=country,series=r['series_id'],month=r['month'],horizon=key):
                        if r['index'] and r[base]:
                            expected=100*(float(r['index'])/float(r[base])-1)
                            self.assertAlmostEqual(float(r[key+'_percent']),expected)
                            self.assertEqual(r['missing_'+key],'False')
                        else:
                            self.assertEqual(r[key+'_percent'],'')
                            self.assertEqual(r['missing_'+key],'True')
        i = d['dates'].index('2026-10')
        june = d['dates'].index('2022-06')
        self.assertAlmostEqual(d['rates']['CA']['all']['CA'][i][1],3.1,places=1)
        self.assertTrue(all(d['rates']['US']['all'][code][i][1] is None for code in d['areas']['US']))
        self.assertAlmostEqual(d['rates']['US']['gasoline']['0000'][i][1],37.4,places=1)
        self.assertAlmostEqual(d['rates']['CA']['gasoline']['PE'][june][1],63.4,places=1)
        self.assertAlmostEqual(d['rates']['US']['gasoline']['0200'][june][1],64.4,places=1)

    def test_weight_vintages_and_units(self):
        d = self.data
        ca=d['weights']['CA']['geography']
        us=d['weights']['US']['geography']
        self.assertAlmostEqual(sum(ca.values()),100,delta=.05)
        self.assertEqual(ca['ON'],39.76)
        self.assertAlmostEqual(sum(us[code] for code in d['areas']['US']),100)
        self.assertEqual(us['0300'],35.228)
        self.assertEqual(d['weights']['US']['product']['gasoline']['0100'],2.401)
        vintages = {r['price_period'] for r in rows('canada_geographical_weights.csv')}
        self.assertEqual(vintages,{'2025','2026-04'})
        self.assertNotIn('NU',d['weights']['CA']['product']['gasoline'])
        self.assertNotIn('NU',d['rates']['CA']['gasoline'])
