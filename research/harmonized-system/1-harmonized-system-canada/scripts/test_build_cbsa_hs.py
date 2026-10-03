#!/usr/bin/env python3
"""Offline regression checks: python3 -m unittest discover -s <scripts-dir>."""
import copy
import json
import unittest
from bs4 import BeautifulSoup
import build_cbsa_hs as hs


def table(rows):
    header = '<tr>' + ''.join('<th>' + c + '</th>' for c in hs.COLUMNS) + '</tr>'
    content = ''.join('<tr>' + ''.join('<td>' + v + '</td>' for v in row) + '</tr>' for row in rows)
    return BeautifulSoup('<table>' + header + content + '</table>', 'lxml')


def chapter(code='06'):
    return dict(code=code, source_url=hs.MASTER_URL, reserved=False, headings=[])


class ParserTests(unittest.TestCase):
    def test_six_and_eight_digits_and_repeated_headers(self):
        c = chapter()
        hs.parse_chapter(c, table([
            hs.COLUMNS,
            ['06.01', '', 'Bulbs', '', '', ''],
            ['0601.10', '', 'Bulbs - Dormant', '', '', ''],
            ['0601.10.11', '00', 'Bulbs - Dormant - Narcissus', 'NMB', '6%', 'GPT 5%'],
            ['06.02', '', 'Other live plants', '', '', ''],
            ['0602.10.00', '00', 'Other live plants - Unrooted cuttings and slips', 'NMB', 'Free', ''],
            ['0602.20.00', '', 'Other live plants - Trees', '', '', ''],
            ['0602.20.00', '10', 'Other live plants - Trees - Fruit trees', '', '', ''],
        ]))
        sub = [s for h in c['headings'] for s in h['subheadings']]
        self.assertEqual([s['code'] for s in sub], ['060110', '060210', '060220'])
        self.assertEqual(sub[1]['description'], 'Unrooted cuttings and slips')
        self.assertEqual(sub[2]['description'], 'Trees')
        self.assertEqual(sub[1]['inferred_from'], ['06021000'])
        self.assertEqual(sub[0]['canadian_tariff_lines'], [{
            'tariff_item': '0601.10.11', 'statistical_suffix': '00',
            'description': 'Bulbs - Dormant - Narcissus', 'unit': 'NMB',
            'mfn_tariff': '6%', 'preferential_tariffs': 'GPT 5%'}])
        self.assertEqual([r['statistical_suffix'] for r in sub[2]['canadian_tariff_lines']], ['', '10'])

    def test_rates_and_blank_cells_are_not_inherited_or_combined(self):
        c = chapter()
        hs.parse_chapter(c, table([
            ['06.01', '', 'Bulbs', '', '', ''],
            ['0601.10', '', 'Bulbs - Dormant', '', '', ''],
            ['0601.10.11', '00', 'Bulbs - Dormant - Narcissus', 'NMB', '6%', 'GPT 5%'],
            ['0601.10.19', '', 'Bulbs - Dormant - Other', '', 'Free', 'GPT: Free'],
            ['0601.10.19', '10', 'Bulbs - Dormant - Other - Tulip', '-', '', ''],
        ]))
        lines = c['headings'][0]['subheadings'][0]['canadian_tariff_lines']
        self.assertEqual(len(lines), 3)
        self.assertEqual([r['mfn_tariff'] for r in lines], ['6%', 'Free', ''])
        self.assertEqual([r['preferential_tariffs'] for r in lines], ['GPT 5%', 'GPT: Free', ''])
        self.assertEqual([r['unit'] for r in lines], ['NMB', '', '-'])

    def test_omitted_heading(self):
        c = chapter('02')
        hs.parse_chapter(c, table([['0205.00.00', '00', 'Meat of horses', '', '', '']]))
        self.assertEqual(c['headings'][0]['code'], '0205')
        self.assertEqual(c['headings'][0]['subheadings'][0]['code'], '020500')

    def test_duplicates_fail(self):
        rows = [['06.01', '', 'Bulbs', '', '', ''], ['0601.10', '', 'Bulbs - Dormant', '', '', '']]
        for duplicate in rows:
            with self.subTest(duplicate=duplicate[0]), self.assertRaisesRegex(ValueError, 'Duplicate'):
                hs.parse_chapter(chapter(), table(rows + [duplicate]))

    def test_ambiguous_national_description_fails(self):
        with self.assertRaisesRegex(ValueError, 'Cannot safely infer'):
            hs.parse_chapter(chapter(), table([['06.01', '', 'Bulbs', '', '', ''], ['0601.10.11', '00', 'Bulbs - Narcissus', '', '', '']]))

    def test_statistical_only_source_warning(self):
        c = chapter('71')
        hs.parse_chapter(c, table([
            ['71.04', '', 'Synthetic stones', '', '', ''],
            ['7104.20.00', '10', 'Synthetic stones - Unworked - null - Diamonds', '', '', ''],
            ['7104.20.00', '90', 'Synthetic stones - Unworked - null - Other', '', '', ''],
        ]))
        s = c['headings'][0]['subheadings'][0]
        self.assertTrue(s['source_warning'])
        self.assertEqual(s['description'], 'Unworked')
        self.assertEqual(len(s['source_row_descriptions']), 2)

    def test_pdf_discovery_from_links(self):
        soup = BeautifulSoup('<main><a href="../ref/a.pdf">PDF</a><a href="../ref/a.pdf">Again</a><a href="b.html">HTML</a></main>', 'lxml')
        _, _, pdfs = hs.discover(soup, 'https://example.test/html/master.html')
        self.assertEqual(pdfs, ['https://example.test/ref/a.pdf'])


class DatasetTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.data = json.loads(hs.OUTPUT.read_text())

    def test_actual_dataset_and_chapter6(self):
        errors, warnings = hs.validate(self.data)
        self.assertEqual(errors, [])
        self.assertEqual(len(warnings), 2)
        self.assertEqual(hs.summarize(self.data)['hs6_subheadings'], 5614)

    def test_chapter15_lard_and_multiple_chapter6_rows(self):
        nodes = {s['code']: s for section in self.data['sections'] for c in section['chapters']
                 for h in c['headings'] for s in h['subheadings']}
        lard = nodes['150110']
        self.assertEqual((lard['chapter_code'], lard['heading_code'], lard['display_code']), ('15', '1501', '1501.10'))
        self.assertEqual(lard['canadian_tariff_lines'], [{
            'tariff_item': '1501.10.00', 'statistical_suffix': '00',
            'description': 'Pig fat (including lard) and poultry fat, other than that of heading 02.09 or 15.03. - Lard',
            'unit': 'KGM', 'mfn_tariff': 'Free',
            'preferential_tariffs': 'CCCT, LDCT, GPT, UST, MXT, CIAT, CT, CRT, IT, PT, COLT, JT, PAT, HNT, KRT, CEUT, UAT, CPTPT, UKT, CPUKT: Free'}])
        rows = nodes['060110']['canadian_tariff_lines']
        self.assertEqual(len(rows), 13)
        self.assertEqual([(r['tariff_item'], r['statistical_suffix']) for r in rows], [
            ('0601.10.11', '00'), ('0601.10.19', ''), ('0601.10.19', '10'), ('0601.10.19', '90'),
            ('0601.10.21', ''), ('0601.10.21', '10'), ('0601.10.21', '21'), ('0601.10.21', '22'),
            ('0601.10.21', '91'), ('0601.10.21', '92'), ('0601.10.21', '93'), ('0601.10.21', '99'),
            ('0601.10.29', '00')])
        self.assertEqual(hs.summarize(self.data)['canadian_tariff_lines'], 12391)

    def test_tariff_line_validation_detects_corruption(self):
        for field, invalid in [('tariff_item', '9999.99.99'), ('statistical_suffix', 0),
                               ('statistical_suffix', '0'), ('mfn_tariff', 0)]:
            with self.subTest(field=field, invalid=invalid):
                data = copy.deepcopy(self.data)
                line = data['sections'][0]['chapters'][0]['headings'][0]['subheadings'][0]['canadian_tariff_lines'][0]
                line[field] = invalid
                self.assertTrue(hs.validate(data)[0])

    def test_corruption_is_detected(self):
        for kind in ['numeric', 'duplicate4', 'duplicate6', 'parent', 'reserved', 'special', 'description', 'source']:
            with self.subTest(kind=kind):
                data = copy.deepcopy(self.data)
                c = data['sections'][0]['chapters'][0]; h = c['headings'][0]; s = h['subheadings'][0]
                if kind == 'numeric': s['code'] = 10121
                elif kind == 'duplicate4': c['headings'].append(copy.deepcopy(h))
                elif kind == 'duplicate6': h['subheadings'].append(copy.deepcopy(s))
                elif kind == 'parent': s['heading_code'] = '9999'
                elif kind == 'reserved':
                    next(c for sec in data['sections'] for c in sec['chapters'] if c['code'] == '77')['reserved'] = False
                elif kind == 'special': data['sections'][0]['chapters'].append(data['special_chapters'][0])
                elif kind == 'description': s['description'] = ''
                elif kind == 'source': c['source_url'] = ''
                errors, _ = hs.validate(data)
                self.assertTrue(errors, kind)


if __name__ == '__main__':
    unittest.main()
