"""Harmonize CPI snapshots and generate static-site data. No network required."""
import csv
from datetime import datetime
import io
import json
import math
import re
import statistics as stats
import zipfile
from html.parser import HTMLParser

from openpyxl import load_workbook

from acquire import ROOT, RAW_CA, RAW_US, US, BATCHES

START, END = '2000-01', '2026-10'
MONTHS = [f'{y}-{m:02}' for y in range(2000, 2027) for m in range(1, 13)
          if START <= f'{y}-{m:02}' <= END]
PROCESSED = ROOT / 'data/processed'
CA = {
    'all': 'All-items', 'core': 'All-items excluding food and energy', 'food': 'Food',
    'shelter': 'Shelter', 'energy': 'Energy', 'gasoline': 'Gasoline', 'apparel': 'Clothing',
    'transport': 'Transportation', 'rent': 'Rent', 'food_home': 'Food purchased from stores',
    'interest': 'Mortgage interest cost', 'replacement': "Homeowners' replacement cost",
    'tax': 'Property taxes and other special charges',
    'insurance': "Homeowners' home and mortgage insurance",
    'repairs': "Homeowners' maintenance and repairs",
}
PROVINCES = ['Newfoundland and Labrador', 'Prince Edward Island', 'Nova Scotia',
             'New Brunswick', 'Quebec', 'Ontario', 'Manitoba', 'Saskatchewan',
             'Alberta', 'British Columbia', 'Whitehorse', 'Yellowknife', 'Iqaluit']
MATCH = {'food': ('Food', 'Food'), 'shelter': ('Shelter', 'Shelter'),
         'apparel': ('Clothing', 'Apparel'), 'transport': ('Transportation', 'Transportation')}
US_NAMES = dict(zip(US, ['All items', 'All items less food and energy', 'Food', 'Shelter', 'Energy',
                         'Gasoline (all types)', 'Apparel', 'Transportation',
                         'Rent of primary residence', "Owners’ equivalent rent of residences", 'Food at home']))
CA_MAJOR = [CA['food'], CA['shelter'], 'Household operations, furnishings and equipment',
            'Clothing and footwear', CA['transport'], 'Health and personal care',
            'Recreation, education and reading', 'Alcoholic beverages, tobacco products and recreational cannabis']
US_MAJOR = ['Food and beverages', 'Housing', 'Apparel', 'Transportation', 'Medical care',
            'Recreation', 'Education and communication', 'Other goods and services']

def json_write(path, data, compact=False):
    path.write_text(json.dumps(data, ensure_ascii=False, allow_nan=False,
                               separators=(',', ':') if compact else None,
                               indent=None if compact else 2) + '\n', encoding='utf-8')

def csv_write(path, rows):
    with path.open('w', encoding='utf-8', newline='') as file:
        writer = csv.DictWriter(file, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)

def cube_rows(code):
    with zipfile.ZipFile(ROOT / RAW_CA / f'{code}-eng.zip') as archive:
        with archive.open(f'{code}.csv') as file:
            yield from csv.DictReader(io.TextIOWrapper(file, encoding='utf-8-sig'))

def pct(new, old):
    return 100 * (new / old - 1) if new is not None and old not in (None, 0) else None

def lagged(values, lag):
    return [None if i < lag else pct(value, values[i - lag]) for i, value in enumerate(values)]

class Tables(HTMLParser):
    """Extract text cells without relying on browser layout or stripping source HTML."""
    def __init__(self, path):
        super().__init__(convert_charrefs=True)
        self.table = None
        self.row = None
        self.cell = None
        self.tables = []
        self.feed(path.read_text(encoding='utf-8', errors='replace'))

    def handle_starttag(self, tag, attrs):
        if tag == 'table':
            self.table = []
        elif tag == 'tr' and self.table is not None:
            self.row = []
        elif tag in ('td', 'th') and self.row is not None:
            self.cell = []

    def handle_data(self, data):
        if self.cell is not None:
            self.cell.append(data)

    def handle_endtag(self, tag):
        if tag in ('td', 'th') and self.cell is not None:
            self.row.append(' '.join(''.join(self.cell).split()))
            self.cell = None
        elif tag == 'tr' and self.row is not None:
            if self.row:
                self.table.append(self.row)
            self.row = None
        elif tag == 'table' and self.table is not None:
            self.tables.append(self.table)
            self.table = None

def number(value):
    try:
        result = float(value.replace(',', '').replace('−', '-'))
        return result if math.isfinite(result) else None
    except (ValueError, AttributeError):
        return None

def get_prices():
    series, rows, metadata = {}, [], {}
    retrievals = {row['path']: row for row in json.loads(
        (ROOT / 'data/metadata/retrievals.json').read_text(encoding='utf-8'))}

    def add(country, key, geo, label, month, value, seasonal, series_id, units, status, source, detail):
        ident = f'{country}_{key}' if geo in ('Canada', 'U.S. city average') else f'CA_all_{geo}'
        series.setdefault(ident, {})[month] = value
        retrieval = retrievals.get(source, {}).get('retrieved', '')
        metadata[ident] = dict(country=country, geography=geo, label=label, id=series_id,
                               adjustment=seasonal, units=units)
        rows.append(dict(country=country, agency='Statistics Canada' if country == 'CA' else 'BLS',
                         category=key, label=label, geography=geo, series_id=series_id,
                         adjustment=seasonal, reference_month=month, value=value, units=units,
                         missing=value is None, status=status, source=source, retrieved=retrieval, source_detail=detail))

    names = {label: key for key, label in CA.items()}
    for cube in ('18100004', '18100006'):
        for row in cube_rows(cube):
            month, geo, label = (row[k] for k in ('REF_DATE', 'GEO', 'Products and product groups'))
            if month < START or month > END or label not in names:
                continue
            if geo != 'Canada' and (geo not in PROVINCES or label != 'All-items' or cube != '18100004'):
                continue
            sa = cube == '18100006'
            if sa and label != 'All-items':
                continue
            parts = {k: row[k] for k in ('DGUID', 'SCALAR_FACTOR', 'SCALAR_ID', 'SYMBOL',
                                         'TERMINATED', 'COORDINATE', 'DECIMALS')}
            add('CA', 'sa' if sa else names[label], geo, label, month, number(row['VALUE']),
                'SA' if sa else 'NSA', row['VECTOR'], row['UOM'], row['STATUS'],
                RAW_CA + f'{cube}-eng.zip', json.dumps(parts))

    ids = {'CUUR0000' + code: key for key, code in US.items()}
    ids['CUSR0000SA0'] = 'sa'
    for first, last in BATCHES:
        source = RAW_US + f'api-{first}-{last}.json'
        data = json.loads((ROOT / source).read_text(encoding='utf-8'))
        for item in data['Results']['series']:
            ident = item['seriesID']
            key = ids[ident]
            entries = dict((p['year'] + '-' + p['period'][1:], p) for p in item['data'])
            for month in MONTHS:
                if not first <= int(month[:4]) <= last:
                    continue
                p = entries.get(month, {})
                flags = [f['code'] for f in p.get('footnotes', []) if f.get('code')]
                reference = 'December 1982=100' if key == 'oer' else '1982–84=100'
                add('US', key, 'U.S. city average', US_NAMES.get(key, 'All items'), month,
                    number(p.get('value')), 'SA' if key == 'sa' else 'NSA', ident, reference,
                    'absent_from_response' if not p else '|'.join(flags), source,
                    json.dumps(p.get('footnotes', [])))
            assert entries, ident
    return {key: [values.get(month) for month in MONTHS] for key, values in series.items()}, rows, metadata

def weights():
    ca, us = {}, {}
    for row in cube_rows('18100007'):
        year = int(row['REF_DATE'])
        label = row['Products and product groups']
        if not 2023 <= year <= 2025 or row['GEO'] != 'Canada':
            continue
        if row['Price period of weight'] != 'Weight at basket link month prices':
            continue
        if row['Geographic distribution of weight'] != 'Distribution to selected geographies':
            continue
        if label in {*CA_MAJOR, *(p[0] for p in MATCH.values())}:
            ca.setdefault(str(year), {})[label] = number(row['VALUE'])
    for year in ('2023', '2024', '2025'):
        tables = Tables(ROOT / RAW_US / f'weights-{year}.html').tables
        targets = {*US_MAJOR, *(p[1] for p in MATCH.values())}
        found = {}
        for table in tables:
            for row in table:
                label = re.sub(r'\s*\(\d+\)$', '', row[0]).strip()
                if label in targets and len(row) >= 3 and number(row[1]) is not None:
                    found.setdefault(label, number(row[1]))
        us[year] = found
    for definitions, collected in ((CA_MAJOR, ca), (US_MAJOR, us)):
        for year, row in collected.items():
            assert set(definitions) <= set(row), (year, set(definitions) - set(row))
            assert abs(sum(row[k] for k in definitions) - 100) < .08, (year, row)
    return {'CA': ca, 'US': us, 'major': {'CA': CA_MAJOR, 'US': US_MAJOR}, 'mapping': MATCH}

def uncertainty():
    results = []
    for year in (2023, 2024, 2025):
        book = load_workbook(ROOT / RAW_US / f'variance-{year}.xlsx', read_only=True, data_only=True)
        for sheet in book:
            rows = list(sheet.values)
            geography = rows[2][2]
            for row in rows[7:]:
                if not isinstance(row[0], int) or not row[1]:
                    continue
                for i, lag in enumerate((1, 2, 6, 12)):
                    median, se = row[2 + 2*i:4 + 2*i]
                    if isinstance(se, (int, float)):
                        results.append(dict(year=year, geography=geography, item=row[1],
                                            months=lag, median_change=median, se_pp=se))
        book.close()
    return results

def response_rates():
    path = ROOT / RAW_US / 'monthly-response-rates.xlsx'
    book = load_workbook(path, read_only=True, data_only=True)
    columns = [('C&S quotes', 5, 7), ('C&S outlets', 5, 7), ('Housing units', 2, 5)]
    records = []
    for sheet, (domain, observed, used) in zip(book, columns):
        for row in list(sheet.values)[3:]:
            try:
                month = datetime.strptime(row[0], '%B %Y').strftime('%Y-%m')
            except (TypeError, ValueError):
                continue
            if month <= END:
                records.append(dict(month=month, domain=domain, targeted=row[1] if isinstance(row[1], int) else None,
                                    observed_pct=100*row[observed] if isinstance(row[observed], (float,int)) else None,
                                    used_pct=100*row[used] if isinstance(row[used], (float,int)) else None))
    book.close()
    assert len(records)>270
    return records

def release_data(series):
    """Cross-check BLS identifiers against native labels and published levels."""
    file = ROOT / RAW_US / 'october-2026.html'
    tables = Tables(file).tables
    clean = lambda label: re.sub(r'\s*\(\d+\)$', '', label).replace('’', "'").strip()
    levels = {clean(r[0]):r for i in (1,3) for r in tables[i] if len(r) == 10}
    id_checks = []
    j = MONTHS.index('2025-10')
    for key, title in US_NAMES.items():
        row = levels[clean(title)]
        actual = series['US_'+key]
        assert [number(row[2]), number(row[3]), number(row[4])] == [actual[j], actual[-2], actual[-1]], key
        id_checks.append(dict(key=key, series_id='CUUR0000'+US[key], label=title,
                              month=END, october_2026=actual[-1], october_2025=actual[j]))
    effects = []
    for row in tables[6]:
        if len(row)>=5 and clean(row[0]) in set(map(clean, US_NAMES.values())):
            effects.append(dict(month=END, label=clean(row[0]), relative_importance_pct=number(row[1]),
                                adjusted_monthly_change_pct=number(row[2]), official_effect_pp=number(row[3]),
                                median_se_pp=number(row[4])))
    return id_checks, effects

def correlation(x, y):
    pairs = [(a,b) for a,b in zip(x,y) if a is not None and b is not None]
    if len(pairs) < 3 or any(len(set(z)) < 2 for z in zip(*pairs)):
        return None
    return stats.correlation(*zip(*pairs))

def diagnostic(series, month_window=36):
    x, y = (lagged(series[key], 1) for key in ('CA_sa', 'US_sa'))
    rows = []
    for i, month in enumerate(MONTHS):
        start = max(0, i + 1 - month_window)
        xx, yy = x[start:i+1], y[start:i+1]
        a, b = ([v for v in z if v is not None] for z in (xx, yy))
        rows.append(dict(month=month, ca_monthly=x[i], us_monthly=y[i],
                         rolling_correlation=correlation(xx, yy) if i >= month_window-1 else None,
                         ca_rolling_mean=stats.mean(a) if len(a)>1 and i>=month_window-1 else None,
                         us_rolling_mean=stats.mean(b) if len(b)>1 and i>=month_window-1 else None,
                         ca_rolling_sd=stats.stdev(a) if len(a)>1 and i>=month_window-1 else None,
                         us_rolling_sd=stats.stdev(b) if len(b)>1 and i>=month_window-1 else None))
    return rows

def main():
    series, rows, metadata = get_prices()
    basket = weights()
    se = uncertainty()
    responses = response_rates()
    checks, effects = release_data(series)
    assert None is series['US_all'][MONTHS.index('2025-10')]
    assert None is lagged(series['US_all'], 12)[-1]
    assert abs(lagged(series['US_gasoline'],12)[-1]-37.4)<.051
    assert abs(pct(series['US_sa'][-1], series['US_sa'][-2]) - .4) < .051
    assert abs(pct(series['CA_all'][-1], series['CA_all'][-13]) - float(RELEASE['CA_yoy'])) < .051
    assert all(series[key][-1] is not None for key in ('US_all', 'CA_all', 'CA_sa', 'US_sa'))
    raw = dict(months=MONTHS, series=series, metadata=metadata, weights=basket,
               response=responses, effects=effects,
               uncertainty=[r for r in se if r['geography']=='U.S. city average' and
                            r['item'] in ('All Items','Food','Shelter','Energy','Apparel')])
    json_write(PROCESSED / 'cpi.json', raw, compact=True)
    (PROCESSED / 'cpi-data.js').write_text('globalThis.CPI_DATA = ' + json.dumps(raw, ensure_ascii=False,
                                                          separators=(',', ':'), allow_nan=False) + ';\n', encoding='utf-8')
    csv_write(PROCESSED / 'cpi-long.csv', rows)
    csv_write(PROCESSED / 'bls-standard-errors.csv', se)
    csv_write(PROCESSED / 'bls-response-rates.csv', responses)
    csv_write(PROCESSED / 'bls-october-effects.csv', effects)
    csv_write(PROCESSED / 'bls-series-check.csv', checks)
    csv_write(PROCESSED / 'rolling-statistics.csv', diagnostic(series))
    rates = {}
    for key, values in series.items():
        rates[key] = (lagged(values, 1), lagged(values, 12))
    rate_rows = [dict(month=date, ca_index=series['CA_all'][i], us_index=series['US_all'][i],
                      ca_monthly=rates['CA_all'][0][i], us_monthly=rates['US_all'][0][i],
                      ca_monthly_sa=rates['CA_sa'][0][i], us_monthly_sa=rates['US_sa'][0][i],
                      ca_annual=rates['CA_all'][1][i], us_annual=rates['US_all'][1][i],
                      annual_gap_pp=(rates['CA_all'][1][i]-rates['US_all'][1][i]
                                     if None not in (rates['CA_all'][1][i], rates['US_all'][1][i]) else None))
                 for i,date in enumerate(MONTHS)]
    csv_write(PROCESSED / 'inflation-rates.csv', rate_rows)
    json_write(PROCESSED / 'release-audit.json', dict(expected=RELEASE, reproduced=rate_rows[-1]))
    print(f'{len(MONTHS)} months, {len(series)} series, {len(rows)} records, {len(se)} standard errors')

# Values transcribed from the November 16 Statistics Canada and November 10 BLS releases.
RELEASE = {'CA_yoy': 3.1, 'CA_monthly': 0.2, 'CA_sa_monthly': 0.2,
           'US_sa_monthly': 0.4, 'US_yoy': None}

if __name__ == '__main__':
    main()

