"""Build regional CPI observations, weights, sample crosswalks and map geometry offline."""
import csv
import gzip
import io
import json
import re
import struct
import unicodedata
from collections import defaultdict
from pathlib import Path
from zipfile import ZipFile

from openpyxl import load_workbook

from acquire import RAW_CA, RAW_US, ROOT
from acquire_geography import API_FIRST, API_LAST, BULK_FALLBACK, ITEMS, US_AREAS, official_series, raw, series_batches
from make_maps import projector
from prepare_data import CA, END, MONTHS, PROCESSED, Tables, csv_write, cube_rows, json_write, number, pct

OUT = PROCESSED / 'geography'
CUTOFF = '2026-11-24'
DATES = [m for m in MONTHS if m >= '2018-01']
LOOKBACK_START = '2017-01'
CA_AREAS = {
    'NL': ('Newfoundland and Labrador', 1), 'PE': ('Prince Edward Island', 1),
    'NS': ('Nova Scotia', 1), 'NB': ('New Brunswick', 1), 'QC': ('Quebec', 3),
    'ON': ('Ontario', 4), 'MB': ('Manitoba', 1), 'SK': ('Saskatchewan', 1),
    'AB': ('Alberta', 1), 'BC': ('British Columbia', 2), 'YT': ('Whitehorse', 1),
    'NT': ('Yellowknife', 1), 'NU': ('Iqaluit', 1),
}
DIVISIONS = {
    '0110': 'CT ME MA NH RI VT', '0120': 'NJ NY PA', '0230': 'IL IN MI OH WI',
    '0240': 'IA KS MN MO NE ND SD', '0350': 'DE DC FL GA MD NC SC VA WV',
    '0360': 'AL KY MS TN', '0370': 'AR LA OK TX', '0480': 'AZ CO ID MT NV NM UT WY',
    '0490': 'AK CA HI OR WA',
}
US_REGIONS = ['0100', '0200', '0300', '0400']
LOCAL_CODES = {c for c in US_AREAS if c.startswith('S')}
CA_CAPITALS = {'YT': (-135.06, 60.72), 'NT': (-114.37, 62.45), 'NU': (-68.52, 63.75)}
CA_ILLUSTRATIVE = {
    'NL': (-52.71, 47.56), 'PE': (-63.13, 46.24), 'NS': (-63.57, 44.65),
    'NB': (-66.06, 45.27), 'QC': (-73.57, 45.50), 'ON': (-79.38, 43.65),
    'MB': (-97.14, 49.90), 'SK': (-106.67, 52.13), 'AB': (-114.07, 51.04),
    'BC': (-123.12, 49.28),
}
US_ITEMS = {'all': 'All items', 'food': 'Food', 'shelter': 'Shelter',
            'transport': 'Transportation', 'energy': 'Energy', 'gasoline': 'Gasoline (all types)'}
RELEASES = {'CA': {'2026-10': '2026-11-16', '2022-06': '2022-07-20'},
            'US': {'2026-10': '2026-11-10', '2022-06': '2022-07-13'}}
NOTES = {'2025-10': 'Index missing where BLS did not publish or could not collect data'}

def read_metadata():
    return {r['path']: r for r in json.loads((ROOT / 'data/metadata/retrievals.json').read_text(encoding='utf-8'))}

def price_csv(observations, meta, country, filename, retrieved, observation_sources=None):
    rates = {}
    rows = []
    for (area, key), series in sorted(observations.items()):
        m = meta[area, key]
        rates.setdefault(key, {})[area] = []
        for month in DATES:
            level, status = series.get(month, (None, ''))
            year, calendar_month = map(int, month.split('-'))
            previous_month = f'{year - (calendar_month == 1)}-{calendar_month-1 or 12:02}'
            previous_year = f'{year-1}-{calendar_month:02}'
            monthly = pct(level, series.get(previous_month, (None, ''))[0])
            annual = pct(level, series.get(previous_year, (None, ''))[0])
            if level is None and not status:
                status = NOTES.get(month, 'No observation') if country == 'US' else 'No observation'
            source = (observation_sources or {}).get((area, key, month), m['path'])
            rows.append(dict(country=country, area_code=area, geography=m['geography'],
                             category=key, official_item=m['label'], series_id=m['id'],
                             month=month, index=level, monthly_percent=monthly, annual_percent=annual,
                             previous_month_index=series.get(previous_month, (None, ''))[0],
                             previous_year_index=series.get(previous_year, (None, ''))[0],
                             missing_index=level is None, missing_monthly=monthly is None, missing_annual=annual is None,
                             units=m['units'], adjustment='Not seasonally adjusted',
                             price_status=status, release_date=RELEASES[country].get(month, ''),
                             retrieved=retrieved.get(source, {}).get('retrieved', ''), source=source))
            rates[key][area].append([None if x is None else round(x, 5) for x in (monthly, annual)])
    csv_write(OUT / filename, rows)
    return rates

def canada_prices(retrieved):
    areas = {'Canada': 'CA'} | {v[0]: k for k, v in CA_AREAS.items()}
    keys = {CA[key]: key for key in ITEMS}
    obs, meta = defaultdict(dict), {}
    for row in cube_rows('18100004'):
        month, place, label = (row[key] for key in ('REF_DATE', 'GEO', 'Products and product groups'))
        if month < LOOKBACK_START or month > END or label not in keys or row['TERMINATED']:
            continue
        area = areas.get(place, next((a for p, a in areas.items() if place.startswith(p+', ')), None))
        if area is None:
            continue
        k = (area, keys[label])
        obs[k][month] = (number(row['VALUE']), row['STATUS'] or row['SYMBOL'])
        meta[k] = dict(geography=place, label=label, id=row['VECTOR'], units=row['UOM'],
                       path=RAW_CA + '18100004-eng.zip')
    return price_csv(obs, meta, 'CA', 'canada_geographical_cpi.csv', retrieved)

def us_prices(retrieved):
    catalogue = {row['series_id']: row for row in official_series()}
    keys = {v: k for k, v in ITEMS.items()}
    obs, meta, observation_sources = defaultdict(dict), {}, {}
    sources = [RAW_US + 'api-2020-2026.json', RAW_US + 'api-2010-2019.json']
    sources += [raw(f'api-{API_FIRST}-{API_LAST}-{i}.json') for i in range(1, len(series_batches())+1)]
    # National prices span two original API pulls. Each observation retains its own source.
    for source in sources:
        if not (ROOT / source).exists():
            continue
        data = json.loads((ROOT / source).read_text(encoding='utf-8'))
        for series in data['Results']['series']:
            row = catalogue.get(series['seriesID'])
            if row is None:
                continue
            area, key = row['area_code'], keys[row['item_code']]
            k = (area, key)
            if k not in meta or source.startswith(RAW_US + 'geo-'):
                meta[k] = dict(geography=US_AREAS[area], label=US_ITEMS[key], id=row['series_id'],
                               units=row['base_period'], path=source)
            for entry in series['data']:
                period = entry['period']
                month = entry['year'] + '-' + period[1:]
                if period != 'M13' and period.startswith('M') and LOOKBACK_START <= month <= END:
                    footnotes = ' '.join(f.get('text', '') for f in entry.get('footnotes', []) if f)
                    obs[k][month] = (number(entry['value']), footnotes)
                    observation_sources[(*k, month)] = source
    last_source = sources[-1]
    if not (ROOT / last_source).exists():
        # BLS limits unregistered API calls. Use exact observations from its flat files.
        fallback_ids = set(series_batches()[-1])
        for tag in BULK_FALLBACK:
            source = raw(f'cu.data.{tag}.gz')
            with gzip.open(ROOT / source, 'rt', encoding='utf-8-sig', newline='') as file:
                reader = csv.reader(file, delimiter='\t')
                next(reader)
                for line in reader:
                    series_id, year, period, value, *footnote = (s.strip() for s in line)
                    if series_id not in fallback_ids or period == 'M13':
                        continue
                    month = year + '-' + period[1:]
                    if not LOOKBACK_START <= month <= END:
                        continue
                    row = catalogue[series_id]
                    area, key = row['area_code'], keys[row['item_code']]
                    k = (area, key)
                    meta[k] = dict(geography=US_AREAS[area], label=US_ITEMS[key], id=series_id,
                                   units=row['base_period'], path=source)
                    obs[k][month] = (number(value), ' '.join(footnote).strip())
                    observation_sources[(*k, month)] = source
    for row in catalogue.values():
        k = (row['area_code'], keys[row['item_code']])
        if k not in meta:
            raise ValueError(f'Missing requested BLS series: {row["series_id"]}')
    meta_by_area = [dict(r, geography=US_AREAS[r['area_code']]) for r in catalogue.values()]
    csv_write(OUT / 'us_series_catalogue.csv', meta_by_area)
    return price_csv(obs, meta, 'US', 'us_regional_cpi.csv', retrieved, observation_sources)

def published_local_titles():
    with (ROOT / raw('cu.series')).open(encoding='utf-8-sig', newline='') as file:
        for r in csv.DictReader(file, delimiter='\t'):
            row = {key.strip(): value.strip() for key,value in r.items()}
            code, series_id = row['area_code'], row['series_id']
            if code.startswith('S') and series_id == 'CUUR'+code+'SA0' and row['end_year'] == '2026':
                title = row['series_title'].split(' in ',1)[1].split(', all urban consumers',1)[0]
                yield code, dict(name=title, id=series_id)

def weights(retrieved):
    ca_geo, ca_product, ca_table, ca_cells = {}, {}, [], []
    ca_id = {r[0]: k for k, r in CA_AREAS.items()} | {'Canada': 'CA'}
    keys = {CA[key]: key for key in ITEMS}
    src = RAW_CA + '18100007-eng.zip'
    for r in cube_rows('18100007'):
        if r['REF_DATE'] != '2025':
            continue
        label = r['Products and product groups']
        if label not in keys:
            continue
        geo = r['GEO']
        code = ca_id.get(geo.split(',')[0])
        if code is None:
            continue
        total = r['Geographic distribution of weight'] == 'Distribution to Canada'
        if total and label != 'All-items':
            continue
        is_link = r['Price period of weight'] == 'Weight at basket link month prices'
        rec = dict(area_code=code, geography=geo, category=keys[label], label=label,
                   percent=number(r['VALUE']), weight_reference_period='2025',
                   price_period='2026-04' if is_link else '2025', release_date='2026-06-15',
                   distribution=r['Geographic distribution of weight'], vector=r['VECTOR'],
                   status=r['STATUS'] or r['SYMBOL'], source=src, retrieved=retrieved[src]['retrieved'])
        if total:
            ca_table.append(rec)
            if is_link and code != 'CA':
                ca_geo[code] = rec['percent']
        else:
            ca_cells.append(rec)
            if is_link and code != 'CA':
                ca_product.setdefault(keys[label], {})[code] = rec['percent']
    csv_write(OUT / 'canada_geographical_weights.csv', ca_table)
    csv_write(OUT / 'canada_geographical_product_weights.csv', ca_cells)

    src = raw('weights-2025.xlsx')
    sheet = load_workbook(ROOT / src, read_only=True, data_only=True)
    us_geo, us_product, us_table, us_cells = {}, {}, [], []
    basics = {v: k for k, v in US_AREAS.items() if not k.startswith('S')}
    basics |= {'Urban Hawaii': 'S49F', 'Urban Alaska': 'S49G'}
    for _, label, value, _ in sheet['Table 7'].values:
        code = basics.get(re.sub(r'\(\d+\)$', '', label or '').strip())
        if code and isinstance(value, (int, float)):
            us_geo[code] = value
            us_table.append(dict(area_code=code, geography=US_AREAS[code], percent=value,
                                 weight_reference_period='2024', price_period='2025-12',
                                 source_table='Table 7, CPI-U', source=src, retrieved=retrieved[src]['retrieved']))
    for row in sheet['Table 4'].values:
        key = next((k for k, label in US_ITEMS.items()
                    if str(row[1]).strip() == label), None)
        if key:
            cells = us_product.setdefault(key, {})
            for j, code in enumerate(US_REGIONS):
                value = row[2+j*2]
                cells[code] = value if isinstance(value, (float, int)) else None
                us_cells.append(dict(area_code=code, geography=US_AREAS[code], category=key,
                                     percent=cells[code], price_period='2025-12',
                                     weight_reference_period='2024', source_table='Table 4, CPI-U',
                                     source=src, retrieved=retrieved[src]['retrieved']))
    csv_write(OUT / 'us_geographical_weights.csv', us_table)
    csv_write(OUT / 'us_geographical_product_weights.csv', us_cells)
    assert abs(sum(ca_geo.values())-100) < .1 and abs(sum(us_geo[a] for a in US_REGIONS)-100) < .01
    return {'CA': {'geography': ca_geo, 'product': ca_product},
            'US': {'geography': us_geo, 'product': us_product}}

def matching_name(name, strip_type=False):
    name = unicodedata.normalize('NFKD', name).casefold()
    name = ''.join(c for c in name if not unicodedata.combining(c))
    if strip_type:
        name = re.sub(r'\s+(county and borough|census area|municipality|parish|county|borough|city)$', '', name)
    return re.sub('[^a-z0-9]', '', name.replace('saint', 'st'))

def county_geometries():
    """Read the Census polygon shapefile and corresponding DBF in record order."""
    with ZipFile(ROOT / raw('counties-2020.zip')) as archive:
        contents = {Path(n).suffix: archive.read(n) for n in archive.namelist()
                    if Path(n).suffix in ['.shp', '.dbf']}
    dbf = contents['.dbf']
    count, header, length = struct.unpack_from('<IHH', dbf, 4)
    fields, offset = {}, 1
    for i in range(32, header-1, 32):
        name = dbf[i:i+11].split(b'\0')[0].decode()
        size = dbf[i+16]
        fields[name] = (offset, offset+size)
        offset += size
    shp = contents['.shp']
    p = 100
    for j in range(count):
        record = dbf[header+j*length:header+(j+1)*length]
        attributes = {k: record[a:b].decode('utf-8', errors='replace').strip() for k, (a, b) in fields.items()}
        size = struct.unpack_from('>i', shp, p+4)[0]*2
        start, p = p+8, p+8+size
        if struct.unpack_from('<i', shp, start)[0] != 5:
            continue
        part_count, point_count = struct.unpack_from('<ii', shp, start+36)
        parts = struct.unpack_from(f'<{part_count}i', shp, start+44)
        points = list(struct.iter_unpack('<dd', shp[start+44+4*part_count:start+44+4*part_count+16*point_count]))
        rings = [points[a:b] for a,b in zip(parts, (*parts[1:], point_count))]
        if rings:
            yield attributes, rings

def map_fit(subset, area, bounds):
    project = projector(area)
    points = [project(v) for _, polygons in subset for polygon in polygons for ring in polygon for v in ring]
    xmin, ymin = map(min, zip(*points))
    xmax, ymax = map(max, zip(*points))
    x, y, w, h = bounds
    scale = min((w-5)/(xmax-xmin), (h-5)/(ymax-ymin))
    x0, y0 = (w-(xmax-xmin)*scale)/2+x, (h-(ymax-ymin)*scale)/2+y
    def fit(v):
        px, py = project(v)
        return round(x0+(px-xmin)*scale, 2), round(y0+(py-ymin)*scale, 2)
    return fit

def path_string(rings, fit):
    paths = []
    for ring in rings:
        pts = [fit(v) for v in ring]
        pts = [p for i, p in enumerate(pts) if i == 0 or p != pts[i-1]]
        if len(pts) >= 4:
            paths.append('M'+' '.join(f'{p[0]:g},{p[1]:g}' for p in pts)+'Z')
    return ''.join(paths)

def state_maps():
    features = json.loads((ROOT / 'data/metadata/ne_admin1.geojson').read_text(encoding='utf-8'))['features']
    areas = {'Canada': [], 'Lower 48': [], 'Alaska': [], 'Hawaii': []}
    for f in features:
        p, g = f['properties'], f['geometry']
        if p['admin'] not in ('Canada', 'United States of America'):
            continue
        area = p['admin'] if p['admin'] == 'Canada' else p['name'] if p['name'] in ('Alaska', 'Hawaii') else 'Lower 48'
        polygons = g['coordinates'] if g['type'] == 'MultiPolygon' else [g['coordinates']]
        areas[area].append(((p['iso_3166_2'][-2:], p['name']), polygons))
    bounds = {'Canada': (0, 0, 460, 248), 'Lower 48': (3, 0, 454, 226),
              'Alaska': (8, 169, 104, 75), 'Hawaii': (345, 183, 99, 59)}
    maps, fits = {'CA': {}, 'US': {}}, {}
    divisions = {state: code for code, states in DIVISIONS.items() for state in states.split()}
    for area, subset in areas.items():
        fit = fits[area] = map_fit(subset, area, bounds[area])
        for (code, name), polygons in subset:
            record = dict(name=name, path=path_string([r for p in polygons for r in p], fit))
            if area == 'Canada':
                label, count = CA_AREAS[code]
                record.update(area=label, strata=count, capital=code in CA_CAPITALS,
                              point=fit((CA_CAPITALS | CA_ILLUSTRATIVE)[code]))
                maps['CA'][code] = record
            else:
                division = divisions[code]
                record.update(division=division, region=division[:2]+'00',
                              point=fit((-77.0365, 38.8977)) if code == 'DC' else None)
                maps['US'][code] = record
    ca_rows = [dict(area_code=k, administrative_name=r['name'], cpi_geography=r['area'],
                    documented_strata=r['strata'], scope='Capital only' if r['capital'] else 'Province',
                    source='Statistics Canada CPI Reference Paper, chapter 4 (2023)')
               for k,r in maps['CA'].items()]
    csv_write(OUT / 'canada_geographical_classifications.csv', ca_rows)
    csv_write(OUT / 'us_geographical_classifications.csv', [
        dict(state=k, name=r['name'], census_region=US_AREAS[r['region']], region_code=r['region'],
             census_division=US_AREAS[r['division']], division_code=r['division'],
             geometry='Natural Earth 5.1.2', source='U.S. Census Bureau region/division classification')
        for k,r in maps['US'].items()])
    assert len(maps['CA']) == 13 and len(maps['US']) == 51
    return maps, fits

def psu_data(fits, retrieved):
    county_points = {}
    source = raw('county-2023.zip')
    with ZipFile(ROOT / source) as archive:
        with archive.open(archive.namelist()[0]) as handle:
            rows = csv.DictReader(io.TextIOWrapper(handle, encoding='utf-8-sig'), delimiter='\t')
            for row in rows:
                r = {k.strip(): v.strip() for k,v in row.items()}
                county_points[r['GEOID']] = (float(r['INTPTLONG']), float(r['INTPTLAT']))
    shapes, independent = {}, {}
    for row, rings in county_geometries():
        geoid = row['GEOID']
        fips = geoid[:2]
        canonical = matching_name(row['NAME'])
        encoded = (geoid, rings)
        if row['LSAD'] == '25':
            independent[fips, canonical] = encoded
        else:
            shapes[fips, canonical] = encoded
        shapes[fips, matching_name(row['NAMELSAD'])] = encoded
    state_fips = dict(zip('AL AK AZ AR CA CO CT DE DC FL GA HI ID IL IN IA KS KY LA ME MD MA MI MN MS MO MT NE NV NH NJ NM NY NC ND OH OK OR PA RI SC SD TN TX UT VT VA WA WV WI WY'.split(),
                         '01 02 04 05 06 08 09 10 11 12 13 15 16 17 18 19 20 21 22 23 24 25 26 27 28 29 30 31 32 33 34 35 36 37 38 39 40 41 42 44 45 46 47 48 49 50 51 53 54 55 56'.split()))
    for fips, alias, canonical in [('12','Dade','Miami-Dade'), ('46','Shannon','Oglala Lakota'),
                                   ('02','Wade Hampton','Kusilvak')]:
        value = shapes.get((fips,matching_name(canonical)))
        if value:
            shapes[fips,matching_name(alias)] = value
    aggregate, group_table = Tables(ROOT / raw('aggregation.html')).tables[:2]
    basic_by_psu = {psu: row[0] for row in group_table[1:] if len(row)==4
                    for psu in row[2].split(', ')}
    basics = set(aggregate[1][3].split(', '))
    appendix = raw('appendix.html')
    rows, county_rows, psus = [], [], []
    population = defaultdict(int)
    for record in Tables(ROOT / appendix).tables[0][1:]:
        if len(record) != 8 or not re.fullmatch(r'[NS]\w{3}', record[2]):
            continue
        region, division, code, label, definition, cnt, share, cycle = record
        r = next(k for k in US_REGIONS if US_AREAS[k] == region.split('-', 1)[1])
        d = next(k for k in DIVISIONS if US_AREAS[k] == division.split('-', 1)[1])
        basic = basic_by_psu.get(code, code)
        count = int(cnt.replace(',', ''))
        population[r] += count
        cert = 'Statewide urban certainty exception' if code in ('S49F','S49G') else (
                'Self-representing metro' if code.startswith('S') else 'Non-self-representing')
        matches = list(re.finditer(r'\b([A-Z]{2}):\s*', definition))
        states, polygons, points, missing = [], [], [], []
        for i, match in enumerate(matches):
            state = match[1]
            states.append(state)
            block = definition[match.end():matches[i+1].start() if i+1 < len(matches) else None].rstrip('. ;')
            if state not in state_fips:
                continue
            fit = fits[{'AK':'Alaska','HI':'Hawaii'}.get(state,'Lower 48')]
            for name in block.split(', '):
                fips = state_fips[state]
                plain = matching_name(name)
                item = shapes.get((fips,plain)) or independent.get((fips,plain))
                if not item:
                    item = shapes.get((fips,matching_name(name,strip_type=True)))
                if not item and state == 'DC':
                    item = shapes.get(('11','districtofcolumbia')) or shapes.get(('11','washington'))
                geoid, rings = item if item else ('', [])
                county_rows.append(dict(psu_code=code, state=state, official_county_text=name,
                                        county_geoid=geoid, matched=bool(item), boundary_vintage='2020',
                                        source=appendix))
                if not item:
                    missing.append(name)
                    continue
                polygons.append(path_string(rings, fit))
                point = county_points.get(geoid)
                if point is None:
                    # Connecticut county boundaries predate the 2023 planning-region geography.
                    ring = max(rings,key=len)
                    point = tuple(sum(v[i] for v in ring)/len(ring) for i in range(2))
                points.append(fit(point))
        if not points:
            print(f'No matched counties for {code}: {definition}')
        entry = dict(code=code, name=label, basic=basic, region=r, division=d, type=cert,
                     states=states, definition=definition, population=count, populationShare=float(share),
                     cycle=cycle, point=([round(sum(v[i] for v in points)/len(points),2) for i in range(2)]
                                         if points else None), paths=polygons, missing=missing)
        psus.append(entry)
        rows.append(dict(psu_code=code, name=label, index_area_code=basic, region=r, division=d,
                         selection_type=cert, states=';'.join(states), official_counties=definition,
                         population_represented=count, population_reference='2010 decennial census',
                         percent_of_index_population=share, pricing_cycle=cycle,
                         representative_x=entry['point'][0] if points else '',
                         representative_y=entry['point'][1] if points else '',
                         unmatched_counties=';'.join(missing), source=appendix,
                         retrieved=retrieved[appendix]['retrieved']))
    assert len(psus) == 75 and len(basics) == 32 and {p['basic'] for p in psus} == basics
    assert next(p for p in psus if p['code']=='S12A')['states'] == ['NJ','NY','PA']
    csv_write(OUT / 'us_cpi_psu_crosswalk.csv', rows)
    csv_write(OUT / 'us_cpi_psu_counties.csv', county_rows)
    csv_write(OUT / 'us_cpi_area_aggregation.csv', [
        dict(area_code=r[0], name=r[1], area_type=r[2], basic_areas=r[3], source=raw('aggregation.html'))
        for r in aggregate[1:] if len(r)==4])
    return psus, dict(population), sorted(basics)

def main():
    OUT.mkdir(parents=True, exist_ok=True)
    retrieved = read_metadata()
    maps, fits = state_maps()
    psus, population, basics = psu_data(fits, retrieved)
    data = dict(dates=DATES, cutoff=CUTOFF, maps=maps, psus=psus, basics=basics,
                weights=weights(retrieved), population=population,
                labels={'US': US_AREAS, 'CA': ({k:r[0] for k,r in CA_AREAS.items()} | {'CA':'Canada'})},
                areas={'US': US_REGIONS, 'CA': list(CA_AREAS)}, divisions=list(DIVISIONS),
                publishedLocal=dict(published_local_titles()),
                metros=sorted(LOCAL_CODES), rates={'CA': canada_prices(retrieved), 'US': us_prices(retrieved)})
    json_write(OUT / 'geography.json', data, compact=True)
    (OUT / 'geography-data.js').write_text(
        'globalThis.CPI_GEO = ' + json.dumps(data, ensure_ascii=False, separators=(',', ':'), allow_nan=False) + ';\n',
        encoding='utf-8')
    print('Prepared regional series, weights, geographical classifications, 75 PSU records and maps.')

if __name__ == '__main__':
    main()
