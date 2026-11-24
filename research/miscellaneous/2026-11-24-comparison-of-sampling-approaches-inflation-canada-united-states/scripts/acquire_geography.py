"""Acquire the geographical supplements; use retained national and Canadian cubes."""
import concurrent.futures
import csv
import gzip
import hashlib
import json

from acquire import API, BLS, RAW_US, ROOT, fetch

US_AREAS = {
    '0000': 'U.S. city average', '0100': 'Northeast', '0200': 'Midwest',
    '0300': 'South', '0400': 'West', '0110': 'New England', '0120': 'Middle Atlantic',
    '0230': 'East North Central', '0240': 'West North Central', '0350': 'South Atlantic',
    '0360': 'East South Central', '0370': 'West South Central', '0480': 'Mountain',
    '0490': 'Pacific', 'S12A': 'New York–Newark–Jersey City',
    'S23A': 'Chicago–Naperville–Elgin', 'S37A': 'Dallas–Fort Worth–Arlington',
    'S49A': 'Los Angeles–Long Beach–Anaheim', 'S49F': 'Urban Hawaii', 'S49G': 'Urban Alaska',
    'S11A': 'Boston–Cambridge–Newton',
}
ITEMS = {'all': 'SA0', 'food': 'SAF1', 'shelter': 'SAH1',
         'transport': 'SAT', 'energy': 'SA0E', 'gasoline': 'SETB01'}
API_FIRST, API_LAST = '2017', '2026'
BULK_FALLBACK = ('9.OtherSouth', '10.OtherWest')
SUPPLEMENTS = {
    'weights-2025.xlsx': BLS + '/cpi/tables/relative-importance/2025.xlsx',
    'appendix.html': BLS + '/cpi/additional-resources/geographic-sample.htm',
    'aggregation.html': BLS + '/cpi/additional-resources/cpi-area-aggregation.htm',
    'revision-2028.html': BLS + '/cpi/additional-resources/geographic-revision-2028.htm',
    'operations-2026.html': BLS + '/cpi/notices/2026/data-collection-update.htm',
    'county-2023.zip': 'https://www2.census.gov/geo/docs/maps-data/data/gazetteer/2023_Gazetteer/2023_Gaz_counties_national.zip',
    'counties-2020.zip': 'https://www2.census.gov/geo/tiger/GENZ2020/shp/cb_2020_us_county_20m.zip',
    'cu.series': 'https://download.bls.gov/pub/time.series/cu/cu.series',
}

def raw(name):
    return RAW_US + 'geo-' + name

def official_series():
    with (ROOT / raw('cu.series')).open(encoding='utf-8-sig', newline='') as file:
        for record in csv.DictReader(file, delimiter='\t'):
            row = {key.strip(): value.strip() for key, value in record.items()}
            if (row['area_code'] in US_AREAS and row['item_code'] in ITEMS.values()
                    and row['series_id'].startswith('CUUR') and int(row['end_year']) >= 2018):
                yield row

def series_batches():
    ids = sorted(row['series_id'] for row in official_series() if row['area_code'] != '0000')
    return [ids[i:i+20] for i in range(0, len(ids), 20)]

def bundle_bulk(tag, record):
    """Compress a raw BLS TSV and retain the hash of its original response."""
    target = ROOT / record['path']
    compressed = gzip.compress(target.read_bytes(), mtime=0)
    destination = target.with_suffix(target.suffix + '.gz')
    destination.write_bytes(compressed)
    target.unlink()
    return dict(record, path=raw(f'cu.data.{tag}.gz'), source_sha256=record['sha256'],
                source_bytes=record['bytes'], sha256=hashlib.sha256(compressed).hexdigest(),
                bytes=len(compressed))

def main():
    manifest = ROOT / 'data/metadata/retrievals.json'
    records = {row['path']: row for row in json.loads(manifest.read_text(encoding='utf-8'))}
    def acquire(requests):
        with concurrent.futures.ThreadPoolExecutor(max_workers=4) as pool:
            jobs = [pool.submit(fetch, *request) for request in requests]
            for job in concurrent.futures.as_completed(jobs):
                row = job.result()
                records[row['path']] = row
                print(row['path'], row['status'], row.get('bytes', row.get('error')), flush=True)
    acquire([(raw(name), url) for name, url in SUPPLEMENTS.items() if not (ROOT / raw(name)).exists()])
    batches = [(raw(f'api-{API_FIRST}-{API_LAST}-{i}.json'), API,
              json.dumps({'seriesid': ids, 'startyear': API_FIRST, 'endyear': API_LAST}).encode())
             for i, ids in enumerate(series_batches(), 1)]
    acquire(batches)
    if not (ROOT / batches[-1][0]).exists():
        for tag in BULK_FALLBACK:
            if (ROOT / raw(f'cu.data.{tag}.gz')).exists():
                continue
            row = fetch(raw(f'cu.data.{tag}'), f'https://download.bls.gov/pub/time.series/cu/cu.data.{tag}')
            if row['status'] == 'retrieved':
                row = bundle_bulk(tag, row)
            records[row['path']] = row
    manifest.write_text(json.dumps(sorted(records.values(), key=lambda r: r['path']), indent=2) + '\n',
                        encoding='utf-8')

if __name__ == '__main__':
    main()
