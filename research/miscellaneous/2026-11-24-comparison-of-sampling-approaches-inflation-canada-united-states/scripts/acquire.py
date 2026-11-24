"""Acquire official snapshots and record byte-level provenance. Python 3.11+."""
import argparse
import concurrent.futures
import hashlib
import json
from datetime import datetime
from pathlib import Path
from urllib.request import Request, urlopen
from urllib.error import URLError
from time import sleep

ROOT = Path(__file__).resolve().parents[1]
SC = 'https://www150.statcan.gc.ca'
BLS = 'https://www.bls.gov'
API = 'https://api.bls.gov/publicAPI/v2/timeseries/data/'
RAW_CA = 'data/raw/canada/'
RAW_US = 'data/raw/united-states/'
US = {
    'all': 'SA0', 'core': 'SA0L1E', 'food': 'SAF1', 'shelter': 'SAH1',
    'energy': 'SA0E', 'gasoline': 'SETB01', 'apparel': 'SAA', 'transport': 'SAT',
    'rent': 'SEHA', 'oer': 'SEHC', 'food_home': 'SAF11'
}
SOURCES = {
    **{RAW_CA + f'{cube}-eng.zip': SC + f'/n1/tbl/csv/{cube}-eng.zip'
       for cube in ['18100004', '18100006', '18100007']},
    RAW_CA + 'october-2026.html': SC + '/n1/daily-quotidien/261116/dq261116a-eng.htm',
    RAW_CA + 'october-2026.pdf': SC + '/n1/daily-quotidien/261116/dq261116a-eng.pdf',
    RAW_US + 'october-2026.html': BLS + '/news.release/archives/cpi_11102026.htm',
    **{RAW_US + f'weights-{year}.html': BLS + f'/cpi/tables/relative-importance/{year}.htm'
       for year in [2023, 2024, 2025]},
    **{RAW_US + f'variance-{year}.xlsx': BLS + f'/cpi/tables/variance-estimates/{year}.xlsx'
       for year in [2023, 2024, 2025]},
    RAW_US + 'response-rates.html': BLS + '/cpi/tables/response-rates/',
    RAW_US + 'monthly-response-rates.xlsx': BLS + '/cpi/tables/response-rates/monthly-response-rates-201901-present.xlsx',
    'references/canada-reference.pdf': SC + '/n1/en/pub/62-553-x/62-553-x2023001-eng.pdf',
    'references/canada-basket-2026.pdf': SC + '/n1/pub/62f0014m/62f0014m2026005-eng.pdf',
    'references/canada-basket-2026.html': SC + '/n1/pub/62f0014m/62f0014m2026005-eng.htm',
    'data/metadata/ne_admin1.geojson': 'https://raw.githubusercontent.com/nvkelso/natural-earth-vector/v5.1.2/geojson/ne_50m_admin_1_states_provinces_lakes.geojson',
}
BATCHES = [(2000, 2009), (2010, 2019), (2020, 2026)]
IDS = ['CUUR0000' + code for code in US.values()] + ['CUSR0000SA0']

def fetch(path, url, body=None):
    destination = ROOT / path
    destination.parent.mkdir(parents=True, exist_ok=True)
    headers = {'User-Agent': 'Mozilla/5.0 CPI comparative research (public data)',
               'Accept': '*/*'}
    if body is not None:
        headers['Content-Type'] = 'application/json'
    for attempt in range(3):
        try:
            with urlopen(Request(url, data=body, headers=headers), timeout=30) as response:
                data = response.read()
                modified = response.headers.get('Last-Modified')
            if destination.suffix in {'.zip', '.xlsx'} and not data.startswith(b'PK'):
                raise ValueError('Expected a ZIP container')
            if destination.suffix == '.pdf' and not data.startswith(b'%PDF'):
                raise ValueError('Expected a PDF')
            if body and json.loads(data).get('status') != 'REQUEST_SUCCEEDED':
                raise ValueError('BLS API did not report success')
            destination.write_bytes(data)
            return {'path': path, 'url': url, 'status': 'retrieved',
                    'retrieved': datetime.now().astimezone().isoformat(timespec='seconds'),
                    'last_modified': modified, 'bytes': len(data),
                    'sha256': hashlib.sha256(data).hexdigest()}
        except (URLError, OSError, ValueError) as err:
            if attempt == 2:
                return {'path': path, 'url': url, 'status': 'unavailable', 'error': str(err)}
            sleep(attempt + 1)

def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--only', default='', help='Only acquire paths containing this text')
    args = parser.parse_args()
    requests = [(path, url, None) for path, url in SOURCES.items()]
    requests += [(RAW_US + f'api-{first}-{last}.json', API,
                  json.dumps({'seriesid': IDS, 'startyear': str(first), 'endyear': str(last)}).encode())
                 for first, last in BATCHES]
    manifest = ROOT / 'data/metadata/retrievals.json'
    records = {row['path']: row for row in json.loads(manifest.read_text(encoding='utf-8'))} if manifest.exists() else {}
    with concurrent.futures.ThreadPoolExecutor(max_workers=5) as pool:
        jobs = [pool.submit(fetch, *request) for request in requests if args.only in request[0]]
        for job in concurrent.futures.as_completed(jobs):
            row = job.result()
            records[row['path']] = row
            print(row['path'], row['status'], row.get('bytes', row.get('error')), flush=True)
    manifest.write_text(json.dumps(sorted(records.values(), key=lambda r: r['path']), indent=2) + '\n', encoding='utf-8')

if __name__ == '__main__':
    main()

