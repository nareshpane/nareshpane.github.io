"""Acquire public inputs and keep the exact original or service query responses."""
import argparse
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
import hashlib
import json
import math
from pathlib import PurePosixPath
import re
import shutil
import zipfile

import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

from common import ROOT, CALGARY, datum, json_write

AB = 'https://geospatial.alberta.ca/mimas/rest/services/'
STAT = 'https://www12.statcan.gc.ca/census-recensement/2021/'
FTP = 'https://ftp.maps.canada.ca/pub/statcan_statcan/'
FIRE = AB + 'wildfire/historical_wildfires_data/FeatureServer'
FLOOD = AB + 'fama_data_distribution/fama_public_data_distribution/MapServer'
ROAD = 'https://geo.statcan.gc.ca/geo_wa/rest/services/2021/Road_Network_File/MapServer'
BOUNDARIES = 'https://geo.statcan.gc.ca/geo_wa/rest/services/2021/Cartographic_boundary_files/MapServer'
HEALTH_2020 = 'https://maps-cartes.services.geo.ca/server2_serveur2/rest/services/StatCan/OpenDatabaseHealthFacilities/MapServer'
BUILD = 'https://services1.arcgis.com/AVP60cs0Q9PEA8rH/ArcGIS/rest/services/Buildings_from_Digital_Aerial_Survey/FeatureServer'
FILES = {
    'boundaries': ('shared', 'Alberta_Census_Boundaries_SHP.zip',
        'https://extranet.gov.ab.ca/srd/geodiscover/srd_pub/boundaries/census/Alberta_Census_Boundaries_SHP.zip'),
    'population': ('shared', '98100015-eng.zip', 'https://www150.statcan.gc.ca/n1/tbl/csv/98100015-eng.zip'),
    'points': ('shared', 'geosuite_da_2021.parquet',
        'https://data.mountainmath.ca/census/geo_attributes/geo_attributes_2021.parquet'),
    'healthcare': ('healthcare', 'odhf_v2_gpkg.zip',
        'https://www150.statcan.gc.ca/n1/pub/13-26-0001/2020001/zip/odhf_v2_gpkg.zip'),
    'profile': ('socioeconomic', 'census_profile.parquet',
        'https://data.source.coop/dataforcanada/d4c-datapkg-statistical/processed/'
        'ca_statcan_2021A000011124_d4c-datapkg-statistical_census_pop_dissemination_areas_2021_v0.1.0-beta.parquet'),
    'incidents': ('wildfire', 'fp-historical-wildfire-data-2006-2025.csv',
        'https://open.alberta.ca/dataset/a221e7a0-4f46-4be7-9c5a-e29de9a3447e/resource/80480824-0c50-456c-9723-f9d4fc136141/download/fp-historical-wildfire-data-2006-2025.csv'),
}
CANADA = 'https://open.canada.ca/en/open-government-licence-canada'
ALBERTA = 'https://open.alberta.ca/licence'
CALGARY_LICENCE = 'https://data.calgary.ca/d/Open-Data-Terms/u45n-7awa'
FILE_LICENCES = {'boundaries': ALBERTA, 'incidents': ALBERTA}
MANUAL = {
    'points': ('2021_92-151_X.zip',
               STAT + 'geo/aip-pia/attribute-attribs/files-fichiers/2021_92-151_X.zip'),
    'profile': ('98-401-X2021006_eng_CSV_Prairies.zip',
                STAT + 'dp-pd/prof/details/download-telecharger/comp/GetFile.cfm?FILETYPE=CSV&GEONO=006_Prairies&Lang=E'),
}


def client():
    session = requests.Session()
    retry = Retry(total=3, backoff_factor=0.8, status_forcelist=(429, 500, 502, 503, 504))
    session.mount('https://', HTTPAdapter(max_retries=retry))
    return session


def validate(path, suffix):
    with path.open('rb') as f:
        head = f.read(2048).lstrip()
    if suffix == '.zip' and zipfile.is_zipfile(path):
        return
    if suffix == '.parquet' and head.startswith(b'PAR1'):
        return
    if suffix == '.gpkg' and head.startswith(b'SQLite format 3'):
        return
    if suffix == '.csv' and b',' in next(iter(head.splitlines()), b'') and not head.startswith(b'<'):
        return
    if suffix in ('.json', '.geojson') and head.startswith(b'{'):
        return
    raise ValueError(f'{path.name}: invalid {suffix} data or HTML error page')


def fingerprint(path, url, licence, previous, extra=None):
    relative = path.relative_to(ROOT).as_posix()
    with path.open('rb') as f:
        digest = hashlib.file_digest(f, 'sha256').hexdigest()
    record = {'path': relative, 'url': url, 'licence': licence,
              'bytes': path.stat().st_size, 'sha256': digest,
              'retrieved_at': datetime.now(timezone.utc).isoformat()}
    old = previous.get(relative, {})
    if old.get('sha256') == digest:
        record['retrieved_at'] = old['retrieved_at']
        record.update({k: v for k, v in old.items() if k not in record})
    if extra:
        record.update(extra)
    return record


def download(key, previous):
    area, name, url = FILES[key]
    if key in MANUAL:
        manual_name, manual_url = MANUAL[key]
        supplied = datum(area, 'raw', manual_name)
        if supplied.exists():
            validate(supplied, '.zip')
            return [fingerprint(supplied, manual_url, CANADA, previous, {'retrieval': 'Manually supplied'})]
    path = datum(area, 'raw', name)
    if not path.exists():
        temporary = path.with_suffix(path.suffix + '.part')
        try:
            with client().get(url, timeout=(12, 180), stream=True) as response:
                response.raise_for_status()
                with temporary.open('wb') as output:
                    for chunk in response.iter_content(1024 * 1024):
                        output.write(chunk)
                validate(temporary, path.suffix)
                temporary.replace(path)
        finally:
            temporary.unlink(missing_ok=True)
    validate(path, path.suffix)
    return [fingerprint(path, url, FILE_LICENCES.get(key, CANADA), previous)]


def extract(path, target, predicate=lambda name: True):
    """Select archive members and refuse ambiguous or escaping paths."""
    target.mkdir(parents=True, exist_ok=True)
    selected = []
    with zipfile.ZipFile(path) as archive:
        for member in archive.infolist():
            name = PurePosixPath(member.filename.replace('\\', '/'))
            if name.is_absolute() or '..' in name.parts or any(':' in part for part in name.parts):
                raise ValueError(f'Unsafe member {member.filename}')
            if member.is_dir() or not predicate(str(name)):
                continue
            destination = target.joinpath(*name.parts)
            if not destination.resolve().is_relative_to(target.resolve()):
                raise ValueError(f'Archive member escapes destination: {name}')
            destination.parent.mkdir(parents=True, exist_ok=True)
            if not destination.exists() or destination.stat().st_size != member.file_size:
                temporary = destination.with_suffix(destination.suffix + '.part')
                with archive.open(member) as source, temporary.open('wb') as output:
                    shutil.copyfileobj(source, output)
                temporary.replace(destination)
            selected.append(destination)
    return selected


def request_json(session, url, **params):
    response = session.get(url, params=params, timeout=(12, 100))
    response.raise_for_status()
    value = response.json()
    if 'error' in value:
        raise ValueError(f"{response.url}: {value['error']}")
    return value


def cached_json(session, path, url, **params):
    if path.exists():
        return json.loads(path.read_text(encoding='utf-8'))
    value = request_json(session, url, **params)
    json_write(path, value)
    return value


def query_layer(session, area, stem, url, columns, licence, previous, extent=None, where='1=1',
                paginate=False, display_offset=None):
    schema_path = datum(area, 'raw', stem + '.schema.json')
    schema = cached_json(session, schema_path, url, f='json')
    fields = {field['name'].casefold(): field['name'] for field in schema.get('fields', [])}
    measured = [fields[name.casefold()] for name in columns]
    oid = schema.get('objectIdField') or next((f['name'] for f in schema['fields']
                                             if f['type'] == 'esriFieldTypeOID'))
    query = {'where': where}
    if extent:
        query.update(geometry=','.join(map(str, extent)), geometryType='esriGeometryEnvelope',
                     inSR=4326, spatialRel='esriSpatialRelIntersects')
    path = datum(area, 'raw', stem + '.geojson')
    if not path.exists():
        ids = request_json(session, url + '/query', f='json', returnIdsOnly='true', **query)['objectIds'] or []
        ids = sorted(set(ids))
        size = min(int(schema.get('maxRecordCount', 1000)), 700 if paginate else 120 if display_offset else 240)
        features = []
        for offset in range(0, len(ids), size):
            group = ids[offset:offset + size]
            pagination = dict(query, resultOffset=offset, resultRecordCount=size, orderByFields=oid) if paginate else {
                'where': '1=1', 'objectIds': ','.join(map(str, group))}
            if display_offset:
                pagination['maxAllowableOffset'] = display_offset
            response = request_json(session, url + '/query', f='geojson',
                                    outFields=','.join(dict.fromkeys([oid] + measured)), outSR=4326,
                                    returnGeometry='true', **pagination)
            page = response['features']
            if {f['properties'].get(oid, f.get('id')) for f in page} != set(group):
                raise ValueError(f'Incomplete ArcGIS response for {stem} at offset {offset}')
            features.extend(page)
        json_write(path, {'type': 'FeatureCollection', 'features': features})
        print(f'{stem}: {len(features):,} features', flush=True)
    return [fingerprint(schema_path, url + '?f=json', licence, previous),
            fingerprint(path, url + '/query', licence, previous,
                        {'selection': query, 'fields': measured, 'display_offset_degrees': display_offset})]


def service_layers(area, url, previous, licence):
    path = datum(area, 'raw', url.split('/')[-2] + '.service.json')
    value = cached_json(client(), path, url, f='json')
    return value['layers'], fingerprint(path, url + '?f=json', licence, previous)


def fire(previous):
    layers, record = service_layers('wildfire', FIRE, previous, ALBERTA)
    jobs = [layer for layer in layers if re.fullmatch(r'Historical Wildfire Perimeter - \d+ Years? Prior', layer['name'])]
    def acquire(layer):
        return query_layer(client(), 'wildfire', 'perimeter_' + str(layer['id']), f"{FIRE}/{layer['id']}",
                           ['YEAR', 'FIRENUMBER', 'BURNCODE'], ALBERTA, previous)
    with ThreadPoolExecutor(max_workers=5) as pool:
        return [record] + [entry for records in pool.map(acquire, jobs) for entry in records]


def flood(previous):
    layers, record = service_layers('flood', FLOOD, previous, ALBERTA)
    jobs = [layer for layer in layers if layer['name'] in
            {'1:20 Open Water Flood', '1:50 Open Water Flood', '1:100 Open Water Flood',
             '1:200 Open Water Flood', '1:500 Open Water Flood', 'Flood Hazard'}]
    def acquire(layer):
        period = re.search(r'1:(\d+)', layer['name'])
        stem = 'flood_' + period[1] if period else 'study_coverage'
        return query_layer(client(), 'flood', stem, f"{FLOOD}/{layer['id']}",
                           ['FloodReturnPeriod', 'StudyName', 'RiverName'] if period else [],
                           ALBERTA, previous, CALGARY if period else None,
                           display_offset=None if period else 0.001)
    with ThreadPoolExecutor(max_workers=6) as pool:
        return [record] + [entry for records in pool.map(acquire, jobs) for entry in records]


def roads(previous):
    layers, record = service_layers('flood', ROAD, previous, CANADA)
    def acquire(layer):
        rank = re.search(r'Rank (\d+)', layer['name'])
        return query_layer(client(), 'flood', 'roads_' + rank[1], f"{ROAD}/{layer['id']}",
                           [], CANADA, previous, CALGARY)
    with ThreadPoolExecutor(max_workers=5) as pool:
        return [record] + [entry for records in pool.map(acquire, layers) for entry in records]


def buildings(previous):
    layers, record = service_layers('flood', BUILD, previous, CALGARY_LICENCE)
    layer = next(x for x in layers if x['name'] == 'DAS_BUILDING')
    return [record] + query_layer(client(), 'flood', 'buildings', f"{BUILD}/{layer['id']}",
                                  [], CALGARY_LICENCE, previous, CALGARY, paginate=True)


def centres(previous):
    layers, record = service_layers('shared', BOUNDARIES, previous, CANADA)
    layer = next(x for x in layers if x['name'].startswith('PC - '))
    return [record] + query_layer(client(), 'shared', 'population_centres',
                                  f"{BOUNDARIES}/{layer['id']}", ['PCUID', 'PCNAME'], CANADA,
                                  previous, (-120.1, 48.95, -109.9, 60.1))


def historical_hospitals(previous):
    layers, record = service_layers('healthcare', HEALTH_2020, previous, CANADA)
    layer = next(x for x in layers if x['name'] == 'Health Facilities in Canada')
    return [record] + query_layer(client(), 'healthcare', 'hospital_2020',
                                  f"{HEALTH_2020}/{layer['id']}",
                                  ['name_nom', 'facility_type', 'province_en', 'latitude', 'longitude'],
                                  CANADA, previous, where="province_en='Alberta' AND facility_type='Hospitals'")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    tasks = {key: lambda old, key=key: download(key, old) for key in FILES}
    tasks.update(perimeters=fire, flood=flood, roads=roads, buildings=buildings, centres=centres,
                 hospitals_2020=historical_hospitals)
    parser.add_argument('sources', nargs='*', metavar='SOURCE', help=', '.join(tasks))
    selected = parser.parse_args().sources or list(tasks)
    if set(selected) - tasks.keys():
        parser.error('Unknown source: ' + ', '.join(set(selected) - tasks.keys()))
    path = ROOT / 'metadata' / 'acquisition_manifest.json'
    manifest = json.loads(path.read_text(encoding='utf-8')) if path.exists() else {'files': {}, 'failures': {}}
    manifest['files'] = {key: record for key, record in manifest['files'].items() if (ROOT / key).exists()}
    with ThreadPoolExecutor(max_workers=8) as pool:
        jobs = {key: pool.submit(tasks[key], manifest['files'].copy()) for key in selected}
        for key, job in jobs.items():
            try:
                for record in job.result():
                    manifest['files'][record['path']] = record
                manifest['failures'].pop(key, None)
                print(f'{key}: ready', flush=True)
            except (OSError, ValueError, KeyError, StopIteration, requests.RequestException) as error:
                manifest['failures'][key] = str(error)
                print(f'{key}: {error}', flush=True)
    json_write(path, manifest)
    return int(any(key in manifest['failures'] for key in selected))


if __name__ == '__main__':
    raise SystemExit(main())


