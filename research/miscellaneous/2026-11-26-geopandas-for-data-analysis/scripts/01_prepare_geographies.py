"""Build 2021 Alberta analysis geographies and reconcile the population join."""
import importlib
import zipfile

import geopandas as gpd
import pandas as pd
from pyproj import Geod

from common import ROOT, AREA, DISTANCE, clean, datum, positive_parts, save, summarize, web

extract = importlib.import_module('00_download_data').extract


def prepare():
    raw = datum('shared', 'raw', 'Alberta_Census_Boundaries_SHP.zip')
    layers = extract(raw, raw.parent / 'boundaries')
    polygons = {}
    for path in sorted(p for p in layers if p.suffix.casefold() == '.shp'):
        frame, repairs = clean(gpd.read_file(path))
        keys = {c.upper(): c for c in frame.columns}
        for name, id_columns, max_count in (
            ('das', ('DAUID', 'DA_UID'), 99999), ('divisions', ('CDUID', 'CD_UID'), 25),
            ('subdivisions', ('CSDUID', 'CSD_UID'), 1000), ('centres', ('PCUID', 'POPCTUID', 'POPCUID'), 150)):
            key = next((keys[c] for c in id_columns if c in keys), None)
            if key and frame[key].is_unique and len(frame) <= max_count and name not in polygons:
                frame = positive_parts(frame, 'Polygon')
                frame['uid'] = frame[key].astype(str).str.split('.').str[0]
                polygons[name] = (frame, repairs, path.name)
    if 'das' not in polygons:
        raise ValueError(f'No unique 2021 dissemination area layer among {[p.name for p in layers if p.suffix == ".shp"]}')
    archive = datum('shared', 'raw', '98100015-eng.zip')
    with zipfile.ZipFile(archive) as source:
        with source.open('98100015.csv') as stream:
            table = pd.read_csv(stream, dtype={'DGUID': str}, encoding='utf-8-sig', low_memory=False)
    alberta = table.loc[table.DGUID.eq('2021A000248')]
    table = table.loc[table.DGUID.str.match(r'^2021S0512(48\d{6})$', na=False)].copy()
    columns = {'Population, 2021': 'population', 'Land area in square kilometres, 2021': 'land_km2',
               'Population density per square kilometre, 2021': 'density'}
    source_columns = {next(c for c in table if f': {key} [' in c): value
                      for key, value in columns.items()}
    attrs = table[['DGUID', *source_columns]].rename(columns=source_columns)
    attrs['uid'] = attrs.DGUID.str[-8:]
    attrs.drop(columns='DGUID').to_csv(datum('shared', 'processed', 'census_counts.csv'), index=False)
    areas, repair_count, layer = polygons['das']
    das = areas[['uid', 'geometry']].merge(attrs, on='uid', how='left', validate='1:1').to_crs(DISTANCE)
    assert das.uid.str.fullmatch(r'48\d{6}').all()
    assert set(attrs.uid) <= set(das.uid)
    province_count = int(alberta[next(iter(source_columns))].iloc[0])
    assert int(das.population.sum()) == province_count
    outline = gpd.GeoDataFrame({'name': ['Alberta']}, geometry=[das.geometry.union_all()], crs=DISTANCE)
    save(das, 'shared', 'das')
    save(outline, 'shared', 'alberta')
    outline.to_file(datum('shared', 'processed', 'alberta_boundary.gpkg'), layer='alberta', index=False)
    web(outline, 'shared', 'outline', 150)
    web(das[['uid', 'density', 'geometry']], 'shared', 'das', 90)

    points = das.copy()
    points.geometry = points.representative_point()
    points['point_method'] = 'Interior point'
    gaf = datum('shared', 'raw', '2021_92-151_X.zip')
    mirror = datum('shared', 'raw', 'geosuite_da_2021.parquet')
    if gaf.exists() or mirror.exists():
        if gaf.exists():
            with zipfile.ZipFile(gaf) as source:
                member = next(n for n in source.namelist() if n.lower().endswith('.csv'))
                with source.open(member) as stream:
                    coordinates = pd.read_csv(stream, encoding='latin1', low_memory=False)
        else:
            coordinates = pd.read_parquet(mirror)
        coordinates.columns = coordinates.columns.str.upper()
        aliases = {'DALAT': 'lat', 'DALONG': 'lon', 'DARPLAT': 'lat', 'DARPLONG': 'lon',
                   'REP_LAT': 'lat', 'REP_LON': 'lon',
                   'LAT': 'lat', 'LON': 'lon', 'DA_RP_LAT': 'lat', 'DA_RP_LONG': 'lon',
                   'DAUID': 'uid', 'GUID': 'uid', 'GEOUID': 'uid'}
        coordinates = coordinates.rename(columns=aliases)
        if not {'uid', 'lat', 'lon'}.issubset(coordinates):
            raise ValueError(f'Representative-point schema: {list(coordinates)}')
        coordinates['uid'] = coordinates.uid.astype(str).str[-8:]
        coordinates = coordinates[['uid', 'lat', 'lon']].dropna().drop_duplicates()
        coordinates = coordinates[coordinates.uid.isin(points.uid)].set_index('uid')
        assert coordinates.index.is_unique
        same = points.uid.isin(coordinates.index)
        subset = coordinates.loc[points.loc[same, 'uid']]
        official = gpd.GeoSeries(gpd.points_from_xy(subset.lon, subset.lat), crs=4269,
                                 index=points.index[same]).to_crs(DISTANCE)
        points.loc[same, 'geometry'] = official
        points.loc[same, 'point_method'] = 'Official representative point'
    save(points, 'shared', 'demand')

    for name in ('divisions', 'subdivisions', 'centres'):
        if name not in polygons:
            continue
        frame = polygons[name][0]
        fields = [c for c in frame if c.upper() in {'CDNAME', 'CSDNAME', 'PCNAME', 'POPCNAME', 'NAME'}]
        frame = frame[['uid', *fields, 'geometry']].to_crs(DISTANCE)
        save(frame, 'shared', name)
        web(frame, 'shared', name, 100)
    centres = datum('shared', 'raw', 'population_centres.geojson')
    if centres.exists():
        frame, _ = clean(gpd.read_file(centres))
        frame = frame.rename(columns={'PCUID': 'uid', 'PCNAME': 'name'})
        frame = positive_parts(gpd.clip(frame.to_crs(DISTANCE), outline), 'Polygon')
        if 'uid' not in frame:
            frame['uid'] = frame.index.astype(str)
        save(frame[['uid', 'name', 'geometry']], 'shared', 'centres')
        web(frame[['uid', 'name', 'geometry']], 'shared', 'centres', 20)
    geod = Geod(ellps='WGS84')
    pair = [-114.0719, 51.0447], [-113.4909, 53.5461]
    straight = gpd.GeoSeries(gpd.points_from_xy([p[0] for p in pair], [p[1] for p in pair]), crs=4326)
    comparisons = {c: float(straight.to_crs(c).iloc[0].distance(straight.to_crs(c).iloc[1]) / 1000)
                   for c in (DISTANCE, 'EPSG:3347', 'EPSG:3857')}
    comparisons['ellipsoidal WGS 84'] = geod.inv(*pair[0], *pair[1])[2] / 1000
    output = {'areas': len(das), 'population': province_count, 'unjoined': int(das.population.isna().sum()),
              'unmatched_uids': das.loc[das.population.isna(), 'uid'].tolist(),
              'outline_area_km2': outline.to_crs(AREA).area.iloc[0] / 1e6, 'repairs': repair_count,
              'points': points.point_method.value_counts().to_dict(), 'source_layer': layer,
              'input_crs': areas.crs.to_string(), 'distances_km': comparisons}
    summarize('shared', output)
    print(output)


if __name__ == '__main__':
    prepare()

