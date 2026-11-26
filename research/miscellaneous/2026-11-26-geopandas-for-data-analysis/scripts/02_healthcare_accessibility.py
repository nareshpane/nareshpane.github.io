"""Compute cross-vintage facility proximity, catchments and candidate scores."""
import geopandas as gpd
import numpy as np
import pandas as pd
from scipy.spatial.distance import cdist
import shapely

from common import ROOT, DISTANCE, clean, datum, json_write, positive_parts, read, save, summarize, web, weighted_quantile

THRESHOLDS = [10, 25, 50, 100]


def xy(gdf):
    return np.column_stack([gdf.geometry.x, gdf.geometry.y])


def facilities():
    root = datum('healthcare', 'raw', 'facilities.geojson').parent
    v2 = next((p for p in sorted(root.iterdir())
               if 'v2' in p.name.lower() and p.suffix.lower() in {'.gpkg', '.zip', '.parquet', '.geojson'}), None)
    if v2:
        if v2.suffix == '.zip':
            import importlib
            files = importlib.import_module('00_download_data').extract(v2, root / 'v2',
                    lambda name: name.lower().endswith(('.gpkg', '.parquet', '.geojson')))
            v2 = next(p for p in files if 'metadata' not in p.name.lower())
        data = gpd.read_parquet(v2) if v2.suffix == '.parquet' else gpd.read_file(v2)
        data.columns = data.columns.str.lower()
        frame = data.loc[data.prov_terr.eq('AB')].rename(columns={
            'name_en': 'name', 'facility_name': 'name', 'facility_type': 'type'})
        version = 'ODHF version 2.0 (September 2026 release)'
    else:
        data = gpd.read_file(root / 'hospital_2020.geojson')
        frame = gpd.GeoDataFrame(data.rename(columns={'name_nom': 'name', 'facility_type': 'type'}),
                                 geometry=gpd.points_from_xy(data.longitude, data.latitude, crs=4326))
        version = 'Official ODHF v1.1 map service (2020 release)'
    frame, repairs = clean(frame)
    types = frame['type'].astype(str).str.casefold().str.strip()
    frame['group'] = np.select([types.isin(['hospital', 'hospitals']),
                                types.isin(['clinic', 'clinics']),
                                types.eq('ambulatory health care services')],
                               ['hospital', 'clinic', 'ambulatory'], default='other')
    frame = frame.loc[frame.group.ne('other')].to_crs(DISTANCE)
    polygon = frame.geom_type.isin(['Polygon', 'MultiPolygon'])
    frame['modelled_as'] = 'Published point'
    frame.loc[polygon, 'geometry'] = frame.loc[polygon].representative_point()
    frame.loc[polygon, 'modelled_as'] = 'Interior locator from published facility polygon'
    frame = frame.loc[frame.geom_type.eq('Point')]
    frame = gpd.clip(frame, read('shared', 'alberta'), keep_geom_type=True)
    frame = frame[['name', 'group', 'modelled_as', 'geometry']].copy()
    historical = root / 'hospital_2020.geojson'
    if v2 and historical.exists():
        old = gpd.read_file(historical)
        history = gpd.GeoDataFrame({'name': old.name_nom, 'group': 'hospital_v11',
            'modelled_as': 'Published point'}, geometry=gpd.points_from_xy(old.longitude, old.latitude),
            crs=4326).to_crs(DISTANCE)
        history, old_repairs = clean(history)
        repairs += old_repairs
        history = gpd.clip(history, read('shared', 'alberta'))
        frame = gpd.GeoDataFrame(pd.concat([frame, history], ignore_index=True), crs=DISTANCE)
    before = len(frame)
    frame = frame.sort_values('name').drop_duplicates(['group', 'geometry']).reset_index(drop=True)
    return frame, version, before - len(frame), repairs


def metrics(distance, weight):
    weight = np.asarray(weight)
    return {'mean_km': float(np.average(distance, weights=weight)),
            'median_km': float(weighted_quantile(distance, weight, .5)),
            'p90_km': float(weighted_quantile(distance, weight, .9)),
            'p95_km': float(weighted_quantile(distance, weight, .95)),
            'population': float(weight.sum()),
            'beyond': {r: float(weight[distance > r].sum()) for r in THRESHOLDS}}


def placement(base, candidate_distances, weights):
    """Exhaustive single-addition objective over the supplied candidate set."""
    new = np.minimum(base[:, None], candidate_distances)
    scores = weights @ new / weights.sum()
    return new, scores


def web_distances(distance):
    """Retain full precision wherever kilometre rounding could change coverage."""
    rounded = np.round(distance, 3)
    crosses = np.isin(rounded, THRESHOLDS) & (distance > rounded)
    return np.where(crosses, distance, rounded).tolist()


def refresh_web_scenarios():
    """Refresh candidate markers and distances from the existing projected build."""
    import json

    points = read('healthcare', 'demand_distances')
    candidates = read('healthcare', 'candidates')
    folder = ROOT / 'maps' / 'healthcare'
    path = folder / 'model.json'
    model = json.loads(path.read_text(encoding='utf-8'))
    assert model['uid'] == points.uid.tolist()
    assert model['candidates'] == candidates[['uid', 'label']].to_dict('records')
    old_base = model['base'].copy()
    for group in old_base:
        model['base'][group] = web_distances(points[group + '_km'].to_numpy())
    if model['base'] != old_base:
        json_write(path, model)
    for i, column in enumerate((cdist(xy(points), xy(candidates)) / 1000).T):
        path = folder / f'candidate_{i}.json'
        new = web_distances(column)
        if json.loads(path.read_text(encoding='utf-8')) != new:
            json_write(path, new)
    export_candidate_markers(candidates)


def export_candidate_markers(candidates):
    # A census population centre code can identify multiple mapped parts.
    web(candidates.assign(candidate_index=range(len(candidates))), 'healthcare', 'candidates')


def run():
    points = read('shared', 'demand').sort_values('uid').reset_index(drop=True)
    points = points.loc[points.population.notna()].reset_index(drop=True)
    attributes = datum('socioeconomic', 'processed', 'census_indicators.csv')
    if attributes.exists() and 'age65' in pd.read_csv(attributes, nrows=0):
        age = pd.read_csv(attributes, dtype={'uid': str}, usecols=['uid', 'age65'])
        points = points.merge(age, on='uid', how='left', validate='1:1')
    else:
        points['age65'] = np.nan
    points['division'] = points.uid.str[:4]
    sites, version, duplicates, repairs = facilities()
    divisions = read('shared', 'divisions')[['uid', 'geometry']].rename(columns={'uid': 'division'})
    sites = gpd.sjoin(sites, divisions, how='left', predicate='intersects').sort_values('index_right')
    sites = sites.loc[~sites.index.duplicated()].drop(columns='index_right').sort_index()
    outline = read('shared', 'alberta')
    centre_path = datum('shared', 'processed', 'centres.parquet')
    if centre_path.exists():
        centres = read('shared', 'centres')
        column = next((c for c in centres if c.lower().endswith('name')), 'uid')
        candidates = centres[['uid', column, 'geometry']].copy() if column != 'uid' else centres[['uid', 'geometry']].assign(label=centres.uid)
        candidates = candidates.rename(columns={column: 'label'}) if column != 'uid' else candidates
        candidates.geometry = candidates.representative_point()
        candidate_method = 'Interior points of 2021 population centres'
    else:
        candidates = points.sort_values('population', ascending=False).groupby('division').head(2).copy()
        candidates['label'] = 'DA ' + candidates.uid
        candidates = candidates[['uid', 'label', 'geometry']]
        candidate_method = 'Two most populous joined DAs per census division'
    candidates = candidates.sort_values('label').reset_index(drop=True)
    candidate_distance = cdist(xy(points), xy(candidates)) / 1000
    output = {'source': version, 'duplicates_removed': duplicates, 'repairs': repairs,
              'polygon_locators': int(sites.modelled_as.ne('Published point').sum()),
              'candidate_method': candidate_method, 'candidates': len(candidates), 'groups': {}}
    model = {'uid': points.uid.tolist(), 'division': points.division.tolist(),
             'population': points.population.astype(int).tolist(),
             'age65': [None if pd.isna(n) else int(n) for n in points.age65], 'base': {},
             'candidates': candidates[['uid', 'label']].to_dict('records')}
    for group, listed in sites.groupby('group'):
        listed = listed.reset_index(drop=True)
        nearest = gpd.sjoin_nearest(points, listed, distance_col='metres')
        nearest = nearest.sort_values('index_right').loc[lambda f: ~f.index.duplicated()].sort_index()
        base = nearest.metres.to_numpy() / 1000
        points[group + '_km'] = base
        results = metrics(base, points.population)
        expanded, scores = placement(base, candidate_distance, points.population.to_numpy())
        best = int(np.argmin(scores))
        results.update(count=len(listed), best=best, best_name=str(candidates.iloc[best].label),
                       best_mean_km=float(scores[best]), gain_km=float(results['mean_km'] - scores[best]))
        coverage = {}
        for radius in THRESHOLDS:
            served = points.population.to_numpy() @ (expanded <= radius)
            best_cover = int(np.argmax(served))
            coverage[radius] = {'candidate': best_cover, 'residents': int(served[best_cover])}
        results['maximum_cover'] = coverage
        age_valid = points.age65.notna()
        if points.age65.fillna(0).sum() > 0:
            results['age65'] = metrics(base[age_valid], points.loc[age_valid, 'age65'])
        results['worst_divisions'] = [
            {'division': key, 'mean_km': metrics(points.loc[ix, group + '_km'], points.loc[ix, 'population'])['mean_km']}
            for key, ix in points.groupby('division').groups.items() if points.loc[ix, 'population'].sum() > 0]
        results['worst_divisions'].sort(key=lambda r: -r['mean_km'])
        results['worst_divisions'] = results['worst_divisions'][:5]
        output['groups'][group] = results
        model['base'][group] = web_distances(base)
        web(listed, 'healthcare', group)
        ordered = shapely.voronoi_polygons(shapely.MultiPoint(list(listed.geometry)),
                                           extend_to=outline.geometry.iloc[0].envelope, ordered=True)
        if len(ordered.geoms) == len(listed):
            partition = listed.set_geometry(gpd.GeoSeries(list(ordered.geoms), crs=DISTANCE))
            partition = positive_parts(gpd.clip(partition, outline), 'Polygon')
            save(partition, 'healthcare', 'voronoi_' + group)
            web(partition, 'healthcare', 'voronoi_' + group, 180)
        for radius in THRESHOLDS:
            buffer = listed.buffer(radius * 1000).union_all()
            buffered = gpd.GeoDataFrame({'radius_km': [radius]}, geometry=[buffer], crs=DISTANCE)
            web(gpd.clip(buffered, outline), 'healthcare', f'buffer_{group}_{radius}', 185)
    if 'hospital' not in output['groups']:
        raise ValueError('No hospital class in acquired source')
    for i, col in enumerate(candidate_distance.T):
        json_write(ROOT / 'maps' / 'healthcare' / f'candidate_{i}.json', web_distances(col))
    save(sites, 'healthcare', 'facilities')
    save(candidates, 'healthcare', 'candidates')
    save(points, 'healthcare', 'demand_distances')
    points.drop(columns='geometry').to_csv(datum('healthcare', 'processed', 'distances.csv'), index=False)
    export_candidate_markers(candidates)
    json_write(ROOT / 'maps' / 'healthcare' / 'model.json', model)
    summarize('healthcare', output)
    print(output)


if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument('--web-scenarios-only', action='store_true')
    if parser.parse_args().web_scenarios_only:
        refresh_web_scenarios()
    else:
        run()

