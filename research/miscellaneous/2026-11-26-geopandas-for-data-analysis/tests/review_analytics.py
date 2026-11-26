"""Reconcile the published atlas with its cached inputs and analytical geometries."""
from collections import Counter
import importlib
import json
from pathlib import Path
import sys
import warnings

import geopandas as gpd
from libpysal.weights import Queen, Rook
import numpy as np
import pandas as pd
from pyproj import CRS
from scipy.ndimage import gaussian_filter
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import connected_components
from scipy.spatial import cKDTree
from scipy.spatial.distance import cdist
from scipy.stats import false_discovery_control, spearmanr
import shapely

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'scripts'))
from common import AREA, CALGARY, DISTANCE, datum, read


def load(path):
    return json.loads((ROOT / path).read_text(encoding='utf-8'))


def xy(geo):
    return np.column_stack((geo.geometry.x, geo.geometry.y))


def web_geometry(path):
    frame = gpd.read_file(ROOT / path)
    assert frame.crs.to_epsg() == 4326
    xmin, ymin, xmax, ymax = frame.total_bounds
    assert -122 < xmin < xmax < -109 and 48 < ymin < ymax < 61
    return frame


def healthcare():
    model_folder = Path('maps/healthcare')
    result = load('results/healthcare.json')
    model = load(model_folder / 'model.json')
    sites = read('healthcare', 'facilities')
    candidates = read('healthcare', 'candidates')
    points = read('healthcare', 'demand_distances')
    demand = read('shared', 'demand').set_index('uid')
    assert demand.index.is_unique and points.uid.is_unique
    assert all(frame.crs.to_epsg() == 3400 for frame in (points, sites, candidates))
    assert model['uid'] == points.uid.tolist() == sorted(demand.index[demand.population.notna()])
    assert model['candidates'] == candidates[['uid', 'label']].to_dict('records')
    np.testing.assert_array_equal(points.population, demand.loc[model['uid'], 'population'])
    np.testing.assert_array_equal(model['population'], points.population)
    census = read('socioeconomic', 'census').set_index('uid')
    assert census.index.is_unique
    np.testing.assert_array_equal(points.age65, census.loc[model['uid'], 'age65'].to_numpy())
    assert model['age65'] == [None if pd.isna(x) else int(x) for x in points.age65]
    assert sites[['group', 'geometry']].duplicated().sum() == 0
    assert set(sites.group) == set(result['groups']) == {'hospital', 'hospital_v11', 'ambulatory', 'clinic'}

    raw = gpd.read_file(datum('healthcare', 'raw', 'v2/odhf_v2.gpkg'))
    raw.columns = raw.columns.str.lower()
    types = raw['facility_type' if 'facility_type' in raw else 'type'].str.strip().str.casefold()
    classes = {'hospital': ('hospital', 'hospitals'), 'clinic': ('clinic', 'clinics'),
               'ambulatory': ('ambulatory health care services',)}
    alberta = read('shared', 'alberta').to_crs(DISTANCE).geometry.iloc[0]
    historical = gpd.read_file(datum('healthcare', 'raw', 'hospital_2020.geojson'))
    recorded = gpd.GeoSeries(gpd.points_from_xy(historical.longitude, historical.latitude), crs=4326).to_crs(DISTANCE)
    assert set(recorded[recorded.intersects(alberta)]) == set(sites.loc[sites.group.eq('hospital_v11')].geometry)
    for group, aliases in classes.items():
        selected = raw.loc[raw.prov_terr.eq('AB') & types.isin(aliases)].to_crs(DISTANCE)
        geometry = shapely.make_valid(selected.geometry.array)
        is_area = np.isin(shapely.get_type_id(geometry), (3, 6))
        geometry[is_area] = shapely.point_on_surface(geometry[is_area])
        geometry = geometry[(shapely.get_type_id(geometry) == 0) & shapely.intersects(geometry, alberta)]
        assert set(geometry) == set(sites.loc[sites.group.eq(group)].geometry)
    exported = web_geometry(model_folder / 'hospital.geojson')
    np.testing.assert_allclose(xy(exported), xy(sites.loc[sites.group.eq('hospital')].to_crs(4326)), atol=5.01e-6, rtol=0)
    markers = web_geometry(model_folder / 'candidates.geojson').set_index('candidate_index').sort_index()
    assert markers.index.tolist() == list(range(len(candidates)))
    assert markers[['uid', 'label']].to_dict('records') == model['candidates']
    np.testing.assert_allclose(xy(markers), xy(candidates.to_crs(4326)), atol=5.01e-6, rtol=0)

    distances = cdist(xy(points), xy(candidates)) / 1000
    web = np.column_stack([load(model_folder / f'candidate_{i}.json') for i in range(len(candidates))])
    np.testing.assert_allclose(distances, web, atol=.00050001, rtol=0)
    pop = points.population.to_numpy()
    cutoffs = (10, 25, 50, 100)
    for cutoff in cutoffs:
        np.testing.assert_array_equal(distances <= cutoff, web <= cutoff)
    ties = {}
    for group, rows in sites.groupby('group'):
        all_distances = cdist(xy(points), xy(rows)) / 1000
        base = all_distances.min(axis=1)
        ties[group] = int(((all_distances == base[:, None]).sum(axis=1) > 1).sum())
        np.testing.assert_allclose(points[group + '_km'], base, atol=1e-10)
        np.testing.assert_allclose(model['base'][group], base, atol=.00050001, rtol=0)
        output = result['groups'][group]
        assert output['count'] == len(rows)
        for weights, values, stats in ((pop, base, output),
                                       (points.age65.dropna().to_numpy(), base[points.age65.notna()], output['age65'])):
            order = np.argsort(values)
            cumulative = np.cumsum(weights[order])
            assert weights.sum() == stats['population']
            np.testing.assert_allclose(np.average(values, weights=weights), stats['mean_km'])
            for q, key in ((.5, 'median_km'), (.9, 'p90_km'), (.95, 'p95_km')):
                np.testing.assert_allclose(values[order[np.searchsorted(cumulative, q * cumulative[-1])]], stats[key])
            for cutoff in cutoffs:
                assert weights[values > cutoff].sum() == stats['beyond'][str(cutoff)]
                np.testing.assert_array_equal(base <= cutoff, np.asarray(model['base'][group]) <= cutoff)
        expanded = np.minimum(base[:, None], distances)
        cost = np.average(expanded, axis=0, weights=pop)
        assert cost.argmin() == output['best']
        np.testing.assert_allclose(cost.min(), output['best_mean_km'])
        for cutoff in cutoffs:
            covered = pop @ (expanded <= cutoff)
            peak = output['maximum_cover'][str(cutoff)]
            assert peak == {'candidate': int(covered.argmax()), 'residents': int(covered.max())}
    hospital = result['groups']['hospital']
    population = int(hospital['population'])
    covered = {str(r): population - int(hospital['beyond'][str(r)]) for r in cutoffs}
    return {'demand_points': len(points), 'missing_population_in_demand_input': int(demand.population.isna().sum()),
            'demand_without_age_65_weight': int(points.age65.isna().sum()), 'population_denominator': population,
            'exact_distance_ties_by_category': ties, 'facility_categories': sites.group.value_counts().to_dict(),
            'candidates': len(candidates), 'candidate_groups_with_multiple_parts': candidates.loc[
                candidates.uid.duplicated(keep=False), 'label'].unique().tolist(),
            'distances_checked': distances.size, 'hospital_coverage': covered,
            'additional_residents_10_to_25km': covered['25'] - covered['10'],
            'coverage_change_percentage_points': 100 * (covered['25'] - covered['10']) / population}


def wildfire():
    result = load('results/wildfire.json')
    source = datum('wildfire', 'raw', 'fp-historical-wildfire-data-2006-2025.csv')
    records = pd.read_csv(source, usecols=['YEAR', 'FIRE_NUMBER']).dropna().drop_duplicates()
    counts = records.groupby('YEAR').size().to_dict()
    assert result['recorded_incidents'] == len(records)
    annual_csv = pd.read_csv(datum('wildfire', 'processed', 'annual.csv')).set_index('year')
    grid = read('wildfire', 'grid_50')
    full_area = np.sqrt(3) / 2 * 50**2
    boundary = grid.km2 < full_area - 1e-6
    samples = grid.loc[[grid.loc[~boundary, 'unique'].idxmax(), grid.loc[boundary, 'unique'].idxmax()]]
    annual_samples = []
    burn_codes = Counter()
    areas = {}
    for layer, year in result['layer_years'].items():
        attributes = gpd.read_file(source.parent / f'{layer}.geojson', columns=['YEAR', 'BURNCODE'],
                                   ignore_geometry=True)
        assert set(attributes.YEAR.astype(int)) == {year}
        burn_codes.update(attributes.BURNCODE.fillna('missing'))
        annual = read('wildfire', f'footprint_{year}')
        assert annual.crs == CRS(AREA) and len(annual) == 1 and annual.is_valid.all()
        areas[year] = annual.area.iloc[0] / 1e6
        parts = shapely.get_parts(annual.geometry.iloc[0])
        tree = shapely.STRtree(parts)
        intersections = [shapely.union_all(shapely.intersection(cell, parts[tree.query(cell)]))
                         for cell in samples.geometry]
        np.testing.assert_allclose(shapely.area(intersections) / samples.area, samples[f'y{year}'],
                                   atol=.000005001, rtol=0)
        annual_samples.append(intersections)
    assert burn_codes == result['burn_classes'] and sum(burn_codes.values()) == result['raw_features']
    for row in result['annual']:
        year = row['year']
        assert counts[year] == row['incidents'] == annual_csv.loc[year, 'incidents']
        if year not in areas:
            assert row['km2'] is None and pd.isna(annual_csv.loc[year, 'km2'])
        else:
            np.testing.assert_allclose([row['km2'], annual_csv.loc[year, 'km2']], areas[year])
    assert sorted(areas) == result['years'] == list(range(2016, 2026))
    total = result['cumulative_km2']
    unique = result['unique_km2']
    np.testing.assert_allclose(sum(areas.values()), total)
    assert max(areas.values()) <= unique <= total
    sample_period_unions = shapely.union_all(annual_samples, axis=0)
    np.testing.assert_allclose(shapely.area(sample_period_unions) / samples.area, samples['unique'], atol=1e-8)

    outline = read('shared', 'alberta').to_crs(AREA)
    province = outline.area.sum()
    verified_grids = {}
    for size in (25, 50):
        cells = read('wildfire', f'grid_{size}')
        assert cells.crs == CRS(AREA) and cells.cell.is_unique and cells.is_valid.all()
        nominal = np.sqrt(3) / 2 * (size * 1000)**2
        cell_area = cells.area.to_numpy()
        np.testing.assert_allclose(cells.km2 * 1e6, cell_area)
        assert (cell_area > 1e6).all() and (cell_area <= nominal + .001).all()
        # Retained cells are disjoint partitions. Dropped boundary slivers bound any
        # difference between their reconstructed union area and the full-province value.
        omitted_km2 = (province - cell_area.sum()) / 1e6
        measured = cells['unique'] @ cell_area / 1e6
        assert -.001 <= unique - measured <= omitted_km2 + .001
        years = cells[[f'y{y}' for y in sorted(areas)]].to_numpy()
        assert ((years >= 0) & (years <= 1)).all() and cells['unique'].between(0, 1).all()
        assert (years <= cells['unique'].to_numpy()[:, None] + .00000501).all()
        row = result['grid'][str(size)]
        assert len(cells) == row['cells']
        assert np.histogram(cells['unique'], bins=[0, .001, .01, .05, .20, .50, 1])[0].tolist() == row['unique_distribution']
        for threshold in (0, 1, 5):
            t = threshold / 100
            frequency = cells[f'frequency_{threshold}'].to_numpy()
            assert ((years > t + .00000501).mean(axis=1) <= frequency).all()
            assert ((years > t - .00000501).mean(axis=1) >= frequency).all()
            assert np.count_nonzero(frequency >= .2) == row['recurrent'][str(threshold)]
        web = web_geometry(Path('maps/wildfire') / f'grid_{size}.geojson').set_index('cell')
        np.testing.assert_allclose(web['unique'], cells.set_index('cell').loc[web.index, 'unique'])
        verified_grids[size] = {'cells': len(cells), 'clipped_cells': int((cell_area < nominal - .001).sum()),
                                'union_km2_in_retained_cells': measured, 'omitted_km2': omitted_km2}

    ignitions = read('wildfire', 'ignitions')
    assert ignitions.crs == CRS(AREA) and len(ignitions) == result['ignitions_used']
    assert not ignitions.duplicated(['YEAR', 'FIRE_NUMBER']).any()
    x0, y0, x1, y1 = outline.total_bounds
    step = 10000
    histogram = np.histogram2d(*xy(ignitions).T, bins=(np.arange(x0, x1 + 2 * step, step),
                                                     np.arange(y0, y1 + 2 * step, step)))[0]
    assert histogram.sum() == len(ignitions)
    kde = web_geometry('maps/wildfire/intensity.geojson').to_crs(AREA)
    positions = xy(kde.set_geometry(kde.representative_point()))
    indices = np.floor((positions - [x0, y0]) / step).astype(int)
    for h in (20, 50, 100):
        expected = gaussian_filter(histogram, h / 10, mode='constant') / 100
        np.testing.assert_allclose(kde[f'h{h}'], expected[tuple(indices.T)], atol=.0000801, rtol=0)
    observed_counts = {y: counts[y] for y in areas}
    return {'incident_years': sorted(counts), 'verified_source_perimeter_years': sorted(areas),
            'most_records_with_perimeters': max(observed_counts, key=observed_counts.get),
            'incidents_2016': counts[2016], 'incidents_2023': counts[2023], 'area_2016': areas[2016],
            'area_2023': areas[2023], 'area_ratio_2023_to_2016': areas[2023] / areas[2016],
            'unique_km2': unique, 'cumulative_km2': total, 'grids': verified_grids,
            'annual_and_period_intersection_samples': samples.cell.tolist(),
            'ignitions': len(ignitions), 'checked_intensity_cells': len(kde), 'intensity_units': 'incidents/km²'}


def flood():
    result = load('results/flood.json')
    roads, roofs = (read('flood', item) for item in ('roads', 'buildings'))
    np.testing.assert_array_equal(result['study_bounds'], CALGARY)
    assert roads.crs.to_epsg() == 3400 and roofs.crs == CRS(AREA)
    for item in (roads, roofs):
        assert item.key.is_unique and not pd.Series(shapely.normalize(item.geometry.array)).duplicated().any()
    denominators = (roads.length.sum(), roofs.area.sum())
    np.testing.assert_allclose(denominators, (1000 * result['total_road_km'], result['total_roof_m2']))
    scenarios = {}
    geometries = {}
    for entry in result['scenarios']:
        period = entry['period']
        print(f'  Flood: 1-in-{period} geometry and assets', flush=True)
        event = read('flood', f'inundation_{period}')
        shape = event.geometry.iloc[0]
        assert event.crs == CRS(AREA) and shape.is_valid
        np.testing.assert_allclose(shape.area / 1e6, entry['footprint_km2'])
        geometries[period] = shape
        raw = gpd.read_file(datum('flood', 'raw', f'flood_{period}.geojson'),
                            bbox=CALGARY, columns=['StudyName'])
        assert raw.crs.to_epsg() == 4326
        box = gpd.GeoSeries([shapely.box(*CALGARY)], crs=4326).to_crs(AREA).iloc[0]
        projected = raw.to_crs(AREA).geometry.array
        near_window = shapely.clip_by_rect(projected, *box.bounds)
        all_shapes = shapely.make_valid(near_window)
        clipped = shapely.intersection(all_shapes, box)
        assert sorted(raw.StudyName[shapely.area(clipped) > 0].dropna().unique()) == entry['studies']
        np.testing.assert_allclose(shapely.union_all(clipped).area, shape.area, atol=.005)
        wet = []
        for kind, items, measure, field in (('roads', roads, shapely.length, 'metres'),
                                             ('roofs', roofs, shapely.area, 'square_metres')):
            parts = shapely.get_parts(event.to_crs(items.crs).geometry.iloc[0])
            part_ids, item_ids = items.sindex.query(parts, predicate='intersects')
            hit = items.iloc[item_ids]
            local_bounds = hit.bounds.to_numpy() + [-.01, -.01, .01, .01]
            local_parts = [shapely.clip_by_rect(part, *bounds)
                           for part, bounds in zip(parts[part_ids], local_bounds)]
            invalid = np.flatnonzero(~shapely.is_valid(local_parts))
            for i in invalid:
                local_parts[i] = parts[part_ids[i]]
            quantity = measure(shapely.intersection(hit.geometry.array, local_parts))
            positive = pd.Series(quantity, index=hit.key).groupby(level=0).sum().loc[lambda rows: rows > 1e-9]
            saved = read('flood', f'exposed_{kind}_{period}').set_index('key')
            assert saved.index.is_unique
            assert set(positive.index) == set(saved.index)
            saved = saved.reindex(positive.index)
            total = pd.Series(measure(items.geometry.array), index=items.key).loc[positive.index]
            np.testing.assert_allclose(saved[field], positive, atol=1e-6)
            np.testing.assert_allclose(saved.fraction, positive / total, atol=1e-7)
            assert saved.fraction.between(0, 1).all()
            wet.append(saved)
        np.testing.assert_allclose([wet[0].metres.sum() / 1000, wet[0].metres.sum() / denominators[0] * 100,
                                    wet[1].square_metres.sum()],
                                   [entry['road_km'], entry['road_pct'], entry['roof_m2']])
        assert [len(w) for w in wet] == [entry['roads_with_exposure'], entry['roofs']]
        for cutoff in (0, .25, .5, .75):
            threshold = next(x for x in result['thresholds']
                             if x['period'] == period and x['minimum_fraction'] == cutoff)
            road = wet[0].loc[wet[0].fraction.ge(cutoff)]
            roof = wet[1].loc[wet[1].fraction.ge(cutoff)]
            np.testing.assert_allclose([road.metres.sum() / 1000, len(roof), roof.square_metres.sum()],
                                       [threshold['road_km'], threshold['roofs'], threshold['roof_m2']])
        scenarios[period] = entry
    web_geometry('maps/flood/event_50.geojson')
    left, right = scenarios[20], scenarios[50]
    assert left['studies'] == right['studies'] and len(left['studies']) == 1 and len(scenarios[100]['studies']) == 2
    return {'scenarios_checked': sorted(scenarios), 'distinct_roads': len(roads), 'distinct_roofs': len(roofs),
            'road_denominator_km': denominators[0] / 1000, 'roof_denominator_m2': denominators[1],
            'common_study_20_and_50': left['studies'],
            'studies_100': scenarios[100]['studies'],
            'area_ratio_50_to_20': right['footprint_km2'] / left['footprint_km2'],
            'road_length_ratio_50_to_20': right['road_km'] / left['road_km'],
            '20_year_outside_50_year_m2': geometries[20].difference(geometries[50]).area}


def socioeconomic():
    result = load('results/socioeconomic.json')
    source_fields = {'age_total': 8, 'age65': 24, 'income': 243, 'low_income': 345, 'labour': 2224,
                     'unemployed': 2226, 'unemployment': 2230, 'housing': 1467, 'degree': 2008}
    columns = {f'count_total_{key}': name for name, key in source_fields.items()}
    source = pd.read_parquet(datum('socioeconomic', 'raw', 'census_profile.parquet'),
                             columns=['da_dguid', *columns]).rename(columns=columns)
    source = source.loc[source.da_dguid.str.fullmatch(r'2021S051248\d{6}', na=False)].copy()
    source['uid'] = source.da_dguid.str[-8:]
    source = source.set_index('uid')
    assert source.index.is_unique
    for key in source_fields:
        clean = pd.to_numeric(source[key], errors='coerce')
        source[key] = clean.where(clean.ge(0))
    da = read('shared', 'das').set_index('uid')
    census = read('socioeconomic', 'census').set_index('uid')
    assert da.index.is_unique and census.index.is_unique
    assert set(da.index) == set(census.index)
    np.testing.assert_allclose(da.loc[census.index, 'density'], census.density)
    joined = source.reindex(census.index)
    np.testing.assert_allclose(census.age_share, 100 * joined.age65 / joined.age_total.replace(0, np.nan))
    for key in source_fields.keys() - {'age_total'}:
        np.testing.assert_allclose(census[key], joined[key])
    divisions = census.dropna(subset=['labour', 'unemployed']).groupby('division')[['labour', 'unemployed']].sum()
    for row in result['division_unemployment']:
        labour, unemployed = divisions.loc[row['division']]
        np.testing.assert_allclose([labour, unemployed, unemployed / labour * 100],
                                   [row['labour'], row['unemployed'], row['rate']])
    folder = Path('maps/socioeconomic')
    values = load(folder / 'values.json')
    permutations = result['permutations']
    seed = result['seed']
    checked = {}
    for indicator, info in result['indicators'].items():
        rows = census.loc[census[indicator].notna()]
        assert len(rows) == info['n'] and len(census) - len(rows) == info['missing']
        assert list(values[indicator]) == rows.index.tolist()
        x = rows[indicator].to_numpy()
        np.testing.assert_allclose(x, list(values[indicator].values()))
        if indicator == 'density':
            x = np.log1p(x)
        z = (x - x.mean()) / x.std()
        n = len(rows)
        expected = -1 / (n - 1)
        for mode, summary in info['weights'].items():
            exported = load(folder / f'{indicator}_{mode}.json')
            assert list(exported) == rows.index.tolist()
            links = [row[6] for row in exported.values()]
            degree = np.array([len(link) for link in links])
            source_id = np.repeat(np.arange(n), degree)
            target_id = rows.index.get_indexer([target for link in links for target in link])
            assert (target_id >= 0).all() and (source_id != target_id).all()
            assert all(len(link) == len(set(link)) for link in links)
            if mode in ('queen', 'rook'):
                with warnings.catch_warnings():
                    warnings.simplefilter('ignore', UserWarning)
                    weights = (Queen if mode == 'queen' else Rook).from_dataframe(rows, use_index=False)
                assert all(set(links[i]) == set(rows.index[weights.neighbors[i]]) for i in range(n))
            else:
                assert rows.crs.to_epsg() == 3400
                positions = xy(rows.set_geometry(rows.representative_point()))
                edge = np.linalg.norm(positions[source_id] - positions[target_id], axis=1).reshape(n, 8)
                distance = cKDTree(positions).query(positions, k=9)[0][:, -1]
                np.testing.assert_allclose(edge.max(axis=1), distance, atol=1e-9)
                assert (degree == 8).all()
            w = csr_matrix((1 / degree[source_id], (source_id, target_id)), shape=(n, n))
            np.testing.assert_allclose(np.ravel(w.sum(axis=1)), degree > 0)
            assert summary['islands'] == np.count_nonzero(degree == 0)
            assert summary['components'] == connected_components(w, directed=False, return_labels=False)
            lag = w @ z
            local = z * lag
            scale = n / w.sum() / np.dot(z, z)
            global_i = scale * np.dot(z, lag)
            np.testing.assert_allclose([global_i, expected], [summary['I'], summary['expected']], atol=1e-12)
            rng = np.random.default_rng(seed)
            simulated = np.array([scale * np.dot(v, w @ v) for v in (rng.permutation(z) for _ in range(permutations))])
            tail = (1 + np.count_nonzero(np.abs(simulated - expected) >= abs(global_i - expected))) / (permutations + 1)
            np.testing.assert_allclose(tail, summary['p'])
            numeric = np.array([row[:5] for row in exported.values()])
            np.testing.assert_allclose(numeric[:, :3], np.column_stack((z, lag, local)), atol=.000050001, rtol=0)
            p, q = numeric[:, 3:].T
            assert ((p >= 2 / (permutations + 1)) & (p <= 1)).all() and (p[degree == 0] == 1).all()
            np.testing.assert_allclose(p * (permutations + 1) / 2, np.round(p * (permutations + 1) / 2))
            adjusted = false_discovery_control(p)
            np.testing.assert_allclose(q, adjusted, atol=.000005001, rtol=0)
            assert np.count_nonzero(p <= .05) == summary['raw_significant']
            assert np.count_nonzero(q <= .05) == np.count_nonzero(adjusted <= .05) == summary['fdr_significant']
            checked[f'{indicator}/{mode}'] = {'I': global_i, 'p': tail, 'islands': summary['islands']}

    # Replay one complete conditional local test against the exact input and seed.
    stats = importlib.import_module('05_socioeconomic_geography')
    unemployment = census.loc[census.unemployment.notna()]
    weights = stats.spatial_weights(unemployment.reset_index(), 'queen')
    _, _, _, _, p, _, quadrants = stats.permutation(unemployment.unemployment.to_numpy(), weights)
    local = load(folder / 'unemployment_queen.json')
    np.testing.assert_allclose(p, [row[3] for row in local.values()])
    np.testing.assert_array_equal(quadrants, [row[5] for row in local.values()])
    elevated = unemployment.unemployment.ge(15)
    high_high = sum(elevated[uid] and row[4] <= .05 and row[5] == 'High–high' for uid, row in local.items())
    components = result['composite']
    complete = census[components].dropna().astype(float)
    assert complete.index.tolist() == list(values['components'])
    oriented = (complete - complete.mean()) / complete.std(ddof=0)
    oriented['income'] *= -1
    np.testing.assert_allclose(oriented, list(values['components'].values()))
    rho = spearmanr(oriented.mean(axis=1), oriented.iloc[:, 0]).statistic
    np.testing.assert_allclose(rho, result['composite_sensitivity']['rank_correlation'])
    return {'census_areas': len(census), 'indicator_weight_models': checked, 'permutations': permutations,
            'local_inference_replayed': 'unemployment/queen', 'unemployment_reported': len(unemployment),
            'unemployment_at_least_15_pct': int(elevated.sum()), 'adjusted_high_high_among_these': high_high,
            'composite_complete_cases': len(complete), 'composite_rank_correlation': rho}


def main():
    distance = CRS(DISTANCE)
    area = CRS(AREA)
    assert distance.to_epsg() == 3400 and distance.axis_info[0].unit_name == 'metre'
    assert area.coordinate_operation.method_name == 'Albers Equal Area' and area.axis_info[0].unit_name == 'metre'
    assert area.datum.name == 'North American Datum 1983'
    assert read('shared', 'alberta').crs == distance
    web_geometry('maps/shared/outline.geojson')
    report = {'crs': {'distance': distance.name, 'area': AREA, 'web_vectors': 'EPSG:4326'},
              'corrections': [{'scope': 'Healthcare candidate coverage in the browser',
                             'changed_exports': ['maps/healthcare/candidate_16.json', 'maps/healthcare/candidate_36.json'],
                             'cause': 'Rounding to three decimal places moved two distances above a cutoff onto it.',
                             'cases': [{'candidate': 'Calgary', 'DA': '48060294', 'cutoff_km': 10, 'residents': 352},
                                       {'candidate': 'Drumheller', 'DA': '48060070', 'cutoff_km': 100, 'residents': 384}]},
                            {'scope': 'Healthcare candidate markers in the browser',
                             'changed_export': 'maps/healthcare/candidates.geojson',
                             'cause': 'Both Lloydminster candidates share the population-centre code 0478.',
                             'fix': 'Use one marker per candidate index and distinguish mapped parts by label.'}],
              'limits': ['Existing cached public sources were used. Source vintage, completeness and field meaning follow their published documentation.',
                         'Wildfire proportions were reconciled with detailed annual footprints and two independent grid resolutions. Raw perimeter geometries were not unioned again from scratch.',
                         'Local Monte Carlo inference was fully replayed for queen unemployment. Other local models were checked for their computed statistics, p-value support and BH adjustment.']}
    for check in (healthcare, flood, socioeconomic, wildfire):
        print(f'Reviewing {check.__name__}', flush=True)
        report[check.__name__] = check()
    path = ROOT / 'metadata' / 'analytical_review.json'
    path.write_text(json.dumps(report, indent=2, ensure_ascii=False, default=lambda v: v.item()) + '\n', encoding='utf-8')
    print(f'Wrote {path.name}', flush=True)


if __name__ == '__main__':
    main()
