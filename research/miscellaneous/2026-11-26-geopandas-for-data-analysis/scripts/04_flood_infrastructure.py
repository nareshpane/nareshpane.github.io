"""Measure mapped Calgary road and roof intersections with flood scenarios."""
import hashlib

import geopandas as gpd
import numpy as np
import pandas as pd
from shapely import box, normalize

from common import AREA, CALGARY, DISTANCE, clean, datum, positive_parts, save, summarize, web


def shape_key(geometry):
    return hashlib.sha1(normalize(geometry).wkb, usedforsecurity=False).hexdigest()


def load_many(paths, crs, kind, extent):
    frames, repairs = [], 0
    for path in paths:
        frame, fixed = clean(gpd.read_file(path))
        frames.append(frame.to_crs(crs))
        repairs += fixed
    frame = gpd.GeoDataFrame(pd.concat(frames, ignore_index=True), crs=crs)
    frame = positive_parts(gpd.clip(frame, extent), kind)
    frame['key'] = frame.geometry.map(shape_key)
    before = len(frame)
    frame = frame.drop_duplicates('key').reset_index(drop=True)
    return frame, repairs, before - len(frame)


def intersect(frame, footprint, kind, field):
    x0, y0, x1, y1 = footprint.bounds
    x, y = np.meshgrid(np.arange(x0, x1, 750), np.arange(y0, y1, 750))
    tiles = gpd.GeoDataFrame(geometry=box(x.ravel(), y.ravel(), x.ravel() + 750, y.ravel() + 750),
                             crs=frame.crs)
    tiles.geometry = tiles.intersection(footprint)
    tiles = positive_parts(tiles, 'Polygon')
    assets = frame.copy()
    assets['_asset'] = np.arange(len(assets))
    assets['_denominator'] = assets.length if kind == 'LineString' else assets.area
    candidates = gpd.overlay(assets, tiles, how='intersection', keep_geom_type=True)
    candidates = candidates.dissolve(by='_asset', as_index=False)
    amount = candidates.length if kind == 'LineString' else candidates.area
    candidates[field] = amount
    candidates['fraction'] = amount / candidates.pop('_denominator')
    assert candidates.fraction.between(0, 1 + 1e-8).all()
    candidates['fraction'] = candidates.fraction.clip(0, 1)
    return candidates.drop(columns='_asset')


def run():
    root = datum('flood', 'raw', 'unused').parent
    study = gpd.GeoDataFrame({'name': ['Central Calgary study window']}, geometry=[box(*CALGARY)], crs=4326)
    study_m = study.to_crs(DISTANCE)
    study_a = study.to_crs(AREA)
    road_paths = sorted(root.glob('roads_*.geojson'))
    if len(road_paths) != 5:
        raise ValueError(f'Expected five road rank files and found {len(road_paths)}')
    roads, road_repairs, duplicates = load_many(road_paths, DISTANCE, 'LineString', study_m)
    road_length = roads.length.sum()
    buildings_path = root / 'buildings.geojson'
    buildings, roof_repairs, roof_duplicates = load_many([buildings_path], AREA, 'Polygon', study_a)
    assert len(roads) and len(buildings)
    save(roads, 'flood', 'roads')
    save(buildings, 'flood', 'buildings')
    web(roads[['key', 'geometry']], 'flood', 'roads', 3)
    web(buildings[['geometry']], 'flood', 'buildings', 1, compressed=True)
    web(study, 'flood', 'study')
    coverage_path = root / 'study_coverage.geojson'
    coverage_quality = None
    if coverage_path.exists():
        coverage = gpd.read_file(coverage_path)
        coverage_input = len(coverage)
        coverage = coverage.loc[coverage.geometry.notna()].to_crs(DISTANCE)
        # ArcGIS display generalization can collapse narrow rings. Repair the
        # display layer without altering the detailed scenario geometries.
        invalid = ~coverage.is_valid
        coverage.loc[invalid, 'geometry'] = coverage.loc[invalid].buffer(0)
        coverage = positive_parts(coverage, 'Polygon')
        web(coverage[['geometry']], 'flood', 'coverage', 100, compressed=True)
        coverage_quality = {'source_features': coverage_input, 'repaired': int(invalid.sum()),
                            'display_features': len(coverage)}
    results, book = [], []
    for path in sorted(root.glob('flood_*.geojson'), key=lambda p: int(p.stem.split('_')[-1])):
        period = int(path.stem.split('_')[-1])
        polys, repairs = clean(gpd.read_file(path))
        clipped = positive_parts(gpd.clip(polys.to_crs(AREA), study_a), 'Polygon')
        if clipped.empty:
            continue
        union = clipped.geometry.union_all()
        area = gpd.GeoDataFrame({'period': [period]}, geometry=[union], crs=AREA)
        names = sorted(clipped.StudyName.dropna().unique().tolist())
        impacted_roads = intersect(roads, area.to_crs(DISTANCE).geometry.iloc[0], 'LineString', 'metres')
        impacted_roofs = intersect(buildings, union, 'Polygon', 'square_metres')
        assert 0 <= impacted_roads.metres.sum() <= road_length + 1e-6
        web(area, 'flood', f'event_{period}', 3)
        web(impacted_roads[['metres', 'fraction', 'geometry']], 'flood', f'road_{period}', 2)
        web(impacted_roofs[['square_metres', 'fraction', 'geometry']], 'flood', f'roof_{period}', .5)
        for label, frame in [('inundation', area), ('exposed_roads', impacted_roads), ('exposed_roofs', impacted_roofs)]:
            save(frame, 'flood', f'{label}_{period}')
        results.append({'period': period, 'studies': names, 'polygon_records': len(clipped),
                        'repairs': repairs, 'footprint_km2': union.area / 1e6,
                        'road_km': impacted_roads.metres.sum() / 1000,
                        'road_pct': impacted_roads.metres.sum() / road_length * 100,
                        'roads_with_exposure': len(impacted_roads),
                        'roofs': len(impacted_roofs), 'roof_m2': impacted_roofs.square_metres.sum()})
        for fraction in (0, .25, .50, .75):
            road_values = impacted_roads.loc[impacted_roads.fraction.ge(fraction), 'metres']
            roof_values = impacted_roofs.loc[impacted_roofs.fraction.ge(fraction), 'square_metres']
            book.append({'period': period, 'minimum_fraction': fraction,
                         'road_km': road_values.sum() / 1000, 'roofs': len(roof_values),
                         'roof_m2': roof_values.sum()})
    if not results:
        raise ValueError('No flooded geometry was returned for the study window')
    pd.DataFrame(results).to_csv(datum('flood', 'processed', 'scenario_summary.csv'), index=False)
    pd.DataFrame(book).to_csv(datum('flood', 'processed', 'threshold_summary.csv'), index=False)
    output = {'study_bounds': CALGARY, 'roads': len(roads), 'buildings': len(buildings),
              'total_road_km': road_length / 1000, 'total_roof_m2': buildings.area.sum(),
              'duplicate_roads': duplicates, 'duplicate_roofs': roof_duplicates,
              'road_repairs': road_repairs, 'roof_repairs': roof_repairs,
              'study_coverage': coverage_path.exists(), 'coverage_quality': coverage_quality,
              'scenarios': results, 'thresholds': book}
    summarize('flood', output)
    print({k: v for k, v in output.items() if k != 'thresholds'})


if __name__ == '__main__':
    run()

