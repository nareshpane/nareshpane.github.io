"""Union observed wildfire footprints and quantify spatial and temporal exposure."""
import math

import geopandas as gpd
import numpy as np
import pandas as pd
import shapely
from scipy.ndimage import gaussian_filter

from common import AREA, DISTANCE, clean, datum, positive_parts, read, save, summarize, web


def hexagons(outline, width):
    radius = width / math.sqrt(3)
    x0, y0, x1, y1 = outline.total_bounds
    dx, dy = width, 1.5 * radius
    angles = np.radians(np.arange(30, 390, 60))
    cells = []
    for r in range(math.ceil((y1 - y0) / dy) + 2):
        for c in range(math.ceil((x1 - x0) / dx) + 2):
            x, y = x0 + (c - 1 + (r % 2) / 2) * dx, y0 + (r - 1) * dy
            cells.append(shapely.Polygon(np.column_stack((x + radius * np.cos(angles),
                                                           y + radius * np.sin(angles)))))
    frame = gpd.GeoDataFrame({'cell': np.arange(len(cells))}, geometry=cells, crs=outline.crs)
    frame = positive_parts(gpd.clip(frame, outline), 'Polygon')
    frame['km2'] = frame.area / 1e6
    return frame.loc[frame.km2.gt(1)].copy()


def grid_exposure(grid, annual, footprint):
    shares = {}
    for year, shape in annual.items():
        shares[str(year)] = grid_fraction(grid, shape)
        grid['y' + str(year)] = shares[str(year)].round(5)
    fractions = pd.DataFrame(shares, index=grid.index)
    grid['unique'] = grid_fraction(grid, footprint)
    for tau in (0, 1, 5):
        grid['frequency_' + str(tau)] = fractions.gt(tau / 100).mean(axis=1)
    return grid


def grid_fraction(grid, shape):
    """Sum areas of exact intersections after indexing disjoint footprint parts."""
    parts = shapely.get_parts(shape)
    rows, hits = shapely.STRtree(parts).query(grid.geometry.array)
    bounds = grid.bounds.to_numpy()[rows].T
    cropped = np.array([shapely.clip_by_rect(geom, *envelope)
                        for geom, envelope in zip(parts[hits], bounds.T)], dtype=object)
    bad = ~shapely.is_valid(cropped)
    if bad.any():
        cropped[bad] = shapely.intersection(parts[hits[bad]], shapely.box(*bounds[:, bad]))
    areas = shapely.area(shapely.intersection(grid.geometry.to_numpy()[rows], cropped))
    totals = np.bincount(rows, weights=areas, minlength=len(grid))
    fractions = totals / grid.area.to_numpy()
    assert np.all(fractions <= 1 + 1e-6)
    return pd.Series(np.clip(fractions, 0, 1), index=grid.index)


def run():
    polygon = read('shared', 'alberta').to_crs(AREA)
    root = datum('wildfire', 'raw', 'unused').parent
    layers = sorted(root.glob('perimeter_*.geojson'))
    if len(layers) != 10:
        raise ValueError(f'Expected ten rolling perimeter layers, acquired {len(layers)}')
    annual, year_mapping, repairs = {}, {}, 0
    codes, raw_features, clipped_features = {}, 0, 0
    boundary = polygon.geometry.iloc[0]
    shapely.prepare(boundary)
    for path in layers:
        original = gpd.read_file(path, columns=['YEAR', 'BURNCODE'])
        years = original.YEAR.astype(int).unique()
        if len(years) != 1:
            raise ValueError(f'Mixed or missing fire years in {path.name}: {years}')
        year = int(years[0])
        year_mapping[path.stem] = year
        raw_features += len(original)
        for code, count in original.BURNCODE.fillna('missing').value_counts().items():
            codes[code] = codes.get(code, 0) + int(count)
        # B and B_* denote wildfire perimeters. PB and PB_* are prescribed burns.
        keep = original.BURNCODE.str.fullmatch(r'B(?:_[A-Z]+)?', na=False)
        layer, fixed = clean(original.loc[keep].to_crs(AREA))
        repairs += fixed
        selected = layer.iloc[layer.sindex.query(boundary, predicate='intersects')].copy()
        crossing = ~shapely.covers(boundary, selected.geometry.array)
        selected.loc[crossing, 'geometry'] = selected.loc[crossing].intersection(boundary)
        clipped = positive_parts(selected, 'Polygon')
        clipped_features += len(clipped)
        annual[year] = clipped[['YEAR', 'geometry']].dissolve(by='YEAR', method='disjoint_subset').geometry.iloc[0]
        print(f'{year}: {len(clipped):,} wildfire polygons unioned', flush=True)
    if len(set(year_mapping.values())) != 10:
        raise ValueError(f'Repeated rolling-layer year: {year_mapping}')
    summary_rows = []
    incidents = pd.read_csv(root / 'fp-historical-wildfire-data-2006-2025.csv', encoding='utf-8-sig')
    identified = incidents.dropna(subset=['YEAR', 'FIRE_NUMBER'])
    unique_incidents = identified.drop_duplicates(['YEAR', 'FIRE_NUMBER'])
    counts = unique_incidents.groupby('YEAR')['FIRE_NUMBER'].count()
    for year, count in counts.items():
        shape = annual.get(int(year))
        summary_rows.append({'year': int(year), 'incidents': int(count),
                             'km2': None if shape is None else shape.area / 1e6})
    features = gpd.GeoDataFrame({'year': list(annual)}, geometry=list(annual.values()), crs=AREA)
    overview = features.assign(measured_km2=features.area / 1e6)
    overview.geometry = [shapely.disjoint_subset_union_all(
        shapely.simplify(shapely.get_parts(geom), 125, preserve_topology=True)) for geom in features.geometry]
    save(overview, 'wildfire', 'annual_footprints')
    for year, layer in features.groupby('year'):
        save(layer, 'wildfire', f'footprint_{year}')
        web(overview.loc[overview.year.eq(year)], 'wildfire', f'year_{year}')
    footprint = shapely.disjoint_subset_union_all(shapely.get_parts(features.geometry.array))
    print('Unique period footprint unioned', flush=True)
    visible = shapely.disjoint_subset_union_all(
        shapely.simplify(shapely.get_parts(footprint), 160, preserve_topology=True))
    web(gpd.GeoDataFrame({'scope': ['Unique mapped footprint']}, geometry=[visible], crs=AREA),
        'wildfire', 'unique')
    population = read('shared', 'demand').to_crs(DISTANCE)
    populated = population.loc[population.population.gt(0), ['uid', 'population', 'geometry']].copy()
    parts = gpd.GeoDataFrame(geometry=shapely.get_parts(footprint), crs=AREA).to_crs(DISTANCE)
    nearest = gpd.sjoin_nearest(populated, parts, distance_col='metres')
    populated['km'] = nearest.groupby(level=0).metres.min() / 1000
    save(populated, 'wildfire', 'population_proximity')
    web(populated[['uid', 'population', 'km', 'geometry']], 'wildfire', 'proximity')
    sensitivity = {}
    for size in (25, 50):
        grid = grid_exposure(hexagons(polygon, size * 1000), annual, footprint)
        sensitivity[size] = {'cells': len(grid),
            'recurrent': {tau: int(grid[f'frequency_{tau}'].ge(0.2).sum()) for tau in (0, 1, 5)},
            'unique_distribution': np.histogram(grid['unique'], bins=[0, .001, .01, .05, .20, .50, 1])[0].tolist()}
        save(grid, 'wildfire', f'grid_{size}')
        web(grid, 'wildfire', f'grid_{size}', 100)
        print(f'{size} km grid: {len(grid):,} hexagons measured', flush=True)

    valid = unique_incidents.dropna(subset=['LONGITUDE', 'LATITUDE'])
    valid = valid[valid.LATITUDE.between(48, 61) & valid.LONGITUDE.between(-122, -109)]
    ignitions = gpd.GeoDataFrame(valid[['YEAR', 'FIRE_NUMBER']],
        geometry=gpd.points_from_xy(valid.LONGITUDE, valid.LATITUDE), crs=4326).to_crs(AREA)
    ignitions = gpd.clip(ignitions, polygon)
    save(ignitions, 'wildfire', 'ignitions')
    # Bin ignition points at 10 km before Gaussian smoothing. Zero padding exposes edge loss.
    x0, y0, x1, y1 = polygon.total_bounds
    cell = 10000
    xs = np.arange(x0, x1 + 2 * cell, cell)
    ys = np.arange(y0, y1 + 2 * cell, cell)
    histogram = np.histogram2d(ignitions.geometry.x, ignitions.geometry.y, bins=[xs, ys])[0]
    x, y = np.meshgrid(xs[:-1], ys[:-1], indexing='ij')
    surface = gpd.GeoDataFrame(geometry=shapely.box(x.ravel(), y.ravel(),
        x.ravel() + cell, y.ravel() + cell), crs=AREA)
    for h in (20, 50, 100):
        surface['h' + str(h)] = (gaussian_filter(histogram, h / 10, mode='constant') / 100).ravel().round(5)
    web(gpd.clip(surface, polygon), 'wildfire', 'intensity', 70)
    web(surface.loc[histogram.ravel() > 0].assign(incidents=histogram.ravel()[histogram.ravel() > 0])
        [['incidents', 'geometry']], 'wildfire', 'counts')
    pd.DataFrame(summary_rows).to_csv(datum('wildfire', 'processed', 'annual.csv'), index=False)
    output = {'years': sorted(annual), 'layer_years': year_mapping, 'annual': summary_rows,
              'unique_km2': footprint.area / 1e6, 'cumulative_km2': sum(g.area for g in annual.values()) / 1e6,
              'raw_features': raw_features, 'clipped_features': clipped_features, 'repairs': repairs,
              'recorded_incidents': int(counts.sum()), 'duplicate_incidents': len(identified) - len(unique_incidents),
              'missing_incident_keys': len(incidents) - len(identified),
              'burn_classes': codes, 'prescribed_perimeter_records_excluded': sum(v for k, v in codes.items() if k.startswith('PB')),
              'ignitions_used': len(ignitions), 'grid': sensitivity,
              'proximity': {d: {'locations': int(populated.km.le(d).sum()),
                                'population': int(populated.loc[populated.km.le(d), 'population'].sum())}
                            for d in (0, 5, 10, 25)}}
    summarize('wildfire', output)
    print({k: v for k, v in output.items() if k not in ('annual', 'layer_years')})


if __name__ == '__main__':
    run()

