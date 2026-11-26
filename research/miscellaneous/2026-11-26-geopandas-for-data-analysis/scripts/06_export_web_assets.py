"""Render static previews, check outputs and benchmark comparable spatial operations."""
import argparse
from importlib.metadata import version
import gzip
from io import BytesIO
import json
from pathlib import Path
import platform
from statistics import median
from tempfile import TemporaryDirectory
from time import perf_counter

import geopandas as gpd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.ticker import FuncFormatter, MaxNLocator
import networkx as nx
import numpy as np
import shapely

from common import ROOT, AREA, DISTANCE, datum, json_write, read, summarize


def summary(name):
    return json.loads((ROOT / 'results' / (name + '.json')).read_text(encoding='utf-8'))


def tidy(ax, title=''):
    ax.set_xticks([])
    ax.set_yticks([])
    for spine in ax.spines.values():
        spine.set_color('#d7e3db')
    if title:
        ax.set_title(title, fontsize=11, color='#334b42', loc='left')


def finish(fig, destination, dpi=154):
    fig.savefig(ROOT / destination, dpi=dpi, bbox_inches='tight', pad_inches=.09,
                facecolor=fig.get_facecolor())
    plt.close(fig)


def foundation_assets():
    outline = gpd.read_file(datum('shared', 'processed', 'alberta_boundary.gpkg'))
    units = outline.crs.axis_info[0].unit_name
    json_write(ROOT / 'figures/foundations/boundary-preview.json',
               {'records': len(outline), 'geometry_type': ', '.join(sorted(outline.geom_type.unique())),
                'geometry_column': outline.geometry.name, 'crs': outline.crs.to_string(),
                'units': 'metres' if units == 'metre' else units})

    canvas = '#fff8f2'
    fig, ax = plt.subplots(figsize=(3.7, 4.25), layout='constrained', facecolor=canvas)
    ax.set_facecolor(canvas)
    outline.plot(ax=ax, color='#bbdccc', edgecolor='#467468', linewidth=.9, zorder=2)
    ax.margins(.08)
    ax.set_title('Alberta boundary', fontsize=12, loc='left', color='#2d4d43', pad=8)
    ax.set_xlabel('Easting (km)', fontsize=10, color='#2d4d43')
    ax.set_ylabel('Northing (km)', fontsize=10, color='#2d4d43')
    for axis in (ax.xaxis, ax.yaxis):
        axis.set_major_locator(MaxNLocator(4, prune='both'))
        axis.set_major_formatter(FuncFormatter(lambda value, _: f'{value / 1000:,.0f}'))
    ax.tick_params(labelsize=9, colors='#4d6158', length=3)
    ax.grid(color='#ddd3c7', linewidth=.55, zorder=0)
    for spine in ax.spines.values():
        spine.set_color('#bdb9aa')
    finish(fig, 'figures/foundations/alberta-basic.png', dpi=220)
    fig, axs = plt.subplots(1, 3, figsize=(9.4, 3), layout='constrained', facecolor=canvas)
    for ax, code, text in zip(axs, ['EPSG:4326', DISTANCE, 'EPSG:3347'],
                             ['WGS 84 · raw axes in degrees', 'Alberta 10-TM · metres', 'Statistics Canada Lambert · metres']):
        outline.to_crs(code).plot(ax=ax, color='#bfd9ca', edgecolor='#547f6d', linewidth=.8,
                                 aspect='equal')
        tidy(ax, text)
        ax.set_facecolor(canvas)
    finish(fig, 'figures/foundations/projections.png', dpi=190)
    return outline

def figure_assets():
    outline = foundation_assets()
    (ROOT / 'results' / 'boundary_example.txt').write_text(
        str(outline.head()) + '\n' + str(outline.crs) + '\n', encoding='utf-8')
    province = outline.to_crs(DISTANCE)
    for kind, background in [('healthcare','#eaf4ee'), ('wildfire','#f6eddb'),
                             ('flood','#eaf2f8'), ('socioeconomic','#f2edf7')]:
        fig, ax = plt.subplots(figsize=(4.2, 2.05), layout='constrained', facecolor=background)
        if kind != 'flood':
            province.plot(ax=ax, color='#f6f9f1', edgecolor='#637e71', linewidth=.7)
        if kind == 'healthcare':
            facilities = read('healthcare', 'facilities').query("group == 'hospital'")
            read('shared', 'divisions').boundary.plot(ax=ax, color='#9ab2a5', linewidth=.45)
            facilities.plot(ax=ax, color='#3c7b66', markersize=12, linewidth=.22, edgecolor='white')
        elif kind == 'wildfire':
            grid = read('wildfire', 'grid_50').to_crs(DISTANCE)
            grid.plot(ax=ax, column='frequency_0', cmap='YlOrBr', linewidth=.14, edgecolor='#b5a78e',
                      vmin=0, vmax=.5)
            province.boundary.plot(ax=ax, color='#6e7159', linewidth=.7)
        elif kind == 'flood':
            read('flood', 'roads').plot(ax=ax, color='#8f9c9d', linewidth=.25)
            flood = read('flood', 'inundation_100').to_crs(DISTANCE)
            flood.plot(ax=ax, color='#7db2d1', alpha=.6)
            read('flood', 'exposed_roads_100').plot(ax=ax, color='#bb7149', linewidth=.7)
        else:
            economics = read('socioeconomic', 'census')
            column = 'low_income' if 'low_income' in economics else 'density'
            visible = economics.loc[economics[column].notna()]
            upper = visible[column].quantile(.95)
            visible.plot(ax=ax, column=column, cmap='Purples', vmin=0, vmax=upper, linewidth=0)
            province.boundary.plot(ax=ax, color='#6e6883', linewidth=.7)
        ax.set_axis_off()
        finish(fig, 'figures/' + kind + '/preview.png')


def elapsed(function, repeats=4):
    times = []
    for _ in range(repeats):
        start = perf_counter()
        function()
        times.append(1000 * (perf_counter() - start))
    return median(times)


def benchmark():
    das = read('shared', 'das').sort_values('uid')
    sample = das[['uid', 'density', 'geometry']].sample(850, random_state=907).reset_index(drop=True)
    points = read('healthcare', 'facilities').query("group == 'hospital'").geometry.buffer(10000).to_numpy()
    geoms = sample.geometry.to_numpy()
    index = shapely.STRtree(geoms)

    def naive():
        return np.count_nonzero(shapely.intersects(points[:, None], geoms[None, :]))

    def indexed():
        return index.query(points, predicate='intersects').shape[1]

    assert naive() == indexed()
    reads = []
    with TemporaryDirectory() as tmp:
        work = Path(tmp)
        for name, write, load in (
                ('GeoJSON', lambda p: sample.to_file(p, driver='GeoJSON', index=False), gpd.read_file),
                ('GeoPackage', lambda p: sample.to_file(p, driver='GPKG', index=False), gpd.read_file),
                ('GeoParquet', lambda p: sample.to_parquet(p, index=False), gpd.read_parquet)):
            extension = {'GeoJSON':'geojson','GeoPackage':'gpkg','GeoParquet':'parquet'}[name]
            path = work / ('sample.' + extension)
            write(path)
            assert len(load(path)) == len(sample)
            reads.append({'format': name, 'bytes': path.stat().st_size,
                          'milliseconds': elapsed(lambda: load(path))})
    full = das[['uid', 'density', 'geometry']].to_crs(4326).to_json(drop_id=True, separators=(',', ':')).encode()
    web = ROOT / 'maps' / 'shared' / 'das.geojson'
    results = {'index': {'pairs': len(geoms) * len(points), 'candidates': index.query(points).shape[1],
                         'matches': indexed(), 'naive_ms': elapsed(naive), 'indexed_ms': elapsed(indexed),
                         'tree_build_ms': elapsed(lambda: shapely.STRtree(geoms))},
               'sample': len(sample), 'reads': reads,
               'geometry': {'original': len(full), 'web': web.stat().st_size},
               'platform': platform.system() + ' ' + platform.release() + ' · ' + platform.processor(),
               'timing': 'Median of four repeated local reads, warm filesystem cache; indexed query excludes tree construction'}
    summarize('benchmark', results)
    return results


def validate():
    shared, health, fire, flood, social = (summary(key) for key in
        ('shared', 'healthcare', 'wildfire', 'flood', 'socioeconomic'))
    census = read('shared', 'das')
    demand = read('shared', 'demand')
    assert census.uid.is_unique and demand.uid.is_unique
    assert census.population.sum() == shared['population']
    assert census.is_valid.all() and demand.is_valid.all()
    assert all(group['gain_km'] >= 0 for group in health['groups'].values())
    assert fire['years'] == list(range(min(fire['years']), max(fire['years']) + 1))
    assert len(fire['years']) == 10
    assert fire['unique_km2'] <= fire['cumulative_km2'] + 1e-7
    assert all(0 <= x['road_pct'] <= 100 for x in flood['scenarios'])
    assert all(r['roof_m2'] <= flood['total_roof_m2'] for r in flood['scenarios'])
    assert all(0 < w['p'] <= 1 for r in social['indicators'].values() for w in r['weights'].values())
    counts = {}
    for item in (ROOT / 'maps').rglob('*.geojson*'):
        layer = gpd.read_file(BytesIO(gzip.decompress(item.read_bytes()))) if item.suffix == '.gz' else gpd.read_file(item)
        assert layer.crs.to_epsg() == 4326
        failed = ~layer.is_valid | layer.is_empty
        assert not failed.any(), f'{item.name}: {failed.sum()} invalid or empty web features'
        counts[item.relative_to(ROOT).as_posix()] = len(layer)
    total = len(counts)
    graph = nx.Graph()
    graph.add_weighted_edges_from([('A','B',3), ('A','C',5), ('B','C',3), ('B','D',3),
                                   ('C','E',3), ('D','E',4), ('D','F',2), ('E','F',4)])
    baseline = nx.dijkstra_path_length(graph, 'A', 'F')
    graph.remove_edge('B', 'D')
    disrupted = nx.dijkstra_path_length(graph, 'A', 'F')
    assert (baseline, disrupted) == (8, 12)
    validation = {'status': 'passed', 'boundary_coverage': shared, 'web_map_files': total,
                  'educational_network': {'baseline': baseline, 'with_b_d_closed': disrupted},
                  'web_feature_counts': counts, 'reconciled_population': int(census.population.sum()),
                  'facility_repairs': health['repairs'], 'facility_duplicates': health['duplicates_removed'],
                  'wildfire_repairs': fire['repairs'], 'flood_road_repeats_removed': flood['duplicate_roads'],
                  'flood_roof_repeats_removed': flood['duplicate_roofs'],
                  'census_indicators': {k:{'n': v['n'], 'missing': v['missing']}
                                       for k,v in social['indicators'].items()}}
    json_write(ROOT / 'metadata' / 'validation_report.json', validation)
    return validation


def main():
    figure_assets()
    stats = benchmark()
    validation = validate()
    packages = ('geopandas','shapely','pyogrio','pyproj','pandas','numpy','scipy','pyarrow',
                'matplotlib','libpysal','esda','networkx','requests','playwright')
    manifest = {'python': platform.python_version(), 'platform': platform.platform(),
                'packages': {p: version(p) for p in packages}, 'distance_crs': DISTANCE,
                'area_crs': AREA, 'web_crs': 'EPSG:4326', 'wildfire_years': summary('wildfire')['layer_years'],
                'healthcare_source': summary('healthcare')['source'], 'profile_source': summary('socioeconomic')['source'],
                'asset_size_bytes': {p.relative_to(ROOT).as_posix(): p.stat().st_size for p in
                    (ROOT / 'maps').rglob('*') if p.is_file()}}
    json_write(ROOT / 'metadata' / 'processing_manifest.json', manifest)
    print(f"Created static figures and validated {validation['web_map_files']} GeoJSON layers.")
    print(stats)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--foundations-only', action='store_true',
                        help='Redraw introductory figures from the existing Alberta GeoPackage')
    if parser.parse_args().foundations_only:
        foundation_assets()
    else:
        main()

