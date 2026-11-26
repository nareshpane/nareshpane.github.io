"""Project paths, serialization and spatial output helpers."""
from pathlib import Path
import gzip
import json

import geopandas as gpd
import numpy as np
import shapely

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / 'data'
AREAS = {'shared': 'shared', 'healthcare': '01-healthcare-accessibility',
         'wildfire': '02-wildfire-exposure', 'flood': '03-flood-infrastructure',
         'socioeconomic': '04-socioeconomic-geography'}
DISTANCE = 'EPSG:3400'
AREA = '+proj=aea +lat_1=49 +lat_2=60 +lat_0=54.5 +lon_0=-115 +datum=NAD83 +units=m +type=crs'
CALGARY = (-114.135, 51.02, -114.013, 51.077)
SEED = 907


def datum(area, kind, name):
    path = DATA / AREAS[area] / kind / name
    path.parent.mkdir(parents=True, exist_ok=True)
    return path


def json_write(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix(path.suffix + '.tmp')
    temp.write_text(json.dumps(value, ensure_ascii=False, allow_nan=False,
                               default=lambda obj: obj.item(), separators=(',', ':')), encoding='utf-8')
    temp.replace(path)


def clean(gdf):
    if gdf.crs is None:
        raise ValueError('Missing source CRS')
    rows = gdf.loc[~shapely.is_missing(gdf.geometry.array) & ~gdf.geometry.is_empty].copy()
    invalid = ~rows.is_valid
    if invalid.any():
        rows.loc[invalid, 'geometry'] = rows.loc[invalid].make_valid()
    return rows, int(invalid.sum())


def save(gdf, area, name):
    gdf.to_parquet(datum(area, 'processed', name + '.parquet'), index=False,
                   compression='zstd', compression_level=7)


def read(area, name):
    return gpd.read_parquet(datum(area, 'processed', name + '.parquet'))


def web(gdf, area, name, tolerance=0, compressed=False):
    frame = gdf.copy()
    if tolerance:
        simplified = frame.geometry.simplify(tolerance, preserve_topology=True)
        usable = ~simplified.is_empty & simplified.is_valid
        frame.geometry = simplified.where(usable, frame.geometry)
    frame = frame.to_crs(4326)
    invalid = ~frame.is_valid
    if invalid.any():
        frame.loc[invalid, 'geometry'] = frame.loc[invalid].make_valid(method='structure', keep_collapsed=False)
        frame = frame.loc[~frame.geometry.is_empty].copy()
    rounded = gpd.GeoSeries(shapely.set_precision(frame.geometry.array, 0.00001, mode='pointwise'),
                            index=frame.index, crs=frame.crs)
    usable = rounded.is_valid & ~rounded.is_empty
    frame.geometry = rounded.where(usable, frame.geometry)
    suffix = '.geojson.gz' if compressed else '.geojson'
    path = ROOT / 'maps' / area / (name + suffix)
    path.parent.mkdir(parents=True, exist_ok=True)
    content = frame.to_json(drop_id=True, na='null', separators=(',', ':'))
    if compressed:
        path.write_bytes(gzip.compress(content.encode(), compresslevel=6, mtime=0))
        path.with_suffix('').unlink(missing_ok=True)
    else:
        path.write_text(content, encoding='utf-8')


def summarize(area, result):
    json_write(ROOT / 'results' / (area + '.json'), result)


def positive_parts(gdf, family):
    """Keep matching primitive components after repairs or intersections."""
    if family not in ('Polygon', 'LineString'):
        raise ValueError(family)
    def parts(geometry):
        if geometry.geom_type == family:
            return [geometry]
        return [part for child in getattr(geometry, 'geoms', ()) for part in parts(child)]
    series = gdf.geometry.map(lambda geometry: shapely.union_all(parts(geometry)))
    return gdf.set_geometry(series).loc[~series.is_empty].copy()


def weighted_quantile(value, weight, probability):
    order = np.argsort(value)
    cumulative = np.cumsum(np.asarray(weight)[order])
    return np.asarray(value)[order][np.searchsorted(cumulative, probability * cumulative[-1])]

