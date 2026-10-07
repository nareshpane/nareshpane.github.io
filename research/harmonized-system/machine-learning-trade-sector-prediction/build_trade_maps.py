"""Derive the animation's inline maps, without touching trade data or models.

Natural Earth v5.1.2, public domain:
https://www.naturalearthdata.com/about/terms-of-use/
50m admin-1 Canadian provinces; 50m physical land for the world.
Canada: Lambert conformal conic, parallels 49/77, central meridian -96.
World: Robinson projection. Both maps use projected-pixel simplification.

Development usage: python build_trade_maps.py --cache <temporary-dir>
                    --output <temporary-json>
The output is embedded in the HTML; there are no runtime geographic requests.
"""
import argparse
import html
import json
import math
from pathlib import Path
import urllib.request

SOURCE = 'https://raw.githubusercontent.com/nvkelso/natural-earth-vector/v5.1.2/geojson/'


def source(cache, name):
    path = cache / (name + '.geojson')
    if not path.exists():
        urllib.request.urlretrieve(SOURCE + path.name, path)
    return json.loads(path.read_text(encoding='utf-8'))


def polygons(geometry):
    if geometry['type'] == 'Polygon':
        return [geometry['coordinates']]
    return geometry['coordinates']


def distance(point, a, b):
    dx, dy = b[0] - a[0], b[1] - a[1]
    if dx == dy == 0:
        return math.hypot(point[0] - a[0], point[1] - a[1])
    u = max(0, min(1, ((point[0]-a[0])*dx+(point[1]-a[1])*dy)/(dx*dx+dy*dy)))
    return math.hypot(point[0]-a[0]-u*dx, point[1]-a[1]-u*dy)


def simplify(points, tolerance):
    if len(points) < 3:
        return points
    maximum, index = max((distance(p, points[0], points[-1]), i)
                         for i, p in enumerate(points[1:-1], 1))
    if maximum <= tolerance:
        return [points[0], points[-1]]
    return simplify(points[:index+1], tolerance)[:-1] + simplify(points[index:], tolerance)


def ring_path(ring, project, tolerance, minimum_area):
    points = [project(lon, lat) for lon, lat, *_ in ring]
    area = abs(sum(a[0]*b[1]-b[0]*a[1] for a, b in zip(points, points[1:]))) / 2
    if area < minimum_area:
        return ''
    points = simplify(points, tolerance)
    if len(points) < 4:
        return ''
    return 'M' + 'L'.join(f'{x:.1f},{y:.1f}' for x, y in points[:-1]) + 'Z'


def geometry_path(geometry, project, tolerance=1, minimum_area=2):
    return ''.join(ring_path(ring, project, tolerance, minimum_area)
                   for poly in polygons(geometry) for ring in poly)


def lambert(lon, lat):
    a, b = map(math.radians, [49, 77])
    n = math.log(math.cos(a)/math.cos(b))/math.log(math.tan(math.pi/4+b/2)/math.tan(math.pi/4+a/2))
    f = math.cos(a) * math.tan(math.pi/4+a/2)**n / n
    rho = f / math.tan(math.pi/4+math.radians(lat)/2)**n
    theta = n * math.radians(lon+96)
    return rho*math.sin(theta), -rho*math.cos(theta)


RX = [1,.9986,.9954,.99,.9822,.973,.96,.9427,.9216,.8962,.8679,.835,.7986,.7597,.7186,.6732,.6213,.5722,.5322]
RY = [0,.062,.124,.186,.248,.31,.372,.434,.4958,.5571,.6176,.6769,.7346,.7903,.8435,.8936,.9394,.9761,1]


def robinson(lon, lat):
    pos = min(abs(lat), 89.999) / 5
    i, fraction = int(pos), pos % 1
    x = RX[i] + (RX[i+1]-RX[i])*fraction
    y = RY[i] + (RY[i+1]-RY[i])*fraction
    return 560 + lon/180*x*518, 240 - math.copysign(y*200, lat)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--cache', required=True, type=Path)
    parser.add_argument('--output', required=True, type=Path)
    args = parser.parse_args()
    args.cache.mkdir(parents=True, exist_ok=True)
    provinces = source(args.cache, 'ne_50m_admin_1_states_provinces')['features']
    canada = [f for f in provinces if f['properties'].get('admin') == 'Canada']
    assert len(canada) == 13, [(f['properties'].get('name')) for f in canada]
    all_points = [lambert(lon,lat) for f in canada for poly in polygons(f['geometry'])
                  for ring in poly for lon, lat, *_ in ring]
    low_x = min(p[0] for p in all_points); high_x = max(p[0] for p in all_points)
    low_y = min(p[1] for p in all_points); high_y = max(p[1] for p in all_points)
    scale = min(630/(high_x-low_x), 335/(high_y-low_y))
    offset_x = 30+(630-(high_x-low_x)*scale)/2
    offset_y = 20+(335-(high_y-low_y)*scale)/2
    def project(lon,lat):
        x,y = lambert(lon,lat)
        return offset_x+(x-low_x)*scale, offset_y+(high_y-y)*scale
    parts = []
    alberta = None
    for f in canada:
        name = f['properties']['name']
        path = geometry_path(f['geometry'], project, .85, 1.5)
        if name == 'Alberta':
            alberta = path
        else:
            parts.append(f'<path data-province="{html.escape(name)}" d="{path}"/>')
    assert alberta
    land = source(args.cache, 'ne_50m_land')['features']
    # Antarctica is outside this illustration's trading network and viewport.
    # Its rings are excluded before projection rather than drawing a clipped band.
    world = ''.join(ring_path(ring, robinson, 1.05, 2.5) for f in land
                    for poly in polygons(f['geometry']) for ring in poly
                    if max(lat for _,lat,*_ in ring) > -60)
    origin = robinson(-114.5,54.8)
    destinations = [('us','United States',-98,38,15.15),('mx','Mexico',-102,23,16.15),
                    ('cn','China',105,35,18),('jp','Japan',139,37,17),
                    ('kr','South Korea',127.8,36,17.6),('uk','United Kingdom',-2,54,19.2)]
    result = {'canada_paths': ''.join(parts), 'alberta_path': alberta,
              'canada_origin': project(-114.5,54.8),
              'neighbors': {label:project(lon,lat) for label,lon,lat in
                            [('BC',-125,55),('SK',-106,54),('NWT',-119,64),('U.S.',-114,47.9)]},
              'world_path': world, 'world_origin': origin,
              'destinations': [{'id':id,'name':name,'point':robinson(lon,lat),'start':start,
                                'lon':lon,'lat':lat} for id,name,lon,lat,start in destinations]}
    args.output.write_text(json.dumps(result,indent=2),encoding='utf-8')
    print('Canada / Alberta / world SVG bytes:',len(result['canada_paths']),len(alberta),len(world))
    print('Alberta centers:',result['canada_origin'],origin)


if __name__ == '__main__':
    main()
