"""Build only the opening illustration's geographic SVG (no analytical data).

Natural Earth v5.1.2, public-domain 50m admin-1 / admin-0 boundaries.
Lambert conformal conic: parallels 49/77, central meridian -96.
Usage: python build_animation_maps.py --cache <directory>
The SVG is local at runtime; coordinates printed here anchor schematic icons.
"""
import argparse
import html
import json
import math
from pathlib import Path
import urllib.request

SOURCE = 'https://raw.githubusercontent.com/nvkelso/natural-earth-vector/v5.1.2/geojson/'


def source(cache, name, alias):
    path = cache / alias
    if not path.exists():
        urllib.request.urlretrieve(SOURCE + name + '.geojson', path)
    return json.loads(path.read_text(encoding='utf-8'))['features']


def rings(geometry):
    polys = [geometry['coordinates']] if geometry['type'] == 'Polygon' else geometry['coordinates']
    return [ring for poly in polys for ring in poly]


def lambert(lon, lat):
    a, b = map(math.radians, (49, 77))
    n = math.log(math.cos(a)/math.cos(b)) / math.log(math.tan(math.pi/4+b/2)/math.tan(math.pi/4+a/2))
    f = math.cos(a) * math.tan(math.pi/4+a/2)**n / n
    rho = f / math.tan(math.pi/4+math.radians(lat)/2)**n
    theta = n * math.radians(lon+96)
    return rho*math.sin(theta), -rho*math.cos(theta)


def simplify(points, tolerance):
    if len(points) < 3:
        return points
    a, b = points[0], points[-1]
    dx, dy = b[0]-a[0], b[1]-a[1]
    def distance(p):
        u = max(0, min(1, ((p[0]-a[0])*dx+(p[1]-a[1])*dy)/(dx*dx+dy*dy))) if dx or dy else 0
        return math.hypot(p[0]-a[0]-u*dx, p[1]-a[1]-u*dy)
    maximum, i = max((distance(p), i) for i, p in enumerate(points[1:-1], 1))
    if maximum < tolerance:
        return [a, b]
    return simplify(points[:i+1], tolerance)[:-1] + simplify(points[i:], tolerance)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--cache', required=True, type=Path)
    args = parser.parse_args()
    args.cache.mkdir(parents=True, exist_ok=True)
    canada = [f for f in source(args.cache, 'ne_50m_admin_1_states_provinces', 'section338-provinces.geojson') if f['properties']['admin'] == 'Canada']
    us = next(f for f in source(args.cache, 'ne_50m_admin_0_countries', 'section338-countries.geojson') if f['properties']['ADMIN'] == 'United States of America')
    assert len(canada) == 13
    points = [lambert(*p[:2]) for f in canada for ring in rings(f['geometry']) for p in ring]
    left, right = min(p[0] for p in points), max(p[0] for p in points)
    bottom, top = min(p[1] for p in points), max(p[1] for p in points)
    scale = min(760/(right-left), 380/(top-bottom))
    def project(lon, lat):
        x, y = lambert(lon, lat)
        return round(60+(x-left)*scale, 1), round(20+(top-y)*scale, 1)
    def path(geometry, mainland=False):
        parts = []
        for ring in rings(geometry):
            if mainland and not any(-125 < p[0] < -66 and 25 < p[1] < 49 for p in ring):
                continue  # Alaska/Hawaii are outside this contiguous-U.S. crop.
            pts = [project(*p[:2]) for p in ring]
            area = abs(sum(a[0]*b[1]-b[0]*a[1] for a, b in zip(pts, pts[1:]))) / 2
            if area < .7:
                continue
            pts = simplify(pts, .6)
            if len(pts) > 3:
                parts.append('M'+'L'.join(f'{x},{y}' for x, y in pts[:-1])+'Z')
        return ''.join(parts)
    abbreviations = {'British Columbia':'BC','Alberta':'AB','Saskatchewan':'SK','Manitoba':'MB','Ontario':'ON','Quebec':'QC','New Brunswick':'NB','Nova Scotia':'NS','Prince Edward Island':'PE','Newfoundland and Labrador':'NL','Yukon':'YT','Northwest Territories':'NT','Nunavut':'NU'}
    svg = ['<svg xmlns="http://www.w3.org/2000/svg"><!-- Natural Earth v5.1.2: https://www.naturalearthdata.com/about/terms-of-use/ ; Lambert conformal conic. -->']
    for f in canada:
        name = f['properties']['name'].replace('Québec', 'Quebec')
        svg.append(f'<g id="{abbreviations[name]}" fill-rule="evenodd"><title>{html.escape(name)}</title><path d="{path(f["geometry"])}"/></g>')
    svg.append(f'<g id="US" fill-rule="evenodd"><path d="{path(us["geometry"], True)}"/></g></svg>')
    output = Path(__file__).resolve().parent / 'animation-maps.svg'
    output.write_text('\n'.join(svg)+'\n', encoding='utf-8')
    anchors = {'BC':(-124,55),'AB':(-114.5,55),'SK':(-106,54),'MB':(-98,55),'ON':(-85,51),'QC':(-71,53),'NB':(-66.5,46.5),'NS':(-63,45),'PE':(-63,46.4),'NL':(-58,53),'YT':(-136,64),'NT':(-121,65),'NU':(-95,69)}
    print('Province anchors:', {k:project(*v) for k,v in anchors.items()})
    print('Transport anchors:', {k:project(*v) for k,v in {'Vancouver':(-123,49.3),'Prairies':(-105,50),'Ontario':(-80,44),'Atlantic':(-63,45),'Seattle':(-122.3,47.6),'CentralUS':(-99,41),'GreatLakes':(-83,41),'NortheastUS':(-71,42),'Halifax':(-63.6,44.6),'Boston':(-71,42.4)}.items()})
    print('Border:', [project(*p) for p in [(-123,49),(-115,49),(-105,49),(-95,49),(-90,48),(-84,46),(-82.5,42),(-79,43),(-75,45),(-71.5,45),(-69,47),(-67,47),(-67,45)]])
    print(output)


if __name__ == '__main__':
    main()
