"""Draw Natural Earth 1:50m admin-1 polygons in local Lambert/equirectangular views."""
import json
import math
from html import escape
from prepare_data import ROOT, json_write

PLAN = {
    'CA': [('Canada', (0, 0, 400, 200), {'Vancouver': (-123.12,49.28),
             'Calgary':(-114.07,51.04),'Winnipeg':(-97.14,49.9),'Toronto':(-79.38,43.65),
             'Montréal':(-73.57,45.5),'Halifax':(-63.57,44.65),'Yellowknife':(-114.37,62.45)})],
    'US': [('Lower 48', (5, 0, 390, 190), {'Seattle':(-122.33,47.61),'Chicago':(-87.63,41.88),
             'New York':(-74.01,40.71),'Houston':(-95.37,29.76),'Los Angeles':(-118.24,34.05)}),
           ('Alaska', (6, 144, 88, 53), {'Anchorage':(-149.9,61.22)}),
           ('Hawaii', (295, 153, 82, 45), {'Honolulu':(-157.86,21.31)})]
}

def projector(area):
    central, ref = {'Canada':(-96,60),'Lower 48':(-96,39),'Alaska':(-154,62),'Hawaii':(-157,20)}[area]
    n = math.sin(math.radians(ref))
    def project(point):
        lon,lat = point
        theta = n * math.radians((lon-central+180)%360-180)
        rho = 1 / math.tan(math.pi/4 + math.radians(lat)/2)**n
        return (rho*math.sin(theta),rho*math.cos(theta))
    return project

def render(features, country):
    colors = ['#f0d1c4','#f8e1d7','#f4d9cc','#fae8e0'] if country=='CA' else ['#bed7ea','#d4e5f2','#c8dded','#dceaf4']
    groups, spots = [], []
    for area, bounds, cities in PLAN[country]:
        subset = []
        for f in features:
            p, g = f['properties'], f['geometry']
            if country=='CA' and p['admin']!='Canada':
                continue
            if country=='US' and (p['admin']!='United States of America' or
                    ((p['name'] if p['name'] in ('Alaska','Hawaii') else 'Lower 48') != area)):
                continue
            polygons = g['coordinates'] if g['type']=='MultiPolygon' else [g['coordinates']]
            subset.append((p['name'], polygons))
        project = projector(area)
        points = [project(v) for _,polygons in subset for polygon in polygons for ring in polygon for v in ring]
        xmin,ymin = map(min,zip(*points))
        xmax,ymax = map(max,zip(*points))
        x,y,w,h = bounds
        scale = min((w-5)/(xmax-xmin),(h-5)/(ymax-ymin))
        ox,oy = (w-(xmax-xmin)*scale)/2+x,(h-(ymax-ymin)*scale)/2+y
        def fit(point):
            a,b = project(point)
            return (round(ox+(a-xmin)*scale,1),round(oy+(b-ymin)*scale,1))
        for j,(name,polygons) in enumerate(subset):
            paths = []
            for polygon in polygons:
                for ring in polygon:
                    pts = [fit(v) for v in ring]
                    distinct = [p for i,p in enumerate(pts) if i==0 or p!=pts[i-1]]
                    if len(distinct)>3:
                        paths.append('M'+' '.join(f'{a},{b}' for a,b in distinct)+'Z')
            groups.append(f'<path fill="{colors[j%4]}" d="{" ".join(paths)}"><title>{escape(name)}</title></path>')
        for name,pos in cities.items():
            ax,ay = fit(pos)
            spots.append([name,ax,ay])
    svg = '<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 400 200" role="img">' + \
          f'<title>{country} subnational boundaries</title><g stroke="#fff" stroke-width=".6" fill-rule="evenodd">' + \
          ''.join(groups) + '</g></svg>'
    (ROOT / 'figures' / f'{country.lower()}-map.svg').write_text(svg,encoding='utf-8')
    return spots

def main():
    source = ROOT / 'data/metadata/ne_admin1.geojson'
    features = json.loads(source.read_text(encoding='utf-8'))['features']
    spots = {country:render(features,country) for country in PLAN}
    (ROOT/'animations/map-cities.js').write_text('globalThis.CPI_CITIES = '+json.dumps(spots)+';\n',encoding='utf-8')

if __name__=='__main__':
    main()

