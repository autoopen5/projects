import json, sys, gzip
from clickhouse_connect import get_client
from pathlib import Path

def ring_clockwise(coords):
    # Площадь кольца (signed). >0 ~ по часовой для (lon,lat)
    area = 0.0
    for i in range(len(coords)-1):
        x1,y1 = coords[i]
        x2,y2 = coords[i+1]
        area += (x2 - x1) * (y2 + y1)
    return area > 0

def normalize_polygon(poly):
    # poly = [ring1, ring2, ...]; ring — список [lon,lat], последний == первый
    out = []
    for i, ring in enumerate(poly):
        r = ring[:-1] if ring[0]==ring[-1] else ring[:]  # убираем дубликат последней точки
        cw = ring_clockwise(r)
        if i == 0:   # внешнее кольцо — по часовой
            if not cw: r = r[::-1]
        else:        # дырки — против часовой
            if cw: r = r[::-1]
        out.append([(float(x), float(y)) for x,y in r])
    return out

def iter_polygons(geom):
    t = geom.get('type')
    if t == 'Polygon':
        yield 1, normalize_polygon(geom['coordinates'])
    elif t == 'MultiPolygon':
        for i, poly in enumerate(geom['coordinates'], start=1):
            yield i, normalize_polygon(poly)
    else:
        return

geojson_path = r'C:/Users/SatyaTR/Desktop/OSMB2.geojson'  # ваш файл

with open(geojson_path, 'r', encoding='utf-8') as f:
    gj = json.load(f)

client = get_client(host='clickhouse.moscow', username='GrushkoIV', port = '8123', password='jNbrvzd1IcF0Yx5I', database='grushko_iv')

rows = []
for feat in gj['features']:
    props = feat.get('properties', {})
    geom  = feat.get('geometry', {})
    fid = str(props.get('id') or props.get('@id') or props.get('osm_id') or props.get('OBJECTID') or '')
    name = props.get('name') or props.get('NAME') or ''
    admin_level = int(props.get('admin_level') or props.get('ADMIN_LVL') or 0)

    for part_idx, polygon in iter_polygons(geom):
        rows.append([fid, name, admin_level, part_idx, polygon])

# вставка пачками
client.insert('grushko_iv.boundaries',
              rows,
              column_names=['id','name','admin_level','part_index','polygon'])
print(f'Inserted {len(rows)} rows')

# import json

# geojson_path = r'C:/Users/SatyaTR/Desktop/OSMB.geojson'  # ваш файл



# print(type(gj), gj.keys())          # например: dict, dict_keys(['type','features'])
# print('features:', len(gj.get('features', [])))