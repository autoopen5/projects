import json, sys, gzip
import pandas as pd
from clickhouse_connect import get_client
from pathlib import Path

client = get_client(host='clickhouse.moscow', username='GrushkoIV', port = '8123', password='jNbrvzd1IcF0Yx5I', database='grushko_iv')

with client:
    df = client.query_df(f'''
            SELECT id, federal_subject_name, lat, lon FROM MDLP.Branches b
            LEFT JOIN MDLP.Branches_add_info bai 
            ON b.address_houseguid = bai.address_houseguid
            WHERE lat IS NOT NULL AND lon IS NOT NULL                  
            ''') 
    
result_records = []

for _, r in df.iterrows():
    print(r['lat'], r['lon'])

    boundaries = client.command(f'''WITH
	      {r['lon']} AS lon,   -- Москва, пример
	   	  {r['lat']} AS lat
	SELECT
	    name
	FROM grushko_iv.boundaries
	WHERE pointInPolygon( (lon, lat), polygon ) AND admin_level = 4
	ORDER BY admin_level DESC, name;
    ''')
    print(boundaries)
    print(r['federal_subject_name'])

    if r['federal_subject_name'] != boundaries:
         result_records.append({
            'id': r['id'],
            'federal_subject': r['federal_subject_name'],
            'boundary_name': boundaries
        })

excel = pd.DataFrame(result_records, columns=['id', 'federal_subject', 'boundary_name'])
out_path = 'results.xlsx'
with pd.ExcelWriter(out_path, engine='openpyxl') as xw:
    excel.to_excel(xw, index=False, sheet_name='results')

    print(f'Excel сохранён: {out_path}')





