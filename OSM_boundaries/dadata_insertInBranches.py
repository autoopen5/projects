from dadata import Dadata
from clickhouse_connect import get_client
import pandas as pd
from openpyxl import load_workbook


# dadata api
token = "0b98b6b33ac216cb85afafb8126e2b84726dd012"
secret = "101e666124845d7cbd05889746cc718f15fed1df"
dadata = Dadata(token, secret)

# clickhouse
client = get_client(host='clickhouse.moscow', username='GrushkoIV', port = '8123', password='jNbrvzd1IcF0Yx5I', database='grushko_iv')

with client:
    df = client.query_df(f'''
            SELECT c.id AS id, c.lat AS lat, c.lon AS lon, b.address_houseguid AS address_houseguid FROM grushko_iv.coords c 
            LEFT JOIN MDLP.Branches b
            ON c.id = b.id
            LEFT JOIN MDLP.Branches_add_info bai 
            ON b.address_houseguid = bai.address_houseguid            
            ''') 

# rows = []    
for _, r in df[5000:].iterrows():
    print(r['address_houseguid'], r['lat'], r['lon'])
    lat = float(r['lat'])
    lon = float(r['lon'])
    # houseguid =  str(r['address_houseguid'])
    result = client.command(f'''ALTER TABLE MDLP.Branches_add_info UPDATE lat = {lat}, lon = {lon} 
                                WHERE address_houseguid = '{r['address_houseguid']}' 
                            ''')
	
    print(f' inserted')