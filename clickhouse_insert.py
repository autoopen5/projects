from clickhouse_connect import get_client


# clickhouse
client = get_client(host='clickhouse.moscow', username='GrushkoIV', port = '8123', password='jNbrvzd1IcF0Yx5I', database='grushko_iv')


result = client.command(f'''INSERT INTO Visits_Effectiveness.SalesVsVisits
SELECT s.*, amount_of_clients, amount_of_contacts, amount_of_visits, CASE WHEN v.amount_of_visits != 0 THEN 'Визит' ELSE 'Без визита' END AS VisitFlag, o.name as org_name, o.type_name as org_type, o.full_address as org_adres, 
b.address_address_description as Drugstore_adres, b.org_name as Drugstore_name, b.inn as Drugstore_inn, bf.contragent_n as Net_name, bf.belonging as Drugstore_type  FROM Visits_Effectiveness.ML_LPUSales_AllBrands s
LEFT JOIN Visits_Effectiveness.ML_VisitsBySKU_All v
ON s.`year` = v.`year` AND s.`month` = v.`month` AND s.organization_id = v.organization_id AND s.sku = v.Brand 
LEFT JOIN  CRM.Organizations o 
ON s.organization_id = o.id 
LEFT JOIN MDLP.Branches b 
ON b.id = s.Drugstore_id 
LEFT JOIN MDLP.BELONGING_FULL bf 
ON bf.INN =  b.inn AND bf.`year` = s.year AND bf.`month` = s.month

                            ''')
	
print(f' inserted')