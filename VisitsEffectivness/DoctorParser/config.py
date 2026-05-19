# config.py — единственный файл который нужно редактировать

CH = {
    "host":     "localhost",   # или IP сервера
    "port":     8123,
    "database": "NSI",
    "user":     "default",
    "password": "",
}

NSI_TABLE     = "NSI.reestr_med_org"

# Колонки под реальную схему reestr_med_org
COL_MO_ID     = "id"
COL_MO_NAME   = "nameFull"
COL_MO_INN    = "inn"
COL_MO_OGRN   = "ogrn"
COL_MO_REGION = "regionName"
COL_MO_CITY   = "areaName"

# Фильтр: не брать удалённые МО (deleteDate IS NOT NULL = удалена)
NSI_EXTRA_FILTER = "AND deleteDate IS NULL"

TBL_QUEUE     = "NSI.pipeline_queue"
TBL_DOCTORS   = "NSI.doctors_parsed"

ANTHROPIC_API_KEY = "sk-ant-..."
ANTHROPIC_MODEL   = "claude-sonnet-4-20250514"

# 2GIS API — получить на dev.2gis.ru (бесплатно, 25 000 запросов/день)
DGIS_API_KEY = "ВАШ_КЛЮЧ_2GIS"

WORKERS         = 2   # поиск чувствителен к rate-limit; для парсинга можно поднять до 5
SEARCH_DELAY_S  = 2.0
FETCH_DELAY_S   = 1.5
FETCH_TIMEOUT_S = 15
MAX_RETRIES     = 3
BATCH_SIZE      = 500
FILTER_REGIONS  = None   # или ["Свердловская область"]
