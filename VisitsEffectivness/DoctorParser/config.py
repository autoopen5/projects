# config.py — единственный файл который нужно редактировать

CH = {
    "host":     "clickhouse.moscow",   # или IP сервера
    "port":     8123,
    "database": "NSI",
    "user":     "GrushkoIV",
    "password": "jNbrvzd1IcF0Yx5I",
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

TBL_QUEUE     = "grushko_iv.pipeline_queue"
TBL_DOCTORS   = "grushko_iv.doctors_parsed"

ANTHROPIC_API_KEY = "sk-ant-..."
ANTHROPIC_MODEL   = "claude-sonnet-4-20250514"

# 2GIS API — получить на dev.2gis.ru (бесплатно, 25 000 запросов/день)
DGIS_API_KEY = "5da76007-847e-4b33-b07e-780da4047899"

# Яндекс Cloud Search API — console.yandex.cloud
# 1. IAM → Сервисные аккаунты → создать → роль search-api.webSearch.user
# 2. Сервисный аккаунт → API-ключи → создать
# 3. Folder ID из URL консоли
YANDEX_SEARCH_KEY    = "AQVNx6L2x-qkgdhEc8Ard-C-lyVl85uI5m3PcVsV"
YANDEX_SEARCH_FOLDER = "b1gn7gbculvpn3qbjs2t"

# Сервисный аккаунт ai-doctor-parser (роль ai.languageModels.user)
YANDEX_GPT_KEY = "AQVN00qqSACarUFubiAqAj1Tng3qZ_XMc6duDSHl"

WORKERS         = 2   # поиск чувствителен к rate-limit; для парсинга можно поднять до 5
SEARCH_DELAY_S  = 2.0
FETCH_DELAY_S   = 1.5
FETCH_TIMEOUT_S = 15
MAX_RETRIES     = 3
BATCH_SIZE      = 500
FILTER_REGIONS  = None   # или ["Свердловская область"]
