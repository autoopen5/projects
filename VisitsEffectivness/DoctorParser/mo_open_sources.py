"""
ТФОМС Scraper — сбор реестров врачей из открытых данных территориальных ФОМС
Источник: региональные порталы ТФОМС публикуют реестры медработников,
          прикреплённых к системе ОМС, в открытом доступе (XLS/CSV/XML).

Также: Яндекс.Карты / 2GIS — легальный доступ через официальные API
       (требуют регистрации, бесплатный тариф есть).

Стратегия для MVP:
1. Реестры ТФОМС → врачи с специальностью + ЛПУ (легально, бесплатно)
2. 2GIS Places API  → координаты + профиль ЛПУ (официальный API)
3. Объединение с CRM клиента → финальная база для скоринга
"""

import requests
import pandas as pd
import time
import logging
from pathlib import Path
from io import BytesIO
from typing import Optional

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger(__name__)

OUTPUT_DIR = Path("output")
OUTPUT_DIR.mkdir(exist_ok=True)


# ─────────────────────────────────────────────
# ИСТОЧНИК 1: 2GIS Places API (официальный)
#
# Регистрация: https://dev.2gis.ru/
# Бесплатный план: 25 000 запросов/сутки
# ─────────────────────────────────────────────

DGIS_PLACES_URL = "https://catalog.api.2gis.com/3.0/items"

MEDICAL_RUBRIC_IDS = {
    "polikliniki":    "159",
    "bolnitsy":       "160",
    "dispansery":     "161",
    "med_centry":     "169",
    "kardiologiya":   "1713",
    "nevrologiya":    "1714",
    "onkologiya":     "1715",
    "endokrinologiya":"1719",
}

def fetch_2gis_clinics(
    api_key: str,
    city: str,
    rubric_id: str = "159",
    max_results: int = 1000,
    delay: float = 0.5,
) -> pd.DataFrame:
    records = []
    page = 1
    page_size = 50

    while len(records) < max_results:
        params = {
            "key":       api_key,
            "q":         f"поликлиника {city}",
            "city_id":   _get_city_id(city),
            "rubric_id": rubric_id,
            "fields":    "items.point,items.address,items.contact_groups,items.rubrics",
            "page":      page,
            "page_size": page_size,
            "locale":    "ru_RU",
        }

        try:
            time.sleep(delay)
            resp = requests.get(DGIS_PLACES_URL, params=params, timeout=15)
            resp.raise_for_status()
            data = resp.json()
        except Exception as e:
            log.error(f"2GIS API ошибка (стр. {page}): {e}")
            break

        items = data.get("result", {}).get("items", [])
        if not items:
            break

        for item in items:
            point = item.get("point", {})
            address = item.get("address_name", "")
            contacts = item.get("contact_groups", [{}])[0].get("contacts", [])
            phone = next((c["value"] for c in contacts if c.get("type") == "phone"), "")
            rubrics = [r.get("name", "") for r in item.get("rubrics", [])]

            records.append({
                "mo_name":  item.get("name", ""),
                "address":  address,
                "city":     city,
                "lat":      point.get("lat"),
                "lon":      point.get("lon"),
                "phone":    phone,
                "url":      item.get("url", ""),
                "rubric":   ", ".join(rubrics),
                "dgis_id":  item.get("id", ""),
                "source":   "2gis_api",
            })

        log.info(f"2GIS {city}: страница {page}, загружено {len(records)} МО")
        page += 1

        total = data.get("result", {}).get("total", 0)
        if len(records) >= total:
            break

    df = pd.DataFrame(records)
    log.info(f"2GIS: итого {len(df)} МО для '{city}'")
    return df


def _get_city_id(city: str) -> str:
    CITY_IDS = {
        "Москва":          "4504222397521793",
        "Санкт-Петербург": "4591396063138816",
        "Новосибирск":     "4504222397521796",
        "Екатеринбург":    "4504222397521797",
        "Казань":          "4504222397521798",
        "Нижний Новгород": "4504222397521801",
        "Челябинск":       "4504222397521805",
        "Самара":          "4504222397521806",
        "Уфа":             "4504222397521807",
        "Ростов-на-Дону":  "4504222397521811",
    }
    return CITY_IDS.get(city, "")


# ─────────────────────────────────────────────
# ИСТОЧНИК 2: ТФОМС — открытые реестры врачей
# ─────────────────────────────────────────────

TFOMS_SOURCES = {
    "moscow": {
        "url": "https://www.mgfoms.ru/sites/default/files/reestr-mo.xlsx",
        "type": "xlsx",
        "description": "Реестр МО МГФОМС (Москва)",
    },
    "spb": {
        "url": "https://www.spboms.ru/kiop/main?page=opendata_list",
        "type": "html_links",
        "description": "Открытые данные ТФОМС СПб",
    },
    "novosibirsk": {
        "url": "https://www.nso-oms.ru/opendata/",
        "type": "html_links",
        "description": "Открытые данные ТФОМС НСО",
    },
}

def fetch_tfoms_xlsx(url: str, source_name: str) -> pd.DataFrame:
    headers = {
        "User-Agent": "MO-Research-Bot/1.0 (contact@example.com)",
    }

    try:
        log.info(f"Скачиваем {source_name}: {url}")
        resp = requests.get(url, headers=headers, timeout=60)
        resp.raise_for_status()

        df = pd.read_excel(BytesIO(resp.content), dtype=str)
        df.columns = df.columns.str.strip().str.lower()

        col_mapping = {}
        for col in df.columns:
            if any(k in col for k in ["наименов", "название", "организаци"]):
                col_mapping[col] = "mo_name"
            elif any(k in col for k in ["адрес", "адрес мо"]):
                col_mapping[col] = "address"
            elif any(k in col for k in ["специальн"]):
                col_mapping[col] = "specialty"
            elif any(k in col for k in ["врач", "фио", "работник"]):
                col_mapping[col] = "doctor_name"
            elif any(k in col for k in ["огрн"]):
                col_mapping[col] = "ogrn"

        df = df.rename(columns=col_mapping)
        df["source"] = source_name
        log.info(f"{source_name}: загружено {len(df)} строк, колонки: {list(df.columns)}")
        return df

    except Exception as e:
        log.error(f"{source_name}: ошибка — {e}")
        return pd.DataFrame()


# ─────────────────────────────────────────────
# ИСТОЧНИК 3: Яндекс Геокодер API (официальный)
# ─────────────────────────────────────────────

YANDEX_GEOCODER_URL = "https://geocode-maps.yandex.ru/1.x/"

def geocode_yandex(
    address: str,
    api_key: str,
    delay: float = 0.1,
) -> tuple[Optional[float], Optional[float]]:
    params = {
        "apikey":  api_key,
        "geocode": address,
        "format":  "json",
        "results": 1,
        "lang":    "ru_RU",
    }

    try:
        time.sleep(delay)
        resp = requests.get(YANDEX_GEOCODER_URL, params=params, timeout=10)
        resp.raise_for_status()
        data = resp.json()

        members = (
            data.get("response", {})
                .get("GeoObjectCollection", {})
                .get("featureMember", [])
        )
        if not members:
            return None, None

        pos = members[0]["GeoObject"]["Point"]["pos"]
        lon, lat = map(float, pos.split())
        return lat, lon

    except Exception as e:
        log.warning(f"Яндекс геокодер: {address[:40]}... — {e}")
        return None, None


# ─────────────────────────────────────────────
# MAIN: объединённый пайплайн
# ─────────────────────────────────────────────

def run_open_data_pipeline(
    cities: list[str],
    dgis_api_key: Optional[str] = None,
    yandex_api_key: Optional[str] = None,
):
    all_dfs = []

    if dgis_api_key:
        for city in cities:
            for rubric_name, rubric_id in list(MEDICAL_RUBRIC_IDS.items())[:3]:
                df = fetch_2gis_clinics(dgis_api_key, city, rubric_id)
                all_dfs.append(df)
                time.sleep(1)

    for source_name, source_info in TFOMS_SOURCES.items():
        if source_info["type"] == "xlsx":
            df = fetch_tfoms_xlsx(source_info["url"], source_name)
            if not df.empty:
                all_dfs.append(df)

    if not all_dfs:
        log.error("Нет данных")
        return pd.DataFrame()

    df_combined = pd.concat(all_dfs, ignore_index=True)
    df_combined = df_combined.drop_duplicates(
        subset=[c for c in ["mo_name", "address"] if c in df_combined.columns]
    )

    if yandex_api_key and "address" in df_combined.columns:
        log.info("Геокодирование через Яндекс API...")
        lats, lons = [], []
        for addr in df_combined["address"]:
            lat, lon = geocode_yandex(str(addr), yandex_api_key)
            lats.append(lat)
            lons.append(lon)
        df_combined["lat"] = lats
        df_combined["lon"] = lons

    out = OUTPUT_DIR / "mo_open_data.csv"
    df_combined.to_csv(out, index=False, encoding="utf-8-sig")
    log.info(f"✓ Сохранено: {out} ({len(df_combined)} МО)")
    return df_combined


if __name__ == "__main__":
    run_open_data_pipeline(
        cities=["Москва", "Санкт-Петербург"],
        dgis_api_key="ВАШ_КЛЮЧ_2GIS",
        yandex_api_key="ВАШ_КЛЮЧ_ЯНДЕКС",
    )
