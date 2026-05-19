"""
МО Collector — сбор базы медицинских организаций из легальных открытых источников
Источники:
  1. Реестр лицензий Росздравнадзора (roszdravnadzor.gov.ru)
  2. Реестр МО из открытых данных data.gov.ru
  3. Геокодирование адресов через Nominatim (OSM, бесплатно)

Результат: CSV с полями:
  mo_id, full_name, short_name, address, region, city,
  license_number, license_date, activity_types,
  lat, lon, source
"""

import requests
import pandas as pd
import time
import json
import re
import logging
from pathlib import Path
from typing import Optional
from urllib.parse import urlencode

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger(__name__)

OUTPUT_DIR = Path("output")
OUTPUT_DIR.mkdir(exist_ok=True)

# ─────────────────────────────────────────────
# 1. РЕЕСТР ЛИЦЕНЗИЙ РОСЗДРАВНАДЗОРА
# ─────────────────────────────────────────────

RZN_BASE = "https://roszdravnadzor.gov.ru/api/licenses"

def fetch_rzn_licenses(
    activity_type: str = "медицинская деятельность",
    region_code: Optional[str] = None,
    max_pages: int = 50,
    delay: float = 2.0,
) -> pd.DataFrame:
    records = []
    page = 1

    headers = {
        "User-Agent": "MO-Research-Bot/1.0 (pharma-analytics; contact@example.com)",
        "Accept": "application/json",
    }

    while page <= max_pages:
        params = {
            "page": page,
            "per_page": 100,
            "activity": activity_type,
        }
        if region_code:
            params["region"] = region_code

        try:
            resp = requests.get(RZN_BASE, params=params, headers=headers, timeout=30)
            resp.raise_for_status()
            data = resp.json()
        except requests.RequestException as e:
            log.error(f"Страница {page}: ошибка запроса — {e}")
            break
        except json.JSONDecodeError:
            log.error(f"Страница {page}: не JSON в ответе")
            break

        items = data.get("items") or data.get("data") or []
        if not items:
            log.info(f"Страница {page}: данных нет, завершаем.")
            break

        for item in items:
            records.append({
                "mo_id":          item.get("id") or item.get("license_id"),
                "full_name":      item.get("organization_name") or item.get("full_name"),
                "short_name":     item.get("short_name", ""),
                "ogrn":           item.get("ogrn", ""),
                "inn":            item.get("inn", ""),
                "address":        item.get("address") or item.get("legal_address", ""),
                "region":         item.get("region", ""),
                "license_number": item.get("license_number", ""),
                "license_date":   item.get("license_date", ""),
                "license_status": item.get("status", ""),
                "activity_types": "; ".join(item.get("activities", [])),
                "source":         "roszdravnadzor",
            })

        log.info(f"Страница {page}: загружено {len(items)} записей (всего {len(records)})")
        page += 1
        time.sleep(delay)

    df = pd.DataFrame(records)
    log.info(f"Росздравнадзор: итого {len(df)} МО")
    return df


# ─────────────────────────────────────────────
# 2. ОТКРЫТЫЕ ДАННЫЕ data.gov.ru
# ─────────────────────────────────────────────

DATAGOV_URL = (
    "https://data.gov.ru/api/json/dataset"
    "/7707778246-medicalorganizations/version/20230101/content"
)

def fetch_datagov_mo(delay: float = 1.0) -> pd.DataFrame:
    headers = {"User-Agent": "MO-Research-Bot/1.0 (contact@example.com)"}

    try:
        log.info("Скачиваем датасет МО с data.gov.ru...")
        resp = requests.get(DATAGOV_URL, headers=headers, timeout=60)
        resp.raise_for_status()
        items = resp.json()
    except Exception as e:
        log.error(f"data.gov.ru: ошибка — {e}")
        return pd.DataFrame()

    records = []
    for item in items:
        records.append({
            "mo_id":      item.get("global_id") or item.get("ID"),
            "full_name":  item.get("FullName") or item.get("ShortName"),
            "short_name": item.get("ShortName", ""),
            "ogrn":       item.get("OGRN", ""),
            "inn":        item.get("INN", ""),
            "address":    item.get("Address", ""),
            "region":     item.get("RegionName") or item.get("Region", ""),
            "mo_type":    item.get("TypeName", ""),
            "mo_profile": item.get("Profile", ""),
            "beds":       item.get("Beds"),
            "source":     "datagov",
        })

    df = pd.DataFrame(records)
    log.info(f"data.gov.ru: загружено {len(df)} МО")
    return df


# ─────────────────────────────────────────────
# 3. ГЕОКОДИРОВАНИЕ через Nominatim (OSM)
# ─────────────────────────────────────────────

NOMINATIM_URL = "https://nominatim.openstreetmap.org/search"

def geocode_address(address: str, delay: float = 1.1) -> tuple[Optional[float], Optional[float]]:
    params = {
        "q":              address + ", Россия",
        "format":         "json",
        "limit":          1,
        "addressdetails": 0,
        "countrycodes":   "ru",
    }
    headers = {"User-Agent": "MO-Research-Bot/1.0 (contact@example.com)"}

    try:
        time.sleep(delay)
        resp = requests.get(NOMINATIM_URL, params=params, headers=headers, timeout=10)
        resp.raise_for_status()
        results = resp.json()
        if results:
            return float(results[0]["lat"]), float(results[0]["lon"])
    except Exception as e:
        log.warning(f"Geocode fail: {address[:50]}... — {e}")

    return None, None


def geocode_dataframe(
    df: pd.DataFrame,
    address_col: str = "address",
    batch_size: int = 500,
    cache_file: str = "output/geocode_cache.json",
) -> pd.DataFrame:
    cache_path = Path(cache_file)
    cache: dict = {}
    if cache_path.exists():
        cache = json.loads(cache_path.read_text(encoding="utf-8"))
        log.info(f"Геокэш загружен: {len(cache)} адресов")

    lats, lons = [], []
    new_geocodes = 0

    for i, addr in enumerate(df[address_col]):
        addr_str = str(addr).strip()

        if not addr_str or addr_str in ("nan", ""):
            lats.append(None)
            lons.append(None)
            continue

        if addr_str in cache:
            lat, lon = cache[addr_str]
            lats.append(lat)
            lons.append(lon)
            continue

        lat, lon = geocode_address(addr_str)
        cache[addr_str] = [lat, lon]
        lats.append(lat)
        lons.append(lon)
        new_geocodes += 1

        if new_geocodes % batch_size == 0:
            cache_path.write_text(json.dumps(cache, ensure_ascii=False, indent=2), encoding="utf-8")
            log.info(f"  Геокэш сохранён ({new_geocodes} новых). Прогресс: {i+1}/{len(df)}")

    cache_path.write_text(json.dumps(cache, ensure_ascii=False, indent=2), encoding="utf-8")

    df = df.copy()
    df["lat"] = lats
    df["lon"] = lons
    df["geocoded"] = df["lat"].notna()
    geocode_rate = df["geocoded"].mean() * 100
    log.info(f"Геокодирование: {geocode_rate:.1f}% адресов успешно ({new_geocodes} новых запросов)")
    return df


# ─────────────────────────────────────────────
# 4. ФИЛЬТРАЦИЯ
# ─────────────────────────────────────────────

TARGET_MO_KEYWORDS = [
    "поликлиник", "больниц", "диспансер", "медицинский центр",
    "клиник", "амбулатор", "госпиталь", "санатор",
]

EXCLUDE_MO_KEYWORDS = [
    "аптек", "лаборатор", "скорая", "станция переливания",
    "судебно-медицин", "патологоанатом", "дом ребёнка",
]

def filter_target_mo(df: pd.DataFrame, name_col: str = "full_name") -> pd.DataFrame:
    df = df.copy()
    name_lower = df[name_col].str.lower().fillna("")

    include_mask = name_lower.str.contains("|".join(TARGET_MO_KEYWORDS), regex=True)
    exclude_mask = name_lower.str.contains("|".join(EXCLUDE_MO_KEYWORDS), regex=True)

    result = df[include_mask & ~exclude_mask].copy()
    log.info(f"Фильтрация МО: {len(df)} → {len(result)} целевых")
    return result


# ─────────────────────────────────────────────
# 5. MAIN PIPELINE
# ─────────────────────────────────────────────

def run_pipeline(
    regions: Optional[list[str]] = None,
    geocode: bool = True,
    max_pages_rzn: int = 100,
):
    dfs = []

    log.info("=== Источник 1: Росздравнадзор ===")
    if regions:
        for reg in regions:
            df_rzn = fetch_rzn_licenses(region_code=reg, max_pages=max_pages_rzn)
            dfs.append(df_rzn)
    else:
        df_rzn = fetch_rzn_licenses(max_pages=max_pages_rzn)
        dfs.append(df_rzn)

    log.info("=== Источник 2: data.gov.ru ===")
    df_gov = fetch_datagov_mo()
    if not df_gov.empty:
        dfs.append(df_gov)

    if not dfs:
        log.error("Нет данных ни из одного источника.")
        return

    df_all = pd.concat(dfs, ignore_index=True)
    log.info(f"Итого до дедупликации: {len(df_all)} записей")

    if "ogrn" in df_all.columns:
        df_all["ogrn"] = df_all["ogrn"].astype(str).str.strip()
        has_ogrn = df_all["ogrn"].str.match(r"^\d{13}$")
        df_dedup = pd.concat([
            df_all[has_ogrn].drop_duplicates(subset=["ogrn"]),
            df_all[~has_ogrn].drop_duplicates(subset=["full_name", "address"]),
        ]).reset_index(drop=True)
    else:
        df_dedup = df_all.drop_duplicates(subset=["full_name", "address"]).reset_index(drop=True)

    log.info(f"После дедупликации: {len(df_dedup)} МО")

    df_filtered = filter_target_mo(df_dedup)

    if geocode:
        log.info("=== Геокодирование (Nominatim, ~1 запрос/сек) ===")
        df_final = geocode_dataframe(df_filtered)
    else:
        df_final = df_filtered

    out_path = OUTPUT_DIR / "mo_database.csv"
    df_final.to_csv(out_path, index=False, encoding="utf-8-sig")
    log.info(f"✓ Сохранено: {out_path} ({len(df_final)} МО)")

    print("\n=== СТАТИСТИКА ===")
    print(f"Всего МО: {len(df_final)}")
    if "geocoded" in df_final.columns:
        print(f"Геокодировано: {df_final['geocoded'].sum()} ({df_final['geocoded'].mean()*100:.1f}%)")
    if "region" in df_final.columns:
        print(f"\nТоп-10 регионов:")
        print(df_final["region"].value_counts().head(10).to_string())

    return df_final


if __name__ == "__main__":
    df = run_pipeline(
        regions=["45000000", "40000000"],
        geocode=False,
        max_pages_rzn=5,
    )
