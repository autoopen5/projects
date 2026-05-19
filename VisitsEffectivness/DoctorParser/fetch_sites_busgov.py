"""
fetch_sites_busgov.py — массовая загрузка сайтов МО из bus.gov.ru.

bus.gov.ru — официальный реестр государственных и муниципальных учреждений РФ.
Содержит официальные сайты, ИНН, ОГРН. Открытый API, без авторизации.

Запуск ОТДЕЛЬНО от основного пайплайна — заполняет site_url в pipeline_queue.
После этого pipeline.py пропускает поиск и сразу парсит врачей.

python fetch_sites_busgov.py --test     # тест на 5 МО
python fetch_sites_busgov.py --limit 1000
python fetch_sites_busgov.py            # все МО из очереди
"""
import argparse
import json
import logging
import sys
import time
import requests

# Корпоративный SSL-прокси — отключаем верификацию сертификата
import urllib3
urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)

import db
from config import FETCH_TIMEOUT_S

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)-8s %(message)s",
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler("fetch_sites.log", encoding="utf-8"),
    ],
)
log = logging.getLogger(__name__)

HEADERS = {
    "User-Agent": "Mozilla/5.0 (compatible; MO-Research/1.0)",
    "Accept": "application/json",
}

BUS_SEARCH = "https://bus.gov.ru/pub/agency/search.json"
BUS_DETAIL = "https://bus.gov.ru/pub/agency/{agency_id}.json"


def search_bus_gov(inn: str = "", ogrn: str = "", name: str = "") -> list[dict]:
    """Ищет учреждение в bus.gov.ru, возвращает список результатов."""
    for query in filter(None, [inn, ogrn, name[:80] if name else ""]):
        try:
            time.sleep(0.5)
            resp = requests.get(
                BUS_SEARCH,
                params={"searchString": query, "page": 0, "size": 5},
                headers=HEADERS,
                timeout=FETCH_TIMEOUT_S,
                verify=False,
            )
            if resp.status_code != 200:
                log.debug(f"bus.gov.ru HTTP {resp.status_code} для '{query}'")
                continue

            data = resp.json()
            items = (
                data.get("agencies")
                or data.get("content")
                or data.get("data")
                or []
            )
            if items:
                return items

        except requests.Timeout:
            log.warning(f"bus.gov.ru таймаут для '{query}'")
        except Exception as e:
            log.warning(f"bus.gov.ru ошибка: {e}")

    return []


def extract_site(agency: dict) -> str:
    """Извлекает URL сайта из записи bus.gov.ru."""
    for field in ["site", "siteUrl", "webSite", "websiteUrl", "url"]:
        val = agency.get(field, "")
        if val and isinstance(val, str):
            val = val.strip().rstrip("/")
            if val and val not in ("—", "-", "нет", "отсутствует"):
                if not val.startswith("http"):
                    val = "https://" + val
                return val
    return ""


def process_batch(rows: list[dict]) -> dict:
    """Обрабатывает пачку МО, обновляет site_url в очереди."""
    found = 0
    not_found = 0

    for i, row in enumerate(rows, 1):
        mo_id   = row["mo_id"]
        mo_name = row["mo_name"]
        inn     = row.get("mo_inn", "")
        ogrn    = row.get("mo_ogrn", "")

        agencies = search_bus_gov(inn=inn, ogrn=ogrn, name=mo_name)

        site_url = ""
        for ag in agencies:
            site_url = extract_site(ag)
            if site_url:
                break

        if site_url:
            db.update_queue_status(mo_id, "url_found", site_url=site_url)
            log.info(f"[{i}/{len(rows)}] ✓ [{mo_id}] {site_url}")
            found += 1
        else:
            db.update_queue_status(mo_id, "url_failed",
                                   error_msg="Не найден в bus.gov.ru")
            log.info(f"[{i}/{len(rows)}] ✗ [{mo_id}] {mo_name[:50]}")
            not_found += 1

        if i % 50 == 0:
            log.info(f"--- Прогресс: {i}/{len(rows)} | найдено: {found} ---")

    return {"found": found, "not_found": not_found}


def run(limit: int | None = None, test: bool = False):
    db.init_tables()

    fetch_n = 5 if test else (limit or 10_000)

    rows = db.fetch_pending(fetch_n, status="pending")
    if not rows:
        log.info("Нет МО для обработки (статус pending). Запусти pipeline.py сначала.")
        return

    log.info(f"Загружаем сайты для {len(rows)} МО из bus.gov.ru...")
    result = process_batch(rows)

    log.info(
        f"Готово: найдено сайтов {result['found']}, "
        f"не найдено {result['not_found']}"
    )
    log.info(
        "Теперь запускай: python pipeline.py --limit 500\n"
        "(pipeline пропустит поиск сайта и сразу будет парсить врачей)"
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--test",  action="store_true", help="Тест на 5 МО")
    parser.add_argument("--limit", type=int, default=None)
    args = parser.parse_args()
    run(limit=args.limit, test=args.test)
