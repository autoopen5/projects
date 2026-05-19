"""
pipeline.py — основной оркестратор.

Запуск:
  python pipeline.py              # полный прогон
  python pipeline.py --stats      # только статистика
  python pipeline.py --retry      # повторить url_failed / parse_failed
  python pipeline.py --limit 100  # обработать только N МО
"""
import argparse
import logging
import re
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone

import config
import db
from searcher import find_mo_site
from extractor import parse_doctors_from_site

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)-8s %(message)s",
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler("pipeline.log", encoding="utf-8"),
    ],
)
log = logging.getLogger(__name__)


# ─────────────────────────────────────────────
# Обработка одной МО
# ─────────────────────────────────────────────

def process_mo(row: dict) -> dict:
    """
    Полный цикл для одной МО:
      pending → url_found → parsed
    """
    mo_id   = row["mo_id"]
    mo_name = row["mo_name"]

    try:
        # ── Шаг 1: поиск сайта ───────────────────────────────
        site_url = row.get("site_url", "").strip()

        if not site_url:
            site_url = find_mo_site(
                mo_id   = mo_id,
                mo_name = mo_name,
                inn     = row.get("mo_inn", ""),
                ogrn    = row.get("mo_ogrn", ""),
                city    = row.get("mo_city", ""),
                region  = row.get("mo_region", ""),
            )

        if not site_url:
            db.update_queue_status(
                mo_id, "url_failed",
                error_msg="Сайт не найден в поиске"
            )
            return {"mo_id": mo_id, "status": "url_failed"}

        db.update_queue_status(mo_id, "url_found", site_url=site_url)

        # ── Шаг 2: парсинг врачей ────────────────────────────
        doctors = parse_doctors_from_site(mo_id, site_url)

        if not doctors:
            db.update_queue_status(
                mo_id, "no_doctors",
                site_url=site_url,
                error_msg="Врачи не найдены на сайте"
            )
            return {"mo_id": mo_id, "status": "no_doctors"}

        # ── Шаг 3: запись в БД ───────────────────────────────
        now = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S")
        rows = [
            {
                "mo_id":       mo_id,
                "mo_name":     mo_name,
                "mo_region":   row.get("mo_region", ""),
                "site_url":    site_url,
                "doctor_name": d["doctor_name"],
                "specialty":   d["specialty"],
                "category":    d["category"],
                "education":   d["education"],
                "experience":  d["experience"],
                "raw_text":    "",
                "parsed_at":   now,
            }
            for d in doctors
        ]
        db.insert_doctors(rows)
        doctors_cnt = len(rows)

        db.update_queue_status(
            mo_id, "parsed",
            site_url=site_url,
            doctors_cnt=doctors_cnt
        )
        return {"mo_id": mo_id, "status": "parsed", "cnt": doctors_cnt}

    except Exception as e:
        msg = re.sub(r"[^\x20-\x7EЀ-ӿ\s]", "?", str(e))[:200]
        log.error(f"[{mo_id}] Необработанная ошибка: {msg}")
        db.update_queue_status(mo_id, "parse_failed", error_msg=msg)
        return {"mo_id": mo_id, "status": "error", "error": msg}


# ─────────────────────────────────────────────
# Основной цикл
# ─────────────────────────────────────────────

def run(limit: int | None = None, retry: bool = False):
    log.info("=== Pipeline start ===")
    db.init_tables()

    added = db.load_mo_into_queue(
        nsi_table  = config.NSI_TABLE,
        col_id     = config.COL_MO_ID,
        col_name   = config.COL_MO_NAME,
        col_inn    = config.COL_MO_INN,
        col_ogrn   = config.COL_MO_OGRN,
        col_region = config.COL_MO_REGION,
        col_city   = config.COL_MO_CITY,
        batch_size = config.BATCH_SIZE,
        filter_regions = config.FILTER_REGIONS,
        extra_filter = config.NSI_EXTRA_FILTER,
    )
    log.info(f"Добавлено в очередь: {added} МО")

    statuses = ["url_failed", "parse_failed", "no_doctors", "url_found"] if retry else ["url_found", "pending"]
    fetch_n  = limit or config.BATCH_SIZE

    rows = []
    for status in statuses:
        rows += db.fetch_pending(fetch_n - len(rows), status=status)
        if len(rows) >= fetch_n:
            break

    if not rows:
        log.info("Нет МО для обработки. Завершено.")
        print_stats()
        return

    log.info(f"Обрабатываем {len(rows)} МО в {config.WORKERS} потоков")

    ok = fail = no_url = no_doc = 0
    t0 = time.time()

    with ThreadPoolExecutor(max_workers=config.WORKERS) as pool:
        futures = {pool.submit(process_mo, row): row for row in rows}
        for i, future in enumerate(as_completed(futures), 1):
            res = future.result()
            s = res.get("status", "")
            if s == "parsed":
                ok += 1
                log.info(f"[{i}/{len(rows)}] ✓ {res['mo_id']} — {res['cnt']} врачей")
            elif s == "url_failed":
                no_url += 1
            elif s == "no_doctors":
                no_doc += 1
            else:
                fail += 1

            if i % 50 == 0:
                elapsed = time.time() - t0
                rps = i / elapsed
                log.info(
                    f"Прогресс: {i}/{len(rows)} | "
                    f"✓{ok} ✗{fail} no_url:{no_url} no_doc:{no_doc} | "
                    f"{rps:.1f} МО/с"
                )

    elapsed = time.time() - t0
    log.info(
        f"=== Готово за {elapsed:.0f}с | "
        f"parsed:{ok} no_url:{no_url} no_doc:{no_doc} fail:{fail} ==="
    )
    print_stats()


def print_stats():
    stats = db.queue_stats()
    print("\n── Статистика очереди ──────────────────")
    total = sum(stats.values())
    for status, cnt in stats.items():
        pct = cnt / total * 100 if total else 0
        print(f"  {status:<15} {cnt:>7,}  ({pct:.1f}%)")
    print(f"  {'ИТОГО':<15} {total:>7,}")
    print("────────────────────────────────────────\n")


# ─────────────────────────────────────────────
# CLI
# ─────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Doctor pipeline")
    parser.add_argument("--stats",  action="store_true", help="Только статистика")
    parser.add_argument("--retry",  action="store_true", help="Повторить failed")
    parser.add_argument("--limit",  type=int, default=None, help="Кол-во МО")
    args = parser.parse_args()

    if args.stats:
        db.init_tables()
        print_stats()
    else:
        run(limit=args.limit, retry=args.retry)
