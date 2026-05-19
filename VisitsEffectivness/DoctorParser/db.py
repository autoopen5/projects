"""
db.py — ClickHouse клиент через HTTP API.
Не требует clickhouse-driver, работает через requests.
"""
import json
import logging
import requests

# Корпоративный SSL-прокси — отключаем верификацию сертификата
import urllib3
urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)
from datetime import datetime, timezone
from config import CH, TBL_QUEUE, TBL_DOCTORS

log = logging.getLogger(__name__)


def ch_query(sql: str, data=None) -> requests.Response:
    url = f"http://{CH['host']}:{CH['port']}/"
    params = {
        "database": CH["database"],
        "user":     CH["user"],
        "password": CH["password"],
        "query":    sql,
    }
    resp = requests.post(url, params=params, data=data, timeout=30, verify=False)
    if resp.status_code != 200:
        raise RuntimeError(f"ClickHouse error [{resp.status_code}]: {resp.text[:300]}")
    return resp


def ch_select(sql: str) -> list[dict]:
    resp = ch_query(sql + " FORMAT JSONEachRow")
    if not resp.text.strip():
        return []
    return [json.loads(line) for line in resp.text.strip().splitlines()]


def ch_execute(sql: str):
    ch_query(sql)


# ── DDL ───────────────────────────────────────────────────────

DDL_QUEUE = f"""
CREATE TABLE IF NOT EXISTS {TBL_QUEUE} (
    mo_id        String,
    mo_name      String,
    mo_inn       String,
    mo_ogrn      String,
    mo_region    String,
    mo_city      String,
    site_url     String   DEFAULT '',
    status       String   DEFAULT 'pending',
    error_msg    String   DEFAULT '',
    doctors_cnt  Int32    DEFAULT 0,
    created_at   DateTime DEFAULT now(),
    updated_at   DateTime DEFAULT now()
) ENGINE = ReplacingMergeTree(updated_at)
ORDER BY mo_id
"""

DDL_DOCTORS = f"""
CREATE TABLE IF NOT EXISTS {TBL_DOCTORS} (
    mo_id        String,
    mo_name      String,
    mo_region    String,
    site_url     String,
    doctor_name  String,
    specialty    String,
    category     String   DEFAULT '',
    education    String   DEFAULT '',
    experience   String   DEFAULT '',
    parsed_at    DateTime DEFAULT now()
) ENGINE = MergeTree()
ORDER BY (mo_id, doctor_name)
PARTITION BY toYYYYMM(parsed_at)
"""


def init_tables():
    ch_execute(DDL_QUEUE)
    ch_execute(DDL_DOCTORS)
    log.info("Таблицы инициализированы")


# ── Очередь ───────────────────────────────────────────────────

def load_mo_into_queue(
    nsi_table: str,
    col_id: str, col_name: str, col_inn: str,
    col_ogrn: str, col_region: str, col_city: str,
    batch_size: int,
    filter_regions=None,
    extra_filter: str = "",
) -> int:
    region_filter = ""
    if filter_regions:
        quoted = ", ".join(f"'{r}'" for r in filter_regions)
        region_filter = f"AND {col_region} IN ({quoted})"

    sql = f"""
    INSERT INTO {TBL_QUEUE}
        (mo_id, mo_name, mo_inn, mo_ogrn, mo_region, mo_city,
         status, created_at, updated_at)
    SELECT
        toString({col_id}),
        toString(coalesce({col_name}, '')),
        toString(coalesce({col_inn},  '')),
        toString(coalesce({col_ogrn}, '')),
        toString(coalesce({col_region}, '')),
        toString(coalesce({col_city},   '')),
        'pending',
        now(),
        now()
    FROM {nsi_table}
    WHERE toString({col_id}) NOT IN (
        SELECT mo_id FROM {TBL_QUEUE}
    )
    {extra_filter}
    {region_filter}
    LIMIT {batch_size}
    """
    ch_execute(sql)

    cnt = ch_select(
        f"SELECT count() as n FROM {TBL_QUEUE} WHERE status = 'pending'"
    )
    return int(cnt[0]["n"]) if cnt else 0


def fetch_pending(limit: int, status: str = "pending") -> list[dict]:
    return ch_select(f"""
        SELECT mo_id, mo_name, mo_inn, mo_ogrn, mo_region, mo_city, site_url
        FROM {TBL_QUEUE}
        WHERE status = '{status}'
        LIMIT {limit}
    """)


def update_queue_status(mo_id: str, status: str, **kwargs):
    """Обновление через вставку новой версии (ReplacingMergeTree)."""
    now = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S")
    rows = ch_select(f"""
        SELECT * FROM {TBL_QUEUE}
        WHERE mo_id = '{mo_id}'
        ORDER BY updated_at DESC
        LIMIT 1
    """)
    if not rows:
        return
    row = rows[0]
    row.update(kwargs)
    row["status"]     = status
    row["updated_at"] = now

    def esc(v):
        return str(v).replace("'", "''")

    cols = ", ".join(row.keys())
    vals = ", ".join(f"'{esc(v)}'" for v in row.values())
    ch_execute(f"INSERT INTO {TBL_QUEUE} ({cols}) VALUES ({vals})")


def insert_doctors(doctors: list[dict]):
    if not doctors:
        return
    cols = list(doctors[0].keys())
    col_str = ", ".join(cols)

    def esc(v):
        return str(v).replace("'", "''")

    rows_sql = []
    for d in doctors:
        vals = ", ".join(f"'{esc(d.get(c, ''))}'" for c in cols)
        rows_sql.append(f"({vals})")

    ch_execute(
        f"INSERT INTO {TBL_DOCTORS} ({col_str}) VALUES {', '.join(rows_sql)}"
    )


def queue_stats() -> dict:
    rows = ch_select(f"""
        SELECT status, count() as cnt
        FROM {TBL_QUEUE}
        GROUP BY status
        ORDER BY cnt DESC
    """)
    return {r["status"]: int(r["cnt"]) for r in rows}
