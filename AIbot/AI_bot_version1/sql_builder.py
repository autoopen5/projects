# sql_builder.py
# -*- coding: utf-8 -*-
"""
Генератор безопасного SQL для одной таблицы MDLP.SHOW_Disposal_reports.

Особенности:
- Корректное кавычение таблиц (db.table) через quote_table
- Безопасное экранирование строк
- Нормализация дат: ISO, now()/today(), now()-N days, YYYY-MM (как месяц)
- НЕ добавляем дефолтный exit_type
- Дефолтный временной фильтр по exit_date добавляем ТОЛЬКО если НЕТ фильтров по date/year/month
- ilike = регистронезависимый подстрочный поиск (LIKE '%value%')
- Валидация AST (только SELECT)
"""

import re
from datetime import datetime, timedelta
from typing import Dict, List, Tuple, Optional

import sqlglot
from sqlglot import exp

from intent_schema import Intent


# ---------- Кавычки и экранирование ----------

def quote_ident(name: str) -> str:
    # Для колонок/алиасов (одна часть). Dotted-имена здесь не трогаем.
    if "." in name:
        return name
    if re.match(r"^[A-Za-z_][A-Za-z0-9_]*$", name):
        return name
    return "`" + name + "`"


def quote_table(name: str) -> str:
    # Для таблиц (db.table). Кавычим только проблемные части.
    parts = name.split(".")
    out = []
    for p in parts:
        if re.match(r"^[A-Za-z_][A-Za-z0-9_]*$", p):
            out.append(p)
        else:
            out.append("`" + p + "`")
    return ".".join(out)


def normalize_value_for_ch(value):
    # Строки -> '...' с экранированием; числа -> как есть.
    if isinstance(value, str):
        return "'" + value.replace("'", "''") + "'"
    if isinstance(value, (int, float)):
        return str(value)
    return str(value)


def escape_like(s: str) -> str:
    # Экранирует %, _ и \ для LIKE (в CH backslash — escape по умолчанию)
    return s.replace("\\", "\\\\").replace("%", "\\%").replace("_", "\\_")


# ---------- Даты ----------

def is_iso_date(s: str) -> bool:
    return bool(re.fullmatch(r"\d{4}-\d{2}-\d{2}", s or ""))


def is_year_month(s: str) -> bool:
    return bool(re.fullmatch(r"\d{4}-\d{2}", s or ""))


def is_first_of_month(s: str) -> bool:
    return bool(re.fullmatch(r"\d{4}-\d{2}-01", s or ""))


def coerce_date_expr(v) -> str:
    """
    Приводит значения к корректному выражению CH:
    - '2024-01-01' -> toDate('2024-01-01')
    - 'YYYY-MM'    -> toDate('YYYY-MM-01') (диапазон месяца разворачивается в build_sql для op='=')
    - now()/today(), now()-N days, today()-N days поддержаны
    - уже валидные выражения (toDate(...), INTERVAL, addDays(...)) — возвращаем как есть
    """
    if isinstance(v, (int, float)):
        return str(v)
    if not isinstance(v, str):
        return "toDate(" + normalize_value_for_ch(v) + ")"

    s = v.strip()
    s_low = s.lower()

    if is_year_month(s):
        return "toDate('" + s + "-01')"

    if is_iso_date(s):
        return "toDate('" + s + "')"

    if s_low in ("now", "now()"):
        return "toDate(now())"
    if s_low in ("today", "today()"):
        return "today()"

    m = re.fullmatch(r"(now|today)\(\)?\s*-\s*(\d+)\s*days?", s_low)
    if m:
        base = m.group(1)
        num = int(m.group(2))
        base_expr = "toDate(now())" if base == "now" else "today()"
        return base_expr + " - INTERVAL " + str(num) + " DAY"

    if ("adddays(" in s_low) or ("interval" in s_low) or ("todate(" in s_low):
        return s

    return "toDate('" + s + "')"


# ---------- Конфиг-вспомогалки ----------

def _time_bounds(intent: Intent, metrics_cfg: Dict) -> Tuple[str, str]:
    days = (metrics_cfg.get("defaults") or {}).get("time_filter_days", 365)
    end = datetime.utcnow().date()
    start = end - timedelta(days=days)
    return start.isoformat(), end.isoformat()


def _time_bucket(column: str, grain: Optional[str]) -> str:
    col = quote_ident(column)
    mapping = {
        None: col,
        "day": col,
        "week": "toStartOfWeek(" + col + ")",
        "month": "toStartOfMonth(" + col + ")",
        "quarter": "toStartOfQuarter(" + col + ")",
        "year": "toStartOfYear(" + col + ")",
    }
    return mapping.get(grain, col)


def _find_measure(name: str, cfg: Dict) -> Tuple[str, str]:
    m = (cfg.get("measures") or {}).get(name)
    if not m:
        raise ValueError("Unknown measure: " + str(name))
    return m["expr"], m["table"]


def _find_dimension(name: str, cfg: Dict) -> Tuple[str, str]:
    d = (cfg.get("dimensions") or {}).get(name)
    if not d:
        # синонимы
        for k, v in (cfg.get("dimensions") or {}).items():
            if name in (v.get("synonyms") or []):
                d = v
                break
    if not d:
        raise ValueError("Unknown dimension: " + str(name))
    return d["column"], d["table"]


def _has_explicit_period(intent: Intent) -> bool:
    """
    Считаем, что период задан явно, если в фильтрах есть поля date/year/month.
    """
    for f in (intent.filters or []):
        if f.field in ("date", "year", "month"):
            return True
    return False


# ---------- Основной билдер ----------

def build_sql(intent: Intent, metrics_cfg: Dict, schema_cfg: Dict) -> str:
    if not intent.measures:
        raise ValueError("At least one measure is required")

    m_expr, m_table = _find_measure(intent.measures[0], metrics_cfg)

    # SELECT / GROUP BY
    select_parts: List[str] = []
    group_by_parts: List[str] = []

    for dname in (intent.dimensions or []):
        col, table = _find_dimension(dname, metrics_cfg)
        if table != m_table:
            raise ValueError("Unexpected table for dimension '" + dname + "': " + table + " != " + m_table)
        if dname == "date" and intent.time_grain:
            bucket = _time_bucket(col, intent.time_grain)
            select_parts.append(bucket + " AS date")
            group_by_parts.append("date")
        else:
            select_parts.append(quote_ident(col) + " AS " + quote_ident(dname))
            group_by_parts.append(quote_ident(dname))

    for meas in intent.measures:
        expr, table = _find_measure(meas, metrics_cfg)
        if table != m_table:
            raise ValueError("Unexpected table for measure '" + meas + "': " + table + " != " + m_table)
        select_parts.append(expr + " AS " + quote_ident(meas))

    # WHERE: дефолт по времени только если нет date/year/month
    time_col = quote_ident((metrics_cfg.get("defaults") or {}).get("time_column", "exit_date"))
    where_clauses: List[str] = []

    if not _has_explicit_period(intent):
        start_iso, end_iso = _time_bounds(intent, metrics_cfg)
        where_clauses.append(
            time_col + " >= toDate('" + start_iso + "') AND " + time_col + " < toDate('" + end_iso + "')"
        )

    # Пользовательские фильтры
    for f in (intent.filters or []):
        try:
            col, _ = _find_dimension(f.field, metrics_cfg)
        except ValueError:
            continue

        qcol = quote_ident(col)
        op = f.op
        val = f.value

        # Спец-логика для даты
        if f.field == "date":
            if op == "between" and isinstance(val, list) and len(val) == 2:
                a = coerce_date_expr(val[0])
                b = coerce_date_expr(val[1])
                where_clauses.append(qcol + " >= " + a + " AND " + qcol + " <= " + b)
                continue
            if op == "=" and isinstance(val, str) and (is_year_month(val) or is_first_of_month(val)):
                base = "toDate('" + (val + "-01" if is_year_month(val) else val) + "')"
                where_clauses.append(qcol + " >= " + base + " AND " + qcol + " < addMonths(" + base + ", 1)")
                continue
            if op in (">", "<", ">=", "<="):
                a = coerce_date_expr(val)
                where_clauses.append(qcol + " " + op + " " + a)
                continue
            if op == "=":
                a = coerce_date_expr(val)
                where_clauses.append(qcol + " = " + a)
                continue

        # Общие случаи
        if op == "between" and isinstance(val, list) and len(val) == 2:
            left = normalize_value_for_ch(val[0])
            right = normalize_value_for_ch(val[1])
            where_clauses.append(qcol + " >= " + left + " AND " + qcol + " <= " + right)
        elif op == "in" and isinstance(val, list):
            items = ",".join(normalize_value_for_ch(v) for v in val)
            where_clauses.append(qcol + " IN (" + items + ")")
        elif op == "contains":
            where_clauses.append("positionCaseInsensitive(" + qcol + ", " + normalize_value_for_ch(val) + ") > 0")
        elif op == "ilike":
            pat = "'%" + escape_like(str(val).lower()) + "%'"
            where_clauses.append("lowerUTF8(" + qcol + ") LIKE " + pat)
        else:
            where_clauses.append(qcol + " " + op + " " + normalize_value_for_ch(val))

    # ORDER BY
    order_parts: List[str] = []
    for s in (intent.sort_by or []):
        order_parts.append(quote_ident(s.field) + " " + s.dir.upper())

    # LIMIT (safety)
    top_n = intent.top_n or (metrics_cfg.get("defaults") or {}).get("top_n", 50)
    top_n = max(1, min(int(top_n), 10000))

    sql = (
        "SELECT\n  " + ", ".join(select_parts) + "\n"
        "FROM " + quote_table(m_table) + "\n"
        + ("WHERE " + " AND ".join(where_clauses) + "\n" if where_clauses else "")
        + ("GROUP BY " + ", ".join(group_by_parts) + "\n" if group_by_parts else "")
        + ("ORDER BY " + ", ".join(order_parts) + "\n" if order_parts else "")
        + "LIMIT " + str(top_n)
    ).strip()

    # Валидация — только SELECT
    try:
        tree = sqlglot.parse_one(sql, read="clickhouse")
    except Exception as e:
        raise ValueError("SQL parse error: " + str(e))

    root = tree
    while hasattr(root, "this") and root.this is not None:
        root = root.this
    if not isinstance(root, exp.Select):
        raise ValueError("Only SELECT queries are allowed")

    forbidden = [
        "INSERT", "UPDATE", "DELETE", "DROP", "ALTER", "TRUNCATE",
        "ATTACH", "DETACH", "RENAME", "OPTIMIZE", "CROSS JOIN"
    ]
    if any(tok in sql.upper() for tok in forbidden):
        raise ValueError("Forbidden statement detected")

    return sql