from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Dict, Optional

import pandas as pd
import clickhouse_connect


# =========================
# CONFIG
# =========================

@dataclass
class CHConfig:
    host: str
    port: int
    username: str
    password: str
    # database: str


@dataclass
class ReportParams:
    date_from: str   # '2025-09-01'
    date_to: str     # '2025-12-01'
    visits_threshold: float = 1.0
    sales_threshold: float = 1.0


# =========================
# CLICKHOUSE
# =========================

def get_client(cfg: CHConfig):
    return clickhouse_connect.get_client(
        host=cfg.host,
        port=cfg.port,
        username=cfg.username,
        password=cfg.password,
        # database=cfg.database,
    )


def run_sql_df(client, sql: str) -> pd.DataFrame:
    result = client.query_df(sql)
    return result


# =========================
# BASE MATRIX SQL
# =========================

def build_base_matrix_sql(date_from: str, date_to: str) -> str:
    return f"""
WITH sales_base AS (
    SELECT
        toDate(concat(toString(year), '-', leftPad(toString(month), 2, '0'), '-01')) AS period_start,
        organization_id,
        org_name,
        sku,
        `Region MMH` AS region_mmh,
        federal_subject_name AS federal_subject_name,
        Drugstore_id,
        sales
    FROM Visits_Effectiveness.SalesVsVisits
    WHERE VisitFlag = 'Визит'
      AND toDate(concat(toString(year), '-', leftPad(toString(month), 2, '0'), '-01')) >= toDate('{date_from}')
      AND toDate(concat(toString(year), '-', leftPad(toString(month), 2, '0'), '-01')) < toDate('{date_to}')
),

sales_agg AS (
    SELECT
        organization_id,
        org_name,
        sku,
        region_mmh,
        federal_subject_name,
        SUM(sales) AS totalsales,
        SUM(sales) / COUNT(DISTINCT Drugstore_id) AS avg_sales,
        COUNT(DISTINCT Drugstore_id) AS drugstore_amount
    FROM sales_base
    GROUP BY
        organization_id,
        org_name,
        sku,
        region_mmh,
        federal_subject_name
),

visits_base AS (
    SELECT
        toDate(concat(toString(year), '-', leftPad(toString(month), 2, '0'), '-01')) AS period_start,
        organization_id,
        Brand,
        amount_of_clients,
        amount_of_contacts,
        amount_of_visits
    FROM Visits_Effectiveness.ML_VisitsBySKU_All
    WHERE toDate(concat(toString(year), '-', leftPad(toString(month), 2, '0'), '-01')) >= toDate('{date_from}')
      AND toDate(concat(toString(year), '-', leftPad(toString(month), 2, '0'), '-01')) < toDate('{date_to}')
),

visits_agg AS (
    SELECT
        organization_id,
        Brand,
        SUM(amount_of_clients) AS amount_of_clients,
        SUM(amount_of_contacts) AS amount_of_contacts,
        SUM(amount_of_visits) AS amount_of_visits
    FROM visits_base
    GROUP BY
        organization_id,
        Brand
),

joined AS (
    SELECT
        s.organization_id,
        s.org_name,
        s.sku,
        s.region_mmh,
        s.federal_subject_name,
        s.totalsales,
        s.avg_sales,
        s.drugstore_amount,
        ifNull(v.amount_of_clients, 0) AS amount_of_clients,
        ifNull(v.amount_of_contacts, 0) AS amount_of_contacts,
        ifNull(v.amount_of_visits, 0) AS amount_of_visits
    FROM sales_agg s
    LEFT JOIN visits_agg v
        ON s.organization_id = v.organization_id
       AND s.sku = v.Brand
),

calc AS (
    SELECT
        *,
        median(avg_sales) OVER (PARTITION BY sku, federal_subject_name) AS median_avg_sales,
        median(amount_of_visits) OVER (PARTITION BY sku, federal_subject_name) AS median_visits
    FROM joined
)

SELECT
    organization_id,
    org_name,
    sku,
    region_mmh,
    federal_subject_name,
    totalsales,
    avg_sales,
    drugstore_amount,
    amount_of_clients,
    amount_of_contacts,
    amount_of_visits,
    median_avg_sales,
    median_visits,
    avg_sales - median_avg_sales AS sales_diff,
    amount_of_visits - median_visits AS visits_diff,
    if(median_avg_sales = 0, NULL, avg_sales / median_avg_sales) AS sales_index,
    if(median_visits = 0, NULL, amount_of_visits / median_visits) AS visits_index,
    row_number() OVER (
        PARTITION BY federal_subject_name, sku
        ORDER BY
            if(median_avg_sales = 0, NULL, avg_sales / median_avg_sales) ASC,
            if(median_visits = 0, NULL, amount_of_visits / median_visits) DESC
    ) AS rank_bad,
    row_number() OVER (
        PARTITION BY federal_subject_name, sku
        ORDER BY
            if(median_avg_sales = 0, NULL, avg_sales / median_avg_sales) DESC,
            if(median_visits = 0, NULL, amount_of_visits / median_visits) ASC
    ) AS rank_good
FROM calc
"""


# =========================
# QUADRANT LOGIC
# =========================

def add_quadrant_flags(
    df: pd.DataFrame,
    visits_threshold: float = 1.0,
    sales_threshold: float = 1.0,
) -> pd.DataFrame:
    out = df.copy()

    out["quadrant"] = "other"
    out.loc[
        (out["visits_index"] > visits_threshold) & (out["sales_index"] < sales_threshold),
        "quadrant"
    ] = "high_visits_low_sales"
    out.loc[
        (out["visits_index"] > visits_threshold) & (out["sales_index"] >= sales_threshold),
        "quadrant"
    ] = "high_visits_high_sales"
    out.loc[
        (out["visits_index"] <= visits_threshold) & (out["sales_index"] >= sales_threshold),
        "quadrant"
    ] = "low_visits_high_sales"
    out.loc[
        (out["visits_index"] <= visits_threshold) & (out["sales_index"] < sales_threshold),
        "quadrant"
    ] = "low_visits_low_sales"

    out["needs_attention_flag"] = (
        (out["visits_index"] > visits_threshold) & (out["sales_index"] < sales_threshold)
    ).astype(int)

    return out


def select_problem_lpus(df: pd.DataFrame) -> pd.DataFrame:
    cols_order = [
        "region_mmh",
        "federal_subject_name",
        "organization_id",
        "org_name",
        "sku",
        "totalsales",
        "avg_sales",
        "drugstore_amount",
        "amount_of_clients",
        "amount_of_contacts",
        "amount_of_visits",
        "median_avg_sales",
        "median_visits",
        "sales_diff",
        "visits_diff",
        "sales_index",
        "visits_index",
        "rank_bad",
        "rank_good",
        "quadrant",
        "needs_attention_flag",
    ]
    existing_cols = [c for c in cols_order if c in df.columns]

    out = df[df["needs_attention_flag"] == 1].copy()
    out = out[existing_cols].sort_values(
        by=["federal_subject_name", "sku", "visits_index", "sales_index"],
        ascending=[True, True, False, True]
    )
    return out


# =========================
# PLACEHOLDERS FOR EXTRA BLOCKS
# =========================

def build_in_clause_str(values: pd.Series) -> str:
    uniq = values.dropna().astype(str).unique().tolist()
    escaped = ["'" + v.replace("'", "''") + "'" for v in uniq]
    return ", ".join(escaped) if escaped else "''"


def build_org_in_clause(values: pd.Series) -> str:
    uniq = values.dropna().astype(str).unique().tolist()
    return ", ".join(uniq) if uniq else "0"


def fetch_pharmacy_supply_block(
    client,
    base_problem_df: pd.DataFrame,
    params: ReportParams,
) -> pd.DataFrame:
    if base_problem_df.empty:
        return pd.DataFrame(
            columns=[
                "organization_id",
                "sku",
                "stock_pharmacy_count",
                "stock_dates_count",
                "availability_rate",
                "avg_stock",
                "avg_stock_positive",
                "stock_sum_total",
            ]
        )

    org_ids_sql = build_org_in_clause(base_problem_df["organization_id"])
    sku_sql = build_in_clause_str(base_problem_df["sku"])

    sql = f"""
    WITH lpu_ds AS (
        SELECT DISTINCT
            organization_id,
            toString(`Drugstore_id`) AS lpu_drugstore_id
        FROM Visits_Effectiveness.LPU_Drugstore_Register_All
        WHERE organization_id IN ({org_ids_sql})
    ),

    mdlp_base AS (
        SELECT
            toDate(date) AS snapshot_date,
            toString(id_md) AS mdlp_drugstore_id,
            sku,
            sum(ifNull(remains_full, 0)) AS stock_qty
        FROM MDLP.SHOW_Remaining_reports
        WHERE toDate(date) >= toDate('{params.date_from}')
          AND toDate(date) < toDate('{params.date_to}')
          AND sku IN ({sku_sql})
        GROUP BY
            snapshot_date,
            mdlp_drugstore_id,
            sku
    ),

    joined AS (
        SELECT
            l.organization_id AS organization_id,
            m.sku AS sku,
            m.mdlp_drugstore_id AS stock_drugstore_id,
            m.snapshot_date AS snapshot_date,
            m.stock_qty AS stock_qty
        FROM lpu_ds l
        INNER JOIN mdlp_base m
            ON l.lpu_drugstore_id = m.mdlp_drugstore_id
    )

    SELECT
        organization_id,
        sku,
        uniqExact(stock_drugstore_id) AS stock_pharmacy_count,
        uniqExact(snapshot_date) AS stock_dates_count,
        avg(if(stock_qty > 0, 1.0, 0.0)) AS availability_rate,
        avg(stock_qty) AS avg_stock,
        avgIf(stock_qty, stock_qty > 0) AS avg_stock_positive,
        sum(stock_qty) AS stock_sum_total
    FROM joined
    GROUP BY
        organization_id,
        sku
    """

    print("\n=== PHARMACY SUPPLY SQL START ===")
    print(sql)
    print("=== PHARMACY SUPPLY SQL END ===\n")

    return run_sql_df(client, sql)

def fetch_doctor_block(
    client,
    base_problem_df: pd.DataFrame,
    params: ReportParams,
) -> pd.DataFrame:
    """
    Блок по врачам:
      - repeat_contact_rate
      - specialty_share

    Сопоставление sku делается через словарь канонических названий.
    """
    if base_problem_df.empty:
        return pd.DataFrame(
            columns=[
                "organization_id",
                "sku",
                "repeat_contact_rate",
                "specialty_share",
            ]
        )

    org_ids_sql = build_org_in_clause(base_problem_df["organization_id"])

    # Канонические названия из основной таблицы + их варианты из CRM/percent
    sku_alias_map = {
        "ПРОСПЕКТА": ["ПРОСПЕКТА", "Проспекта"],
        "РАФАМИН": ["РАФАМИН", "Рафамин"],
        "Тенотен детский": ["Тенотен детский", "Тенотен дет"],
        "Анаферон детский": ["Анаферон детский", "Анаферон дет"],
        # при желании сюда можно добавлять новые пары
    }

    # Все sku, которые реально есть в problem_df
    target_skus = sorted(base_problem_df["sku"].dropna().astype(str).unique().tolist())

    # Строим mapping SQL только для нужных sku
    sku_map_sql_parts = []

    for sku in target_skus:
        aliases = sku_alias_map.get(sku, [sku])
        for alias in aliases:
            sku_original = sku.replace("'", "''")
            alias_norm = alias.lower().replace("'", "''")
            sku_map_sql_parts.append(
                f"SELECT '{sku_original}' AS sku_original, '{alias_norm}' AS sku_norm"
            )

    sku_map_sql = "\nUNION ALL\n".join(sku_map_sql_parts)

    sql = f"""
    WITH target_skus AS (
        {sku_map_sql}
    ),

    percent_union AS (
        SELECT
            year_,
            cycle_,
            month_,
            BU,
            partner_,
            Brand__,
            AVG(percent) AS percent
        FROM MedRepVisits.percent
        GROUP BY
            year_, cycle_, month_, BU, partner_, Brand__

        UNION ALL

        SELECT
            year_,
            cycle_,
            month_,
            BU,
            partner_,
            Brand__,
            AVG(percent) AS percent
        FROM MedRepVisits.percent_
        GROUP BY
            year_, cycle_, month_, BU, partner_, Brand__
    ),

    raw_events AS (
        SELECT
            E.organization_id AS organization_id,
            ts.sku_original AS sku,
            E.Specialiti_n AS specialty,
            E.Client_FIO AS doctor_fio,
            count() AS contacts_cnt,
            sum(ifNull(p.percent, 0)) AS weighted_visits_cnt
        FROM CRM.CRM_EVENT_DETAILS E
        LEFT JOIN percent_union p
            ON p.year_ = E.event_year
           AND p.month_ = E.event_month
           AND toInt32(substring(p.cycle_, 1, 1)) = E.event_cycle
           AND p.BU = substring(E.BUnit_n, 4)
           AND p.partner_ = E.Specialiti_n
        INNER JOIN target_skus ts
            ON lowerUTF8(p.Brand__) = ts.sku_norm
        WHERE E.organization_id IS NOT NULL
          AND ifNull(E.org_lat, 0) <> 0
          AND ifNull(E.org_lon, 0) <> 0
          AND E.Organization_Type_n IN ('Амбулаторное учреждение', 'Стационар')
          AND E.country_n IN ('РФ', 'Россия')
          AND E.BUnit_n IN (
              '8. РВА', '8. СВА', '8. СВАЭ', '8. СВАЭ Москва',
              '8. РЕСПИ', '8. СПЕЦ', '8. НЕВРО', '8. НЕВРО2', '8. НЕВРО1',
              '8. ПРОРЕСПИ', '8. ПРОСПЕЦ', '8. ПРОСПЕЦ_РФ', '8. ПРОСПЕЦ_МОСКВА',
              '8. ПРОРЕСПИ_РФ', '8. ПРОРЕСПИ_МОСКВА'
          )
          AND E.event_type_n = 'Визит'
          AND E.Specialiti_n IS NOT NULL
          AND toDate(E.date_completed) >= toDate('{params.date_from}')
          AND toDate(E.date_completed) < toDate('{params.date_to}')
          AND E.organization_id IN ({org_ids_sql})
        GROUP BY
            organization_id,
            sku,
            specialty,
            doctor_fio
    ),

    agg_main AS (
        SELECT
            organization_id,
            sku,
            uniqExact(doctor_fio) AS doctor_unique_count,
            sum(contacts_cnt) AS doctor_contacts_total
        FROM raw_events
        GROUP BY
            organization_id,
            sku
    ),

    specialty_agg AS (
        SELECT
            organization_id,
            sku,
            specialty,
            sum(weighted_visits_cnt) AS specialty_weighted_visits
        FROM raw_events
        GROUP BY
            organization_id,
            sku,
            specialty
    ),

    specialty_ranked AS (
        SELECT
            organization_id,
            sku,
            specialty,
            specialty_weighted_visits,
            sum(specialty_weighted_visits) OVER (PARTITION BY organization_id, sku) AS total_weighted_visits
        FROM specialty_agg
    ),

    specialty_text AS (
        SELECT
            organization_id,
            sku,
            arrayStringConcat(
                groupArray(
                    concat(
                        specialty,
                        ': ',
                        toString(
                            round(
                                if(total_weighted_visits = 0, 0, specialty_weighted_visits / total_weighted_visits * 100),
                                1
                            )
                        ),
                        '%'
                    )
                ),
                '; '
            ) AS specialty_share
        FROM specialty_ranked
        GROUP BY
            organization_id,
            sku
    )

    SELECT
        a.organization_id,
        a.sku,
        if(
            a.doctor_contacts_total = 0,
            NULL,
            (a.doctor_contacts_total - a.doctor_unique_count) / a.doctor_contacts_total
        ) AS repeat_contact_rate,
        s.specialty_share
    FROM agg_main a
    LEFT JOIN specialty_text s
        ON a.organization_id = s.organization_id
       AND a.sku = s.sku
    """

    return run_sql_df(client, sql)


def enrich_with_blocks(client, base_problem_df: pd.DataFrame, params: ReportParams) -> pd.DataFrame:
    result = base_problem_df.copy()

    extra_blocks = [
        fetch_pharmacy_supply_block,
        fetch_doctor_block,
    ]

    for fetcher in extra_blocks:
        block_df = fetcher(client, result, params)
        if not block_df.empty:
            result = result.merge(
                block_df,
                on=["organization_id", "sku"],
                how="left"
            )

    return result

# =========================
# DIAGNOSIS / RECOMMENDATION PLACEHOLDER
# =========================

def add_recommendation_columns(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()

    # -------------------------
    # Базовые колонки вывода
    # -------------------------
    out["primary_issue"] = "to_be_defined"
    out["recommendation_main"] = "Требуется дополнительный анализ"
    out["recommendation_comment"] = (
        "Проверить остатки в аптеках, структуру специальностей и повторяемость визитов"
    )

    # -------------------------
    # Нормализация числовых полей
    # -------------------------
    numeric_cols = [
        "availability_rate",
        "avg_stock",
        "avg_stock_positive",
        "stock_sum_total",
        "stock_pharmacy_count",
        "stock_dates_count",
        "drugstore_amount",
        "amount_of_clients",
        "amount_of_contacts",
        "amount_of_visits",
        "sales_index",
        "visits_index",
        "totalsales",
        "repeat_contact_rate",
    ]

    for col in numeric_cols:
        if col in out.columns:
            out[col] = pd.to_numeric(out[col], errors="coerce")

    # -------------------------
    # Вспомогательные флаги
    # -------------------------
    if "availability_rate" in out.columns:
        out["availability_bucket"] = pd.cut(
            out["availability_rate"],
            bins=[-0.01, 0.30, 0.70, 1.01],
            labels=["low", "medium", "high"]
        ).astype("string")
    else:
        out["availability_bucket"] = pd.Series(pd.NA, index=out.index, dtype="string")

    # По умолчанию
    has_stock_cols = {"availability_rate", "stock_pharmacy_count", "stock_sum_total"}.issubset(out.columns)
    has_doctor_cols = {"repeat_contact_rate", "specialty_share"}.issubset(out.columns)

    # -------------------------
    # 1. Нет данных по остаткам
    # -------------------------
    if has_stock_cols:
        no_stock_data_mask = (
            out["stock_pharmacy_count"].isna() |
            (out["stock_pharmacy_count"] == 0)
        )

        out.loc[no_stock_data_mask, "primary_issue"] = "no_stock_data"
        out.loc[no_stock_data_mask, "recommendation_main"] = "Недостаточно данных по остаткам"
        out.loc[no_stock_data_mask, "recommendation_comment"] = (
            "Для аптек рядом с ЛПУ не найдены данные MDLP по остаткам за выбранный период. "
            "Сначала проверьте полноту данных по аптекам и остаткам."
        )

    # -------------------------
    # 2. Проблема с остатками
    # -------------------------
    if has_stock_cols:
        supply_problem_mask = (
            out["availability_rate"].fillna(0) < 0.30
        )

        partial_supply_problem_mask = (
            (out["availability_rate"].fillna(0) >= 0.30) &
            (out["availability_rate"].fillna(0) < 0.70)
        )

        out.loc[supply_problem_mask, "primary_issue"] = "supply_problem"
        out.loc[supply_problem_mask, "recommendation_main"] = "Проверить наличие в аптеках"
        out.loc[supply_problem_mask, "recommendation_comment"] = (
            "Низкая доступность остатков рядом с ЛПУ за выбранный период. "
            "Не увеличивать визиты, пока не улучшено наличие и стабильность поставок."
        )

        out.loc[partial_supply_problem_mask, "primary_issue"] = "partial_supply_problem"
        out.loc[partial_supply_problem_mask, "recommendation_main"] = "Проверить стабильность остатков"
        out.loc[partial_supply_problem_mask, "recommendation_comment"] = (
            "Товар присутствует нестабильно. Нужно проверить провалы по неделям, "
            "ключевые аптеки рядом с ЛПУ и возможные проблемы с дистрибуцией."
        )

    # -------------------------
    # 3. Остатки нормальные, но возможно ходим к одним и тем же врачам
    # -------------------------
    if has_stock_cols and has_doctor_cols:
        overfocus_same_doctors_mask = (
            (out["availability_rate"].fillna(1) >= 0.70) &
            (out["repeat_contact_rate"].fillna(0) > 0.60)
        )

        out.loc[overfocus_same_doctors_mask, "primary_issue"] = "overfocus_same_doctors"
        out.loc[overfocus_same_doctors_mask, "recommendation_main"] = (
            "Проверить концентрацию визитов на одних и тех же врачах"
        )
        out.loc[overfocus_same_doctors_mask, "recommendation_comment"] = (
            "Остатки рядом с ЛПУ выглядят приемлемо, но высокая доля повторных контактов "
            "может означать, что визитная активность сосредоточена на слишком узком круге врачей. "
            "Проверьте возможность расширить охват."
        )

    # -------------------------
    # 4. Остатки нормальные, повторяемость не критична:
    #    проверить структуру специальностей
    # -------------------------
    if has_stock_cols and has_doctor_cols:
        targeting_review_mask = (
            (out["availability_rate"].fillna(1) >= 0.70) &
            (out["sales_index"].fillna(0) < 1.0) &
            (out["visits_index"].fillna(0) > 1.0) &
            (out["repeat_contact_rate"].fillna(0) <= 0.60)
        )

        out.loc[targeting_review_mask, "primary_issue"] = "targeting_review_needed"
        out.loc[targeting_review_mask, "recommendation_main"] = (
            "Проверить структуру посещаемых специальностей"
        )
        out.loc[targeting_review_mask, "recommendation_comment"] = (
            "Остатки рядом с ЛПУ выглядят нормальными, но продажи остаются низкими "
            "при высокой визитной активности. Проверьте, соответствует ли структура "
            "посещаемых специальностей ожидаемому таргету по бренду."
        )

    # -------------------------
    # 5. Если мало аптек с данными — пометка
    # -------------------------
    if "stock_pharmacy_count" in out.columns:
        low_pharmacy_coverage_mask = (
            out["stock_pharmacy_count"].fillna(0) <= 1
        )

        out.loc[
            low_pharmacy_coverage_mask & (out["primary_issue"] == "to_be_defined"),
            "primary_issue"
        ] = "low_pharmacy_coverage"

        out.loc[
            low_pharmacy_coverage_mask & (out["recommendation_main"] == "Требуется дополнительный анализ"),
            "recommendation_main"
        ] = "Проверить покрытие аптек рядом с ЛПУ"

        out.loc[
            low_pharmacy_coverage_mask &
            (out["recommendation_comment"] == "Проверить остатки в аптеках, структуру специальностей и повторяемость визитов"),
            "recommendation_comment"
        ] = (
            "Рядом с ЛПУ мало аптек с данными по остаткам. Выводы по supply делать осторожно."
        )

    # -------------------------
    # 6. Если ничего не сработало, но кейс всё равно проблемный
    # -------------------------
    unresolved_mask = (
        (out["primary_issue"] == "to_be_defined") &
        (out["sales_index"].fillna(0) < 1.0) &
        (out["visits_index"].fillna(0) > 1.0)
    )

    out.loc[unresolved_mask, "primary_issue"] = "needs_manual_review"
    out.loc[unresolved_mask, "recommendation_main"] = "Провести ручной анализ ЛПУ"
    out.loc[unresolved_mask, "recommendation_comment"] = (
        "ЛПУ попало в квадрант 'высокие визиты / низкие продажи', "
        "но явная причина не определена автоматически. "
        "Проверьте остатки, врачебный таргет, конкурентов и качество визитов."
    )

    # -------------------------
    # 7. Приоритет кейса
    # -------------------------
    if {"visits_index", "sales_index", "totalsales"}.issubset(out.columns):
        out["priority_score"] = (
            (out["visits_index"].fillna(0) - 1).clip(lower=0)
            * (1 - out["sales_index"].fillna(1)).clip(lower=0)
            * out["totalsales"].fillna(0)
        )
    else:
        out["priority_score"] = 0

    # -------------------------
    # 8. Короткая человекочитаемая рекомендация
    # -------------------------
    out["recommendation_short"] = out["recommendation_main"].fillna("")

    return out


# =========================
# MAIN PIPELINE
# =========================

def build_problem_lpu_report(client, params: ReportParams) -> pd.DataFrame:
    base_sql = build_base_matrix_sql(params.date_from, params.date_to)
    base_df = run_sql_df(client, base_sql)

    if base_df.empty:
        return base_df

    # 1. Квадранты
    base_df = add_quadrant_flags(
        base_df,
        visits_threshold=params.visits_threshold,
        sales_threshold=params.sales_threshold,
    )

    # 2. Берём только проблемные ЛПУ
    problem_df = select_problem_lpus(base_df)

    if problem_df.empty:
        return problem_df

    # 3. Дотягиваем дополнительные блоки
    enriched_df = enrich_with_blocks(client, problem_df, params)

    # 4. Ставим первичный диагноз и рекомендации
    final_df = add_recommendation_columns(enriched_df)

    # 5. Сортировка для удобства анализа
    sort_cols = [c for c in [
        "federal_subject_name",
        "sku",
        "primary_issue",
        "priority_score",
        "visits_index",
        "sales_index",
    ] if c in final_df.columns]

    ascending = []
    for c in sort_cols:
        if c in ["priority_score", "visits_index"]:
            ascending.append(False)
        elif c == "sales_index":
            ascending.append(True)
        else:
            ascending.append(True)

    if sort_cols:
        final_df = final_df.sort_values(by=sort_cols, ascending=ascending)

    # 6. Переставляем колонки в удобный порядок
    desired_order = [
        "region_mmh",
        "federal_subject_name",
        "organization_id",
        "org_name",
        "sku",

        "quadrant",
        "needs_attention_flag",
        "primary_issue",
        "recommendation_main",
        "recommendation_comment",
        "priority_score",

        "totalsales",
        "avg_sales",
        "drugstore_amount",
        "amount_of_clients",
        "amount_of_contacts",
        "amount_of_visits",

        "median_avg_sales",
        "median_visits",
        "sales_diff",
        "visits_diff",
        "sales_index",
        "visits_index",
        "rank_bad",
        "rank_good",

        "stock_pharmacy_count",
        "stock_dates_count",
        "availability_rate",
        "availability_bucket",
        "avg_stock",
        "avg_stock_positive",
        "stock_sum_total",

        "repeat_contact_rate",
        "specialty_share",
    ]

    final_cols = [c for c in desired_order if c in final_df.columns]
    other_cols = [c for c in final_df.columns if c not in final_cols]

    final_df = final_df[final_cols + other_cols]

    return final_df


# =========================
# SAVE
# =========================

def save_report(df: pd.DataFrame, path: str = "problem_lpu_report.xlsx") -> None:
    with pd.ExcelWriter(path, engine="openpyxl") as writer:
        df.to_excel(writer, sheet_name="problem_lpu_report", index=False)


# =========================
# RUN
# =========================

if __name__ == "__main__":
    cfg = CHConfig(
        host="clickhouse.moscow",
        port=8123,
        username="GrushkoIV",
        password="jNbrvzd1IcF0Yx5I",
        # database="Visits_Effectiveness",
    )

    params = ReportParams(
        date_from="2025-09-01",
        date_to="2025-12-31",
        visits_threshold=1.0,
        sales_threshold=1.0,
    )

    client = get_client(cfg)

    report_df = build_problem_lpu_report(client, params)
    save_report(report_df, "problem_lpu_report.xlsx")

    print("Done. Rows:", len(report_df))