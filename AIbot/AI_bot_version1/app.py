# -*- coding: utf-8 -*-
import yaml
import clickhouse_connect
from llm_intent import call_llm_to_intent
from sql_builder import build_sql

# Загружаем конфиги
METRICS = yaml.safe_load(open("metrics.yaml", "r", encoding="utf-8"))
SCHEMA  = yaml.safe_load(open("schema_catalog.yaml", "r", encoding="utf-8"))

# Подключение к ClickHouse (read-only аккаунт!)
client = clickhouse_connect.get_client(
    host='clickhouse.moscow',
    port=8123,  # 8123 для HTTP, 9000 для Native (TCP)
    username='GrushkoIV',
    password='jNbrvzd1IcF0Yx5I',
)

def run_nl_query(natural_query: str, model: str = "mistral"):
    intent = call_llm_to_intent(
        natural_query=natural_query,
        metrics_yaml=yaml.safe_dump(METRICS, allow_unicode=True),
        schema_yaml=yaml.safe_dump(SCHEMA, allow_unicode=True),
        model=model,
    )
    sql = build_sql(intent, METRICS, SCHEMA)
    print("\n— NL question:", natural_query)
    print("— Intent:", intent.model_dump())
    print("— SQL:\n", sql)

    data = client.query(sql)
    return {"columns": data.column_names, "rows": data.result_rows, "intent": intent.model_dump()}

if __name__ == "__main__":
    examples = [
        "Какие были продажи Эргоферона в декабре 2024",
        "Покажи продажи Ренгалина в разбивке по регионам за январь-апрель 2025г"
        # "Топ-10 SKU по продажам в Москве за 2024",
        # "Помесячная динамика по региону северо-запад по SKU Эргоферон",
        # "Сколько уникальных контрагентов за последний год по типу выбытия Продажа",
    ]
    for q in examples:
        try:
            res = run_nl_query(q)
            print("— Result columns:", res["columns"])
            print("— First 3 rows:", res["rows"][:3])
        except Exception as e:
            print("⚠️ Error:", e)