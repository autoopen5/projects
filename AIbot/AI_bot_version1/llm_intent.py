# llm_intent.py
# -*- coding: utf-8 -*-
"""
NL → JSON-намерение (Intent) для генерации SQL по одной таблице MDLP.SHOW_Disposal_reports.

Изменения:
- В SYSTEM_PROMPT: правило — если запрос формата «по региону X / в регионе X», вернуть фильтр region_mmh=X
  и НЕ включать region_mmh в dimensions, если явно не просили "по регионам".
- Постобработка Intent: из текста запроса вытягиваем регион (по региону/в регионе ...),
  добавляем фильтр {"field":"region_mmh","op":"ilike","value": "..."} и удаляем region_mmh из dimensions.
- ilike = регистронезависимый ПОДСТРОЧНЫЙ поиск (LIKE '%value%') — согласовано с sql_builder.py.
"""

from __future__ import annotations

import json
import re
from typing import Any, Dict, Optional, Tuple, List

import requests
import yaml

from intent_schema import Intent


# ------------------------------
# Настройки Ollama
# ------------------------------
OLLAMA_URL = "http://localhost:11434/api/generate"
DEFAULT_MODEL = "mistral"
DEFAULT_OPTIONS = {"temperature": 0, "num_ctx": 4096}


# ------------------------------
# Ollama вызов
# ------------------------------
def _compose_prompt(user_prompt: str, system: Optional[str]) -> str:
    return f"<<SYS>>\n{system.strip()}\n<</SYS>>\n\n{user_prompt}" if system else user_prompt


def ask_ollama(prompt: str,
               model: str = DEFAULT_MODEL,
               options: Optional[Dict[str, Any]] = None,
               system: Optional[str] = None) -> str:
    payload: Dict[str, Any] = {
        "model": model,
        "prompt": _compose_prompt(prompt, system),
        "stream": True,
        "options": options or DEFAULT_OPTIONS,
    }
    with requests.post(OLLAMA_URL, json=payload, stream=True) as r:
        r.raise_for_status()
        chunks = []
        for line in r.iter_lines(decode_unicode=True):
            if not line:
                continue
            try:
                obj = json.loads(line)
            except Exception:
                continue
            if "response" in obj:
                chunks.append(obj["response"])
            if obj.get("done"):
                break
        return "".join(chunks).strip()


# ------------------------------
# JSON разбор/ремонт
# ------------------------------
def strip_code_fences(text: str) -> str:
    text = re.sub(r"^```(?:json)?\s*", "", text.strip(), flags=re.IGNORECASE)
    text = re.sub(r"\s*```$", "", text.strip())
    return text.strip()


def extract_json_block(text: str) -> str:
    text = strip_code_fences(text)
    start = text.find("{")
    end = text.rfind("}")
    if start == -1 or end == -1 or end <= start:
        return text
    return text[start:end + 1]


def _is_valid_json(s: str) -> bool:
    try:
        json.loads(s)
        return True
    except Exception:
        return False


def repair_json_minimal(text: str) -> str:
    t = extract_json_block(text)
    if _is_valid_json(t):
        return t
    t = re.sub(r",\s*([}\]])", r"\1", t)  # лишние запятые
    if not _is_valid_json(t) and "'" in t and '"' not in t:
        cand = t.replace("'", '"')
        if _is_valid_json(cand):
            return cand
    t = t.replace("\u0000", "")
    return t


def parse_intent_json(raw: str, metrics_yaml: str) -> Intent:
    cand = extract_json_block(raw)
    for variant in (cand, repair_json_minimal(raw)):
        try:
            data = json.loads(variant)
            return Intent(**data)
        except Exception:
            continue
    return _fallback_intent(metrics_yaml)


def _fallback_intent(metrics_yaml: str) -> Intent:
    try:
        cfg = yaml.safe_load(metrics_yaml) or {}
    except Exception:
        cfg = {}
    measures = list(((cfg.get("measures") or {}).keys()))
    measure = measures[0] if measures else "sales_units"
    dims_cfg = (cfg.get("dimensions") or {})
    dims = ["date"] if "date" in dims_cfg else []
    return Intent(
        measures=[measure],
        dimensions=dims,
        filters=[],
        time_grain="month" if "date" in dims else None,
        sort_by=[{"field": measure, "dir": "desc"}],
        top_n=(cfg.get("defaults") or {}).get("top_n", 50),
        explain="Fallback: не удалось распарсить JSON",
        confidence=0.5
    )


# ------------------------------
# Allowed lists
# ------------------------------
def _extract_allowed(metrics_yaml: str) -> Tuple[List[str], List[str]]:
    try:
        cfg = yaml.safe_load(metrics_yaml) or {}
    except Exception:
        cfg = {}
    measures = list(((cfg.get("measures") or {}).keys()))
    dimensions = list(((cfg.get("dimensions") or {}).keys()))
    return measures, dimensions


# ------------------------------
# Промпты
# ------------------------------
SYSTEM_PROMPT = """
Ты — планировщик аналитических запросов для ClickHouse по одной таблице MDLP.SHOW_Disposal_reports.
Верни СТРОГО ВАЛИДНЫЙ JSON без текста вокруг со следующими полями:
{
  "measures": ["<measure>", ...],
  "dimensions": ["<dimension>", ...],
  "filters": [{"field":"<dimension>","op":"=|!=|>|<|>=|<=|between|in|contains|ilike","value": <...>}],
  "time_grain": "day"|"week"|"month"|"quarter"|"year"|null,
  "sort_by": [{"field":"<measure|dimension>","dir":"asc"|"desc"}],
  "top_n": <int|null>,
  "explain": "краткое объяснение на русском",
  "confidence": <0..1>
}

Правила:
- Используй ТОЛЬКО measures/dimensions из каталога метрик. Ничего не выдумывай.
- Таблица всегда одна: MDLP.SHOW_Disposal_reports. JOIN-ов нет.
- Если в запросе звучит «продажи», «реализация», «продал» и т.п., добавь фильтр по типу выбытия:
  предпочтительно {"field":"exit_type","op":"=","value":"Продажа"}, допускается {"op":"ilike"}.
- Период:
  • Для формулировок «в <месяц> <год>» предпочитай Year=<год>, Month=<номер месяца>.
    Допустимо альтернативно: {"field":"date","op":"=","value":"YYYY-MM"}.
  • Для временных рядов добавляй dimension "date" и ставь time_grain (обычно "month").
- Регионы:
  • Если пользователь говорит «по региону X» или «в регионе X», добавь фильтр {"field":"region_mmh","op":"ilike","value":"X"}.
    В этом случае НЕ добавляй "region_mmh" в dimensions, если явно не сказано «по регионам/в разрезе регионов».
  • Если пользователь просит «по регионам» (разбивку), тогда добавь "region_mmh" в dimensions и НЕ добавляй фильтр.
- ilike трактуем как регистронезависимый ПОДСТРОЧНЫЙ поиск (LIKE '%значение%').
- top_n указывай только если явно просят «топ».
- Если период явно задан (date/year/month) — система НЕ будет добавлять дефолтный фильтр по exit_date.
- Верни ТОЛЬКО JSON, без комментариев и кодовых блоков.
""".strip()


FEW_SHOTS = """
Вопрос: "Помесячная динамика по региону северо-запад по SKU Эргоферон"
Ответ:
{
  "measures": ["sales_units"],
  "dimensions": ["date"],
  "filters": [
    {"field":"exit_type","op":"=","value":"Продажа"},
    {"field":"region_mmh","op":"ilike","value":"северо-запад"},
    {"field":"sku","op":"ilike","value":"Эргоферон"}
  ],
  "time_grain": "month",
  "sort_by": [{"field":"date","dir":"asc"}],
  "top_n": null,
  "explain": "Помесячная динамика продаж по региону Северо-Запад (фильтр), SKU Эргоферон",
  "confidence": 0.88
}

Вопрос: "Помесячная динамика по регионам по SKU Эргоферон"
Ответ:
{
  "measures": ["sales_units"],
  "dimensions": ["date", "region_mmh"],
  "filters": [
    {"field":"exit_type","op":"=","value":"Продажа"},
    {"field":"sku","op":"ilike","value":"Эргоферон"}
  ],
  "time_grain": "month",
  "sort_by": [{"field":"date","dir":"asc"}],
  "top_n": null,
  "explain": "Разбивка по регионам (без фильтра по региону), SKU Эргоферон",
  "confidence": 0.86
}
""".strip()


def build_user_prompt(natural_query: str, metrics_yaml: str, schema_yaml: str) -> str:
    measures, dimensions = _extract_allowed(metrics_yaml)
    allowed = f"""
РАЗРЕШЁННЫЕ MEASURES:
{json.dumps(measures, ensure_ascii=False)}

РАЗРЕШЁННЫЕ DIMENSIONS:
{json.dumps(dimensions, ensure_ascii=False)}
""".strip()
    parts = [
        "Каталог метрик (YAML):",
        "```",
        metrics_yaml.strip(),
        "```",
        "",
        "Каталог схем (YAML):",
        "```",
        schema_yaml.strip(),
        "```",
        "",
        allowed,
        "",
        "Примеры JSON-намерений:",
        "```",
        FEW_SHOTS,
        "```",
        "",
        f'Вопрос пользователя: "{natural_query.strip()}"',
        "",
        "Верни СТРОГО ТОЛЬКО JSON-намерение без текста вокруг."
    ]
    return "\n".join(parts)


# ------------------------------
# Постобработка: авто-фильтр по региону из текста
# ------------------------------
_REGION_ALIASES = {
    "северо-запад": ["северо-запад", "северо запад", "северо-западный", "сзфо"],
    "москва": ["москва", "мск"],
    "санкт-петербург": ["санкт-петербург", "санкт петербург", "спб", "питер"],
    "волга": ["волга", "поволжье"],
    "юг": ["юг"],
    "урал": ["урал"],
    "сибирь": ["сибирь"],
    "дальний восток": ["дальний восток", "дфо"],
    "центр": ["центр", "цфо"],
}

def _extract_region_from_text(text: str) -> Optional[str]:
    """
    Ищем конструкции 'по региону X' | 'в регионе X' | 'по округу X' | 'в округе X'.
    Возвращаем каноническое имя из _REGION_ALIASES, иначе саму найденную фразу (lowercased).
    """
    t = (text or "").lower()
    # Если пользователь явно просит разбивку по регионам — не фильтруем
    if ("по регионам" in t) or ("разрезе регионов" in t) or ("по региональным" in t):
        return None

    # Простые попадания по словарю
    for canon, variants in _REGION_ALIASES.items():
        if any(v in t for v in variants):
            # убедимся, что есть указание "по региону/в регионе/по округу/в округе" (чтобы не ловить случайные слова)
            if ("по региону" in t) or ("в регионе" in t) or ("по округу" in t) or ("в округе" in t):
                return canon

    # Пытаемся вытащить фразу после "по региону"/"в регионе"/"по округу"/"в округе"
    m = re.search(r"(по\s+регион[ауе]|в\s+регионе|по\s+округ[ауе]|в\s+округе)\s+([^\.,;:]+)", t)
    if m:
        raw = m.group(2).strip()
        # обрежем хвост типа "по SKU ..." если он примыкает
        raw = re.split(r"\s+по\s+", raw)[0].strip()
        raw = re.split(r"\s+и\s+", raw)[0].strip()
        # маппинг на канон, если попадается
        for canon, variants in _REGION_ALIASES.items():
            if any(v in raw for v in variants):
                return canon
        return raw  # вернем как есть (ilike подстрока справится)

    return None


def _adjust_intent_with_region(natural_query: str, intent: Intent) -> Intent:
    """
    Если в вопросе 'по региону X' и фильтра по region_mmh нет — добавим.
    И если region_mmh оказался в dimensions, а разбиение по регионам не просили — уберём.
    """
    region_val = _extract_region_from_text(natural_query)
    if not region_val:
        return intent

    has_region_filter = any(f.field == "region_mmh" for f in (intent.filters or []))
    if not has_region_filter:
        # добавим фильтр
        new_filters = list(intent.filters or [])
        new_filters.append({"field": "region_mmh", "op": "ilike", "value": region_val})
        intent.filters = new_filters  # pydantic сам приведёт к FilterSpec

    # если не просили "по регионам" — уберём измерение region_mmh из dimensions
    t = (natural_query or "").lower()
    if ("по регионам" not in t) and ("разрезе регионов" not in t):
        intent.dimensions = [d for d in (intent.dimensions or []) if d != "region_mmh"]

    return intent


# ------------------------------
# Публичная функция
# ------------------------------
def call_llm_to_intent(natural_query: str,
                       metrics_yaml: str,
                       schema_yaml: str,
                       model: str = DEFAULT_MODEL,
                       options: Optional[Dict[str, Any]] = None) -> Intent:
    user_prompt = build_user_prompt(natural_query, metrics_yaml, schema_yaml)
    raw = ask_ollama(
        prompt=user_prompt,
        model=model,
        options=options or DEFAULT_OPTIONS,
        system=SYSTEM_PROMPT
    )
    intent = parse_intent_json(raw, metrics_yaml)

    # Фикс регионов из текста
    intent = _adjust_intent_with_region(natural_query, intent)

    # Мини-гигиена на случай пустых measures
    if not intent.measures:
        try:
            cfg = yaml.safe_load(metrics_yaml) or {}
            first_measure = next(iter((cfg.get("measures") or {}).keys()), "sales_units")
        except Exception:
            first_measure = "sales_units"
        intent.measures = [first_measure]

    return intent


# ------------------------------
# Локальный тест (опционально)
# ------------------------------
if __name__ == "__main__":
    METRICS_YAML = """
defaults:
  time_filter_days: 365
  top_n: 50
  time_column: exit_date
measures:
  sales_units:
    expr: "sum(cnt)"
    table: "MDLP.SHOW_Disposal_reports"
    format: "integer"
dimensions:
  date:     { table: "MDLP.SHOW_Disposal_reports", column: "exit_date",      type: "date" }
  year:     { table: "MDLP.SHOW_Disposal_reports", column: "Year",           type: "categorical" }
  month:    { table: "MDLP.SHOW_Disposal_reports", column: "Month",          type: "categorical" }
  sku:      { table: "MDLP.SHOW_Disposal_reports", column: "SKU",            type: "categorical" }
  region_mmh:{ table: "MDLP.SHOW_Disposal_reports", column: "Region MMH",    type: "categorical" }
  exit_type:{ table: "MDLP.SHOW_Disposal_reports", column: "exit_type",      type: "categororical" }
""".strip()

    SCHEMA_YAML = """
tables:
  MDLP.SHOW_Disposal_reports:
    columns:
      - { name: "exit_date", type: "Date" }
      - { name: "Year", type: "UInt16" }
      - { name: "Month", type: "UInt8" }
      - { name: "SKU", type: "String" }
      - { name: "Region MMH", type: "String" }
      - { name: "exit_type", type: "String" }
      - { name: "cnt", type: "Int64" }
""".strip()

    q = "Помесячная динамика по региону северо-запад по SKU Эргоферон"
    # Для реальной проверки нужен поднятый Ollama.
    print("Готово. В проде вызывайте call_llm_to_intent(...) из app.py")