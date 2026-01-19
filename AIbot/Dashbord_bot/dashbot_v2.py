# dashbot_v1.py
# Простая консольная версия: спрашивает запрос → понимает его через Ollama(Mistral) →
# тянет грид дашбордов из ClickHouse → решает показывать/не показывать по простому скорингу →
# печатает список "Название — URL".
#
# Таблица: grushko_iv.dashboard_catalog со схемой из задания.
#
# Зависимости: clickhouse-connect, rapidfuzz, requests
# Запуск: python dashbot_v1.py

import os
import sys
import json
import math
import requests
from datetime import datetime
from typing import Dict, Any, List

import clickhouse_connect
from rapidfuzz import fuzz


# ---------------------------
# Конфиг
# ---------------------------
CH_HOST = os.getenv("CLICKHOUSE_HOST", "clickhouse.moscow")
CH_PORT = int(os.getenv("CLICKHOUSE_PORT", "8123"))
CH_USER = os.getenv("CLICKHOUSE_USER", "GrushkoIV")
CH_PASSWORD = os.getenv("CLICKHOUSE_PASSWORD", "jNbrvzd1IcF0Yx5I")
CH_DB = os.getenv("CLICKHOUSE_DB", "grushko_iv")

CATALOG_TABLE = "dashboard_catalog"

OLLAMA_BASE = os.getenv("OLLAMA_BASE", "http://localhost:11434").rstrip("/")
OLLAMA_MODEL = os.getenv("OLLAMA_MODEL", "mistral")

TOP_K = int(os.getenv("TOP_K", "5"))
SHOW_DEBUG = os.getenv("DASHBOT_DEBUG", "0") == "1"

# Порог, после которого считаем, что дашборд "подходит"
SHOW_THRESHOLD = float(os.getenv("SHOW_THRESHOLD", "0.33"))

# Весовые коэффициенты (простая интерпретация)
W_SEMANTIC = 0.70   # семантика: fuzzy между запросом и полями карточки
W_INTENT   = 0.15   # бонус за совпадение info_type
W_KEYWORD  = 0.10   # суммарный бонус за совпадение извлечённых ключевых слов
W_FRESH    = 0.03   # бонус за свежесть (last_refreshed_at vs freshness_hours)
W_OPENED   = 0.02   # бонус за популярность (monthly_opens, log1p скейл)

# --- новые флаги/веса ---
EXPLAIN = os.getenv("EXPLAIN", "1") == "1"   # печатать оценки и причины
W_NEG = 0.20                                  # штраф за анти-совпадения

STRICT_INTENT = os.getenv("STRICT_INTENT", "1") == "1"
BLOCK_ANTI = os.getenv("BLOCK_ANTI", "1") == "1"

# Для fallback-эвристики если LLM недоступен
INTENT_SYNONYMS = {
    "sales":   ["выбытие", "продажа", "продажи", "реализация", "экспорт", "списание"],
    "stocks":  ["остатки", "остаток", "запасы"],
    "movement":["движение", "перемещение", "поступление", "приход", "расход", "трансфер"],
}

# ---------------------------
# Утилиты
# ---------------------------

def normalize(x: Any) -> str:
    if x is None:
        return ""
    return str(x).strip()

def as_list(x: Any) -> List[str]:
    if x is None:
        return []
    if isinstance(x, list):
        return [normalize(i) for i in x if normalize(i)]
    return [normalize(x)] if normalize(x) else []

def now_utc_naive() -> datetime:
    # простая "naive" метка времени
    return datetime.utcnow()

# ---------------------------
# LLM: парсинг запроса через Ollama (Mistral)
# ---------------------------

def parse_query_with_llm(user_query: str) -> Dict[str, Any]:
    """
    Возвращает словарь
    {
      "intent": "sales|stocks|movement|other",
      "metrics": [..],
      "dimensions": [..],
      "regions": [..],
      "brands": [..],
      "systems": [..],
      "data_sources": [..],
      "must_have": [..],   # обязательно хорошо бы встретить в карточке
      "nice_to_have": [..] # желательно встретить в карточке
    }
    """
    prompt = f"""
Ты — системный аналитик BI. Пользователь задает вопрос о том, где найти данные/метрики в дашбордах.
Задача — сжато выделить суть запроса, НО вернуть строго JSON без лишних слов.

Поле intent выбери из: "sales", "stocks", "movement", "other".
- "sales" — выбытие/продажи/реализация/экспорт/списание
- "stocks" — остатки/запасы
- "movement" — движение/приход/расход/перемещения
- "other" — если неочевидно

Верни JSON с ключами:
{{
  "intent": "...",
  "metrics": ["..."],
  "dimensions": ["..."],
  "regions": ["..."],
  "brands": ["..."],
  "systems": ["..."],
  "data_sources": ["..."],
  "must_have": ["..."],
  "nice_to_have": ["..."]
}}

Правила:
- Коротко, по смыслу.
- Все списки — массивы строк (могут быть пустыми).
- Никаких комментариев снаружи JSON.

Запрос: \"\"\"{user_query}\"\"\" 
""".strip()

    try:
        resp = requests.post(
            f"{OLLAMA_BASE}/api/generate",
            json={
                "model": OLLAMA_MODEL,
                "prompt": prompt,
                "stream": False,
                "format": "json"
            },
            timeout=45
        )
        resp.raise_for_status()
        data = resp.json()
        raw = data.get("response", "{}")
        parsed = json.loads(raw)
        # нормализуем
        return {
            "intent": parsed.get("intent", "other"),
            "metrics": as_list(parsed.get("metrics")),
            "dimensions": as_list(parsed.get("dimensions")),
            "regions": as_list(parsed.get("regions")),
            "brands": as_list(parsed.get("brands")),
            "systems": as_list(parsed.get("systems")),
            "data_sources": as_list(parsed.get("data_sources")),
            "must_have": as_list(parsed.get("must_have")),
            "nice_to_have": as_list(parsed.get("nice_to_have")),
            "raw_query": user_query,
        }
    except Exception as e:
        if SHOW_DEBUG:
            print(f"[WARN] LLM parse failed: {e}", file=sys.stderr)
        # Fallback: очень простая эвристика
        ql = user_query.lower()
        intent = "other"
        for k, syns in INTENT_SYNONYMS.items():
            if any(s in ql for s in syns):
                intent = k
                break
        return {
            "intent": intent,
            "metrics": [],
            "dimensions": [],
            "regions": [],
            "brands": [],
            "systems": [],
            "data_sources": [],
            "must_have": [],
            "nice_to_have": [],
            "raw_query": user_query,
        }

# ---------------------------
# ClickHouse
# ---------------------------

def ch_client():
    return clickhouse_connect.get_client(
        host=CH_HOST, port=CH_PORT, username=CH_USER, password=CH_PASSWORD, database=CH_DB
    )

def load_catalog(client) -> List[Dict[str, Any]]:
    # Берём ключевые поля; embedding не используется в v1
    q = f"""
    SELECT
      id, system, url, title, description,
      business_questions, metrics, dimensions, tags, data_sources, region_scope,
      owner_name, owner_contact, access_policy,
      freshness_hours, last_refreshed_at, monthly_opens,
      is_active, info_type
    FROM {CH_DB}.{CATALOG_TABLE}
    WHERE is_active = 1
    """
    res = client.query(q)
    cols = res.column_names
    out = []
    for row in res.result_rows:
        item = {cols[i]: row[i] for i in range(len(cols))}
        out.append(item)
    return out

# ---------------------------
# Скоринг
# ---------------------------
def anti_hits(user_intent: str, hay: str):
    """
    Анти-слова: все синонимы НЕ целевого интента.
    Для intent='sales' дополнительно жёстко блокируем явные 'остатк/запас' без упоминаний продаж.
    """
    hay_l = hay.lower()
    anti = []
    for k, syns in INTENT_SYNONYMS.items():
        if k != user_intent:
            anti += [s.lower() for s in syns]
    hits = [w for w in anti if w in hay_l]

    strong_block = False
    if user_intent == "sales":
        if ("остат" in hay_l or "запас" in hay_l or "stock" in hay_l) and not (
            "продаж" in hay_l or "реализац" in hay_l or "sales" in hay_l
        ):
            strong_block = True
    return hits, strong_block


def matched_keywords(parsed: Dict[str, Any], hay: str) -> Dict[str, List[str]]:
    """Показывает, какие слова из парсинга реально встретились в карточке."""
    hay_l = hay.lower()

    def filt(lst: List[str]) -> List[str]:
        out = []
        for w in lst or []:
            w = (w or "").strip()
            if not w:
                continue
            # простая подстрока, без морфологии; при желании можно заменить на нормализацию
            if w.lower() in hay_l:
                out.append(w)
        return out

    res = {}
    for key in ["must_have","metrics","dimensions","regions","brands","systems","data_sources","nice_to_have"]:
        res[key] = filt(parsed.get(key, []))
    return res

def make_card_text(card: Dict[str, Any]) -> str:
    # склеиваем основные поля в одну строку для fuzzy-поиска
    parts = [
        normalize(card.get("title")),
        normalize(card.get("description")),
        " ".join(as_list(card.get("business_questions"))),
        " ".join(as_list(card.get("metrics"))),
        " ".join(as_list(card.get("dimensions"))),
        " ".join(as_list(card.get("tags"))),
        " ".join(as_list(card.get("data_sources"))),
        " ".join(as_list(card.get("region_scope"))),
        normalize(card.get("system")),
        normalize(card.get("owner_name")),
    ]
    return " ".join([p for p in parts if p]).strip()

def intent_bonus(user_intent: str, card_info_type: str) -> float:
    if not user_intent or not card_info_type:
        return 0.0
    u = user_intent.lower()
    c = str(card_info_type).lower()
    return 1.0 if (u == c or u in c or c in u) else 0.0

def keywords_bonus(parsed: Dict[str, Any], hay: str) -> float:
    hay_l = hay.lower()
    score = 0.0
    # must_have сильнее
    for w in parsed.get("must_have", []):
        if w and w.lower() in hay_l:
            score += 0.10
    # nice_to_have + метрики/измерения/регионы/бренды/системы/источники чуточку
    buckets = ["nice_to_have", "metrics", "dimensions", "regions", "brands", "systems", "data_sources"]
    for b in buckets:
        for w in parsed.get(b, []):
            if w and w.lower() in hay_l:
                score += 0.02
    return min(score, 0.30)  # ограничим вклад

def freshness_bonus(card: Dict[str, Any]) -> float:
    try:
        fr_h = float(card.get("freshness_hours") or 0.0)
        last = card.get("last_refreshed_at")
        if not last:
            return 0.0
        # last_refreshed_at обычно datetime; clickhouse-connect вернёт python datetime
        if not isinstance(last, datetime):
            return 0.0
        hours_ago = (now_utc_naive() - last).total_seconds() / 3600.0
        return 1.0 if (fr_h > 0 and hours_ago <= fr_h) else 0.0
    except Exception:
        return 0.0

def opened_bonus(card: Dict[str, Any]) -> float:
    try:
        mo = float(card.get("monthly_opens") or 0.0)
        return min(1.0, math.log1p(max(0.0, mo)) / 6.0)  # мягкая нормализация
    except Exception:
        return 0.0

def score_card(parsed: Dict[str, Any], card: Dict[str, Any]) -> Dict[str, Any]:
    hay = make_card_text(card)

    # компоненты
    sem = fuzz.token_set_ratio(parsed["raw_query"], hay) / 100.0
    ib  = intent_bonus(parsed.get("intent",""), normalize(card.get("info_type")))
    kb  = keywords_bonus(parsed, hay)
    fb  = freshness_bonus(card)
    ob  = opened_bonus(card)

    anti_list, strong_block = anti_hits(parsed.get("intent",""), hay)
    neg = min(1.0, 0.05 * len(anti_list))  # мягкий штраф за кол-во анти-слов

    # итоговый скор
    score = (W_SEMANTIC*sem) + (W_INTENT*ib) + (W_KEYWORD*kb) + (W_FRESH*fb) + (W_OPENED*ob) - (W_NEG*neg)

    # матчи ключевых слов
    m = matched_keywords(parsed, hay)

    return {
        "score": float(score),
        "parts": {
            "sem": sem, "sem_x": W_SEMANTIC*sem,
            "intent": ib, "intent_x": W_INTENT*ib,
            "kw": kb, "kw_x": W_KEYWORD*kb,
            "fresh": fb, "fresh_x": W_FRESH*fb,
            "opens": ob, "opens_x": W_OPENED*ob,
            "anti": neg, "anti_x": W_NEG*neg,
        },
        "details": {
            "info_type": normalize(card.get("info_type")),
            "exact_match": normalize(card.get("info_type")).lower() == parsed.get("intent","").lower(),
            "anti_hits": anti_list,
            "strong_block": strong_block,
            "matched": m,
        }
    }

# ---------------------------
# Печать результатов
# ---------------------------

def print_results(ranked: List[Dict[str, Any]]):
    if not ranked:
        print("Ничего не нашёл. Попробуйте переформулировать запрос.")
        return

    for i, r in enumerate(ranked, 1):
        c = r["card"]
        title = normalize(c.get("title"))
        url = normalize(c.get("url"))
        score = r["score"]
        line = f"{i}) {title}" + (f" — {url}" if url else "")
        print(line)

        if EXPLAIN:
            p = r["parts"]
            d = r["details"]
            m = d["matched"]

            # компактное разложение вклада
            print(
                "    "
                f"score={score:.3f} | "
                f"sem={p['sem']:.2f}×{W_SEMANTIC:.2f}→{p['sem_x']:.3f}; "
                f"intent={p['intent']:.2f}×{W_INTENT:.2f}→{p['intent_x']:.3f}; "
                f"kw={p['kw']:.2f}×{W_KEYWORD:.2f}→{p['kw_x']:.3f}; "
                f"fresh={p['fresh']:.2f}×{W_FRESH:.2f}→{p['fresh_x']:.3f}; "
                f"opens={p['opens']:.2f}×{W_OPENED:.2f}→{p['opens_x']:.3f}; "
                f"anti={p['anti']:.2f}×{W_NEG:.2f}→-{p['anti_x']:.3f}"
            )
            # причины: что совпало и что помешало
            matched_str = []
            for k in ["must_have","metrics","dimensions","regions","brands","systems","data_sources","nice_to_have"]:
                if m.get(k):
                    matched_str.append(f"{k}={m[k]}")
            if matched_str:
                print(f"    matched: {', '.join(matched_str)}")
            if d["anti_hits"]:
                print(f"    anti_hits: {d['anti_hits']}")
            print(f"    info_type={d['info_type']}; exact_intent={d['exact_match']}; strong_block={d['strong_block']}")

# ---------------------------
# Main
# ---------------------------

def main():
    client = ch_client()

    # 1) спросим запрос
    try:
        user_query = input("Введите ваш запрос: ").strip()
    except EOFError:
        user_query = ""

    if not user_query:
        print("Пустой запрос. Завершение.")
        return

    # 2) распарсим запрос через LLM
    parsed = parse_query_with_llm(user_query)
    if SHOW_DEBUG:
        print("[DEBUG] Parsed:", json.dumps(parsed, ensure_ascii=False, indent=2))

    # 3) загрузим активный каталог
    cards = load_catalog(client)
    if not cards:
        print("Каталог пустой или не найден.")
        return

    # 4) посчитаем скор и отберём
    # 4) посчитаем скор и решим, что показать
    scored = []
    for card in cards:
        s = score_card(parsed, card)

        # жёсткий блок по анти-словам
        if BLOCK_ANTI and s["details"]["strong_block"]:
            if SHOW_DEBUG:
                print(f"[DEBUG] DROP (strong anti) → {card.get('title')}")
            continue

        scored.append({"card": card, **s})

    # если STRICT_INTENT=1 и есть хотя бы одна exact_match карточка — оставляем только их
    if STRICT_INTENT and any(it["details"]["exact_match"] for it in scored):
        scored = [it for it in scored if it["details"]["exact_match"]]

    # применяем порог и сортируем
    ranked = [it for it in scored if it["score"] >= SHOW_THRESHOLD]
    ranked.sort(key=lambda x: x["score"], reverse=True)
    ranked = ranked[:TOP_K]

    # 5) выведем
    print_results(ranked)


if __name__ == "__main__":
    main()