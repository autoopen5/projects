# dashbot_v2_explain.py
# Консольный бот: спрашивает запрос → парсит через Ollama (Mistral) → тянет каталог из ClickHouse →
# скорит карточки → печатает рекомендации И подробные оценки (почему карточка попала в выдачу).

import os
import sys
import json
import math
import requests
from datetime import datetime
from typing import Dict, Any, List

import clickhouse_connect
from rapidfuzz import fuzz


# ───────────────────────────────────────────────────────────────────────────────
# Конфиг
# ───────────────────────────────────────────────────────────────────────────────
CH_HOST = os.getenv("CLICKHOUSE_HOST", "clickhouse.moscow")
CH_PORT = int(os.getenv("CLICKHOUSE_PORT", "8123"))
CH_USER = os.getenv("CLICKHOUSE_USER", "GrushkoIV")
CH_PASSWORD = os.getenv("CLICKHOUSE_PASSWORD", "jNbrvzd1IcF0Yx5I")
CH_DB = os.getenv("CLICKHOUSE_DB", "grushko_iv")
CATALOG_TABLE = "dashboard_catalog"

OLLAMA_BASE = os.getenv("OLLAMA_BASE", "http://localhost:11434").rstrip("/")
OLLAMA_MODEL = os.getenv("OLLAMA_MODEL", "mistral")
OLLAMA_EMB_MODEL = os.getenv("OLLAMA_EMB_MODEL", "nomic-embed-text")
W_EMB = float(os.getenv("W_EMB", "0.50"))   # вес эмбеддинг-схожести в итоговом скоре

TOP_K = int(os.getenv("TOP_K", "5"))
SHOW_THRESHOLD = float(os.getenv("SHOW_THRESHOLD", "0.33"))

EXPLAIN = os.getenv("EXPLAIN", "1") == "1"           # печатать разложение оценок
STRICT_INTENT = os.getenv("STRICT_INTENT", "1") == "1"
BLOCK_ANTI = os.getenv("BLOCK_ANTI", "1") == "1"

SHOW_DEBUG = os.getenv("DASHBOT_DEBUG", "0") == "1"

# Весовые коэффициенты
W_SEMANTIC = 0.70   # fuzzy похожесть запроса и карточки
W_INTENT   = 0.15   # совпадение info_type и intent
W_KEYWORD  = 0.10   # бонус за совпавшие must/metrics/dimensions/regions/brands/systems/data_sources
W_FRESH    = 0.03   # свежесть (last_refreshed_at vs freshness_hours)
W_OPENED   = 0.02   # популярность (monthly_opens)
W_NEG      = 0.20   # штраф за «анти-слова» чужого интента

# Синонимы для интентов
INTENT_SYNONYMS = {
    "sales":   ["выбытие","продажа","продажи","реализация","экспорт","списание","sales","sell","outflow"],
    "stocks":  ["остатки","остаток","запасы","stock","stocks","на остатках"],
    "movement":["движение","перемещение","поступление", "закуп","приход","расход","трансфер","movement","inflow","transfer"],
}


# ───────────────────────────────────────────────────────────────────────────────
# Утилиты
# ───────────────────────────────────────────────────────────────────────────────
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
    return datetime.utcnow()

# ───────────────────────────────────────────────────────────────────────────────
# Функции для эмбеддингов и косинуса
# ───────────────────────────────────────────────────────────────────────────────

def get_embedding_ollama(text: str) -> List[float]:
    if not text.strip():
        return []
    try:
        r = requests.post(
            f"{OLLAMA_BASE}/api/embeddings",
            json={"model": OLLAMA_EMB_MODEL, "prompt": text},
            timeout=30
        )
        r.raise_for_status()
        return r.json().get("embedding", []) or []
    except Exception as e:
        if SHOW_DEBUG:
            print(f"[WARN] embedding failed: {e}", file=sys.stderr)
        return []

def cosine_sim(a: List[float], b: List[float]) -> float:
    # безопасный косинус 0..1; при пустых — 0
    if not a or not b or len(a) != len(b):
        return 0.0
    s_ab = 0.0; s_a = 0.0; s_b = 0.0
    for i in range(len(a)):
        x = float(a[i]); y = float(b[i])
        s_ab += x*y; s_a += x*x; s_b += y*y
    denom = (s_a**0.5)*(s_b**0.5)
    return (s_ab/denom) if denom > 0 else 0.0


# ───────────────────────────────────────────────────────────────────────────────
# LLM: парсинг запроса через Ollama (Mistral)
# ───────────────────────────────────────────────────────────────────────────────
def parse_query_with_llm(user_query: str) -> Dict[str, Any]:
    """
    Возвращает строго JSON-структуру:
    {
      "intent": "sales|stocks|movement|other",
      "metrics": [...],
      "dimensions": [...],
      "regions": [...],
      "brands": [...],
      "systems": [...],
      "data_sources": [...],
      "must_have": [...],
      "nice_to_have": [...]
    }
    """
    prompt = f"""
    Ты — системный аналитик BI. Выдели суть запроса и верни строго JSON (без комментариев снаружи).

    intent ∈ {{ "sales","stocks","movement","other" }}
    - "sales" — выбытие/продажи/реализация/экспорт/списание
    - "stocks" — остатки/запасы
    - "movement" — движение/приход/расход/перемещения/закуп/закупки/перемещение

    Схема ответа:
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

    Запрос: \"\"\"{user_query}\"\"\"
    """.strip()

    try:
        resp = requests.post(
            f"{OLLAMA_BASE}/api/generate",
            json={"model": OLLAMA_MODEL, "prompt": prompt, "stream": False, "format": "json"},
            timeout=45
        )
        resp.raise_for_status()
        data = resp.json()
        raw = data.get("response", "{}")
        parsed = json.loads(raw)
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
        # Простой fallback по ключевым словам
        ql = user_query.lower()
        intent = "other"
        for k, syns in INTENT_SYNONYMS.items():
            if any(s in ql for s in syns):
                intent = k
                break
        return {
            "intent": intent,
            "metrics": [], "dimensions": [], "regions": [], "brands": [],
            "systems": [], "data_sources": [], "must_have": [], "nice_to_have": [],
            "raw_query": user_query,
        }


# ───────────────────────────────────────────────────────────────────────────────
# ClickHouse
# ───────────────────────────────────────────────────────────────────────────────
def ch_client():
    return clickhouse_connect.get_client(
        host=CH_HOST, port=CH_PORT, username=CH_USER, password=CH_PASSWORD, database=CH_DB
    )

def load_catalog(client) -> List[Dict[str, Any]]:
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
        out.append({cols[i]: row[i] for i in range(len(cols))})
    return out


# ───────────────────────────────────────────────────────────────────────────────
# Скоринг
# ───────────────────────────────────────────────────────────────────────────────
def make_card_text(card: Dict[str, Any]) -> str:
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
    for w in parsed.get("must_have", []):
        if w and w.lower() in hay_l:
            score += 0.10
    buckets = ["nice_to_have","metrics","dimensions","regions","brands","systems","data_sources"]
    for b in buckets:
        for w in parsed.get(b, []):
            if w and w.lower() in hay_l:
                score += 0.02
    return min(score, 0.30)

def freshness_bonus(card: Dict[str, Any]) -> float:
    try:
        fr_h = float(card.get("freshness_hours") or 0.0)
        last = card.get("last_refreshed_at")
        if not last or not isinstance(last, datetime):
            return 0.0
        hours_ago = (now_utc_naive() - last).total_seconds() / 3600.0
        return 1.0 if (fr_h > 0 and hours_ago <= fr_h) else 0.0
    except Exception:
        return 0.0

def opened_bonus(card: Dict[str, Any]) -> float:
    try:
        mo = float(card.get("monthly_opens") or 0.0)
        return min(1.0, math.log1p(max(0.0, mo)) / 6.0)
    except Exception:
        return 0.0

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
    """Какие слова из парсинга реально встретились в карточке."""
    hay_l = hay.lower()
    def filt(lst: List[str]) -> List[str]:
        out = []
        for w in lst or []:
            w = (w or "").strip()
            if w and w.lower() in hay_l:
                out.append(w)
        return out
    res = {}
    for key in ["must_have","metrics","dimensions","regions","brands","systems","data_sources","nice_to_have"]:
        res[key] = filt(parsed.get(key, []))
    return res

def score_card(parsed: Dict[str, Any], card: Dict[str, Any], query_emb: List[float]) -> Dict[str, Any]:
    hay = make_card_text(card)

    # семантика по тексту (fuzzy + расширение запроса, если у тебя уже есть expand_query — используй её)
    aug_query = parsed["raw_query"]  # если используешь expand_query, подмени здесь
    sem = fuzz.token_set_ratio(aug_query, hay) / 100.0

    ib  = intent_bonus(parsed.get("intent",""), normalize(card.get("info_type")))
    kb  = keywords_bonus(parsed, hay)
    fb  = freshness_bonus(card)
    ob  = opened_bonus(card)

    # --- эмбеддинги ---
    card_emb = card.get("embedding") or []  # Array(Float32) из БД
    emb_sim = cosine_sim(query_emb, card_emb) if query_emb and card_emb else 0.0

    anti_list, strong_block = anti_hits(parsed.get("intent",""), hay)
    neg = min(1.0, 0.05 * len(anti_list))

    score = (
        (W_SEMANTIC*sem) + (W_INTENT*ib) + (W_KEYWORD*kb) +
        (W_FRESH*fb) + (W_OPENED*ob) + (W_EMB*emb_sim) -
        (W_NEG*neg)
    )

    m = matched_keywords(parsed, hay)

    return {
        "score": float(score),
        "parts": {
            "sem": sem,       "sem_x": W_SEMANTIC*sem,
            "intent": ib,     "intent_x": W_INTENT*ib,
            "kw": kb,         "kw_x": W_KEYWORD*kb,
            "fresh": fb,      "fresh_x": W_FRESH*fb,
            "opens": ob,      "opens_x": W_OPENED*ob,
            "emb": emb_sim,   "emb_x": W_EMB*emb_sim,
            "anti": neg,      "anti_x": W_NEG*neg,
        },
        "details": {
            "info_type": normalize(card.get("info_type")),
            "exact_match": normalize(card.get("info_type")).lower() == parsed.get("intent","").lower(),
            "anti_hits": anti_list,
            "strong_block": strong_block,
            "matched": m,
            "has_card_embedding": bool(card_emb),
        }
    }


# ───────────────────────────────────────────────────────────────────────────────
# Вывод результатов
# ───────────────────────────────────────────────────────────────────────────────
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
            print(
                "    "
                f"score={score:.3f} | "
                f"sem={p['sem']:.2f}×{W_SEMANTIC:.2f}→{p['sem_x']:.3f}; "
                f"intent={p['intent']:.2f}×{W_INTENT:.2f}→{p['intent_x']:.3f}; "
                f"kw={p['kw']:.2f}×{W_KEYWORD:.2f}→{p['kw_x']:.3f}; "
                f"emb={p['emb']:.2f}×{W_EMB:.2f}→{p['emb_x']:.3f}; "   # <── добавили
                f"fresh={p['fresh']:.2f}×{W_FRESH:.2f}→{p['fresh_x']:.3f}; "
                f"opens={p['opens']:.2f}×{W_OPENED:.2f}→{p['opens_x']:.3f}; "
                f"anti={p['anti']:.2f}×{W_NEG:.2f}→-{p['anti_x']:.3f}"
            )

            matched_str = []
            for k in ["must_have","metrics","dimensions","regions","brands","systems","data_sources","nice_to_have"]:
                if m.get(k):
                    matched_str.append(f"{k}={m[k]}")
            if matched_str:
                print(f"    matched: {', '.join(matched_str)}")
            if d["anti_hits"]:
                print(f"    anti_hits: {d['anti_hits']}")
            print(f"    info_type={d['info_type']}; exact_intent={d['exact_match']}; strong_block={d['strong_block']}")


# ───────────────────────────────────────────────────────────────────────────────
# Main
# ───────────────────────────────────────────────────────────────────────────────
def main():
    client = ch_client()

    try:
        user_query = input("Введите ваш запрос: ").strip()
    except EOFError:
        user_query = ""

    if not user_query:
        print("Пустой запрос. Завершение.")
        return

    parsed = parse_query_with_llm(user_query)
    query_emb = get_embedding_ollama(parsed["raw_query"])
    if SHOW_DEBUG:
        print("[DEBUG] Parsed:", json.dumps(parsed, ensure_ascii=False, indent=2))

    cards = load_catalog(client)
    if not cards:
        print("Каталог пустой или не найден.")
        return

    scored = []
    for card in cards:
        s = score_card(parsed, card, query_emb)
        if BLOCK_ANTI and s["details"]["strong_block"]:
            if SHOW_DEBUG:
                print(f"[DEBUG] DROP (strong anti) → {card.get('title')}")
            continue
        scored.append({"card": card, **s})

    if STRICT_INTENT and any(it["details"]["exact_match"] for it in scored):
        scored = [it for it in scored if it["details"]["exact_match"]]

    ranked = [it for it in scored if it["score"] >= SHOW_THRESHOLD]
    ranked.sort(key=lambda x: x["score"], reverse=True)
    ranked = ranked[:TOP_K]

    print_results(ranked)


if __name__ == "__main__":
    main()