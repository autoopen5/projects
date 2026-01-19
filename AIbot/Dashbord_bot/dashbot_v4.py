# dashbot_v2_explain.py
# Консольный бот: спрашивает запрос → считает эмбеддинг (Ollama) → тянет каталог из ClickHouse →
# считает cosine similarity только по эмбеддингам → печатает ТОЛЬКО близкие карточки с пояснениями.

import os
import sys
import json
import requests
from datetime import datetime
from typing import Dict, Any, List

import clickhouse_connect

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
OLLAMA_EMB_MODEL = os.getenv("OLLAMA_EMB_MODEL", "nomic-embed-text")

# Порог близости (косинусная схожесть 0..1). Всё, что ниже — отбрасываем.
SIM_THRESHOLD = float(os.getenv("SIM_THRESHOLD", os.getenv("SHOW_THRESHOLD", "0.65")))

EXPLAIN = os.getenv("EXPLAIN", "1") == "1"
SHOW_DEBUG = os.getenv("DASHBOT_DEBUG", "0") == "1"


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
# Эмбеддинги и косинус
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
        emb = r.json().get("embedding", []) or []
        # Приводим к float, на всякий случай
        return [float(v) for v in emb]
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
# ClickHouse
# ───────────────────────────────────────────────────────────────────────────────
def ch_client():
    return clickhouse_connect.get_client(
        host=CH_HOST, port=CH_PORT, username=CH_USER, password=CH_PASSWORD, database=CH_DB
    )

def load_catalog(client) -> List[Dict[str, Any]]:
    # ВАЖНО: тянем колонку embedding!
    q = f"""
    SELECT
      id, system, url, title, description,
      business_questions, metrics, dimensions, tags, data_sources, region_scope,
      owner_name, owner_contact, access_policy,
      freshness_hours, last_refreshed_at, monthly_opens,
      is_active, info_type, embedding
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
# Скоринг ТОЛЬКО эмбеддингами
# ───────────────────────────────────────────────────────────────────────────────
def make_card_text(card: Dict[str, Any]) -> str:
    # Используется только для explain (показать matched поля), не влияет на счёт
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

def score_card(card: Dict[str, Any], query_emb: List[float]) -> Dict[str, Any]:
    card_emb = card.get("embedding") or []
    emb_sim = cosine_sim(query_emb, card_emb) if query_emb and card_emb else 0.0

    return {
        "score": float(emb_sim),
        "parts": {
            "emb": emb_sim,   # единственный вклад
        },
        "details": {
            "has_card_embedding": bool(card_emb),
            "info_type": normalize(card.get("info_type")),
            "title": normalize(card.get("title")),
            "url": normalize(card.get("url")),
        }
    }


# ───────────────────────────────────────────────────────────────────────────────
# Вывод результатов
# ───────────────────────────────────────────────────────────────────────────────
def print_results(ranked: List[Dict[str, Any]]):
    if not ranked:
        print("Ничего близкого к запросу не нашёл (по порогу схожести).")
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
            print(
                "    "
                f"score={score:.3f} | emb={p['emb']:.3f} (cosine similarity)"
            )
            if d.get("info_type"):
                print(f"    info_type={d['info_type']}")
            print(f"    has_embedding={d['has_card_embedding']}")


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

    query_emb = get_embedding_ollama(user_query)
    if not query_emb:
        print("Не удалось получить эмбеддинг запроса — проверьте Ollama и модель эмбеддингов.")
        return

    cards = load_catalog(client)
    if not cards:
        print("Каталог пустой или не найден.")
        return

    scored = []
    for card in cards:
        s = score_card(card, query_emb)
        # фильтруем только близкие
        if s["score"] >= SIM_THRESHOLD:
            scored.append({"card": card, **s})

    # сортируем по убыванию близости, БЕЗ top-k
    ranked = sorted(scored, key=lambda x: x["score"], reverse=True)

    print_results(ranked)


if __name__ == "__main__":
    main()