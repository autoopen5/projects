# dashbot_v2_dist_all.py
# Запрос → embedding (SentenceTransformer) → загрузка каталога из ClickHouse →
# расчёт КОСИНУСНОГО РАССТОЯНИЯ до КАЖДОЙ карточки → печать всего списка.

import os
import sys
from typing import Any, List, Dict

import clickhouse_connect
from sentence_transformers import SentenceTransformer

# ───────────────────────────────────────────────────────────────────────────────
# Конфиг через переменные окружения
# ───────────────────────────────────────────────────────────────────────────────
CH_HOST = os.getenv("CLICKHOUSE_HOST", "clickhouse.moscow")
CH_PORT = int(os.getenv("CLICKHOUSE_PORT", "8123"))
CH_USER = os.getenv("CLICKHOUSE_USER", "GrushkoIV")
CH_PASSWORD = os.getenv("CLICKHOUSE_PASSWORD", "jNbrvzd1IcF0Yx5I")
CH_DB = os.getenv("CLICKHOUSE_DB", "grushko_iv")
CATALOG_TABLE = os.getenv("CATALOG_TABLE", "dashboard_catalog")

# Модель ДОЛЖНА совпадать с той, которой индексировали каталог
EMBED_MODEL = os.getenv("EMBED_MODEL", "sentence-transformers/all-MiniLM-L6-v2")
# Если при индексации делали normalize_embeddings=True — поставьте NORMALIZE=1
NORMALIZE = os.getenv("NORMALIZE", "0") == "1"

# Печатать ли отладочную инфу
DEBUG = os.getenv("DASHBOT_DEBUG", "0") == "1"

# ───────────────────────────────────────────────────────────────────────────────
# Вспомогательные функции
# ───────────────────────────────────────────────────────────────────────────────
def normalize(x: Any) -> str:
    return "" if x is None else str(x).strip()

def cosine_sim(a: List[float], b: List[float]) -> float:
    """Косинусная схожесть ∈ [-1, 1]. При пустых/разной размерности — 0."""
    if not a or not b or len(a) != len(b):
        return 0.0
    s_ab = 0.0
    s_a = 0.0
    s_b = 0.0
    for i in range(len(a)):
        x = float(a[i]); y = float(b[i])
        s_ab += x * y
        s_a += x * x
        s_b += y * y
    denom = (s_a ** 0.5) * (s_b ** 0.5)
    return (s_ab / denom) if denom > 0 else 0.0

def cosine_distance(sim: float) -> float:
    """Косинусное расстояние по определению ClickHouse: 1 - cosine_similarity."""
    return 1.0 - sim

# ───────────────────────────────────────────────────────────────────────────────
# Инициализация
# ───────────────────────────────────────────────────────────────────────────────
def ch_client():
    return clickhouse_connect.get_client(
        host=CH_HOST, port=CH_PORT, username=CH_USER, password=CH_PASSWORD, database=CH_DB
    )

def load_catalog(client) -> List[Dict[str, Any]]:
    """
    ВАЖНО: выбираем колонку embedding из таблицы. Предполагается, что это Array(Float32).
    """
    q = f"""
    SELECT
        id, system, url, title, description,
        business_questions, metrics, dimensions, tags, data_sources, region_scope,
        info_type, monthly_opens, embedding
    FROM {CH_DB}.{CATALOG_TABLE}
    WHERE is_active = 1
    """
    res = client.query(q)
    return res.named_results()

def load_st_model() -> SentenceTransformer:
    if DEBUG:
        print(f"[DEBUG] Loading SentenceTransformer: {EMBED_MODEL}, normalize={NORMALIZE}")
    return SentenceTransformer(EMBED_MODEL)

def embed_text(model: SentenceTransformer, text: str) -> List[float]:
    if not text.strip():
        return []
    vec = model.encode(text, normalize_embeddings=NORMALIZE)
    return [float(v) for v in vec.tolist()]

# ───────────────────────────────────────────────────────────────────────────────
# Расчёт расстояний и вывод
# ───────────────────────────────────────────────────────────────────────────────
def compute_distances(cards: List[Dict[str, Any]], q_emb: List[float]) -> Dict[str, List[Dict[str, Any]]]:
    rows_valid: List[Dict[str, Any]] = []
    rows_invalid: List[Dict[str, Any]] = []

    for card in cards:
        title = normalize(card.get("title"))
        url = normalize(card.get("url"))
        info_type = normalize(card.get("info_type"))
        system = normalize(card.get("system"))
        c_emb = card.get("embedding") or []

        if not c_emb:
            rows_invalid.append({
                "title": title, "url": url, "info_type": info_type, "system": system,
                "reason": "no_embedding"
            })
            continue

        try:
            c_emb = [float(v) for v in c_emb]
        except Exception:
            rows_invalid.append({
                "title": title, "url": url, "info_type": info_type, "system": system,
                "reason": "bad_embedding_values"
            })
            continue

        if len(c_emb) != len(q_emb):
            rows_invalid.append({
                "title": title, "url": url, "info_type": info_type, "system": system,
                "reason": f"dim_mismatch({len(c_emb)} vs {len(q_emb)})"
            })
            continue

        sim = cosine_sim(q_emb, c_emb)
        dist = cosine_distance(sim)
        rows_valid.append({
            "title": title,
            "url": url,
            "info_type": info_type,
            "system": system,
            "similarity": sim,
            "distance": dist
        })

    # сортируем по возрастанию distance (т.е. «ближайшие» сначала)
    rows_valid.sort(key=lambda r: r["distance"])
    return {"valid": rows_valid, "invalid": rows_invalid}

def print_all_distances(result: Dict[str, List[Dict[str, Any]]]):
    valid = result["valid"]
    invalid = result["invalid"]

    if not valid and not invalid:
        print("Каталог пуст или нет ни одной карточки.")
        return

    if valid:
        print("=== Косинусное расстояние до ВСЕХ дашбордов (считалось как 1 - cosine_similarity) ===")
        for i, r in enumerate(valid, 1):
            line = f"{i}) {r['title']}"
            if r['url']:
                line += f" — {r['url']}"
            print(line)
            print(f"    distance={r['distance']:.6f}; similarity={r['similarity']:.6f}"
                  f"{' ; info_type='+r['info_type'] if r['info_type'] else ''}"
                  f"{' ; system='+r['system'] if r['system'] else ''}")
    else:
        print("Нет карточек с корректными эмбеддингами и совпадающими размерностями.")

    if invalid:
        print("\n--- Без значения расстояния (причина) ---")
        for r in invalid:
            line = f"- {r['title']}"
            if r.get('url'):
                line += f" — {r['url']}"
            print(line + f"    [{r['reason']}]")

# ───────────────────────────────────────────────────────────────────────────────
# Main
# ───────────────────────────────────────────────────────────────────────────────
def main():
    # 1) Текст запроса
    if len(sys.argv) > 1:
        user_query = " ".join(sys.argv[1:]).strip()
    else:
        try:
            user_query = input("Введите ваш запрос: ").strip()
        except EOFError:
            user_query = ""

    if not user_query:
        print("Пустой запрос. Завершение.")
        return

    # 2) Модель и embedding запроса
    model = load_st_model()
    q_emb = embed_text(model, user_query)
    if not q_emb:
        print("Не удалось получить эмбеддинг запроса. Проверьте установку sentence-transformers/torch.")
        return

    # 3) Загрузка карточек
    client = ch_client()
    cards = load_catalog(client)
    if not cards:
        print("Каталог пустой или не найден.")
        return

    # 4) Расчёт расстояний и вывод
    result = compute_distances(cards, q_emb)
    print_all_distances(result)

if __name__ == "__main__":
    main()