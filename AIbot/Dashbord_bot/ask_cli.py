# ask_cli.py — с debug и fallback
import os, sys, numpy as np
import clickhouse_connect
from sentence_transformers import SentenceTransformer

TABLE = "grushko_iv.dashboard_catalog"
MODEL_NAME = os.getenv("EMBED_MODEL", "sentence-transformers/all-MiniLM-L6-v2")
TOP_K = 3
THRESHOLD_MAIN = 0.50
THRESHOLD_FALLBACK = 0.35
DEBUG = os.getenv("DEBUG", "0") == "1"

INTENT_TOKENS = {
    "stocks":   {"остат", "остатки", "запас", "запасы", "stock", "onhand", "на складе", "на сегодня"},
    "movement": {"движен", "перемещ", "приход", "расход", "in_cnt", "out_cnt", "закупки", "закуп", "продажи дистрuбьютора"},
    "sales":    {"продаж", "выбыти", "экспорт", "реализац", "avg_price", "выручк"},
}

def detect_intent(q: str):
    ql = q.lower()
    for it, toks in INTENT_TOKENS.items():
        if any(t in ql for t in toks):
            return it
    return None

def cosine(a, b):
    a = np.asarray(a, dtype="float32"); b = np.asarray(b, dtype="float32")
    na = np.linalg.norm(a); nb = np.linalg.norm(b)
    if na == 0.0 or nb == 0.0: return 0.0
    return float(np.dot(a/na, b/nb))

def toks(x):
    if x is None: return set()
    if isinstance(x, str): s = x.lower()
    else: s = " ".join(map(str, x)).lower()
    for c in ",;:/_-|()[]{}":
        s = s.replace(c, " ")
    return set(s.split())

def metadata_gate(query_tokens, title, desc, mets, dims, tags, data_sources):
    # достаточно совпадения хотя бы где-то
    if (toks((title or "") + " " + (desc or "")) & query_tokens): return True
    if ({str(m).lower() for m in (mets or [])} & query_tokens): return True
    if ({str(d).lower() for d in (dims or [])} & query_tokens): return True
    if ({str(t).lower() for t in (tags or [])} & query_tokens): return True
    if ({str(ds).lower() for ds in (data_sources or [])} & query_tokens): return True
    # дополнительная эвристика по стему "остат"
    if any(tok.startswith("остат") for tok in query_tokens) and (
        "stock_units" in (mets or []) or "остатки" in (tags or [])
    ):
        return True
    return False

def fetch_rows(client):
    return client.query(f"""
      SELECT title, url, description, business_questions, metrics, dimensions, tags,
             access_policy, last_refreshed_at, freshness_hours, monthly_opens, embedding, info_type, data_sources
      FROM {TABLE}
      WHERE is_active=1
    """).result_rows

def run_ranking(rows, q_emb, query_tokens, intent, threshold, use_meta_gate=True):
    results = []
    for (title, url, desc, qs, mets, dims, tags, access, last_ref, fresh_h, opens, emb, info_type, data_sources) in rows:
        if info_type != intent:
            if DEBUG: print(f"SKIP[{title}]: info_type={info_type} != {intent}")
            continue
        if not emb:
            if DEBUG: print(f"SKIP[{title}]: empty embedding")
            continue
        if use_meta_gate and not metadata_gate(query_tokens, title, desc, mets, dims, tags, data_sources):
            if DEBUG: print(f"SKIP[{title}]: meta_gate failed")
            continue

        sem = cosine(q_emb, emb)
        if sem < threshold:
            if DEBUG: print(f"SKIP[{title}]: sem {sem:.3f} < {threshold:.3f}")
            continue
        final = sem  # без популярности, чтобы не тянулось нерелевантное
        results.append((final, sem, title, url))
    results.sort(key=lambda x: x[0], reverse=True)
    return results[:TOP_K]

def main():
    query = " ".join(sys.argv[1:]).strip() or input("Вопрос: ").strip()
    if not query:
        print("Пустой вопрос."); return

    intent = detect_intent(query)
    if not intent:
        print(f"\nВопрос: {query}\nНе распознан тип запроса (нужно упоминание: продажи/остатки/движение).\n")
        return

    client = clickhouse_connect.get_client(
    host='clickhouse.moscow',
    port=8123,  # 8123 для HTTP, 9000 для Native (TCP)
    username='GrushkoIV',
    password='jNbrvzd1IcF0Yx5I',
    )
    rows = fetch_rows(client)
    if not rows:
        print("В каталоге нет активных дашбордов."); return

    model = SentenceTransformer(MODEL_NAME)
    q_emb = model.encode(query).astype("float32")
    query_tokens = toks(query)

    # 1) основной прогон (строгий)
    results = run_ranking(rows, q_emb, query_tokens, intent, THRESHOLD_MAIN, use_meta_gate=True)

    # 2) если пусто — fallback (без meta_gate и с меньшим порогом)
    if not results:
        if DEBUG: print("Fallback: выключаю meta_gate и снижаю порог.")
        results = run_ranking(rows, q_emb, query_tokens, intent, THRESHOLD_FALLBACK, use_meta_gate=False)

    print(f"\nВопрос: {query}\nИнтент: {intent}\n")
    if not results:
        print("Ничего не найдено. Проверь:\n- есть ли карточки info_type='stocks'\n- у них заполнен embedding\n- в title/metrics/tags есть что-то про остатки (например 'stock_units', 'Остатки').")
        return

    for i, (final, sem, title, url) in enumerate(results, 1):
        print(f"{i}. {title}\n   {url}\n   score={final:.3f} (sem={sem:.3f})\n")

if __name__ == "__main__":
    main()