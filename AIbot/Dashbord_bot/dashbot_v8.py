# -*- coding: utf-8 -*-
import os, sys, re, numpy as np
import clickhouse_connect
from typing import List, Tuple
from sentence_transformers import SentenceTransformer

TABLE = "grushko_iv.dashboard_catalog"
MODEL_NAME = os.getenv("EMBED_MODEL", "sentence-transformers/all-MiniLM-L6-v2")
TOP_K = 5

# Пороги
THRESH_DOMAIN = 0.30     # "вопрос про MDLP/аналитику?"
THRESH_SEM_MAIN = 0.40   # минимальная семантика к карточке
THRESH_SEM_STRICT = 0.50 # строже, если кандидатов много

# Якоря для определения интента/домена (эмбеддинг-маркеры)
ANCHORS = {
    "domain": "mdlp честный знак аналитика дашборд sku регион клиент аптека остатки продажи движение выбытий",
    "sales":  "выбытие продажи экспорт реализация выручка средняя цена avg price чек клиент регион sku",
    "stocks": "остатки запасы наличие на складе on hand на сегодня склад остаток регион sku",
    "movement":"движение приход расход перемещение поступление transfer inflow outflow регион sku"
}

# Утилиты
def cosine(a, b):
    a = np.asarray(a, dtype="float32"); b = np.asarray(b, dtype="float32")
    na = np.linalg.norm(a); nb = np.linalg.norm(b)
    if na == 0.0 or nb == 0.0: return 0.0
    return float(np.dot(a/na, b/nb))

def normalize_text(s: str) -> str:
    s = (s or "").lower().replace("ё", "е")
    s = re.sub(r"[^a-z0-9а-я\s\-]", " ", s)
    return re.sub(r"\s+", " ", s).strip()

def short_overlap(query: str, text: str, max_terms=6) -> str:
    q = set(normalize_text(query).split())
    t = set(normalize_text(text).split())
    inter = [w for w in q if w in t and len(w) >= 3]
    return ", ".join(inter[:max_terms]) if inter else ""

def connect():
    return clickhouse_connect.get_client(
        host=os.getenv("CH_HOST","localhost"),
        port=int(os.getenv("CH_PORT","8123")),
        username=os.getenv("CH_USER","default"),
        password=os.getenv("CH_PASSWORD","")
    )

def load_catalog():
    client = connect()
    res = client.query(f"""
      SELECT
        toString(id) AS id,
        title,
        url,
        ifNull(description,'') AS description,
        ifNull(business_questions, CAST([], 'Array(String)')) AS business_questions,
        ifNull(metrics, CAST([], 'Array(String)')) AS metrics,
        ifNull(dimensions, CAST([], 'Array(String)')) AS dimensions,
        ifNull(tags, CAST([], 'Array(String)')) AS tags,
        ifNull(data_sources, CAST([], 'Array(String)')) AS data_sources,
        ifNull(info_type,'') AS info_type,
        embedding
      FROM {TABLE}
      WHERE is_active = 1
    """)
    rows = res.result_rows
    return rows

def build_passport(title, desc, bq, metrics, dims, tags, srcs) -> str:
    parts = [
        title or "",
        desc or "",
        "Вопросы: " + "; ".join(bq or []),
        "Метрики: " + ", ".join(metrics or []),
        "Измерения: " + ", ".join(dims or []),
        "Теги: " + ", ".join(tags or []),
        "Источники: " + ", ".join(srcs or []),
    ]
    return "\n".join([p for p in parts if p])

def detect_intent(model, q_emb):
    # по якорям определяем домен и тип
    emb_domain = model.encode(ANCHORS["domain"]).astype("float32")
    emb_sales  = model.encode(ANCHORS["sales"]).astype("float32")
    emb_stocks = model.encode(ANCHORS["stocks"]).astype("float32")
    emb_move   = model.encode(ANCHORS["movement"]).astype("float32")

    s_domain = cosine(q_emb, emb_domain)
    s_sales  = cosine(q_emb, emb_sales)
    s_stocks = cosine(q_emb, emb_stocks)
    s_move   = cosine(q_emb, emb_move)

    if s_domain < THRESH_DOMAIN:
        return None, s_domain, {"sales": s_sales, "stocks": s_stocks, "movement": s_move}

    # выбираем наибольший
    best = max(("sales","stocks","movement"), key=lambda k: {"sales":s_sales,"stocks":s_stocks,"movement":s_move}[k])
    return best, s_domain, {"sales": s_sales, "stocks": s_stocks, "movement": s_move}

def main():
    query = " ".join(sys.argv[1:]).strip() or input("Вопрос: ").strip()
    if not query:
        print("Пустой вопрос."); return

    rows = load_catalog()
    if not rows:
        print("В каталоге нет активных дашбордов."); return

    model = SentenceTransformer(MODEL_NAME)
    q_emb = model.encode(query).astype("float32")

    # 1) доменный гейт + определение интента
    intent, s_domain, intent_scores = detect_intent(model, q_emb)
    print(f"\nВопрос: {query}")
    print(f"Домен-похожесть: {s_domain:.3f}")
    if intent is None:
        print("Похоже, вопрос не про MDLP/аналитику — ничего рекомендовать.")
        return
    print(f"Интент: {intent}  (sales={intent_scores['sales']:.3f}, stocks={intent_scores['stocks']:.3f}, movement={intent_scores['movement']:.3f})\n")

    # 2) Ранжирование
    candidates: List[Tuple[float, float, str, str, str, str]] = []
    many = 0
    for (id_, title, url, desc, bq, metrics, dims, tags, srcs, info_type, emb) in rows:
        if not emb: 
            continue

        # жёстко/мягко фильтруем по типу: если в таблице есть info_type — используем его
        if info_type and info_type != intent:
            continue
        # если info_type пуст — позволим пройти (будем судить только по семантике)

        # семантическая близость
        s = cosine(q_emb, emb)
        if s >= THRESH_SEM_MAIN:
            many += 1
        candidates.append((s, 0.0, title, url, desc, build_passport(title, desc, bq, metrics, dims, tags, srcs)))

    if not candidates:
        print("Подходящих карточек нет (проверь info_type и наличие embedding).")
        return

    # если кандидатов много, применим более строгий порог
    cutoff = THRESH_SEM_STRICT if many >= 5 else THRESH_SEM_MAIN
    candidates = [c for c in candidates if c[0] >= cutoff]
    if not candidates:
        print("Ничего не набрало семантический порог. Попробуй переформулировать запрос.")
        return

    candidates.sort(key=lambda x: x[0], reverse=True)
    top = candidates[:TOP_K]

    # 3) Объяснения: покажем пересечение токенов запроса с «паспортом» дашборда
    for i, (s, _, title, url, desc, passport) in enumerate(top, 1):
        hit = short_overlap(query, passport)
        why = f"по семантике (cos={s:.3f})"
        if hit:
            why += f"; пересеклись термины: {hit}"
        print(f"{i}. {title}\n   {url}\n   {why}\n")

if __name__ == "__main__":
    main()