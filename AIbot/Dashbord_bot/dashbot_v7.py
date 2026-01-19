# -*- coding: utf-8 -*-
import os, re, json
from dataclasses import dataclass, field
from typing import Any, Dict, List, Set, Tuple

import clickhouse_connect

# ──────────────────────────────────────────────────────────────────────────────
# 0) Триггеры → канонические теги (корни/фразы, матч по префиксам слов)
#    !!! ВАЖНО: используем КАНОНИЧЕСКИЕ теги 'sales' | 'stocks' | 'movement'
#    чтобы можно было жёстко маршрутизировать дашборды по типу данных.
# ──────────────────────────────────────────────────────────────────────────────
TRIGGERS: Dict[str, List[str]] = {
    "sales": ["продаж", "выбыт", "реализац", "экспорт", "sell", "sales", "выручк", "avg_price"],
    "stocks": ["остатк", "налич", "stock", "сток", "на складе", "на сегодня", "запас"],
    "movement": ["движен", "перемещ", "поступлен", "приход", "расход", "трансфер", "movement", "inflow", "transfer"],

    # Доп. доменные/ограничивающие
    "shipments": ["отгруз", "shipment", "ship"],
    "distributor": ["дистриб", "дистр", "опт"],
    "purchases": ["закуп", "покуп", "тендер"],
    "pharmacy_chains": ["аптечн", "сеть", "сетей", "ритейл"],
    "mdlp": ["мдлп", "честнзнак", "честный знак"],
    "doctor": ["врач", "доктор", "лпу", "поликлинник"],
    "drugstore": ["аптека", "ас"],
    "CRM": ["crm"],
    "visits": ["визит", "визитная активность"],
}

# Набор "доменных" тегов, по которым делаем ЖЁСТКУЮ маршрутизацию
PRIMARY_INTENTS = ("stocks", "movement", "sales")

# Приоритет интентов, если в вопросе встретилось несколько
INTENT_PRIORITY = ("stocks", "movement", "sales")

# ──────────────────────────────────────────────────────────────────────────────
# 1) Модель карточки
# ──────────────────────────────────────────────────────────────────────────────
@dataclass
class Dashboard:
    id: str
    title: str
    url: str
    tags: List[str]                          # ожидаем наличие 'sales'|'stocks'|'movement'
    description: str = ""
    exclude_phrases: List[str] = field(default_factory=list)
    exclude_if_query_tags_all_of: List[List[str]] = field(default_factory=list)

# ──────────────────────────────────────────────────────────────────────────────
# 2) Нормализация и матчинг по корням/фразам
# ──────────────────────────────────────────────────────────────────────────────
def normalize(text: str) -> str:
    text = (text or "").lower().replace("ё", "е")
    text = re.sub(r"[^a-z0-9а-я\s\-]", " ", text)
    return re.sub(r"\s+", " ", text).strip()

def keyword_hit(nq: str, keyword: str) -> bool:
    """Совпадение по корням/фразам: каждый токен встречается как префикс слова."""
    tokens = normalize(keyword).split()
    if not tokens:
        return False
    for t in tokens:
        if not re.search(rf"\b{re.escape(t)}\w*\b", nq):
            return False
    return True

phrase_hit = keyword_hit  # минус-фразы матчатся тем же способом

# ──────────────────────────────────────────────────────────────────────────────
# 3) Извлечение тегов и объяснений (кто именно сработал)
# ──────────────────────────────────────────────────────────────────────────────
def infer_query_tags(query: str) -> Tuple[Set[str], Dict[str, List[str]]]:
    nq = normalize(query)
    matched_tags: Set[str] = set()
    fired: Dict[str, List[str]] = {}
    for tag, keywords in TRIGGERS.items():
        hits = [kw for kw in keywords if keyword_hit(nq, kw)]
        if hits:
            matched_tags.add(tag)
            fired[tag] = hits
    return matched_tags, fired

def detect_primary_intent(qtags: Set[str]) -> str | None:
    # Берём первый по приоритету интент, который есть в qtags
    for intent in INTENT_PRIORITY:
        if intent in qtags:
            return intent
    return None

# ──────────────────────────────────────────────────────────────────────────────
# 4) ClickHouse загрузка каталога
# ──────────────────────────────────────────────────────────────────────────────
def get_ch_client():
    host = os.getenv("CLICKHOUSE_HOST", "clickhouse.moscow")
    port = int(os.getenv("CLICKHOUSE_PORT", "8123"))
    user = os.getenv("CLICKHOUSE_USER", "GrushkoIV")
    password = os.getenv("CLICKHOUSE_PASSWORD", "jNbrvzd1IcF0Yx5I")
    return clickhouse_connect.get_client(host=host, port=port, username=user, password=password)

def safe_json_load(x: Any):
    if x is None:
        return None
    if isinstance(x, (list, dict)):
        return x
    if isinstance(x, str):
        x = x.strip()
        if not x:
            return None
        try:
            return json.loads(x)
        except Exception:
            return None
    return None

def to_list_list(obj: Any) -> List[List[str]]:
    if obj is None:
        return []
    if isinstance(obj, list):
        out: List[List[str]] = []
        for el in obj:
            if el is None:
                continue
            if isinstance(el, list):
                out.append([str(s) for s in el])
            else:
                out.append([str(el)])
        return out
    js = safe_json_load(obj)
    if js is None:
        return []
    return to_list_list(js)

def to_list(obj: Any) -> List[str]:
    if obj is None:
        return []
    if isinstance(obj, list):
        return [str(x) for x in obj]
    if isinstance(obj, str):
        js = safe_json_load(obj)
        if isinstance(js, list):
            return [str(x) for x in js]
        if "," in obj:
            return [s.strip() for s in obj.split(",") if s.strip()]
        return [obj.strip()] if obj.strip() else []
    return [str(obj)]

def load_catalog_from_clickhouse() -> List[Dashboard]:
    client = get_ch_client()
    query_variants = [
        """
        SELECT
          toString(id) AS id,
          title,
          url,
          tags,
          ifNull(description, '') AS description,
          ifNull(exclude_phrases, CAST([], 'Array(String)')) AS exclude_phrases,
          ifNull(exclude_if_query_tags_all_of, NULL) AS exclude_if_query_tags_all_of
        FROM grushko_iv.dashboard_catalog
        WHERE is_active = 1
        """,
        """
        SELECT
          toString(id) AS id,
          title,
          url,
          tags,
          ifNull(description, '') AS description
        FROM grushko_iv.dashboard_catalog
        WHERE is_active = 1
        """
    ]
    rows, columns, last_err = None, None, None
    for q in query_variants:
        try:
            res = client.query(q)
            rows = res.result_rows
            columns = res.column_names
            break
        except Exception as e:
            last_err = e
            continue
    if rows is None:
        raise RuntimeError(f"Не удалось загрузить каталог: {last_err}")

    idx = {name: i for i, name in enumerate(columns)}
    catalog: List[Dashboard] = []
    for r in rows:
        id_ = str(r[idx["id"]]) if "id" in idx else ""
        title = r[idx["title"]] if "title" in idx else ""
        url = r[idx["url"]] if "url" in idx else ""
        tags = to_list(r[idx["tags"]]) if "tags" in idx else []
        description = r[idx["description"]] if "description" in idx else ""

        exclude_phrases = []
        exclude_if_query_tags_all_of = []
        if "exclude_phrases" in idx:
            exclude_phrases = [normalize(p) for p in to_list(r[idx["exclude_phrases"]]) if p]
        if "exclude_if_query_tags_all_of" in idx:
            raw = r[idx["exclude_if_query_tags_all_of"]]
            parsed = safe_json_load(raw)
            exclude_if_query_tags_all_of = to_list_list(parsed if parsed is not None else raw)

        catalog.append(
            Dashboard(
                id=id_ or "",
                title=title or "",
                url=url or "",
                tags=[t.strip() for t in tags if t and str(t).strip()],
                description=description or "",
                exclude_phrases=exclude_phrases or [],
                exclude_if_query_tags_all_of=exclude_if_query_tags_all_of or [],
            )
        )
    return catalog

# ──────────────────────────────────────────────────────────────────────────────
# 5) Матчинг, исключения и объяснения
# ──────────────────────────────────────────────────────────────────────────────
def score_dashboard(d: Dashboard, qtags: Set[str]) -> int:
    # простой скор: сколько канонических тегов совпало
    return len(set(d.tags) & qtags)

def should_exclude_by_tag_combos(d: Dashboard, qtags: Set[str]) -> Tuple[bool, List[str]]:
    for combo in d.exclude_if_query_tags_all_of:
        if set(combo).issubset(qtags):
            return True, combo
    return False, []

def explain_hits_for_dashboard(d: Dashboard, fired: Dict[str, List[str]]) -> Dict[str, List[str]]:
    """
    Вернём только те "сработавшие триггеры", у которых канонический тег есть в d.tags.
    Например: {'stocks': ['остатк','stock'], 'mdlp': ['мдлп']}
    """
    explanation: Dict[str, List[str]] = {}
    d_tags = set(t.lower() for t in d.tags)
    for tag, hits in fired.items():
        if tag.lower() in d_tags:
            explanation[tag] = hits
    return explanation

# ──────────────────────────────────────────────────────────────────────────────
# 6) Поиск/ранжирование с ЖЁСТКОЙ маршрутизацией по интенту
# ──────────────────────────────────────────────────────────────────────────────
def search_dashboards(query: str, catalog: List[Dashboard], top_k: int = 10):
    nq = normalize(query)
    qtags, fired = infer_query_tags(query)

    # Если вопрос вообще не доменный (нет ни одного тега) — честный выход
    if not qtags:
        return qtags, fired, [], ["Вопрос не содержит доменных маркеров (ничего рекомендовать)."]

    # Определяем первичный интент (stocks/movement/sales)
    primary_intent = detect_primary_intent(qtags)

    scored = []
    excluded_notes = []

    for d in catalog:
        # 0) Жёсткая маршрутизация: если интент определён, карточка должна иметь этот тег
        if primary_intent and primary_intent not in set(t.lower() for t in d.tags):
            excluded_notes.append(f"− {d.title}: info_type≠{primary_intent}")
            continue

        # 1) Исключения по минус-фразам
        if any(phrase_hit(nq, ph) for ph in d.exclude_phrases):
            excluded_notes.append(f"− {d.title}: исключено по минус-фразе из карточки")
            continue

        # 2) Исключения по наборам тегов (если поле присутствует)
        excluded, combo = should_exclude_by_tag_combos(d, qtags)
        if excluded:
            excluded_notes.append(f"− {d.title}: исключено по набору тегов {combo}")
            continue

        # 3) Считаем скор по совпавшим каноническим тегам
        s = score_dashboard(d, qtags)

        # 4) Собираем объяснение — именно по тем тегам, которые есть у карточки
        expl = explain_hits_for_dashboard(d, fired)

        scored.append((s, d, expl, primary_intent))

    # Сортируем по скору, затем по длине объяснения (больше конкретики — выше)
    scored.sort(key=lambda x: (x[0], sum(len(v) for v in x[2].values())), reverse=True)

    nonzero = [item for item in scored if item[0] > 0]
    results = nonzero if nonzero else scored
    return qtags, fired, results[:top_k], excluded_notes

# ──────────────────────────────────────────────────────────────────────────────
# 7) REPL
# ──────────────────────────────────────────────────────────────────────────────
def main():
    print("🔗 Подключаюсь к ClickHouse и загружаю каталог…")
    catalog = load_catalog_from_clickhouse()
    print(f"📚 Загружено карточек: {len(catalog)}")

    print("\n🔎 Поиск по доменным тегам и строгой маршрутизации (stocks/movement/sales).")
    print("Введите запрос (пустая строка или 'exit' — выход).")
    try:
        while True:
            query = input("\nВведите ваш запрос: ").strip()
            if query == "" or query.lower() in {"exit", "quit"}:
                print("👋 Выход.")
                break

            qtags, fired, results, excluded_notes = search_dashboards(query, catalog, top_k=10)

            print(f"\nВопрос: {query}")
            print(f"Распознанные теги: {sorted(qtags) if qtags else '—'}")
            if fired:
                print("Сработавшие триггеры:")
                for tag, hits in fired.items():
                    print(f"  {tag}: {', '.join(hits)}")

            if excluded_notes:
                print("\nИсключены:")
                for note in excluded_notes:
                    print(" ", note)

            print("\nРезультаты:")
            if not results:
                print("— Ничего не найдено.")
            else:
                for score, d, expl, primary_intent in results:
                    # объяснение: какие теги карточки совпали и какие именно слова их дали
                    if expl:
                        parts = [f"{t}: [{', '.join(words)}]" for t, words in expl.items()]
                        why = f"совпали теги карточки с вопросом → " + "; ".join(parts)
                    else:
                        why = "совпадений по тегам карточки нет; выбран по общим доменным тегам запроса"

                    print(f"- [{d.title}]({d.url}) — score={score} — {why}")
    except KeyboardInterrupt:
        print("\n👋 Выход.")

if __name__ == "__main__":
    main()