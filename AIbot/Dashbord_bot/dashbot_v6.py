# # tag_search.py
# # Простой поиск дашбордов по тегам + русские триггеры из запроса.
# # Запуск:  python tag_search.py "где посмотреть отгрузки дистрибьюторов"

# from dataclasses import dataclass
# from typing import List, Dict, Set, Tuple
# import sys
# import re

# # ──────────────────────────────────────────────────────────────────────────────
# # 1) Каталог дашбордов (упростите/расширяйте по месту)
# #    tags — КЛЮЧЕВОЕ: ставьте смысловые ярлыки, по которым хотите выдачу
# # ──────────────────────────────────────────────────────────────────────────────
# @dataclass
# class Dashboard:
#     id: str
#     title: str
#     url: str
#     tags: List[str]  # напр. ["sales","mdlp"] или ["shipments","distributor"]

# CATALOG: List[Dashboard] = [
#     Dashboard(
#         id="176",
#         title="МДЛП Выбытие (продажи)",
#         url="https://superset.example.org/superset/dashboard/176/",
#         tags=["sales", "mdlp"],
#     ),
#     Dashboard(
#         id="177",
#         title="МДЛП Остатки",
#         url="https://superset.example.org/superset/dashboard/177/",
#         tags=["stocks", "mdlp"],
#     ),
#     Dashboard(
#         id="240",
#         title="Отгрузки дистрибьюторов",
#         url="https://superset.example.org/superset/dashboard/240/",
#         tags=["shipments", "distributor"],
#     ),
#     Dashboard(
#         id="310",
#         title="Закупки аптечных сетей",
#         url="https://superset.example.org/superset/dashboard/310/",
#         tags=["purchases", "pharmacy_chains"],
#     ),
#     Dashboard(
#         id="410",
#         title="Движение",
#         url="https://superset.example.org/superset/dashboard/310/",
#         tags=["movement", "mdlp"],
#     ),
# ]

# # ──────────────────────────────────────────────────────────────────────────────
# # 2) Словарь русских триггеров → канонические теги
# #    Добавляйте свои доменные слова/синонимы
# # ──────────────────────────────────────────────────────────────────────────────
# TRIGGERS: Dict[str, List[str]] = {
#     # продажи
#     "sales": ["продаж", "выбыт", "реализац", "sell", "sales"],
#     # остатки
#     "stocks": ["остатк", "налич", "stock"],
#     # отгрузки (shipment)
#     "shipments": ["отгруз", "shipment", "ship"],
#     # дистрибьюторы
#     "distributor": ["дистриб", "дистр", "опт"],
#     # закупки аптечных сетей
#     "purchases": ["закуп", "покуп", "тендер"],
#     "pharmacy_chains": ["аптечн", "сеть", "сетей", "ритейл"],
#     # mdlp (если хотите привязать к источнику)
#     "mdlp": ["мдлп", "честнзнак", "честный знак"],
#     "movement": ["перемещение", "закуп", "движение", "поступление", "приход", "расход", "трансфер"],
# }

# # ──────────────────────────────────────────────────────────────────────────────
# # 3) Нормализация и извлечение тегов из запроса
# # ──────────────────────────────────────────────────────────────────────────────
# def normalize(text: str) -> str:
#     text = text.lower().replace("ё", "е")
#     # убираем всё, кроме букв/цифр/пробелов
#     text = re.sub(r"[^a-z0-9а-я\s]", " ", text)
#     # схлопываем пробелы
#     text = re.sub(r"\s+", " ", text).strip()
#     return text

# def infer_query_tags(query: str) -> Tuple[Set[str], Dict[str, List[str]]]:
#     """
#     Возвращает:
#       matched_tags: множество канонических тегов, найденных в запросе
#       fired: {tag: [какие триггеры сработали]}
#     """
#     nq = normalize(query)
#     matched_tags: Set[str] = set()
#     fired: Dict[str, List[str]] = {}

#     for tag, keywords in TRIGGERS.items():
#         hits = [kw for kw in keywords if kw in nq]
#         if hits:
#             matched_tags.add(tag)
#             fired[tag] = hits
#     return matched_tags, fired

# # ──────────────────────────────────────────────────────────────────────────────
# # 4) Ранжирование дашбордов по совпадениям тегов
# # ──────────────────────────────────────────────────────────────────────────────
# def score_dashboard(d: Dashboard, qtags: Set[str]) -> int:
#     # простой скор: сколько тегов дашборда попало в теги запроса
#     return len(set(d.tags) & qtags)

# def search_dashboards(query: str, top_k: int = 10):
#     qtags, fired = infer_query_tags(query)

#     # если явных тегов не нашли — попробуем очень простой эвристикой:
#     # ищем по подстроке названий тегов в самом запросе
#     if not qtags:
#         nq = normalize(query)
#         # если в запросе есть "дистриб" и "отгруз" — логически это shipments+distributor
#         if "дистриб" in nq and "отгруз" in nq:
#             qtags.update(["shipments", "distributor"])
#         # базовые эвристики
#         for tag in ["sales", "stocks", "shipments", "purchases"]:
#             if tag in TRIGGERS and any(kw in nq for kw in TRIGGERS[tag]):
#                 qtags.add(tag)

#     # считаем очки
#     scored = [(score_dashboard(d, qtags), d) for d in CATALOG]
#     scored.sort(key=lambda x: x[0], reverse=True)

#     # фильтруем нулевые, но если вообще ничего — покажем всё с нулями (для дебага)
#     nonzero = [item for item in scored if item[0] > 0]
#     results = nonzero if nonzero else scored
#     return qtags, fired, results[:top_k]

# # ──────────────────────────────────────────────────────────────────────────────
# # 5) CLI
# # ──────────────────────────────────────────────────────────────────────────────
# def main():
#     print("🔎 Поиск дашбордов по тегам. Введите запрос (пустая строка или 'exit' — выход).")
#     try:
#         while True:
#             query = input("\nВведите ваш запрос: ").strip()
#             if query == "" or query.lower() in {"exit", "quit"}:
#                 print("👋 Выход.")
#                 break

#             qtags, fired, results = search_dashboards(query, top_k=10)

#             print(f"\nВопрос: {query}")
#             print(f"Распознанные теги запроса: {sorted(qtags) if qtags else '—'}")
#             if fired:
#                 print("Сработавшие триггеры:")
#                 for tag, hits in fired.items():
#                     print(f"  {tag}: {', '.join(hits)}")

#             print("\nРезультаты:")
#             for score, d in results:
#                 matched = sorted(set(d.tags) & qtags)
#                 why = f"совпавшие теги: {', '.join(matched)}" if matched else "совпадений не найдено"
#                 print(f"- [{d.title}]({d.url}) — score={score} — {why}")
#     except KeyboardInterrupt:
#         print("\n👋 Выход.")

# if __name__ == "__main__":
#     main()


# tag_search_simple.py
# Поиск дашбордов по тегам с поддержкой корней и "минус-выражений" на уровне карточек.
# Запуск:  python tag_search_simple.py  → затем вводите запросы построчно.

# tag_search_simple.py
# Поиск дашбордов по тегам (корни/фразы) + исключения на уровне карточек:
#  - exclude_phrases: список минус-фраз (по корням/фразам)
#  - exclude_if_query_tags_all_of: список наборов тегов; если все есть в запросе — карточку скрываем
#
# Запуск:  python tag_search_simple.py  → вводите запросы строками.

from dataclasses import dataclass, field
from typing import List, Dict, Set, Tuple, Any
import os
import re
import json

import clickhouse_connect

# ──────────────────────────────────────────────────────────────────────────────
# Каталог дашбордов (пример: замените на свои)
# ──────────────────────────────────────────────────────────────────────────────
# @dataclass
# class Dashboard:
#     id: str
#     title: str
#     url: str
#     tags: List[str]
#     exclude_phrases: List[str] = field(default_factory=list)
#     exclude_if_query_tags_all_of: List[List[str]] = field(default_factory=list)


# CH_CFG = dict(
#     host='clickhouse.moscow',
#     port=8123,                   # 8123 HTTP, 9000 TCP
#     username='GrushkoIV',
#     password='jNbrvzd1IcF0Yx5I',
#     secure=False,                # True если HTTPS
#     connect_timeout=10
# )
# CATALOG = 'grushko_iv.dashboard_catalog'
# CATALOG: List[Dashboard] = [
#     Dashboard(
#         id="176",
#         title="МДЛП Выбытие (продажи)",
#         url="https://superset.example.org/superset/dashboard/176/",
#         tags=["sales", "mdlp"],
#         # 1) минус-фраза по корням
#         exclude_phrases=["продаж дистр"],
#         # 2) общее правило в виде данных: если в запросе одновременно есть эти теги — скрыть
#         exclude_if_query_tags_all_of=[["sales", "distributor"]],
#     ),
#     Dashboard(
#         id="177",
#         title="МДЛП Остатки",
#         url="https://superset.example.org/superset/dashboard/177/",
#         tags=["stocks", "mdlp"],
#     ),
#     Dashboard(
#         id="240",
#         title="Отгрузки дистрибьюторов",
#         url="https://superset.example.org/superset/dashboard/240/",
#         tags=["shipments", "distributor"],
#     ),
#     Dashboard(
#         id="310",
#         title="Закупки аптечных сетей",
#         url="https://superset.example.org/superset/dashboard/310/",
#         tags=["purchases", "pharmacy_chains"],
#     ),
#        Dashboard(
#         id="440",
#         title="МДЛП Движение",
#         url="https://superset.example.org/superset/dashboard/440/",
#         tags=["movement", "mdlp"],
#     ),
# ]



# ──────────────────────────────────────────────────────────────────────────────
# Триггеры → канонические теги (корни/фразы, матч по префиксам слов)
# ──────────────────────────────────────────────────────────────────────────────
TRIGGERS: Dict[str, List[str]] = {
    "sales": ["продаж", "выбыт", "реализац", "экспорт", "sell", "sales"],
    "stocks": ["остатк", "налич", "stock", 'сток'],
    "shipments": ["отгруз", "shipment", "ship"],
    "distributor": ["дистриб", "дистр", "опт"],
    "purchases": ["закуп", "покуп", "тендер"],
    "pharmacy_chains": ["аптечн", "сеть", "сетей", "ритейл"],
    "mdlp": ["мдлп", "честнзнак", "честный знак"],
    "movement": ["движение","перемещение","поступление", "закуп","приход","расход","трансфер","movement","inflow","transfer"],
    "doctor": ["врач", "доктор", "ЛПУ", "поликлинника" ],
    "drugstore": ["аптека", "АС"],
    "CRM": ["CRM"],
    "visits": ["визит", "визитная активность"]
}

# ──────────────────────────────────────────────────────────────────────────────
# 2) Модель данных карточки
# ──────────────────────────────────────────────────────────────────────────────
@dataclass
class Dashboard:
    id: str
    title: str
    url: str
    tags: List[str]
    description: str = ""
    exclude_phrases: List[str] = field(default_factory=list)
    exclude_if_query_tags_all_of: List[List[str]] = field(default_factory=list)


# ──────────────────────────────────────────────────────────────────────────────
# Нормализация и матчинг по корням/фразам
# ──────────────────────────────────────────────────────────────────────────────
def normalize(text: str) -> str:
    text = text.lower().replace("ё", "е")
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
# Извлечение тегов из запроса
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

# ──────────────────────────────────────────────────────────────────────────────
# 5) Подключение к ClickHouse и загрузка каталога
# ──────────────────────────────────────────────────────────────────────────────
def get_ch_client():
    host = os.getenv("CLICKHOUSE_HOST", "clickhouse.moscow")
    port = int(os.getenv("CLICKHOUSE_PORT", "8123"))  # 8123 HTTP, 9000 TCP
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
    """Преобразует значение в List[List[str]] (для exclude_if_query_tags_all_of)."""
    if obj is None:
        return []
    if isinstance(obj, list):
        # уже список; элементы могут быть строками или списками строк
        out: List[List[str]] = []
        for el in obj:
            if el is None:
                continue
            if isinstance(el, list):
                out.append([str(s) for s in el])
            else:
                out.append([str(el)])
        return out
    # если пришла строка формата JSON
    js = safe_json_load(obj)
    if js is None:
        return []
    return to_list_list(js)

def to_list(obj: Any) -> List[str]:
    """Преобразует значение в List[str] (для tags/exclude_phrases)."""
    if obj is None:
        return []
    if isinstance(obj, list):
        return [str(x) for x in obj]
    if isinstance(obj, str):
        js = safe_json_load(obj)
        if isinstance(js, list):
            return [str(x) for x in js]
        # строка не JSON-массив — попробуем разбить по запятой
        if "," in obj:
            return [s.strip() for s in obj.split(",") if s.strip()]
        return [obj.strip()] if obj.strip() else []
    return [str(obj)]

def load_catalog_from_clickhouse() -> List[Dashboard]:
    client = get_ch_client()

    # Пытаемся вытащить расширенный набор полей; если каких-то нет — падаем в упрощённый SELECT
    query_variants = [
        """
        SELECT
          toString(id) AS id,
          title,
          url,
          tags,
          ifNull(description, '') AS description,
          -- поля могут быть Array(String) или JSON/Nullable(String):
          ifNull(exclude_phrases, CAST([], 'Array(String)')) AS exclude_phrases,
          ifNull(exclude_if_query_tags_all_of, NULL) AS exclude_if_query_tags_all_of
        FROM grushko_iv.dashboard_catalog
        """,
        """
        SELECT
          toString(id) AS id,
          title,
          url,
          tags,
          ifNull(description, '') AS description
        FROM grushko_iv.dashboard_catalog
        """
    ]

    rows = None
    columns = None
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

    # Индексы колонок
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
            exclude_phrases = to_list(r[idx["exclude_phrases"]])

        if "exclude_if_query_tags_all_of" in idx:
            raw = r[idx["exclude_if_query_tags_all_of"]]
            parsed = safe_json_load(raw)
            exclude_if_query_tags_all_of = to_list_list(parsed if parsed is not None else raw)

        catalog.append(
            Dashboard(
                id=id_,
                title=title or "",
                url=url or "",
                tags=[t.strip() for t in tags if t and str(t).strip()],
                description=description or "",
                exclude_phrases=[normalize(p) for p in exclude_phrases if p],
                exclude_if_query_tags_all_of=exclude_if_query_tags_all_of or [],
            )
        )

    return catalog

# ──────────────────────────────────────────────────────────────────────────────
# 6) Поиск/ранжирование
# ──────────────────────────────────────────────────────────────────────────────
def score_dashboard(d: Dashboard, qtags: Set[str]) -> int:
    # Простой скор: число совпавших канонических тегов
    return len(set(d.tags) & qtags)

def should_exclude_by_tag_combos(d: Dashboard, qtags: Set[str]) -> Tuple[bool, List[str]]:
    """True, [комбо], если хотя бы один набор из exclude_if_query_tags_all_of ⊆ qtags."""
    for combo in d.exclude_if_query_tags_all_of:
        if set(combo).issubset(qtags):
            return True, combo
    return False, []

def search_dashboards(query: str, catalog: List[Dashboard], top_k: int = 10):
    nq = normalize(query)
    qtags, fired = infer_query_tags(query)

    scored = []
    excluded_notes = []

    for d in catalog:
        # 1) исключения по минус-фразам карточки
        if any(phrase_hit(nq, ph) for ph in d.exclude_phrases):
            excluded_notes.append(f"− {d.title}: исключено по фразе из карточки")
            continue
        # 2) исключения по наборам тегов (если поле присутствует)
        excluded, combo = should_exclude_by_tag_combos(d, qtags)
        if excluded:
            excluded_notes.append(f"− {d.title}: исключено по набору тегов {combo}")
            continue

        scored.append((score_dashboard(d, qtags), d))

    scored.sort(key=lambda x: x[0], reverse=True)
    nonzero = [item for item in scored if item[0] > 0]
    results = nonzero if nonzero else scored
    return qtags, fired, results[:top_k], excluded_notes


# ──────────────────────────────────────────────────────────────────────────────
# 7) REPL
# ──────────────────────────────────────────────────────────────────────────────
def main():
    print("🔗 Подключение к ClickHouse и загрузка каталога…")
    catalog = load_catalog_from_clickhouse()
    print(f"📚 Загружено карточек: {len(catalog)}")

    print("\n🔎 Поиск по тегам (корни/фразы).")
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
                for score, d in results:
                    matched = sorted(set(d.tags) & qtags)
                    why = f"совпавшие теги: {', '.join(matched)}" if matched else "совпадений по тегам нет"
                    print(f"- [{d.title}]({d.url}) — score={score} — {why}")
    except KeyboardInterrupt:
        print("\n👋 Выход.")

if __name__ == "__main__":
    main()