"""
MVP навигатор по дашбордам для ClickHouse.
Запуск:
  python dashbot.py "где посмотреть остатки по Эргоферону в СЗФО?"

Настройте подключение к CH в секции CONFIG.
"""

import sys, re, math
from typing import List, Dict, Any
import clickhouse_connect
# from rapidfuzz import fuzz, process
import difflib

# ==== CONFIG: укажите доступ к ClickHouse ====
CH_CFG = dict(
    host='clickhouse.moscow',
    port=8123,                   # 8123 HTTP, 9000 TCP
    username='GrushkoIV',
    password='jNbrvzd1IcF0Yx5I',
    secure=False,                # True если HTTPS
    connect_timeout=10
)
CATALOG_TABLE = 'grushko_iv.dashboard_catalog'

# Какие колонки попытаемся прочитать (что найдется — то используем)
CANDIDATE_COLS = [
    ('id',         ['dashboard_id','id','uuid']),
    ('title',      ['title','name']),
    ('subtitle',   ['subtitle']),
    ('description',['description','desc']),
    ('url',        ['url','link','href']),
    ('tags',       ['tags']),
    ('systems',    ['systems','system']),
    ('last_updated',['last_updated','updated_at','update_time']),
    ('popularity', ['popularity_score','views','hits'])
]

# Синонимы/онто-словарь для «интентов»
INTENT_SYNONYMS = {
    'sales':    ['продажа','продажи','реализация','выбытие','sales'],
    'stocks':   ['остаток','остатки','stock','stocks', 'запапс'],
    'movement': ['движение','перемещение','перемещения','movement','move']
}

def detect_intent(tokens: List[str]) -> str:
    for intent, syns in INTENT_SYNONYMS.items():
        if any(s in tokens for s in syns):
            return intent
    return 'generic'

def tokenize(text: str) -> List[str]:
    return [t for t in re.split(r'[^a-zA-Zа-яА-Я0-9%]+', (text or '').lower()) if t]

def build_text_blob(row: Dict[str, Any]) -> str:
    parts = []
    for k in ['title','subtitle','description']:
        if row.get(k):
            parts.append(str(row[k]))
    for k in ['tags','systems']:
        if isinstance(row.get(k), list):
            parts.append(' '.join([str(x) for x in row[k]]))
    return ' '.join(parts)

def score_row(query: str, q_tokens: List[str], intent: str, row: Dict[str, Any]) -> Dict[str, Any]:
    title = row.get('title','') or ''
    desc  = row.get('description','') or ''
    tags  = row.get('tags',[]) if isinstance(row.get('tags'), list) else []
    text  = build_text_blob(row)

    # 1) keyword hits
    def hits_in(s: str) -> int:
        s_low = s.lower()
        return sum(1 for t in q_tokens if t and t in s_low)

    title_hits = hits_in(title)
    desc_hits  = hits_in(desc)
    tags_hits  = sum(1 for t in q_tokens for tag in tags if t and isinstance(tag,str) and t in tag.lower())

    # 2) fuzzy similarity (для коротких запросов)
    fuzzy = difflib.SequenceMatcher(None, ' '.join(q_tokens), text.lower()).ratio()

    # 3) intent boost — если теги/системы содержат ключевые слова интента
    def has_intent(words):
        words_low = [w.lower() for w in words]
        return any(any(k in w for w in words_low) for k in INTENT_SYNONYMS.get(intent, []))

    intent_boost = 0
    if intent != 'generic':
        tag_ok = has_intent(tags)
        sys_ok = has_intent(row.get('systems',[]) if isinstance(row.get('systems'), list) else [])
        intent_boost = 2 if (tag_ok or sys_ok) else 0

    # 4) recency/popularity (если есть)
    popularity = float(row.get('popularity') or 0.0)
    recency_boost = 0.0  # можно доработать вычислением возраста

    score = title_hits*3 + tags_hits*2 + desc_hits*1 + fuzzy*2 + intent_boost + min(popularity, 3) + recency_boost

    # explain
    why = []
    if title_hits: why.append(f"title={title_hits}")
    if tags_hits:  why.append(f"tags={tags_hits}")
    if desc_hits:  why.append(f"desc={desc_hits}")
    if intent_boost: why.append(f"intent={intent}")
    if fuzzy > 0.2: why.append(f"fuzzy≈{round(fuzzy,2)}")

    row['_score'] = round(float(score), 3)
    row['_why'] = " | ".join(why) if why else "символьная близость"
    return row

def main():
    if len(sys.argv) < 2:
        print("Usage: python dashbot.py \"ваш вопрос\"")
        sys.exit(1)
    query = sys.argv[1]
    q_tokens = tokenize(query)
    intent = detect_intent(q_tokens)

    client = clickhouse_connect.get_client(**CH_CFG)

    # Определим доступные колонки таблицы
    desc = client.query(f"DESCRIBE TABLE {CATALOG_TABLE}").result_rows
    have_cols = {row[0] for row in desc}

    # Построим SELECT только с существующими колонками
    select_cols = []
    col_map = {}
    for logical, candidates in CANDIDATE_COLS:
        found = None
        for c in candidates:
            if c in have_cols:
                found = c
                break
        if found:
            select_cols.append(found)
            col_map[logical] = found

    if not select_cols:
        print("Не удалось определить колонки каталога. Проверьте таблицу.")
        sys.exit(2)

    sql = f"SELECT {', '.join(select_cols)} FROM {CATALOG_TABLE} WHERE 1"
    # можно отфильтровать неактивные: AND ifNull(is_active,1)=1

    rows = [dict(zip(select_cols, r)) for r in client.query(sql).result_rows]

    # Приведём к логическим именам
    norm_rows = []
    for r in rows:
        nr = {}
        for logical, phys in col_map.items():
            nr[logical] = r.get(phys)
        # нормализуем tags/systems в list[str]
        for k in ['tags','systems']:
            v = nr.get(k)
            if v is None:
                nr[k] = []
            elif isinstance(v, str):
                # если хранили как строку с запятыми
                nr[k] = [x.strip() for x in v.split(',') if x.strip()]
        norm_rows.append(nr)

    # Скоим
    scored = [score_row(query, q_tokens, intent, r) for r in norm_rows]
    scored.sort(key=lambda x: x['_score'], reverse=True)
    top = scored[:7]

    # Порог для "пустых" результатов
    threshold = 1.0
    top = [r for r in top if r['_score'] >= threshold]

    if not top:
        print("Ничего релевантного не нашёл. Попробуйте уточнить запрос (бренд, период, регион).")
        sys.exit(0)

    # Печать ответов
    print(f"\nЗапрос: {query}")
    print(f"Интент: {intent}\n")
    for i, r in enumerate(top, 1):
        title = r.get('title') or '(без названия)'
        url   = r.get('url') or ''
        why   = r.get('_why')
        score = r.get('_score')
        print(f"{i}. {title}\n   {url}\n   score={score} | {why}\n")

if __name__ == "__main__":
    main()