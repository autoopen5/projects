import os, numpy as np
import clickhouse_connect
from fastapi import FastAPI, Query
from pydantic import BaseModel
from sentence_transformers import SentenceTransformer
from dotenv import load_dotenv

load_dotenv()
client = clickhouse_connect.get_client(
    host='clickhouse.moscow',
    port=8123,  # 8123 для HTTP, 9000 для Native (TCP)
    username='GrushkoIV',
    password='jNbrvzd1IcF0Yx5I',
)

model = SentenceTransformer(os.getenv('EMBED_MODEL','sentence-transformers/all-MiniLM-L6-v2'))
DEFAULT_ROLES = set((os.getenv('DEFAULT_ROLES','public') or '').split(','))

app = FastAPI(title='MDLP Router Assistant (sales only)')

class Card(BaseModel):
    title: str
    url: str
    why: str
    suggested_filters: dict

class Response(BaseModel):
    query: str
    intent: dict
    cards: list[Card]

METRICS = {'sales_units','sales_value','avg_price'}
DIMENSIONS = {'date','date_month','region_mmh','sku','client'}
TAGS = {'Продажи','MDLP','Аптеки'}

def extract_intent(q: str):
    ql = q.lower()
    metrics = [m for m in METRICS if m in ql]
    dims = [d for d in DIMENSIONS if d in ql]
    period = 'last_12_months'
    if 'квартал' in ql: period = 'last_4_quarters'
    if 'недел' in ql: period = 'last_12_weeks'
    return {'metrics': metrics, 'dimensions': dims, 'period': period, 'filters': {}}

def cosine(a, b):
    a = np.array(a, dtype='float32'); b = np.array(b, dtype='float32')
    a = a / (np.linalg.norm(a) + 1e-9); b = b / (np.linalg.norm(b) + 1e-9)
    return float((a @ b))

@app.get('/ask', response_model=Response)
def ask(q: str = Query(...), roles: str = Query('')):
    intent = extract_intent(q)
    q_emb = model.encode(q).astype('float32').tolist()
    user_roles = set([r.strip() for r in roles.split(',') if r.strip()]) or DEFAULT_ROLES

    rs = client.query('''
        SELECT title, url, description, business_questions, metrics, dimensions, tags,
               access_policy, last_refreshed_at, freshness_hours, monthly_opens, embedding
        FROM grushko_iv.dashboard_catalog
        WHERE is_active = 1
    ''').result_rows

    cards = []
    for title, url, desc, qs, mets, dims, tags, access, last_ref, fresh_h, opens, emb in rs:
        if access != 'public' and access not in user_roles:  # простые доступы
            continue
        if not emb:  # без эмбеддинга пропускаем
            continue
        sem = cosine(q_emb, emb)
        toks = set(q.lower().replace(',', ' ').split())
        kw = 0.0
        if any((mets or []) and (m.lower() in toks) for m in (mets or [])): kw += 0.5
        if any((dims or []) and (d.lower() in toks) for d in (dims or [])): kw += 0.3
        if any((tags or []) and (t.lower() in toks) for t in (tags or [])): kw += 0.2
        kw = min(1.0, kw)
        fresh = 1.0
        pop = min(1.0, (opens or 0)/500.0)
        final = 0.55*sem + 0.15*kw + 0.10*fresh + 0.10*pop + 0.10*1.0
        why = []
        if kw>0: why.append('совпадение по метаданным')
        if sem>0.5: why.append('высокая семантическая близость')
        if pop>0.4: why.append('популярный дашборд')
        if fresh>0.9: why.append('свежие данные')
        cards.append((final, Card(
            title=title, url=url, why='; '.join(why) or 'наилучшее совпадение',
            suggested_filters={'period': intent['period']}
        )))
    cards = [c for _, c in sorted(cards, key=lambda x: x[0], reverse=True)[:5]]
    return Response(query=q, intent=intent, cards=cards)