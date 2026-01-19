import os
import clickhouse_connect
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

rows = client.query('''
SELECT id, title, description, business_questions, metrics, dimensions, tags, data_sources
FROM grushko_iv.dashboard_catalog
WHERE is_active = 1
''').result_rows

updates = []
for id_, title, desc, q, m, d, t, src in rows:
    text = "\n".join([
        title or '', desc or '',
        'Вопросы: ' + '; '.join(q or []),
        'Метрики: ' + ', '.join(m or []),
        'Измерения: ' + ', '.join(d or []),
        'Теги: ' + ', '.join(t or []),
        'Источники: ' + ', '.join(src or []),
    ])
    emb = model.encode(text).astype('float32').tolist()
    updates.append((emb, id_))

# пакетный апдейт
for emb, id_ in updates:
    client.command(
        "ALTER TABLE grushko_iv.dashboard_catalog "
        "UPDATE embedding = %(emb)s "
        "WHERE id = %(id)s",
        {"emb": emb, "id": id_}
    )
print(f'Updated embeddings for {len(updates)} dashboards')