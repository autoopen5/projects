import os
import re
import sys
import json
import uuid
from dataclasses import dataclass
from typing import List, Tuple

import numpy as np
from docx import Document
from tqdm import tqdm
from sklearn.metrics.pairwise import cosine_similarity

# ──────────────────────────────────────────────────────────────────────────────
# Конфиг
# ──────────────────────────────────────────────────────────────────────────────
DOC_PATH = os.getenv("DOC_PATH", r"C:/Users/SatyaTR/Desktop/1.docx")
EMBED_MODEL = os.getenv ("EMBED_MODEL", "sentence-transformers/all-MiniLM-L6-v2")
CHUNK_SIZE = int(os.getenv("CHUNK_SIZE", 900))        # символов
CHUNK_OVERLAP = int(os.getenv("CHUNK_OVERLAP", 200))  # символов
TOP_K = int(os.getenv("TOP_K", 4))
USE_OPENAI = bool(os.getenv("OPENAI_API_KEY", ""))

# ──────────────────────────────────────────────────────────────────────────────
# Модель эмбеддингов
# ──────────────────────────────────────────────────────────────────────────────
class Embedder:
    def __init__(self, model_name: str):
        from sentence_transformers import SentenceTransformer
        self.model = SentenceTransformer(model_name)

    def encode(self, texts: List[str]) -> np.ndarray:
        # Возвращаем np.array, shape: (n, dim)
        vecs = self.model.encode(texts, normalize_embeddings=True, show_progress_bar=False)
        return np.array(vecs, dtype=np.float32)

# ──────────────────────────────────────────────────────────────────────────────
# Чтение и препроцесс документа
# ──────────────────────────────────────────────────────────────────────────────
def read_docx(path: str) -> str:
    d = Document(path)
    paras = []
    for p in d.paragraphs:
        t = p.text.strip()
        if t:
            paras.append(t)
    raw = "\n".join(paras)
    return normalize_spaces(raw)

def normalize_spaces(s: str) -> str:
    s = s.replace("\xa0", " ")
    s = re.sub(r"[ \t]+", " ", s)
    s = re.sub(r"\n{3,}", "\n\n", s)
    return s.strip()

# ──────────────────────────────────────────────────────────────────────────────
# Чанкирование
# ──────────────────────────────────────────────────────────────────────────────
@dataclass
class Chunk:
    id: str
    text: str
    start: int
    end: int

def make_chunks(text: str, chunk_size: int = 900, overlap: int = 200) -> List[Chunk]:
    chunks: List[Chunk] = []
    i = 0
    n = len(text)
    while i < n:
        j = min(i + chunk_size, n)
        # стараться резать по границе предложений
        slice_text = text[i:j]
        # если не конец, попытаемся сместить до ближайшей точки
        if j < n:
            m = re.search(r"[.!?]\s+[А-ЯA-Z]", text[i:j][::-1])
            if m and m.start() > 20:
                # m найден в обратной строке
                cut_back = m.start()
                j = i + (j - i - cut_back)
                slice_text = text[i:j]
        chunks.append(Chunk(id=str(uuid.uuid4()), text=slice_text.strip(), start=i, end=j))
        if j == n:
            break
        i = max(j - overlap, 0)
    return chunks

# ──────────────────────────────────────────────────────────────────────────────
# Индекс
# ──────────────────────────────────────────────────────────────────────────────
@dataclass
class VectorIndex:
    texts: List[str]
    vectors: np.ndarray
    meta: List[Chunk]

def build_index(doc_path: str, embedder: Embedder) -> VectorIndex:
    print(f"Загружаю документ: {doc_path}")
    full_text = read_docx(doc_path)
    print(f"Размер текста: {len(full_text):,} симв.")
    chunks = make_chunks(full_text, CHUNK_SIZE, CHUNK_OVERLAP)
    print(f"Чанков: {len(chunks)}")

    texts = [c.text for c in chunks]
    vectors = embedder.encode(texts)
    return VectorIndex(texts=texts, vectors=vectors, meta=chunks)

# ──────────────────────────────────────────────────────────────────────────────
# Поиск
# ──────────────────────────────────────────────────────────────────────────────
def retrieve(query: str, index: VectorIndex, embedder: Embedder, top_k: int = 4) -> List[Tuple[float, Chunk]]:
    q_vec = embedder.encode([query])
    sims = cosine_similarity(q_vec, index.vectors)[0]  # (n,)
    top_idx = np.argsort(-sims)[:top_k]
    results = [(float(sims[i]), index.meta[i]) for i in top_idx]
    return results

# ──────────────────────────────────────────────────────────────────────────────
# Генерация ответа
# ──────────────────────────────────────────────────────────────────────────────
SYS_PROMPT = (
    "Ты помощник по внутреннему документу компании. Отвечай строго на русском, кратко и по делу, "
    "опираясь только на переданный контекст. Если в контексте нет ответа — скажи об этом."
)

def synthesize_with_openai(question: str, contexts: List[str]) -> str:
    """Опционально: если задан OPENAI_API_KEY, используем gpt-4o-mini для краткого ответа."""
    from openai import OpenAI
    client = OpenAI()
    ctx = "\n\n---\n\n".join(contexts)
    messages = [
        {"role": "system", "content": SYS_PROMPT},
        {"role": "user", "content": f"Вопрос: {question}\n\nКонтекст:\n{ctx}\n\nОтвет:"},
    ]
    resp = client.chat.completions.create(model="gpt-4o-mini", messages=messages, temperature=0.0)
    return resp.choices[0].message.content.strip()

def extractive_answer(question: str, hits: List[Tuple[float, Chunk]]) -> str:
    """Простой экстрактивный ответ: возвращаем самый релевантный фрагмент и краткий summary-одной фразой."""
    if not hits:
        return "В документе не найдено информации для ответа."
    best = hits[0][1].text
    # Попробуем выделить 1-3 предложения, где чаще встречаются слова вопроса
    q_words = set(re.findall(r"\w{3,}", question.lower()))
    sents = re.split(r"(?<=[.!?])\s+", best)
    scored = []
    for s in sents:
        sw = set(re.findall(r"\w{3,}", s.lower()))
        scored.append((len(q_words & sw), s))
    scored.sort(key=lambda x: (-x[0], -len(x[1])))
    top = " ".join([s for _, s in scored[:3]]).strip()
    return top if top else best

def answer(question: str, index: VectorIndex, embedder: Embedder, top_k: int = TOP_K) -> dict:
    hits = retrieve(question, index, embedder, top_k=top_k)
    contexts = [c.text for _, c in hits]
    if USE_OPENAI:
        final = synthesize_with_openai(question, contexts)
    else:
        final = extractive_answer(question, hits)

    return {
        "question": question,
        "answer": final,
        "sources": [
            {
                "similarity": round(score, 4),
                "start": chunk.start,
                "end": chunk.end,
                "excerpt": chunk.text[:500] + ("…" if len(chunk.text) > 500 else "")
            }
            for score, chunk in hits
        ]
    }

# ──────────────────────────────────────────────────────────────────────────────
# CLI
# ──────────────────────────────────────────────────────────────────────────────
def main():
    if not os.path.exists(DOC_PATH):
        print(f"Файл не найден: {DOC_PATH}")
        sys.exit(1)

    embedder = Embedder(EMBED_MODEL)
    index = build_index(DOC_PATH, embedder)

    print("\nГотово. Задавайте вопросы по документу (выход: пустая строка).")
    while True:
        try:
            q = input("\nQ: ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\nПока!")
            break
        if not q:
            print("Пока!")
            break
        resp = answer(q, index, embedder)
        print("\nA:", resp["answer"])
        print("\nИсточники:")
        for s in resp["sources"]:
            print(f"  • sim={s['similarity']}, [{s['start']}:{s['end']}] {s['excerpt']}")
        # при необходимости можно распечатать JSON
        # print(json.dumps(resp, ensure_ascii=False, indent=2))

if __name__ == "__main__":
    main()