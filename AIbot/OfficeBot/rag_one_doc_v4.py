# OPENAI_API_KEY = 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

import os
import re
import sys
import uuid
from dataclasses import dataclass
from typing import List, Tuple

import numpy as np
from docx import Document
from tqdm import tqdm
from sklearn.metrics.pairwise import cosine_similarity

# ============ Конфиг через ENV ============
DOC_PATH = os.getenv("DOC_PATH", r"C:/Users/SatyaTR/Desktop/1.docx")
EMBED_MODEL = os.getenv("EMBED_MODEL", "sentence-transformers/all-MiniLM-L6-v2")

# Для поиска
CHUNK_SIZE = int(os.getenv("CHUNK_SIZE", 900))        # символов
CHUNK_OVERLAP = int(os.getenv("CHUNK_OVERLAP", 200))  # символов
TOP_K = int(os.getenv("TOP_K", 5))

# LLM провайдеры
PROVIDER = os.getenv("PROVIDER", "openai").lower()  # "openai" | "mistral"
OPENAI_MODEL = os.getenv("OPENAI_MODEL", "gpt-4o-mini")
MISTRAL_MODEL = os.getenv("MISTRAL_MODEL", "mistral-small-latest")

# Ограничение контекста (в символах) чтобы не взорвать prompt
CTX_CHAR_BUDGET = int(os.getenv("CTX_CHAR_BUDGET", 12000))

# ============ Эмбеддинги ============
class Embedder:
    def __init__(self, model_name: str):
        from sentence_transformers import SentenceTransformer
        self.model = SentenceTransformer(model_name)

    def encode(self, texts: List[str]) -> np.ndarray:
        vecs = self.model.encode(texts, normalize_embeddings=True, show_progress_bar=False)
        return np.asarray(vecs, dtype=np.float32)

# ============ Чтение документа ============
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

# ============ Чанкирование ============
@dataclass
class Chunk:
    id: str
    text: str
    start: int
    end: int

def make_chunks(text: str, chunk_size: int, overlap: int) -> List[Chunk]:
    chunks: List[Chunk] = []
    i, n = 0, len(text)
    while i < n:
        j = min(i + chunk_size, n)
        slice_text = text[i:j]
        if j < n:
            # стараемся резать по границе предложений
            back = slice_text[::-1]
            m = re.search(r"[.!?]\s+[А-ЯA-Z]", back)
            if m and m.start() > 20:
                cut_back = m.start()
                j = i + (j - i - cut_back)
                slice_text = text[i:j]
        chunks.append(Chunk(str(uuid.uuid4()), slice_text.strip(), i, j))
        if j == n:
            break
        i = max(j - overlap, 0)
    return chunks

# ============ Индекс ============
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
    return VectorIndex(texts, vectors, chunks)

# ============ Ретрив ============
def retrieve(query: str, index: VectorIndex, embedder: Embedder, top_k: int) -> List[Tuple[float, Chunk]]:
    qv = embedder.encode([query])
    sims = cosine_similarity(qv, index.vectors)[0]
    order = np.argsort(-sims)[:top_k]
    return [(float(sims[i]), index.meta[i]) for i in order]

# ============ Prompt & LLM вызов ============
SYSTEM_PROMPT = (
    "Ты помощник по корпоративному документу (Положение о служебных командировках и поездках). "
    "Отвечай строго на русском, кратко и по делу. Используй только предоставленный контекст. "
    "Если ответа в контексте нет — скажи об этом и не выдумывай."
)

def build_context_block(hits: List[Tuple[float, Chunk]]) -> str:
    # Собираем чанки, пока не исчерпаем бюджет
    ctx = []
    used = 0
    for score, ch in hits:
        t = ch.text.strip()
        if not t:
            continue
        add = f"[sim={score:.3f} | span={ch.start}:{ch.end}]\n{t}"
        if used + len(add) + 2 > CTX_CHAR_BUDGET:
            break
        ctx.append(add)
        used += len(add) + 2
    return "\n\n---\n\n".join(ctx)

def call_openai(question: str, context_block: str) -> str:
    from openai import OpenAI
    client = OpenAI()
    msgs = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user",
         "content": f"Вопрос:\n{question}\n\nКонтекст (фрагменты документа):\n{context_block}\n\nОтвет:"}
    ]
    resp = client.chat.completions.create(
        model=OPENAI_MODEL,
        messages=msgs,
        temperature=0.0,
    )
    return resp.choices[0].message.content.strip()

# def call_mistral(question: str, context_block: str) -> str:
#     # Требуется MISTRAL_API_KEY
#     from mistralai import Mistral
#     client = Mistral(api_key=os.getenv("MISTRAL_API_KEY"))
#     msgs = [
#         {"role": "system", "content": SYSTEM_PROMPT},
#         {"role": "user",
#          "content": f"Вопрос:\n{question}\n\nКонтекст (фрагменты документа):\n{context_block}\n\nОтвет:"}
#     ]
#     resp = client.chat.complete(model=MISTRAL_MODEL, messages=msgs, temperature=0.0)
#     return resp.choices[0].message["content"].strip()

def answer_with_llm(question: str, hits: List[Tuple[float, Chunk]]) -> str:
    context_block = build_context_block(hits)
    if not context_block.strip():
        return "В предоставленном контексте нет данных для ответа."
    # if PROVIDER == "mistral":
    #     return call_mistral(question, context_block)
    # по умолчанию OpenAI
    return call_openai(question, context_block)

# ============ CLI ============
def main():
    if not os.path.exists(DOC_PATH):
        print(f"Файл не найден: {DOC_PATH}")
        sys.exit(1)

    # Быстрая проверка ключей
    if PROVIDER == "openai" and not os.getenv("OPENAI_API_KEY"):
        print("Внимание: PROVIDER=openai, но переменная OPENAI_API_KEY не задана.")
    # if PROVIDER == "mistral" and not os.getenv("MISTRAL_API_KEY"):
    #     print("Внимание: PROVIDER=mistral, но переменная MISTRAL_API_KEY не задана.")

    embedder = Embedder(EMBED_MODEL)
    index = build_index(DOC_PATH, embedder)

    print("\nГотово. Задавайте вопросы (пустая строка — выход).")
    while True:
        try:
            q = input("\nQ: ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\nПока!")
            break
        if not q:
            print("Пока!")
            break

        hits = retrieve(q, index, embedder, TOP_K)
        final = answer_with_llm(q, hits)

        print("\nA:", final)
        print("\nИсточники (top-K):")
        for s, ch in hits:
            excerpt = ch.text[:260].replace("\n", " ")
            suffix = "…" if len(ch.text) > 260 else ""
            print(f"  • sim={s:.3f} | span={ch.start}:{ch.end} | {excerpt}{suffix}")

if __name__ == "__main__":
    main()