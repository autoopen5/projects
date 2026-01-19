# OPENAI_API_KEY = "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"

import os
import re
import sys
import json
import uuid
from dataclasses import dataclass
from typing import List, Tuple, Dict, Optional

import numpy as np
from docx import Document
from sklearn.metrics.pairwise import cosine_similarity

# ============ Конфиг ============
# Несколько путей через запятую или через ; (поддержим оба варианта)
DOC_PATHS = os.getenv("DOC_PATHS", r"C:/Users/SatyaTR/Desktop/1.docx;C:/Users/SatyaTR/Desktop/2.docx").replace(";", ",").split(",")
DOC_PATHS = [p.strip() for p in DOC_PATHS if p.strip()]

EMBED_MODEL = os.getenv("EMBED_MODEL", "sentence-transformers/all-MiniLM-L6-v2")
CHUNK_SIZE = int(os.getenv("CHUNK_SIZE", 900))
CHUNK_OVERLAP = int(os.getenv("CHUNK_OVERLAP", 200))
TOP_K = int(os.getenv("TOP_K", 6))  # немного больше, т.к. документов несколько

PROVIDER = os.getenv("PROVIDER", "openai").lower()  # "openai" | "mistral"
OPENAI_MODEL = os.getenv("OPENAI_MODEL", "gpt-4o-mini")
MISTRAL_MODEL = os.getenv("MISTRAL_MODEL", "mistral-small-latest")

CTX_CHAR_BUDGET = int(os.getenv("CTX_CHAR_BUDGET", 14000))  # общий бюджет контекста

#  Алиасы имён документов (чтобы красиво подписывать источники)
# Пример: {"1.docx": "Положение о служебных командировках и поездках",
#          "2.docx": "Положение по тендерам"}
SOURCE_ALIASES = {"1.docx": "Положение о служебных командировках и поездках",
         "2.docx": "Положение по тендерам"}
if os.getenv("SOURCE_ALIASES"):
    try:
        SOURCE_ALIASES = json.loads(os.getenv("SOURCE_ALIASES"))
    except Exception:
        print("[WARN] Не удалось разобрать SOURCE_ALIASES как JSON — игнорируем.")

CACHE_DIR = os.getenv("CACHE_DIR", ".rag_cache")
os.makedirs(CACHE_DIR, exist_ok=True)

# ============ Эмбеддинги ============
class Embedder:
    def __init__(self, model_name: str):
        from sentence_transformers import SentenceTransformer
        self.model = SentenceTransformer(model_name)

    def encode(self, texts: List[str]) -> np.ndarray:
        vecs = self.model.encode(texts, normalize_embeddings=True, show_progress_bar=False)
        return np.asarray(vecs, dtype=np.float32)

# ============ Чтение DOCX с позициями строк ============
@dataclass
class PlainDoc:
    text: str
    line_starts: List[int]   # позиция начала каждой строки в text
    lines: List[str]         # сами строки

def read_docx_with_index(path: str) -> PlainDoc:
    d = Document(path)
    lines = []
    for p in d.paragraphs:
        t = p.text.replace("\xa0", " ").strip()
        if t:
            lines.append(t)
    # склеиваем в один текст с \n, фиксируем стартовые индексы строк
    text_parts = []
    line_starts = []
    pos = 0
    for ln in lines:
        line_starts.append(pos)
        text_parts.append(ln)
        pos += len(ln) + 1  # + '\n'
    full = "\n".join(lines)
    full = re.sub(r"[ \t]+", " ", full)
    full = re.sub(r"\n{3,}", "\n\n", full).strip()
    # пересчитаем line_starts после нормализации
    norm_lines = full.split("\n")
    line_starts = []
    pos = 0
    for ln in norm_lines:
        line_starts.append(pos)
        pos += len(ln) + 1
    return PlainDoc(text=full, line_starts=line_starts, lines=norm_lines)

# ============ Извлечение номера пункта ============
SECTION_RE = re.compile(r"^\s*((?:\d+\.){1,4}\d*|\d+(?:\.\d+){1,3})(?:\)|\.|\s)")

def build_section_index(doc: PlainDoc) -> List[Tuple[int, str]]:
    """
    Возвращает список (char_start, section_label) — «маяки»,
    указывающие, что с этого места действует данный номер пункта.
    """
    idx = []
    current = None
    for i, line in enumerate(doc.lines):
        m = SECTION_RE.match(line)
        if m:
            current = m.group(1)  # например "3.2.1" или "2.4"
            idx.append((doc.line_starts[i], current))
        elif current is None:
            # иногда первый раздел без цифр — пропускаем
            pass
    # если ничего не нашли — вернём пустой, chunk-и будут без номеров
    return idx

def find_section_for_pos(section_idx: List[Tuple[int, str]], pos: int) -> Optional[str]:
    """
    Бинарный поиск по списку (start, label) — найдём последний start <= pos.
    """
    if not section_idx:
        return None
    lo, hi = 0, len(section_idx) - 1
    if pos < section_idx[0][0]:
        return None
    while lo <= hi:
        mid = (lo + hi) // 2
        if section_idx[mid][0] <= pos:
            lo = mid + 1
        else:
            hi = mid - 1
    return section_idx[hi][1] if hi >= 0 else None

# ============ Чанкирование ============
@dataclass
class Chunk:
    id: str
    text: str
    start: int
    end: int
    source_id: str
    section: Optional[str]  # номер пункта (например, "3.2.1")

def make_chunks(doc: PlainDoc, source_id: str, chunk_size: int, overlap: int) -> List[Chunk]:
    section_idx = build_section_index(doc)
    chunks: List[Chunk] = []
    i, n = 0, len(doc.text)
    while i < n:
        j = min(i + chunk_size, n)
        slice_text = doc.text[i:j]
        if j < n:
            back = slice_text[::-1]
            m = re.search(r"[.!?]\s+[А-ЯA-Z]", back)
            if m and m.start() > 20:
                cut_back = m.start()
                j = i + (j - i - cut_back)
                slice_text = doc.text[i:j]
        sec = find_section_for_pos(section_idx, i)
        chunks.append(Chunk(str(uuid.uuid4()), slice_text.strip(), i, j, source_id, sec))
        if j == n:
            break
        i = max(j - overlap, 0)
    return chunks

# ============ Индексация и кэш ============
@dataclass
class VectorIndex:
    texts: List[str]
    vectors: np.ndarray
    meta: List[Chunk]

def cache_path_for(source_id: str) -> str:
    safe = re.sub(r"[^a-zA-Z0-9_.-]+", "_", source_id)
    return os.path.join(CACHE_DIR, f"{safe}.npz")

def build_or_load_for_doc(doc_path: str, embedder: Embedder) -> Tuple[List[str], np.ndarray, List[Chunk]]:
    source_id = os.path.basename(doc_path) or doc_path
    cp = cache_path_for(source_id)
    if os.path.exists(cp) and os.path.getsize(cp) > 0:
        try:
            data = np.load(cp, allow_pickle=True)
            texts = list(data["texts"])
            vectors = data["vectors"]
            meta_list = data["meta"].tolist()
            meta = [Chunk(**m) if isinstance(m, dict) else m for m in meta_list]
            return texts, vectors, meta
        except Exception:
            print(f"[WARN] Кэш повреждён, пересобираем: {cp}")

    pd = read_docx_with_index(doc_path)
    chunks = make_chunks(pd, source_id, CHUNK_SIZE, CHUNK_OVERLAP)
    texts = [c.text for c in chunks]
    vectors = embedder.encode(texts)

    np.savez_compressed(
        cp,
        texts=np.array(texts, dtype=object),
        vectors=vectors,
        meta=np.array([c.__dict__ for c in chunks], dtype=object),
    )
    return texts, vectors, chunks

def build_index(doc_paths: List[str], embedder: Embedder) -> VectorIndex:
    all_texts: List[str] = []
    all_vecs: List[np.ndarray] = []
    all_meta: List[Chunk] = []

    for p in doc_paths:
        if not os.path.exists(p):
            print(f"[WARN] Файл не найден и будет пропущен: {p}")
            continue
        t, v, m = build_or_load_for_doc(p, embedder)
        all_texts.extend(t)
        all_vecs.append(v)
        all_meta.extend(m)

    if not all_texts:
        print("Нет валидных документов для индексации.")
        sys.exit(1)

    vectors = np.vstack(all_vecs)
    return VectorIndex(texts=all_texts, vectors=vectors, meta=all_meta)

# ============ Ретрив ============
def retrieve(query: str, index: VectorIndex, embedder: Embedder, top_k: int) -> List[Tuple[float, Chunk]]:
    qv = embedder.encode([query])
    sims = cosine_similarity(qv, index.vectors)[0]
    order = np.argsort(-sims)[:top_k]
    return [(float(sims[i]), index.meta[i]) for i in order]

# ============ Утилиты источников ============
def pretty_source_name(source_id: str) -> str:
    if source_id in SOURCE_ALIASES:
        return SOURCE_ALIASES[source_id]
    low = source_id.lower()
    if "командиров" in low:
        return "Положение о служебных командировках и поездках"
    if "тендер" in low or "закуп" in low or "торг" in low:
        return "Положение по тендерам"
    return source_id

# ============ LLM ============
SYSTEM_PROMPT = (
    "Ты помощник по корпоративным документам. Отвечай кратко и по делу. "
    "Используй только предоставленный контекст (фрагменты с указанием источника и номера пункта). "
    "Если точного ответа нет в контексте — так и скажи. Не придумывай."
)

def group_hits_by_source(hits: List[Tuple[float, Chunk]]) -> Dict[str, List[Tuple[float, Chunk]]]:
    by_src: Dict[str, List[Tuple[float, Chunk]]] = {}
    for score, ch in hits:
        by_src.setdefault(ch.source_id, []).append((score, ch))
    return by_src

def build_context_block(hits: List[Tuple[float, Chunk]]) -> str:
    """
    Включаем в контекст и источник, и номер пункта (если определён),
    чтобы LLM мог опираться на конкретные «п.».
    """
    by_src = group_hits_by_source(hits)
    ordered_sources = sorted(by_src.items(), key=lambda kv: -max(s for s, _ in kv[1]))
    used = 0
    blocks = []
    for src, pairs in ordered_sources:
        pairs = sorted(pairs, key=lambda x: -x[0])
        src_header = f"### Источник: {pretty_source_name(src)}"
        chunk_texts = []
        for score, ch in pairs:
            sec = f"п. {ch.section}" if ch.section else "пункт не распознан"
            piece = f"[{sec} | sim={score:.3f} | span={ch.start}:{ch.end}]\n{ch.text.strip()}"
            if used + len(src_header) + len(piece) + 6 > CTX_CHAR_BUDGET:
                break
            chunk_texts.append(piece)
            used += len(piece) + 2
        if chunk_texts:
            blocks.append(src_header + "\n" + "\n\n".join(chunk_texts))
            used += len(src_header) + 2
        if used >= CTX_CHAR_BUDGET:
            break
    return "\n\n---\n\n".join(blocks)

def call_openai(question: str, context_block: str) -> str:
    from openai import OpenAI
    client = OpenAI()
    msgs = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user",
         "content": f"Вопрос:\n{question}\n\nКонтекст:\n{context_block}\n\nОтвет:"}
    ]
    resp = client.chat.completions.create(
        model=OPENAI_MODEL,
        messages=msgs,
        temperature=0.0,
    )
    return resp.choices[0].message.content.strip()

def call_mistral(question: str, context_block: str) -> str:
    from mistralai import Mistral
    client = Mistral(api_key=os.getenv("MISTRAL_API_KEY"))
    msgs = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user",
         "content": f"Вопрос:\n{question}\n\nКонтекст:\n{context_block}\n\nОтвет:"}
    ]
    resp = client.chat.complete(model=MISTRAL_MODEL, messages=msgs, temperature=0.0)
    return resp.choices[0].message["content"].strip()

def answer_with_llm(question: str, hits: List[Tuple[float, Chunk]]) -> Tuple[str, List[Tuple[str, List[str]]]]:
    ctx = build_context_block(hits)
    if not ctx.strip():
        return "В предоставленном контексте нет данных для ответа.", []

    text = call_mistral(question, ctx) if PROVIDER == "mistral" else call_openai(question, ctx)

    # Атрибуция: документ -> список пунктов (уникальные)
    by_src = group_hits_by_source(hits)
    contrib = []
    for src, pairs in by_src.items():
        total = float(sum(s for s, _ in pairs))
        secs = []
        for s, ch in pairs:
            if ch.section and ch.section not in secs:
                secs.append(ch.section)
        contrib.append((total, src, secs))
    contrib.sort(reverse=True)

    # Подготовим список [(pretty_source_name, [пункты...]), ...]
    sources_with_sections: List[Tuple[str, List[str]]] = []
    if contrib:
        first_total = contrib[0][0]
        for total, src, secs in contrib:
            if total >= 0.4 * first_total:  # разумная отсечка слабых следов
                sources_with_sections.append((pretty_source_name(src), secs))
    return text, sources_with_sections

# ============ CLI ============
def main():
    if not DOC_PATHS:
        print("Не задан DOC_PATHS.")
        sys.exit(1)

    for p in DOC_PATHS:
        if not os.path.exists(p):
            print(f"[WARN] Нет файла: {p}")

    if PROVIDER == "openai" and not os.getenv("OPENAI_API_KEY"):
        print("Внимание: PROVIDER=openai, но OPENAI_API_KEY не задан.")
    if PROVIDER == "mistral" and not os.getenv("MISTRAL_API_KEY"):
        print("Внимание: PROVIDER=mistral, но MISTRAL_API_KEY не задан.")

    print("Индексирую документы:")
    for p in DOC_PATHS:
        print(f"  • {p}")
    embedder = Embedder(EMBED_MODEL)
    index = build_index(DOC_PATHS, embedder)
    print(f"Всего чанков: {len(index.meta)}")

    print("\nГотово. Введите вопрос (пустая строка — выход).")
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
        final_text, srcs_secs = answer_with_llm(q, hits)

        print("\nA:", final_text)
        if srcs_secs:
            # Красиво выведем: Источник(+ы) и пункты
            parts = []
            for name, secs in srcs_secs:
                if secs:
                    parts.append(f"{name} (пп.: {', '.join(secs)})")
                else:
                    parts.append(f"{name}")
            print("Источник:", "; ".join(parts))

        print("\nИсточники (top-K):")
        for s, ch in hits:
            sec = f"п. {ch.section}" if ch.section else "пункт не распознан"
            preview = ch.text[:220].replace("\n", " ")
            suffix = "…" if len(ch.text) > 220 else ""
            print(f"  • {pretty_source_name(ch.source_id)} | {sec} | sim={s:.3f} | span={ch.start}:{ch.end} | {preview}{suffix}")

if __name__ == "__main__":
    main()