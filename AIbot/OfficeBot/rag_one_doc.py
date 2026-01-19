# rag_one_doc.py
from __future__ import annotations
import argparse
import json
import re
import hashlib
from pathlib import Path
from typing import List, Dict, Any

import numpy as np
from docx import Document
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity

# ───────────────────────────────────────────────────────────────────────────────
# Конфиг по умолчанию
# ───────────────────────────────────────────────────────────────────────────────
MODEL_NAME = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
CHUNK_MAX_CHARS = 1200
CHUNK_OVERLAP_CHARS = 150
TOP_K = 5

SENT_SPLIT_RE = re.compile(r"(?<=[\.\!\?])\s+")


def read_docx_text(path: Path) -> List[Dict[str, Any]]:
    """Читает .docx, возвращает список блоков текста с простыми метаданными."""
    if not path.exists():
        raise FileNotFoundError(f"Документ не найден: {path}")
    doc = Document(str(path))
    blocks = []
    section_idx = 0
    para_idx = 0
    for p in doc.paragraphs:
        t = (p.text or "").strip()
        if not t:
            para_idx += 1
            continue
        is_header = (len(t) < 120 and (t.isupper() or t.endswith(":")))
        if is_header:
            section_idx += 1
            para_idx = 0
        blocks.append({"text": t, "meta": {"section": section_idx, "para": para_idx}})
        para_idx += 1
    return blocks


def chunk_blocks(blocks: List[Dict[str, Any]],
                 max_chars=CHUNK_MAX_CHARS,
                 overlap=CHUNK_OVERLAP_CHARS) -> List[Dict[str, Any]]:
    """Объединяет параграфы в чанки ограниченной длины с overlap."""
    chunks: List[Dict[str, Any]] = []
    buf = ""
    start_idx = 0
    for i, b in enumerate(blocks):
        candidate = (buf + "\n" + b["text"]).strip() if buf else b["text"]
        if len(candidate) <= max_chars:
            buf = candidate
        else:
            if buf:
                chunks.append({"text": buf, "span": {"start": start_idx, "end": i-1}})
            if len(buf) > overlap:
                buf_tail = buf[-overlap:]
                buf = (buf_tail + "\n" + b["text"]).strip()
            else:
                buf = b["text"]
            start_idx = max(0, i-1)
    if buf:
        chunks.append({"text": buf, "span": {"start": start_idx, "end": len(blocks)-1}})

    for idx, ch in enumerate(chunks):
        ch["chunk_id"] = f"chunk_{idx:04d}"
        ch["hash"] = hashlib.sha256(ch["text"].encode("utf-8")).hexdigest()
    return chunks


class IndexStore:
    """Хранит индекс рядом с файлом в папке .rag_index/<doc_hash>"""
    def __init__(self, doc_path: Path):
        self.doc_path = doc_path
        with open(doc_path, "rb") as f:
            doc_bytes = f.read()
        self.doc_hash = hashlib.sha256(doc_bytes).hexdigest()[:16]
        self.base_dir = doc_path.parent / ".rag_index" / self.doc_hash
        self.base_dir.mkdir(parents=True, exist_ok=True)
        self.chunks_json = self.base_dir / "chunks.json"
        self.emb_npz = self.base_dir / "embeddings.npz"
        self.meta_json = self.base_dir / "meta.json"

    def save(self, chunks: List[Dict[str, Any]], emb: np.ndarray):
        with open(self.chunks_json, "w", encoding="utf-8") as f:
            json.dump(chunks, f, ensure_ascii=False, indent=2)
        np.savez_compressed(self.emb_npz, embeddings=emb)
        with open(self.meta_json, "w", encoding="utf-8") as f:
            json.dump({"file": str(self.doc_path), "doc_hash": self.doc_hash,
                       "model": MODEL_NAME, "dim": int(emb.shape[1])}, f, ensure_ascii=False, indent=2)

    def load(self):
        if not (self.chunks_json.exists() and self.emb_npz.exists()):
            raise FileNotFoundError("Индекс не найден. Постройте его флагом --build.")
        with open(self.chunks_json, "r", encoding="utf-8") as f:
            chunks = json.load(f)
        emb = np.load(self.emb_npz)["embeddings"]
        return chunks, emb


class RAGOneDoc:
    def __init__(self, doc_path: Path, model_name: str = MODEL_NAME):
        self.doc_path = doc_path
        self.store = IndexStore(doc_path)
        self.model = SentenceTransformer(model_name)

    def build(self) -> Dict[str, Any]:
        blocks = read_docx_text(self.doc_path)
        chunks = chunk_blocks(blocks)
        texts = [c["text"] for c in chunks]
        emb = self.model.encode(texts, batch_size=32, show_progress_bar=False, normalize_embeddings=True)
        emb = np.asarray(emb, dtype=np.float32)
        self.store.save(chunks, emb)
        return {"chunks": len(chunks), "dim": int(emb.shape[1]), "index_dir": str(self.store.base_dir)}

    def ask(self, query: str, top_k: int = TOP_K) -> Dict[str, Any]:
        chunks, emb = self.store.load()
        q_vec = self.model.encode([query], normalize_embeddings=True)[0].reshape(1, -1)
        sims = cosine_similarity(q_vec, emb)[0]
        top_idx = np.argsort(-sims)[:top_k]
        top = [{"chunk_id": chunks[i]["chunk_id"], "similarity": float(sims[i]), "text": chunks[i]["text"]}
               for i in top_idx]
        answer = self._extractive_answer(query, top)
        return {"query": query, "top": top, **answer}

    @staticmethod
    def _split_sentences(text: str) -> List[str]:
        return [s.strip() for s in SENT_SPLIT_RE.split(text) if s.strip()]

    def _extractive_answer(self, query: str, top_chunks: List[Dict[str, Any]], max_sentences: int = 6) -> Dict[str, Any]:
        q_terms = set(re.findall(r"[а-яА-Яa-zA-Z0-9-]{3,}", query.lower()))
        scored: List[tuple[float, str, str]] = []
        for ch in top_chunks:
            for s in self._split_sentences(ch["text"]):
                s_terms = set(re.findall(r"[а-яА-Яa-zA-Z0-9-]{3,}", s.lower()))
                inter = len(q_terms & s_terms)
                if inter > 0:
                    scored.append((inter / max(1, len(q_terms)), s, ch["chunk_id"]))
        if not scored:
            fallback = []
            if top_chunks:
                for s in self._split_sentences(top_chunks[0]["text"])[:3]:
                    fallback.append((0.0, s, top_chunks[0]["chunk_id"]))
            scored = fallback
        scored.sort(key=lambda x: x[0], reverse=True)
        picked = scored[:max_sentences]
        answer_text = " ".join(p[1] for p in picked) if picked else ""
        citations = sorted(list({p[2] for p in picked})) if picked else []
        return {"answer": answer_text, "citations": citations}


def main():
    DEFAULT_FILE = r"C:/Users/SatyaTR/Desktop/1.docx"
    ap = argparse.ArgumentParser(description="RAG по одному .docx документу (без LLM).")
    ap.add_argument("--file", default=DEFAULT_FILE, help="Путь к .docx файлу (по умолчанию используется DEFAULT_FILE)")
    ap.add_argument("--build", action="store_true", help="Построить/перестроить индекс")
    ap.add_argument("--ask", type=str, help="Вопрос пользователЯ")
    ap.add_argument("--topk", type=int, default=TOP_K, help="Сколько чанков возвращать (по умолчанию 5)")
    args = ap.parse_args()

    # 🧠 Используем путь, даже если аргумент не передан
    doc_path = Path(args.file)
    rag = RAGOneDoc(doc_path, MODEL_NAME)

    if args.build:
        stats = rag.build()
        print(json.dumps({"status": "ok", "index": stats}, ensure_ascii=False, indent=2))

    if args.ask:
        result = rag.ask(args.ask, top_k=args.topk)
        print(json.dumps(result, ensure_ascii=False, indent=2))

    if not args.build and not args.ask:
        ap.error("Нужно указать хотя бы один из флагов: --build или --ask.")


if __name__ == "__main__":
    main()