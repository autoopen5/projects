#  rag_one_doc_v1.py
from __future__ import annotations
import argparse
import json
import re
import hashlib
from pathlib import Path
from typing import List, Dict, Any, Tuple

import warnings
warnings.filterwarnings("ignore", message="_cross_entropy is deprecated")

import numpy as np
from docx import Document
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.feature_extraction.text import TfidfVectorizer
import joblib
from scipy.sparse import csr_matrix

# ───────────────────────────────────────────────────────────────────────────────
# Конфиг по умолчанию
# ───────────────────────────────────────────────────────────────────────────────
MODEL_NAME = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
CHUNK_MAX_CHARS = 1200
CHUNK_OVERLAP_CHARS = 150
TOP_K = 5

SENT_SPLIT_RE = re.compile(r"(?<=[\.\!\?])\s+")


def is_noise_sentence(s: str, tfidf_vectorizer, global_idf_threshold=1.0):
    s_clean = re.sub(r"[^а-яА-Яa-zA-Z\s]", " ", s)
    if len(s_clean.split()) < 4: return True
    upper_ratio = sum(1 for c in s if c.isupper()) / max(1, len(s))
    if upper_ratio > 0.5: return True
    idf_scores = []
    for w in s_clean.lower().split():
        if w in tfidf_vectorizer.vocabulary_:
            idx = tfidf_vectorizer.vocabulary_[w]
            idf = tfidf_vectorizer.idf_[idx]
            idf_scores.append(idf)
    if idf_scores and np.mean(idf_scores) < global_idf_threshold:
        return True
    return False

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
        # Простая эвристика заголовков
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
                chunks.append({"text": buf, "span": {"start": start_idx, "end": i - 1}})
            if len(buf) > overlap:
                buf_tail = buf[-overlap:]
                buf = (buf_tail + "\n" + b["text"]).strip()
            else:
                buf = b["text"]
            start_idx = max(0, i - 1)
    if buf:
        chunks.append({"text": buf, "span": {"start": start_idx, "end": len(blocks) - 1}})

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

        # TF-IDF артефакты
        self.tfidf_vec_pkl = self.base_dir / "tfidf_vectorizer.joblib"
        self.tfidf_npz = self.base_dir / "tfidf_matrix.npz"
        self.vocab_json = self.base_dir / "tfidf_vocab.json"

    def save(self,
             chunks: List[Dict[str, Any]],
             emb: np.ndarray,
             tfidf_vec: TfidfVectorizer,
             tfidf_mat: csr_matrix):
        with open(self.chunks_json, "w", encoding="utf-8") as f:
            json.dump(chunks, f, ensure_ascii=False, indent=2)

        np.savez_compressed(self.emb_npz, embeddings=emb)

        # TF-IDF
        joblib.dump(tfidf_vec, self.tfidf_vec_pkl)
        np.savez_compressed(self.tfidf_npz,
                            data=tfidf_mat.data,
                            indices=tfidf_mat.indices,
                            indptr=tfidf_mat.indptr,
                            shape=tfidf_mat.shape)
        with open(self.vocab_json, "w", encoding="utf-8") as f:
            json.dump(tfidf_vec.vocabulary_, f, ensure_ascii=False)

        with open(self.meta_json, "w", encoding="utf-8") as f:
            json.dump({"file": str(self.doc_path),
                       "doc_hash": self.doc_hash,
                       "model": MODEL_NAME,
                       "dim": int(emb.shape[1])}, f, ensure_ascii=False, indent=2)

    def load(self) -> Tuple[List[Dict[str, Any]], np.ndarray, TfidfVectorizer, csr_matrix]:
        if not (self.chunks_json.exists() and self.emb_npz.exists() and
                self.tfidf_vec_pkl.exists() and self.tfidf_npz.exists()):
            raise FileNotFoundError("Индекс не найден. Постройте его флагом --build.")
        with open(self.chunks_json, "r", encoding="utf-8") as f:
            chunks = json.load(f)
        emb = np.load(self.emb_npz)["embeddings"]

        tfidf_vec = joblib.load(self.tfidf_vec_pkl)
        tfidf_npz = np.load(self.tfidf_npz)
        tfidf_mat = csr_matrix((tfidf_npz["data"], tfidf_npz["indices"], tfidf_npz["indptr"]),
                               shape=tfidf_npz["shape"])
        return chunks, emb, tfidf_vec, tfidf_mat


class RAGOneDoc:
    def __init__(self, doc_path: Path, model_name: str = MODEL_NAME):
        self.doc_path = doc_path
        self.store = IndexStore(doc_path)
        self.model = SentenceTransformer(model_name)

    def build(self) -> Dict[str, Any]:
        blocks = read_docx_text(self.doc_path)
        chunks = chunk_blocks(blocks)
        texts = [c["text"] for c in chunks]

        # Эмбеддинги
        emb = self.model.encode(texts, batch_size=32, show_progress_bar=False, normalize_embeddings=True)
        emb = np.asarray(emb, dtype=np.float32)

        # TF-IDF (ru+en стоп-слова и н-грамы)
        stop_ru = {"и","в","во","не","что","он","на","я","с","со","как","а","то","все","она","так","его","но","да","ты","к",
                   "у","же","вы","за","бы","по","ее","мне","есть","если","или","ни","тем","нет","чтобы","при","это","из",
                   "уже","для","до","от","также","после","без"}
        stop_en = {"the","and","is","to","of","in","a","on","for","with","as","by","at","from","that","this","it","an",
                   "be","are","or"}
        tfidf_vec = TfidfVectorizer(lowercase=True, ngram_range=(1, 2), min_df=1,
                                    stop_words=list(stop_ru | stop_en))
        tfidf_mat = tfidf_vec.fit_transform(texts)

        self.store.save(chunks, emb, tfidf_vec, tfidf_mat)
        return {"chunks": len(chunks), "dim": int(emb.shape[1]), "index_dir": str(self.store.base_dir)}

    # ───────────────────────────────────────────────────────────────────────────
    # Трансформация запроса (query expansion) по TF-IDF
    # ───────────────────────────────────────────────────────────────────────────
    def _expand_query(self,
                      query: str,
                      tfidf_vec: TfidfVectorizer,
                      tfidf_mat: csr_matrix,
                      texts: List[str],
                      top_m_chunks: int = 8,
                      add_terms: int = 6) -> str:
        q_vec = tfidf_vec.transform([query])
        tf_sims = cosine_similarity(q_vec, tfidf_mat)[0]
        idxs = np.argsort(-tf_sims)[:max(1, top_m_chunks)]

        # средние TF-IDF веса по выбранным чанкам
        mean_weights = None
        for i in idxs:
            row = tfidf_mat.getrow(i)
            row_dense = row.toarray()[0]
            if mean_weights is None:
                mean_weights = row_dense
            else:
                mean_weights += row_dense
        mean_weights = mean_weights / max(1, len(idxs))

        vocab = tfidf_vec.vocabulary_  # term -> idx
        inv_vocab = {i: t for t, i in vocab.items()}

        top_idx = np.argsort(-mean_weights)[:add_terms * 3]  # возьмём с запасом, потом отфильтруем
        query_lc = set(re.findall(r"[а-яА-Яa-zA-Z0-9-]{3,}", query.lower()))
        candidates: List[str] = []
        for j in top_idx:
            term = inv_vocab.get(int(j))
            if not term:
                continue
            words = term.split()
            if any(w in query_lc for w in words):
                continue
            if len(term) < 3:
                continue
            candidates.append(term)

        expanded_terms = candidates[:add_terms]
        if not expanded_terms:
            return query
        return query + " " + " ".join(expanded_terms)

    # ───────────────────────────────────────────────────────────────────────────
    # Экстрактивный ответ на основе предложений + MMR
    # ───────────────────────────────────────────────────────────────────────────
    @staticmethod
    def _split_sentences(text: str) -> List[str]:
        return [s.strip() for s in SENT_SPLIT_RE.split(text) if s.strip()]

    @staticmethod
    def _mmr_select(doc_vectors: np.ndarray, query_vector: np.ndarray, k: int = 5, lambda_div: float = 0.6):
        """
        Maximal Marginal Relevance для выбора k предложений.
        lambda_div ~ 0.6: баланс между релевантностью и разнообразием.
        """
        rel = cosine_similarity(doc_vectors, query_vector.reshape(1, -1)).ravel()
        n = len(rel)
        if n == 0:
            return []
        selected = []
        candidates = set(range(n))
        # Нормируем и предрассчитываем попарные косинусы между предложениями
        doc_norm = doc_vectors / (np.linalg.norm(doc_vectors, axis=1, keepdims=True) + 1e-12)
        sim_docs = np.clip(doc_norm @ doc_norm.T, 0.0, 1.0)

        # Инициализация: самый релевантный
        first = int(np.argmax(rel))
        selected.append(first)
        candidates.remove(first)

        while len(selected) < min(k, n) and candidates:
            best_i, best_score = None, -1e9
            for i in list(candidates):
                div = np.max(sim_docs[i, selected]) if selected else 0.0
                score = lambda_div * rel[i] - (1 - lambda_div) * div
                if score > best_score:
                    best_score, best_i = score, i
            selected.append(best_i)
            candidates.remove(best_i)
        return selected

    def _extractive_answer(self,
                           query: str,
                           top_chunks: List[Dict[str, Any]],
                           max_sentences: int = 5,
                           min_sim_threshold: float = 0.42) -> Dict[str, Any]:
        # собираем предложения
        sentences: List[str] = []
        sent_meta: List[str] = []
        for ch in top_chunks:
            for s in self._split_sentences(ch["text"]):
                if len(s) < 20:
                    continue
                sentences.append(s)
                sent_meta.append(ch["chunk_id"])
        if not sentences:
            return {"answer": "", "citations": []}

        # эмбеддинги предложений и запроса
        q_vec = self.model.encode([query], normalize_embeddings=True)
        s_vecs = self.model.encode(sentences, batch_size=64, show_progress_bar=False, normalize_embeddings=True)
        sims = cosine_similarity(q_vec, s_vecs)[0]

        # если совсем низкая релевантность — аккуратный fallback
        top_sim = float(np.max(sims)) if len(sims) else 0.0
        if top_sim < min_sim_threshold:
            fallback = []
            if top_chunks:
                first = top_chunks[0]["text"]
                first_sents = self._split_sentences(first)
                for s in first_sents:
                    if len(s) > 60:
                        fallback.append(s)
                    if len(fallback) >= 2:
                        break
            answer_text = " ".join(fallback) if fallback else ""
            citations = [top_chunks[0]["chunk_id"]] if top_chunks else []
            return {"answer": answer_text, "citations": citations}

        # выбираем предложения по MMR
        picked_idx = self._mmr_select(s_vecs, q_vec[0], k=max_sentences, lambda_div=0.6)
        picked = [(float(sims[i]), sentences[i], sent_meta[i]) for i in picked_idx]
        picked.sort(key=lambda x: x[0], reverse=True)

        answer_text = " ".join(p[1] for p in picked)
        citations = sorted(list({p[2] for p in picked}))
        return {"answer": answer_text, "citations": citations}

    # ───────────────────────────────────────────────────────────────────────────
    # Основной метод запроса: Query Expansion + Гибридный скоринг
    # ───────────────────────────────────────────────────────────────────────────
    def ask(self, query: str, top_k: int = TOP_K) -> Dict[str, Any]:
        chunks, emb, tfidf_vec, tfidf_mat = self.store.load()
        texts = [c["text"] for c in chunks]

        # 1) расширяем запрос по TF-IDF
        expanded_query = self._expand_query(query, tfidf_vec, tfidf_mat, texts)

        # 2) эмбеддинговый косинус по расширенному запросу
        q_vec_emb = self.model.encode([expanded_query], normalize_embeddings=True)[0].reshape(1, -1)
        emb_sims = cosine_similarity(q_vec_emb, emb)[0]

        # 3) TF-IDF косинус по расширенному запросу
        q_vec_tfidf = tfidf_vec.transform([expanded_query])
        tfidf_sims = cosine_similarity(q_vec_tfidf, tfidf_mat)[0]

        # 4) гибрид: эмбеддинги (семантика) + TF-IDF (точные термины)
        alpha = 0.6
        sims = alpha * emb_sims + (1.0 - alpha) * tfidf_sims

        top_idx = np.argsort(-sims)[:top_k]
        top = [{"chunk_id": chunks[i]["chunk_id"], "similarity": float(sims[i]), "text": chunks[i]["text"]}
               for i in top_idx]

        answer = self._extractive_answer(expanded_query, top)
        return {"query": query, "expanded_query": expanded_query, "top": top, **answer}


def main():
    DEFAULT_FILE = r"C:/Users/SatyaTR/Desktop/1.docx"
    ap = argparse.ArgumentParser(description="RAG по одному .docx документу (без LLM).")
    ap.add_argument("--file", default=DEFAULT_FILE, help="Путь к .docx файлу (по умолчанию используется DEFAULT_FILE)")
    ap.add_argument("--build", action="store_true", help="Построить/перестроить индекс")
    ap.add_argument("--ask", type=str, help="Вопрос пользователЯ")
    ap.add_argument("--topk", type=int, default=TOP_K, help="Сколько чанков возвращать (по умолчанию 5)")
    args = ap.parse_args()

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