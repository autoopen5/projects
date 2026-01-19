# rag_one_doc_v2.py
from __future__ import annotations
import argparse, json, re, hashlib, warnings
from pathlib import Path
from typing import List, Dict, Any, Tuple

warnings.filterwarnings("ignore", message="_cross_entropy is deprecated")

import numpy as np
from docx import Document
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.feature_extraction.text import TfidfVectorizer
from scipy.sparse import csr_matrix
import joblib

# optional: KeyBERT (авто-ключевые фразы)
try:
    from keybert import KeyBERT
    _HAS_KEYBERT = True
except Exception:
    _HAS_KEYBERT = False

# ───────────────────────────────────────────────────────────────────────────────
# Конфиг
# ───────────────────────────────────────────────────────────────────────────────
MODEL_NAME = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
CHUNK_MAX_CHARS = 500
CHUNK_OVERLAP_CHARS = 220
TOP_K = 5

SENT_SPLIT_RE = re.compile(r"(?<=[\.\!\?])\s+")
UPPER_LINE_RE = re.compile(r"^[A-ZА-Я0-9 ,«»\"()№:;._\-–—/]{6,}$")  # «капслоковые» строки/шапки

DOC_TYPES = ["Положение","Инструкция","Регламент","Политика","Порядок"]
# подсказки предметной области (чуть повышаем шанс их попадания в топ)
DOMAIN_HINTS = [
    "командировк","поездк","суточ","транспорт","такси","парков","отчет","заявк",
    "соглас","оплат","прожив","аэропорт","вокзал","авиа","жд","служебн","компенсац","расход"
]

# стоп-слова (ru+en) — базовые
STOP_RU = {
    "и","в","во","не","что","он","на","я","с","со","как","а","то","все","она","так","его","но","да","ты","к",
    "у","же","вы","за","бы","по","ее","мне","есть","если","или","ни","тем","нет","чтобы","при","это","из",
    "уже","для","до","от","также","после","без","поэтому","таким","образом","данный","данная","данные"
}
STOP_EN = {
    "the","and","is","to","of","in","a","on","for","with","as","by","at","from","that","this","it","an",
    "be","are","or","into","within","about","via","per"
}

# ───────────────────────────────────────────────────────────────────────────────
# Утилиты очистки/фильтрации
# ───────────────────────────────────────────────────────────────────────────────
def _clean_text_for_answer(s: str) -> str:
    s = re.sub(r"\s+", " ", s).strip()
    # срежем «шаблонные хвосты»: пустые даты/номера
    s = re.sub(r"\b(__\.){1,}\b", "", s)
    s = re.sub(r"\b№\s*[_\-–—]*\b", "", s)
    if UPPER_LINE_RE.match(s):
        return ""
    return s

def _is_noise_sentence(s: str, tfidf_vec: TfidfVectorizer,
                       min_words: int = 4,
                       upper_ratio_thresh: float = 0.5,
                       global_idf_threshold: float = 1.0) -> bool:
    s_strip = s.strip()
    if not s_strip:
        return True
    if len(s_strip.split()) < min_words:
        return True
    # доля заглавных
    letters = [c for c in s_strip if c.isalpha()]
    if letters:
        upper_ratio = sum(1 for c in letters if c.isupper()) / len(letters)
        if upper_ratio > upper_ratio_thresh:
            return True
    # средний IDF слов в предложении (низкий — «общие/шумные» слова)
    tokens = re.findall(r"[а-яА-Яa-zA-Z]{2,}", s_strip.lower())
    idf_scores = []
    vocab = tfidf_vec.vocabulary_
    idfs = getattr(tfidf_vec, "idf_", None)
    if idfs is not None:
        for w in tokens:
            j = vocab.get(w)
            if j is not None:
                idf_scores.append(idfs[j])
    if idf_scores and float(np.mean(idf_scores)) < global_idf_threshold:
        return True
    return False

def _score_term_for_domain(term: str) -> int:
    tl = term.lower()
    return sum(1 for h in DOMAIN_HINTS if h in tl)

# ───────────────────────────────────────────────────────────────────────────────
# Чтение и чанкинг
# ───────────────────────────────────────────────────────────────────────────────
def read_docx_text(path: Path) -> List[Dict[str, Any]]:
    if not path.exists():
        raise FileNotFoundError(f"Документ не найден: {path}")
    doc = Document(str(path))
    blocks, section_idx, para_idx = [], 0, 0
    for p in doc.paragraphs:
        t = (p.text or "").strip()
        if not t:
            para_idx += 1
            continue
        is_header = (len(t) < 120 and (t.isupper() or t.endswith(":"))) or UPPER_LINE_RE.match(t)
        if is_header:
            section_idx += 1
            para_idx = 0
        blocks.append({"text": t, "meta": {"section": section_idx, "para": para_idx}})
        para_idx += 1
    return blocks

def chunk_blocks(blocks: List[Dict[str, Any]],
                 max_chars=CHUNK_MAX_CHARS,
                 overlap=CHUNK_OVERLAP_CHARS) -> List[Dict[str, Any]]:
    chunks, buf, start_idx = [], "", 0
    for i, b in enumerate(blocks):
        candidate = (buf + "\n" + b["text"]).strip() if buf else b["text"]
        if len(candidate) <= max_chars:
            buf = candidate
        else:
            if buf:
                chunks.append({"text": buf, "span": {"start": start_idx, "end": i-1}})
            tail = buf[-overlap:] if len(buf) > overlap else ""
            buf = (tail + "\n" + b["text"]).strip()
            start_idx = max(0, i-1)
    if buf:
        chunks.append({"text": buf, "span": {"start": start_idx, "end": len(blocks)-1}})
    for idx, ch in enumerate(chunks):
        ch["chunk_id"] = f"chunk_{idx:04d}"
        ch["hash"] = hashlib.sha256(ch["text"].encode("utf-8")).hexdigest()
    return chunks

# ───────────────────────────────────────────────────────────────────────────────
# Хранилище индекса
# ───────────────────────────────────────────────────────────────────────────────
class IndexStore:
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
        self.tfidf_vec_pkl = self.base_dir / "tfidf_vectorizer.joblib"
        self.tfidf_npz = self.base_dir / "tfidf_matrix.npz"
        self.vocab_json = self.base_dir / "tfidf_vocab.json"

    def save(self, chunks, emb, tfidf_vec: TfidfVectorizer, tfidf_mat: csr_matrix):
        with open(self.chunks_json, "w", encoding="utf-8") as f:
            json.dump(chunks, f, ensure_ascii=False, indent=2)
        np.savez_compressed(self.emb_npz, embeddings=emb)
        joblib.dump(tfidf_vec, self.tfidf_vec_pkl)
        np.savez_compressed(self.tfidf_npz, data=tfidf_mat.data, indices=tfidf_mat.indices,
                            indptr=tfidf_mat.indptr, shape=tfidf_mat.shape)
        with open(self.vocab_json, "w", encoding="utf-8") as f:
            json.dump(tfidf_vec.vocabulary_, f, ensure_ascii=False)
        with open(self.meta_json, "w", encoding="utf-8") as f:
            json.dump({"file": str(self.doc_path), "doc_hash": self.doc_hash,
                       "model": MODEL_NAME, "dim": int(emb.shape[1])}, f, ensure_ascii=False, indent=2)

    def load(self):
        if not (self.chunks_json.exists() and self.emb_npz.exists()
                and self.tfidf_vec_pkl.exists() and self.tfidf_npz.exists()):
            raise FileNotFoundError("Индекс не найден. Постройте его флагом --build.")
        with open(self.chunks_json, "r", encoding="utf-8") as f:
            chunks = json.load(f)
        emb = np.load(self.emb_npz)["embeddings"]
        tfidf_vec = joblib.load(self.tfidf_vec_pkl)
        tfidf_npz = np.load(self.tfidf_npz)
        tfidf_mat = csr_matrix((tfidf_npz["data"], tfidf_npz["indices"], tfidf_npz["indptr"]),
                               shape=tfidf_npz["shape"])
        return chunks, emb, tfidf_vec, tfidf_mat

# ───────────────────────────────────────────────────────────────────────────────
# Модель RAG
# ───────────────────────────────────────────────────────────────────────────────
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

        tfidf_vec = TfidfVectorizer(lowercase=True, ngram_range=(1,2), min_df=1,
                                    stop_words=list(STOP_RU | STOP_EN))
        tfidf_mat = tfidf_vec.fit_transform(texts)

        self.store.save(chunks, emb, tfidf_vec, tfidf_mat)
        return {"chunks": len(chunks), "dim": int(emb.shape[1]), "index_dir": str(self.store.base_dir)}

    # ——— helpers ———
    @staticmethod
    def _split_sentences(text: str) -> List[str]:
        return [s.strip() for s in SENT_SPLIT_RE.split(text) if s.strip()]

    @staticmethod
    def _mmr_select(doc_vectors: np.ndarray, query_vector: np.ndarray, k: int = 5, lambda_div: float = 0.6):
        rel = cosine_similarity(doc_vectors, query_vector.reshape(1, -1)).ravel()
        n = len(rel)
        if n == 0: return []
        selected, candidates = [], set(range(n))
        doc_norm = doc_vectors / (np.linalg.norm(doc_vectors, axis=1, keepdims=True) + 1e-12)
        sim_docs = np.clip(doc_norm @ doc_norm.T, 0.0, 1.0)
        first = int(np.argmax(rel)); selected.append(first); candidates.remove(first)
        while len(selected) < min(k, n) and candidates:
            best_i, best_score = None, -1e9
            for i in list(candidates):
                div = float(np.max(sim_docs[i, selected])) if selected else 0.0
                score = lambda_div * rel[i] - (1 - lambda_div) * div
                if score > best_score: best_score, best_i = score, i
            selected.append(best_i); candidates.remove(best_i)
        return selected

    def _is_about_query(self, q: str) -> bool:
        q = q.lower().strip()
        pats = [r"\bо\s*ч[её]м\s+документ", r"\bчто\s+за\s+документ", r"\bкакова\s+тематика\b",
                r"\bкратко\s+о\s+документе\b", r"\babout\b.*\bdocument\b"]
        return any(re.search(p, q) for p in pats)

    # ——— AUTO «о чём документ» ———
    def _about_summary_auto(self, chunks, tfidf_vec, tfidf_mat, max_terms: int = 6) -> Dict[str, Any]:
        texts = [c["text"] for c in chunks]
        # берём первые ~10% чанков (но не меньше 2 и не больше 5), там чаще всего преамбула/цели
        prefix_n = min(max(2, len(chunks)//10), 5)
        prefix_text = " ".join(texts[:prefix_n])
        sents = self._split_sentences(prefix_text)

        # динамическая фильтрация шума
        clean_sents = []
        for s in sents:
            if _is_noise_sentence(s, tfidf_vec): 
                continue
            s2 = _clean_text_for_answer(s)
            if s2:
                clean_sents.append(s2)
        # если после фильтра пусто — возьмём 1–2 длинных предложения из первого чанка
        if not clean_sents:
            first_s = [ss for ss in self._split_sentences(texts[0]) if len(ss) > 30 and not UPPER_LINE_RE.match(ss)]
            clean_sents = first_s[:2] if first_s else [texts[0][:200]]

        clean_text = " ".join(clean_sents)

        # ключевые фразы: KeyBERT → fallback TF-IDF
        topics: List[str] = []
        if _HAS_KEYBERT:
            try:
                kw_model = KeyBERT(self.model)
                kws = kw_model.extract_keywords(
                    clean_text,
                    keyphrase_ngram_range=(1,2),
                    stop_words='russian',
                    top_n=max(10, max_terms*2)
                )
                topics = [kw for kw, score in kws]
            except Exception:
                topics = []

        if not topics:
            # fallback: TF-IDF на базе уже обученного vectorizer'а
            try:
                qv = tfidf_vec.transform([clean_text])
                arr = qv.toarray()[0]
                inv_vocab = {i: t for t, i in tfidf_vec.vocabulary_.items()}
                top_idx = np.argsort(-arr)[:max(20, max_terms*3)]
                cands = []
                for j in top_idx:
                    term = inv_vocab.get(int(j))
                    if not term: continue
                    if len(term) < 3: continue
                    if term.lower() in STOP_RU or term.lower() in STOP_EN: continue
                    cands.append(term)
                # доменное ранжирование: приоритет биграммам и предметным корням
                cands.sort(key=lambda t: (_score_term_for_domain(t), " " in t, len(t)), reverse=True)
                topics = cands
            except Exception:
                topics = []

        # финальная сборка краткого ответа
        # найдём возможный тип документа (по первым чанкам)
        header = " ".join(texts[:2])
        doc_type = None
        for t in DOC_TYPES:
            if re.search(rf"\b{re.escape(t)}\b", header, flags=re.I):
                doc_type = t; break
        lead = f"Это {doc_type} компании" if doc_type else "Это внутренний документ компании"

        # чистим темы и ограничиваем
        cleaned, seen = [], set()
        for t in topics:
            t2 = _clean_text_for_answer(t)
            if not t2: continue
            base = t2.lower()
            if base in seen: continue
            seen.add(base)
            cleaned.append(t2)
        if cleaned:
            summary = f"{lead} о {', '.join(cleaned[:max_terms])}."
        else:
            # fallback: краткая первая «чистая» фраза
            s0 = _clean_text_for_answer(clean_sents[0]) if clean_sents else ""
            summary = f"{lead}: {s0}" if s0 else f"{lead}."

        # цитаты: заголовочный чанк + наиболее «тематический» по TF-IDF к summary
        citations = ["chunk_0000"] if chunks and chunks[0]["chunk_id"] == "chunk_0000" else [chunks[0]["chunk_id"]]
        try:
            qv_sum = tfidf_vec.transform([summary])
            sims = cosine_similarity(qv_sum, tfidf_mat)[0]
            best_i = int(np.argmax(sims))
            if chunks[best_i]["chunk_id"] not in citations:
                citations.append(chunks[best_i]["chunk_id"])
        except Exception:
            pass

        return {"answer": summary, "citations": citations}

    # ——— query expansion (для обычных вопросов) ———
    def _expand_query(self, query: str, tfidf_vec: TfidfVectorizer, tfidf_mat: csr_matrix,
                      texts: List[str], top_m_chunks: int = 8, add_terms: int = 6) -> str:
        q_vec = tfidf_vec.transform([query])
        tf_sims = cosine_similarity(q_vec, tfidf_mat)[0]
        idxs = np.argsort(-tf_sims)[:max(1, top_m_chunks)]
        mean_weights = None
        for i in idxs:
            row = tfidf_mat.getrow(i).toarray()[0]
            mean_weights = row if mean_weights is None else (mean_weights + row)
        mean_weights = mean_weights / max(1, len(idxs))
        vocab = tfidf_vec.vocabulary_; inv_vocab = {i: t for t, i in vocab.items()}
        top_idx = np.argsort(-mean_weights)[:add_terms * 3]
        query_lc = set(re.findall(r"[а-яА-Яa-zA-Z0-9-]{3,}", query.lower()))
        candidates = []
        for j in top_idx:
            term = inv_vocab.get(int(j))
            if not term: continue
            if any(w in query_lc for w in term.split()): continue
            if len(term) < 3: continue
            candidates.append(term)
        expanded_terms = candidates[:add_terms]
        return query if not expanded_terms else query + " " + " ".join(expanded_terms)

    # ——— экстрактивный ответ (MMR) ———
    def _extractive_answer(self, query: str, top_chunks: List[Dict[str, Any]],
                           max_sentences: int = 5, min_sim_threshold: float = 0.42) -> Dict[str, Any]:
        sentences, sent_meta = [], []
        for ch in top_chunks:
            for s in self._split_sentences(ch["text"]):
                s_raw = s.strip()
                if _is_noise_sentence(s_raw, tfidf_vec=None if True else None, global_idf_threshold=0):  # базовая очистка
                    pass  # ниже ещё раз почистим
                s_cln = _clean_text_for_answer(s_raw)
                if not s_cln:
                    continue
                if len(s_cln) < 20:
                    continue
                sentences.append(s_cln); sent_meta.append(ch["chunk_id"])
        if not sentences:
            return {"answer": "", "citations": []}

        q_vec = self.model.encode([query], normalize_embeddings=True)
        s_vecs = self.model.encode(sentences, batch_size=64, show_progress_bar=False, normalize_embeddings=True)
        sims = cosine_similarity(q_vec, s_vecs)[0]
        top_sim = float(np.max(sims)) if len(sims) else 0.0
        if top_sim < min_sim_threshold:
            fallback = []
            for s in sentences:
                if len(s) > 60:
                    fallback.append(s)
                if len(fallback) >= 2: break
            ans = " ".join(fallback)
            ans = re.sub(r"\s+", " ", ans).strip()
            parts = re.split(r"(?<=[\.\!\?])\s+", ans)
            return {"answer": " ".join(parts[:3]), "citations": list(sorted(set(sent_meta[:1])))}  # аккуратно

        picked_idx = self._mmr_select(s_vecs, q_vec[0], k=max_sentences, lambda_div=0.6)
        picked = [(float(sims[i]), sentences[i], sent_meta[i]) for i in picked_idx]
        picked.sort(key=lambda x: x[0], reverse=True)
        answer_text = " ".join(p[1] for p in picked)
        answer_text = re.sub(r"\s+", " ", answer_text).strip()
        parts = re.split(r"(?<=[\.\!\?])\s+", answer_text)
        answer_text = " ".join(parts[:3])
        citations = sorted(list({p[2] for p in picked}))
        return {"answer": answer_text, "citations": citations}

    # ——— основной запрос ———
    def ask(self, query: str, top_k: int = TOP_K) -> Dict[str, Any]:
        chunks, emb, tfidf_vec, tfidf_mat = self.store.load()
        texts = [c["text"] for c in chunks]

        # спец-режим: «о чём документ?»
        if self._is_about_query(query):
            about = self._about_summary_auto(chunks, tfidf_vec, tfidf_mat, max_terms=6)
            # подберём для прозрачности топ чанков по TF-IDF к резюме
            qv = tfidf_vec.transform([about["answer"]])
            sims = cosine_similarity(qv, tfidf_mat)[0]
            top_idx = np.argsort(-sims)[:top_k]
            top = [{"chunk_id": chunks[i]["chunk_id"], "similarity": float(sims[i]), "text": chunks[i]["text"]}
                   for i in top_idx]
            return {"query": query, "mode": "aboutness_auto", "top": top, **about}

        # обычный режим: Query Expansion + гибридный скоринг
        expanded_query = self._expand_query(query, tfidf_vec, tfidf_mat, texts)
        q_vec_emb = self.model.encode([expanded_query], normalize_embeddings=True)[0].reshape(1, -1)
        emb_sims = cosine_similarity(q_vec_emb, emb)[0]
        q_vec_tfidf = tfidf_vec.transform([expanded_query])
        tfidf_sims = cosine_similarity(q_vec_tfidf, tfidf_mat)[0]
        alpha = 0.6
        sims = alpha * emb_sims + (1.0 - alpha) * tfidf_sims

        top_idx = np.argsort(-sims)[:top_k]
        top = [{"chunk_id": chunks[i]["chunk_id"], "similarity": float(sims[i]), "text": chunks[i]["text"]}
               for i in top_idx]

        answer = self._extractive_answer(expanded_query, top)
        return {"query": query, "expanded_query": expanded_query, "top": top, **answer}

# ───────────────────────────────────────────────────────────────────────────────
def main():
    DEFAULT_FILE = r"C:/Users/SatyaTR/Desktop/1.docx"
    ap = argparse.ArgumentParser(description="RAG по одному .docx документу с авто-аннотацией тематики.")
    ap.add_argument("--file", default=DEFAULT_FILE, help="Путь к .docx файлу (по умолчанию используется DEFAULT_FILE)")
    ap.add_argument("--build", action="store_true", help="Построить/перестроить индекс")
    ap.add_argument("--ask", type=str, help="Вопрос пользователя")
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