# -*- coding: utf-8 -*-
"""
MVP: поиск дашбордов по эмбеддингу запроса
-----------------------------------------
Зависимости:
  pip install clickhouse-connect sentence-transformers numpy

Переменные окружения:
  CLICKHOUSE_HOST, CLICKHOUSE_PORT=8123, CLICKHOUSE_USERNAME, CLICKHOUSE_PASSWORD, CLICKHOUSE_DATABASE (опц.)
  CATALOG_TABLE=grushko_iv.dashboard_catalog (по умолч.)
  EMBED_MODEL=sentence-transformers/all-MiniLM-L6-v2 (по умолч.)

Примеры:
  python dashbot_mvp.py "где посмотреть продажи" --k 5
  python dashbot_mvp.py "остатки по бренду" --dump-all-distances
  python dashbot_mvp.py "продажи эргоферон сзфо" --require "продаж,выбыт" --require-mode any
"""

import os
import sys
import argparse
import numpy as np
from typing import List, Dict, Any, Tuple

import clickhouse_connect
from sentence_transformers import SentenceTransformer


# ───────────────────────────────────────────────────────────────────────────────
# Конфиг
# ───────────────────────────────────────────────────────────────────────────────
CLICKHOUSE_HOST = os.getenv("CLICKHOUSE_HOST", "clickhouse.moscow")
CLICKHOUSE_PORT = int(os.getenv("CLICKHOUSE_PORT", "8123"))
CLICKHOUSE_USERNAME = os.getenv("CLICKHOUSE_USERNAME", "GrushkoIV")
CLICKHOUSE_PASSWORD = os.getenv("CLICKHOUSE_PASSWORD", "jNbrvzd1IcF0Yx5I")
CLICKHOUSE_DATABASE = os.getenv("CLICKHOUSE_DATABASE", "")
CATALOG_TABLE = os.getenv("CATALOG_TABLE", "grushko_iv.dashboard_catalog")
EMBED_MODEL_NAME = os.getenv("EMBED_MODEL", "sentence-transformers/all-MiniLM-L6-v2")


# ───────────────────────────────────────────────────────────────────────────────
# Утилиты
# ───────────────────────────────────────────────────────────────────────────────
def norm_vec(x: np.ndarray) -> np.ndarray:
    """L2-нормализация вектора/матрицы по строкам."""
    if x.ndim == 1:
        n = np.linalg.norm(x) + 1e-12
        return x / n
    n = np.linalg.norm(x, axis=1, keepdims=True) + 1e-12
    return x / n


def to_text(obj) -> str:
    if obj is None:
        return ""
    if isinstance(obj, (list, tuple)):
        return " ".join([str(x) for x in obj if x is not None])
    return str(obj)


def build_text_index(row: Dict[str, Any]) -> str:
    """Собираем удобный текстовый индекс для быстрых keyword-проверок/подсветки (не для эмбеддинга)."""
    parts = [
        row.get("title", ""),
        row.get("description", ""),
        to_text(row.get("business_questions")),
        to_text(row.get("metrics")),
        to_text(row.get("dimensions")),
        to_text(row.get("tags")),
        to_text(row.get("system")),
        to_text(row.get("data_sources")),
        to_text(row.get("region_scope")),
    ]
    return " ".join(p for p in parts if p).lower()


def parse_require_list(s: str) -> List[str]:
    return [t.strip().lower() for t in s.split(",") if t.strip()] if s else []


# ───────────────────────────────────────────────────────────────────────────────
# Доступ к ClickHouse
# ───────────────────────────────────────────────────────────────────────────────
def ch_client():
    return clickhouse_connect.get_client(
        host=CLICKHOUSE_HOST,
        port=CLICKHOUSE_PORT,
        username=CLICKHOUSE_USERNAME,
        password=CLICKHOUSE_PASSWORD,
        database=CLICKHOUSE_DATABASE or None,
    )


def load_catalog() -> List[Dict[str, Any]]:
    """Загружаем все карточки дашбордов, включая эмбеддинги."""
    client = ch_client()
    # Поля подстраивайте под свою таблицу; лишние будут просто пустыми
    sql = f"""
    SELECT
      id,
      system,
      url,
      title,
      description,
      business_questions,
      metrics,
      dimensions,
      tags,
      data_sources,
      region_scope,
      embedding
    FROM {CATALOG_TABLE}
    """
    rows = client.query(sql).named_results()
    catalog = []
    for r in rows:
        item = dict(r)
        # Приводим embed к numpy
        emb = item.get("embedding")
        if emb is None:
            # пропускаем карточки без эмбеддинга
            continue
        item["embedding"] = np.array(emb, dtype=np.float32)
        item["text_index"] = build_text_index(item)
        catalog.append(item)
    return catalog


# ───────────────────────────────────────────────────────────────────────────────
# Поиск
# ───────────────────────────────────────────────────────────────────────────────
class DashSearch:
    def __init__(self, items: List[Dict[str, Any]], model_name: str):
        if not items:
            raise RuntimeError("Каталог пуст или не удалось загрузить записи с эмбеддингами.")
        self.items = items
        self.model = SentenceTransformer(model_name)
        # Подготовим матрицу эмбеддингов (нормализованную) для быстрого поиска
        self.emb_matrix = np.vstack([it["embedding"] for it in self.items])
        self.emb_matrix = norm_vec(self.emb_matrix)

    def query_embed(self, q: str) -> np.ndarray:
        vec = self.model.encode([q], normalize_embeddings=True)[0]  # уже нормализован
        return vec.astype(np.float32)

    def search(
        self,
        q: str,
        top_k: int = 10,
        require: List[str] = None,
        require_mode: str = "any",
        dump_all_distances: bool = False,
    ) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
        """
        Возвращает:
          top_results: список top_k с полями + explain
          all_distances: если dump_all_distances=True — список всех с distance (иначе пустой)
        """
        require = require or []
        q_vec = self.query_embed(q)  # (d,)
        # косинусная similarity = матричное произведение (уже всё нормировано)
        sims = self.emb_matrix @ q_vec  # (N,)
        # distance = 1 - similarity
        dists = 1.0 - sims

        # Фильтрация по обязательным словам/фразам (по text_index)
        mask = np.ones(len(self.items), dtype=bool)
        if require:
            ti = [it["text_index"] for it in self.items]
            if require_mode == "all":
                mask = np.array([all(tok in ti[i] for tok in require) for i in range(len(self.items))])
            else:  # any
                mask = np.array([any(tok in ti[i] for tok in require) for i in range(len(self.items))])

        # Индексация отфильтрованных
        idxs = np.where(mask)[0]
        if len(idxs) == 0:
            return [], (
                [{"id": it.get("id"), "title": it.get("title"), "cosine_distance": float(dists[i])}
                 for i, it in enumerate(self.items)]
                if dump_all_distances else []
            )

        # Сортировка по расстоянию
        order = idxs[np.argsort(dists[idxs])]
        top = order[:top_k]

        def explain(i: int) -> Dict[str, Any]:
            it = self.items[i]
            q_tokens = [t for t in q.lower().split() if len(t) > 2]
            metrics = [m.lower() for m in (it.get("metrics") or [])]
            dims = [d.lower() for d in (it.get("dimensions") or [])]
            tags = [t.lower() for t in (it.get("tags") or [])]

            matched_metrics = [m for m in metrics for qt in q_tokens if qt in m]
            matched_dims = [d for d in dims for qt in q_tokens if qt in d]
            matched_tags = [t for t in tags for qt in q_tokens if qt in t]

            return {
                "cosine_similarity": float(sims[i]),
                "cosine_distance": float(dists[i]),
                "matched_metrics": sorted(set(matched_metrics))[:5],
                "matched_dimensions": sorted(set(matched_dims))[:5],
                "matched_tags": sorted(set(matched_tags))[:5],
            }

        top_results = []
        for i in top:
            it = self.items[i]
            top_results.append({
                "id": it.get("id"),
                "title": it.get("title"),
                "url": it.get("url"),
                "system": it.get("system"),
                "score": float(sims[i]),
                "explain": explain(i),
            })

        all_distances = []
        if dump_all_distances:
            all_distances = [
                {
                    "id": it.get("id"),
                    "title": it.get("title"),
                    "cosine_distance": float(dists[i]),
                    "cosine_similarity": float(sims[i]),
                }
                for i, it in sorted(enumerate(self.items), key=lambda x: dists[x[0]])
            ]

        return top_results, all_distances


# ───────────────────────────────────────────────────────────────────────────────
# CLI
# ───────────────────────────────────────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser(description="Поиск дашбордов по эмбеддингу запроса (MVP).")
    parser.add_argument("query", nargs="*", help="Текст запроса (например: где посмотреть продажи)")
    parser.add_argument("--k", type=int, default=10, help="Сколько результатов вернуть (top-K)")
    parser.add_argument("--require", type=str, default="", help="Обязательные слова/фразы через запятую (any/all)")
    parser.add_argument("--require-mode", type=str, choices=["any", "all"], default="any", help="Логика обязательных слов")
    parser.add_argument("--dump-all-distances", action="store_true", help="Вывести косинусные расстояния до всех дашбордов")

    args = parser.parse_args()
    query = " ".join(args.query).strip()
    if not query:
        try:
            query = input("Введите ваш запрос: ").strip()
        except KeyboardInterrupt:
            sys.exit(0)

    # 1) Грузим каталог
    print("Загрузка каталога из ClickHouse…")
    items = load_catalog()
    if not items:
        print("Каталог пуст или нет полей embed. Проверьте таблицу и данные.")
        sys.exit(1)

    # 2) Инициализируем поиск
    print(f"Загрузка модели эмбеддингов: {EMBED_MODEL_NAME}")
    searcher = DashSearch(items, EMBED_MODEL_NAME)

    # 3) Поиск
    require_list = parse_require_list(args.require)
    top_results, all_distances = searcher.search(
        q=query,
        top_k=args.k,
        require=require_list,
        require_mode=args.require_mode,
        dump_all_distances=args.dump_all_distances
    )

    # 4) Вывод
    print("\nЗапрос:", query)
    if require_list:
        print(f"Фильтр обязательных слов ({args.require_mode}): {require_list}")

    if not top_results:
        print("Ничего не найдено по текущим условиям.")
    else:
        print(f"\nTop-{len(top_results)} результатов:")
        for i, r in enumerate(top_results, 1):
            e = r["explain"]
            print(f"\n{i}) {r['title']}  [{r.get('system')}]")
            print(f"   URL: {r['url']}")
            print(f"   cosine_similarity: {e['cosine_similarity']:.4f} | cosine_distance: {e['cosine_distance']:.4f}")
            mm = ", ".join(e["matched_metrics"]) or "—"
            md = ", ".join(e["matched_dimensions"]) or "—"
            mt = ", ".join(e["matched_tags"]) or "—"
            print(f"   matched: metrics[{mm}] dims[{md}] tags[{mt}]")

    if args.dump_all_distances:
        print("\nВсе дашборды по возрастанию косинусного расстояния (distance = 1 - similarity):")
        for r in all_distances:
            print(f" - {r['title']}: distance={r['cosine_distance']:.6f} | similarity={r['cosine_similarity']:.6f}")


if __name__ == "__main__":
    main()