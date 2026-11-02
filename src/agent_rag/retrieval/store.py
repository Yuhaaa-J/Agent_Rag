"""向量库封装（Chroma 持久化）。

只暴露 5 个方法，上层不关心 Chroma 的 API 怎么变。以后换 pgvector，
只要再实现一个同样接口的类。
"""

from __future__ import annotations

import logging
from collections.abc import Callable, Sequence
from functools import lru_cache
from pathlib import Path

import chromadb

from ..config import get_settings

log = logging.getLogger(__name__)

_BATCH = 128
Embedder = Callable[[list[str]], list[list[float]]]


class ChromaStore:
    def __init__(self, path: Path, collection_name: str) -> None:
        path.mkdir(parents=True, exist_ok=True)
        self._client = chromadb.PersistentClient(path=str(path))
        # cosine 距离：bge 系列都是归一化向量，用 cosine 而不是 l2
        self._collection = self._client.get_or_create_collection(
            name=collection_name,
            metadata={"hnsw:space": "cosine"},
        )

    # ---------------- 写 ----------------
    def add_chunks(
        self,
        ids: Sequence[str],
        texts: Sequence[str],
        metadatas: Sequence[dict],
        embedder: Embedder | None = None,
    ) -> int:
        if not ids:
            return 0
        if embedder is None:
            from .embed import embed_texts

            embedder = embed_texts

        vectors = embedder(list(texts))
        for start in range(0, len(ids), _BATCH):
            stop = start + _BATCH
            self._collection.upsert(
                ids=list(ids[start:stop]),
                documents=list(texts[start:stop]),
                embeddings=vectors[start:stop],
                metadatas=list(metadatas[start:stop]),
            )
        return len(ids)

    def delete_source(self, source: str) -> int:
        """按来源删除 —— 这是"重传同一个文件不会留下旧版本"的关键。"""
        existing = self._collection.get(where={"source": source}, include=["metadatas"])
        ids = existing.get("ids") or []
        if ids:
            self._collection.delete(ids=list(ids))
        return len(ids)

    # ---------------- 读 ----------------
    def count(self) -> int:
        return self._collection.count()

    def all_sources(self) -> set[str]:
        data = self._collection.get(include=["metadatas"])
        metadatas = data.get("metadatas") or []
        return {str(m.get("source")) for m in metadatas if m and m.get("source")}

    def all_chunks(self) -> tuple[list[str], list[str], list[dict]]:
        data = self._collection.get(include=["documents", "metadatas"])
        return (
            list(data.get("ids") or []),
            list(data.get("documents") or []),
            list(data.get("metadatas") or []),
        )

    def search(self, query_vector: list[float], k: int = 5) -> list[tuple[str, str, dict, float]]:
        if self.count() == 0:
            return []
        result = self._collection.query(
            query_embeddings=[query_vector],
            n_results=max(1, min(k, self.count())),
            include=["documents", "metadatas", "distances"],
        )
        ids = (result.get("ids") or [[]])[0]
        docs = (result.get("documents") or [[]])[0]
        metas = (result.get("metadatas") or [[]])[0]
        dists = (result.get("distances") or [[]])[0]
        return [
            (ids[i], docs[i], metas[i] or {}, float(dists[i]))
            for i in range(len(ids))
        ]


@lru_cache(maxsize=1)
def get_store() -> ChromaStore:
    settings = get_settings()
    return ChromaStore(settings.chroma_dir, settings.collection)
