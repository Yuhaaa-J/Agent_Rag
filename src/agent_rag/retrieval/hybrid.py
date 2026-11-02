"""混合检索：向量召回 + BM25 召回，用 RRF 融合。

为什么必须加 BM25：
    纯向量检索对"专有名词 / 编号 / 人名"很不敏感。公司制度里的
    「差旅标准 B 类」「工号 10086」「张伟」这类词，字面匹配经常比语义匹配准。
    而中文又不能用空格分词，所以这里写了一个 CJK 字符 + 二元组的分词器。

RRF（Reciprocal Rank Fusion）：
    不比较两路分数的绝对大小（量纲不同，没法比），只看排名：
    score = Σ 1 / (rrf_k + rank)。工业界最常用的无参数融合法。
"""

from __future__ import annotations

import logging
import re
from dataclasses import dataclass, field

from ..config import get_settings
from .store import ChromaStore, get_store

log = logging.getLogger(__name__)

_LATIN_RE = re.compile(r"[a-z0-9_]+")
_CJK_RE = re.compile(r"[\u4e00-\u9fff]")


def tokenize(text: str) -> list[str]:
    """中文没有空格，用 单字 + 相邻二元组 近似分词，够 BM25 用了。"""
    lowered = text.lower()
    latin = _LATIN_RE.findall(lowered)
    cjk = _CJK_RE.findall(lowered)
    bigrams = [f"{cjk[i]}{cjk[i + 1]}" for i in range(len(cjk) - 1)]
    return latin + cjk + bigrams


@dataclass(slots=True)
class Hit:
    chunk_id: str
    text: str
    metadata: dict = field(default_factory=dict)
    score: float = 0.0
    matched_by: tuple[str, ...] = ()

    @property
    def source(self) -> str:
        return str(self.metadata.get("source", "?"))

    @property
    def heading(self) -> str:
        return str(self.metadata.get("heading", ""))

    def citation(self) -> str:
        page = self.metadata.get("page")
        parts = [self.source]
        if self.heading:
            parts.append(self.heading)
        if page:
            parts.append(f"p.{page}")
        return " / ".join(parts)


class HybridRetriever:
    def __init__(self, store: ChromaStore | None = None) -> None:
        self._store = store or get_store()
        self._bm25 = None
        self._bm25_ids: list[str] = []
        self._bm25_payload: list[tuple[str, dict]] = []
        self._indexed_count = -1

    # ---------------- BM25 索引 ----------------
    def _ensure_bm25(self) -> None:
        from rank_bm25 import BM25Okapi

        count = self._store.count()
        if self._indexed_count == count and self._bm25 is not None:
            return

        ids, texts, metas = self._store.all_chunks()
        self._bm25_ids = ids
        self._bm25_payload = [
            (texts[i], metas[i] or {}) for i in range(len(ids))
        ]
        self._bm25 = BM25Okapi([tokenize(t) for t in texts]) if texts else None
        self._indexed_count = count
        log.debug("BM25 索引已重建: %d chunks", len(ids))

    # ---------------- 检索 ----------------
    def retrieve(self, query: str, k: int | None = None) -> list[Hit]:
        settings = get_settings()
        k = k or settings.top_k
        rrf_k = settings.rrf_k
        use_bm25 = settings.use_bm25

        from .embed import embed_query

        vector_hits: list[tuple[str, str, dict]] = []
        for chunk_id, text, meta, _distance in self._store.search(embed_query(query), k=k * 2):
            vector_hits.append((chunk_id, text, meta))

        bm25_hits: list[tuple[str, str, dict]] = []
        if use_bm25:
            self._ensure_bm25()
            if self._bm25 is not None:
                scores = self._bm25.get_scores(tokenize(query))
                order = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)
                for i in order[: k * 2]:
                    if scores[i] <= 0:
                        break
                    chunk_id = self._bm25_ids[i]
                    text, meta = self._bm25_payload[i]
                    bm25_hits.append((chunk_id, text, meta))

        fused: dict[str, Hit] = {}
        for label, hits in (("vector", vector_hits), ("bm25", bm25_hits)):
            for rank, (chunk_id, text, meta) in enumerate(hits, start=1):
                hit = fused.get(chunk_id)
                if hit is None:
                    hit = Hit(chunk_id=chunk_id, text=text, metadata=meta)
                    fused[chunk_id] = hit
                hit.score += 1.0 / (rrf_k + rank)
                hit.matched_by = (*hit.matched_by, label)

        ranked = sorted(fused.values(), key=lambda h: h.score, reverse=True)
        return ranked[:k]


def format_context(hits: list[Hit]) -> str:
    """给 LLM 的上下文：每段都带出处，让它能引用而不用编。"""
    if not hits:
        return "（知识库中没有检索到相关内容）"
    blocks = []
    for i, hit in enumerate(hits, start=1):
        blocks.append(f"[{i}] 出处: {hit.citation()}\n{hit.text}")
    return "\n\n".join(blocks)
