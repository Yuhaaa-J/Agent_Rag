"""检索评测。

没有评测的 RAG 只能靠"感觉好像还行"。这里先用最朴素、零依赖的指标：
- source_recall@k     正确答案所在的文件，有没有出现在前 k 条里
- mrr                 正确文件的排名倒数的平均值（越靠近第 1 名越好）
- keyword_coverage    关键事实（金额、天数、电话）有没有出现在召回的上下文里

这三个指标不需要 LLM，跑一次几秒钟，可以每天在 CI 里跑。
答案正确率（需要 LLM 判分）留到后面再接。
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from pathlib import Path

import yaml

from .retrieval.hybrid import HybridRetriever

log = logging.getLogger(__name__)


@dataclass
class CaseResult:
    case_id: str
    question: str
    source_hit: bool
    reciprocal_rank: float
    keyword_hits: int
    keyword_total: int
    top_source: str = ""
    retrieved_sources: list[str] = field(default_factory=list)


def load_cases(path: Path | str) -> list[dict]:
    data = yaml.safe_load(Path(path).read_text(encoding="utf-8"))
    if not isinstance(data, list):
        raise ValueError("golden_set.yaml 必须是一个列表")
    return data


def _evaluate_case(case: dict, hits, k: int) -> CaseResult:
    sources = [hit.source for hit in hits]
    expected = set(case.get("expected_sources") or [])

    reciprocal_rank = 0.0
    for rank, source in enumerate(sources, start=1):
        if source in expected:
            reciprocal_rank = 1.0 / rank
            break

    context = "\n".join(hit.text for hit in hits)
    keywords = case.get("must_contain") or []
    keyword_hits = sum(1 for word in keywords if word in context)

    return CaseResult(
        case_id=str(case.get("id", "?")),
        question=str(case["question"]),
        source_hit=bool(expected & set(sources)),
        reciprocal_rank=reciprocal_rank,
        keyword_hits=keyword_hits,
        keyword_total=len(keywords),
        top_source=sources[0] if sources else "",
        retrieved_sources=sources,
    )


def evaluate(cases: list[dict], k: int = 5, retriever: HybridRetriever | None = None) -> dict:
    retriever = retriever or HybridRetriever()
    results: list[CaseResult] = []
    for case in cases:
        hits = retriever.retrieve(str(case["question"]), k=k)
        results.append(_evaluate_case(case, hits, k))

    total = len(results) or 1
    keyword_total = sum(r.keyword_total for r in results) or 1

    metrics = {
        "cases": len(results),
        "k": k,
        "source_recall": sum(r.source_hit for r in results) / total,
        "mrr": sum(r.reciprocal_rank for r in results) / total,
        "keyword_coverage": sum(r.keyword_hits for r in results) / keyword_total,
    }
    return {"metrics": metrics, "results": results}


def format_report(report: dict) -> str:
    lines = [
        f"{'id':<10}{'召回':<6}{'RR':<8}{'关键词':<8}top1 来源",
        "-" * 72,
    ]
    for row in report["results"]:
        lines.append(
            f"{row.case_id:<10}"
            f"{'✓' if row.source_hit else '✗':<6}"
            f"{row.reciprocal_rank:<8.2f}"
            f"{row.keyword_hits}/{row.keyword_total:<6}"
            f"{row.top_source or '-'}"
        )
    metrics = report["metrics"]
    lines.append("-" * 72)
    lines.append(
        f"k={metrics['k']}  cases={metrics['cases']}  "
        f"source_recall={metrics['source_recall']:.1%}  "
        f"mrr={metrics['mrr']:.3f}  "
        f"keyword_coverage={metrics['keyword_coverage']:.1%}"
    )
    return "\n".join(lines)
