"""命令行入口：uv run agent-rag --help"""

from __future__ import annotations

import logging
import uuid
from pathlib import Path

import typer
from langchain_core.messages import HumanMessage
from langgraph.types import Command

from .agent.graph import get_graph
from .agent.tools import set_retriever
from .config import get_settings
from .db import seed_demo_balances, session
from .evaluation import evaluate, format_report, load_cases
from .ingest.pipeline import delete_source, ingest_paths
from .logging_setup import setup_logging
from .retrieval.hybrid import HybridRetriever

log = logging.getLogger(__name__)

app = typer.Typer(add_completion=False, help="agent-rag：知识库智能体命令行")


@app.callback()
def _main() -> None:
    setup_logging(get_settings().log_level)


# --------------------------------------------------------------------------- #
@app.command()
def ingest(
    directory: Path = typer.Argument(Path("data/sample"), help="要入库的目录或单个文件"),
    reset: bool = typer.Option(False, "--reset", help="忽略 hash，全部重新向量化"),
    seed: bool = typer.Option(True, help="同时写入演示用的年假余额"),
) -> None:
    """把文档灌进向量库（幂等，重复跑不会重复写）。"""
    settings = get_settings()
    settings.ensure_dirs()
    if seed:
        seed_demo_balances()

    base = directory if directory.is_dir() else directory.parent
    stats = ingest_paths([directory], base=base, reset=reset)
    typer.echo(f"✅ {stats.summary()}")


@app.command()
def documents() -> None:
    """列出已经入库的文档。"""
    with session() as conn:
        rows = conn.execute(
            "SELECT source, chunk_count, indexed_at FROM documents ORDER BY source"
        ).fetchall()
    if not rows:
        typer.echo("（还没有任何文档，先跑 agent-rag ingest data/raw）")
        return
    for row in rows:
        typer.echo(f"{row['source']:<40} {row['chunk_count']:>4} chunks  {row['indexed_at']}")


@app.command()
def remove(source: str) -> None:
    """按 source 删除一个文档及其向量。"""
    removed = delete_source(source)
    typer.echo(f"删除 {source}：{removed} 个 chunk")


@app.command()
def ask(
    question: str = typer.Argument(..., help="你的问题"),
    user: str = typer.Option("demo", "--user", help="当前登录用户，工具用它鉴权"),
    thread: str = typer.Option(None, "--thread", help="会话 id，留空则新建"),
) -> None:
    """单轮提问（含人工确认流程）。"""
    set_retriever(HybridRetriever())
    thread_id = thread or uuid.uuid4().hex
    config = {"configurable": {"thread_id": thread_id, "user": user}}
    graph = get_graph()

    state = graph.invoke({"messages": [HumanMessage(content=question)]}, config)

    while interrupts := state.get("__interrupt__"):
        payload = interrupts[0].value or {}
        typer.secho(f"\n⚠️  {payload.get('message', '需要确认')}", fg="yellow")
        typer.echo(f"   待执行：{payload.get('summary', '')}")
        approved = typer.confirm("确认执行？", default=False)
        state = graph.invoke(Command(resume={"approved": approved}), config)

    messages = state.get("messages") or []
    if messages:
        typer.echo(f"\n🤖 {messages[-1].content}")


@app.command("eval")
def eval_cmd(
    golden: Path = typer.Option(Path("evals/golden_set.yaml")),
    k: int = typer.Option(5, "-k"),
    min_recall: float = typer.Option(0.0, help="低于该值退出码为 1，可用于 CI 卡口"),
) -> None:
    """跑检索评测。"""
    set_retriever(HybridRetriever())
    cases = load_cases(golden)
    report = evaluate(cases, k=k)
    typer.echo(format_report(report))
    if report["metrics"]["source_recall"] < min_recall:
        raise typer.Exit(code=1)


@app.command()
def serve(
    host: str = typer.Option(None, help="默认取 .env 的 API_HOST"),
    port: int = typer.Option(None, help="默认取 .env 的 API_PORT"),
    reload: bool = typer.Option(False, "--reload"),
) -> None:
    """启动 HTTP 服务。"""
    import uvicorn

    settings = get_settings()
    uvicorn.run(
        "agent_rag.api.app:app",
        host=host or settings.api_host,
        port=port or settings.api_port,
        reload=reload,
        log_level=settings.log_level.lower(),
    )


if __name__ == "__main__":
    app()
