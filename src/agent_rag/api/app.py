"""FastAPI 服务。

和旧版 demo 的三处关键差异：
1. 全链路 async，工具用 httpx 而不是 requests，不阻塞事件循环；
2. 流式用 LangGraph 的 stream_mode=["messages","updates"]，
   而不是从 astream_events 里捞 on_tool_start 去猜工具有没有跑；
3. 人工确认走 interrupt / Command(resume=...)，写操作在确认之前真的没执行。
"""

from __future__ import annotations

import json
import logging
import uuid
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI, File, Header, HTTPException, Query, UploadFile
from fastapi.responses import StreamingResponse
from langchain_core.messages import HumanMessage
from langgraph.types import Command

from ..agent.graph import get_graph
from ..agent.tools import set_retriever
from ..config import get_settings
from ..db import session
from ..ingest.loaders import SUPPORTED_SUFFIXES
from ..ingest.pipeline import delete_source, ingest_paths
from ..retrieval.hybrid import HybridRetriever
from .schemas import (
    ChatRequest,
    ConfirmRequest,
    ConfirmResponse,
    DocumentInfo,
    UploadResponse,
)

log = logging.getLogger(__name__)

MAX_UPLOAD_BYTES = 20 * 1024 * 1024


@asynccontextmanager
async def lifespan(app: FastAPI):
    settings = get_settings()
    settings.ensure_dirs()
    set_retriever(HybridRetriever())
    log.info("服务启动：chroma=%s sqlite=%s", settings.chroma_dir, settings.sqlite_path)
    yield
    log.info("服务关闭")


app = FastAPI(title="agent-rag", version="0.1.0", lifespan=lifespan)


def _sse(payload: dict) -> str:
    return f"data: {json.dumps(payload, ensure_ascii=False)}\n\n"


def _graph_config(thread_id: str, user: str) -> dict:
    return {"configurable": {"thread_id": thread_id, "user": user}}


def _text_of(chunk) -> str:
    content = getattr(chunk, "content", "")
    if isinstance(content, str):
        return content
    if isinstance(content, list):  # 某些模型返回结构化块
        return "".join(
            part.get("text", "") if isinstance(part, dict) else str(part) for part in content
        )
    return ""


# --------------------------------------------------------------------------- #
# 基础
# --------------------------------------------------------------------------- #
@app.get("/health")
async def health() -> dict:
    from ..retrieval.store import get_store

    return {"status": "ok", "chunks": get_store().count()}


@app.get("/documents", response_model=list[DocumentInfo])
async def list_documents() -> list[DocumentInfo]:
    with session() as conn:
        rows = conn.execute(
            "SELECT source, chunk_count, indexed_at FROM documents ORDER BY indexed_at DESC"
        ).fetchall()
    return [
        DocumentInfo(source=r["source"], chunks=r["chunk_count"], indexed_at=r["indexed_at"])
        for r in rows
    ]


@app.post("/documents", response_model=UploadResponse)
async def upload_document(file: UploadFile = File(...)) -> UploadResponse:
    settings = get_settings()
    original = Path(file.filename or "upload.bin").name  # 去掉用户传来的路径部分
    suffix = Path(original).suffix.lower()
    if suffix not in SUPPORTED_SUFFIXES:
        raise HTTPException(400, f"不支持的文件类型 {suffix}，仅支持 {sorted(SUPPORTED_SUFFIXES)}")

    settings.data_dir.mkdir(parents=True, exist_ok=True)
    target = settings.data_dir / original

    payload = await file.read()
    if len(payload) > MAX_UPLOAD_BYTES:
        raise HTTPException(413, "文件超过 20MB 限制")
    target.write_bytes(payload)

    stats = ingest_paths([target], base=settings.data_dir, prune=False)
    return UploadResponse(
        source=original,
        chunks=stats.chunks,
        message=f"{original} 已入库，{stats.chunks} 个 chunk",
    )


@app.delete("/documents")
async def remove_document(source: str = Query(..., description="入库时的 source")) -> dict:
    removed = delete_source(source)
    if removed == 0:
        raise HTTPException(404, f"未找到 {source}")
    return {"status": "ok", "removed_chunks": removed}


# --------------------------------------------------------------------------- #
# 对话
# --------------------------------------------------------------------------- #
@app.post("/chat")
async def chat(req: ChatRequest, x_user: str = Header(default="demo")):
    thread_id = req.thread_id or uuid.uuid4().hex
    config = _graph_config(thread_id, x_user)
    graph = get_graph()

    async def event_stream():
        yield _sse({"type": "start", "thread_id": thread_id})
        try:
            async for mode, payload in graph.astream(
                {"messages": [HumanMessage(content=req.message)]},
                config=config,
                stream_mode=["messages", "updates"],
            ):
                if mode == "messages":
                    chunk, meta = payload
                    if meta.get("langgraph_node") != "agent":
                        continue
                    text = _text_of(chunk)
                    if text:
                        yield _sse({"type": "token", "text": text})
                    continue

                # updates：拿 interrupt 和工具执行结果
                if "__interrupt__" in payload:
                    for item in payload["__interrupt__"]:
                        value = item.value or {}
                        yield _sse(
                            {
                                "type": "confirmation_required",
                                "thread_id": thread_id,
                                **value,
                            }
                        )
                    continue

                for node, update in payload.items():
                    if node == "tools" and isinstance(update, dict):
                        for msg in update.get("messages", []):
                            yield _sse(
                                {
                                    "type": "tool_result",
                                    "tool": getattr(msg, "name", ""),
                                    "preview": _text_of(msg)[:200],
                                }
                            )
        except Exception as exc:  # noqa: BLE001
            log.exception("对话失败")
            yield _sse({"type": "error", "message": str(exc)})
        finally:
            yield _sse({"type": "done"})

    return StreamingResponse(
        event_stream(),
        media_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
    )


@app.post("/confirm", response_model=ConfirmResponse)
async def confirm(req: ConfirmRequest, x_user: str = Header(default="demo")) -> ConfirmResponse:
    config = _graph_config(req.thread_id, x_user)
    graph = get_graph()
    try:
        state = await graph.ainvoke(Command(resume={"approved": req.approved}), config=config)
    except Exception as exc:  # noqa: BLE001
        log.exception("恢复执行失败")
        raise HTTPException(500, f"恢复执行失败：{exc}") from exc

    messages = state.get("messages") or []
    reply = _text_of(messages[-1]) if messages else ""
    return ConfirmResponse(status="ok", reply=reply or "已完成")
