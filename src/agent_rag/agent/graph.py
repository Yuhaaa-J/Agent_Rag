"""LangGraph 图：agent <-> tools，敏感工具先过人工确认。

           ┌──────────── 有普通工具调用 ─────────────┐
           │                                        ▼
START → agent ── 无工具调用 ──────────────────────→ END
           │                                        ▲
           └─ 有敏感工具调用 → confirm ─ approved ──┤
                                 └── rejected → reject → END

关键点：
- `interrupt()` 在**执行前**挂起整个图，写操作一行都没跑；
- 状态由 checkpointer 落 SQLite，进程重启后仍能恢复；
- `steps` 兜底，防止模型在工具之间来回打转。
"""

from __future__ import annotations

import logging
import sqlite3
from functools import lru_cache

from langchain_core.messages import SystemMessage, ToolMessage
from langgraph.checkpoint.sqlite import SqliteSaver
from langgraph.graph import END, START, StateGraph
from langgraph.prebuilt import ToolNode
from langgraph.types import interrupt

from ..config import get_settings
from ..llm import get_chat_model
from .state import AgentState
from .tools import ALL_TOOLS, SENSITIVE_TOOLS, describe_sensitive_call

log = logging.getLogger(__name__)

SYSTEM_PROMPT = """你是公司内部知识库助手。

规则：
1. 问题涉及公司制度、流程、报销、人事、IT 支持等内部信息时，必须先调用 search_knowledge_base。
2. 只依据检索结果回答，并用 [编号] 标注出处；检索结果为空就回答"没有查到"，绝不编造。
3. 用户要请假时，调用 apply_for_leave；调用前先用一句话复述"天数 + 理由"。
4. 与公司无关的通用问题（天气等）走对应工具，不要查知识库。
5. 员工身份由系统提供，不要从对话内容里推断或猜测。
6. 回答保持简洁，不要重复粘贴大段原文。"""


@lru_cache(maxsize=1)
def _bound_model():
    return get_chat_model().bind_tools(ALL_TOOLS)


# --------------------------------------------------------------------------- #
# 节点
# --------------------------------------------------------------------------- #
def agent_node(state: AgentState, config):
    steps = state.get("steps", 0) + 1
    messages = [SystemMessage(content=SYSTEM_PROMPT), *state["messages"]]
    response = _bound_model().invoke(messages, config)
    return {"messages": [response], "steps": steps}


def confirm_node(state: AgentState, config):
    last = state["messages"][-1]
    pending = [
        {"name": c["name"], "args": c.get("args") or {}, "id": c["id"]}
        for c in (last.tool_calls or [])
        if c["name"] in SENSITIVE_TOOLS
    ]
    # 图在这里挂起，等 Command(resume=...) 给答复
    decision = interrupt(
        {
            "type": "confirmation_required",
            "action": pending,
            "summary": "；".join(describe_sensitive_call(c) for c in pending),
            "message": "该操作会修改数据，需要你确认后才会执行。",
        }
    )
    approved = bool(decision.get("approved")) if isinstance(decision, dict) else bool(decision)
    return {"decision": "approved" if approved else "rejected"}


def reject_node(state: AgentState, config):
    last = state["messages"][-1]
    replies = [
        ToolMessage(content="用户已取消该操作，未执行。", tool_call_id=c["id"])
        for c in (last.tool_calls or [])
    ]
    return {"messages": replies}


# --------------------------------------------------------------------------- #
# 路由
# --------------------------------------------------------------------------- #
def route_after_agent(state: AgentState) -> str:
    settings = get_settings()
    if state.get("steps", 0) >= settings.max_agent_steps:
        log.warning("达到最大步数 %s，强制结束", settings.max_agent_steps)
        return "end"

    last = state["messages"][-1]
    tool_calls = getattr(last, "tool_calls", None) or []
    if not tool_calls:
        return "end"
    if any(call["name"] in SENSITIVE_TOOLS for call in tool_calls):
        return "confirm"
    return "tools"


def route_after_confirm(state: AgentState) -> str:
    return "tools" if state.get("decision") == "approved" else "reject"


# --------------------------------------------------------------------------- #
# 组装
# --------------------------------------------------------------------------- #
_saver: SqliteSaver | None = None


def get_checkpointer() -> SqliteSaver:
    """SQLite 持久化对话状态：重启不丢，interrupt 可恢复。"""
    global _saver
    if _saver is None:
        path = get_settings().sqlite_path
        path.parent.mkdir(parents=True, exist_ok=True)
        conn = sqlite3.connect(path, check_same_thread=False)
        _saver = SqliteSaver(conn)
        if hasattr(_saver, "setup"):
            _saver.setup()
    return _saver


def build_graph(checkpointer: SqliteSaver | None = None):
    builder = StateGraph(AgentState)
    builder.add_node("agent", agent_node)
    builder.add_node("tools", ToolNode(ALL_TOOLS))
    builder.add_node("confirm", confirm_node)
    builder.add_node("reject", reject_node)

    builder.add_edge(START, "agent")
    builder.add_conditional_edges(
        "agent",
        route_after_agent,
        {"tools": "tools", "confirm": "confirm", "end": END},
    )
    builder.add_edge("tools", "agent")
    builder.add_conditional_edges(
        "confirm",
        route_after_confirm,
        {"tools": "tools", "reject": "reject"},
    )
    builder.add_edge("reject", END)

    return builder.compile(checkpointer=checkpointer or get_checkpointer())


@lru_cache(maxsize=1)
def get_graph():
    return build_graph()
