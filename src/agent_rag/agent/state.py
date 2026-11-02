"""图的状态定义。"""

from __future__ import annotations

from typing import Annotated, TypedDict

from langchain_core.messages import AnyMessage
from langgraph.graph.message import add_messages


class AgentState(TypedDict, total=False):
    """整个图共享的"背包"。

    messages: add_messages 会自动去重 + 追加，不要自己拼 list。
    steps:    已经走了几步，用来兜底防止工具调用死循环（旧版没有这个，会一直烧钱）。
    decision: 人工确认的结果，approve / reject。
    """

    messages: Annotated[list[AnyMessage], add_messages]
    steps: int
    decision: str | None
