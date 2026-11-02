"""人工确认的集成验证（需要真实 API key，所以不进单元测试）。

    uv run python scripts/verify_hitl.py

断言三件事：
1. interrupt 挂起时，年假余额没有变化 —— 说明写操作真的没执行；
2. 确认之后，余额只扣一次；
3. 同样的请求再发一遍，余额不再变化 —— 幂等生效。

知识库检索用测试桩（返回空），所以这个脚本不依赖向量库和 embedding 模型。
"""

from __future__ import annotations

import os
import shutil
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
VAR = ROOT / "var" / "hitl_check"

if VAR.exists():
    shutil.rmtree(VAR)
VAR.mkdir(parents=True)

# 必须在 import agent_rag.config 之前设置，get_settings 会缓存
os.environ["SQLITE_PATH"] = str(VAR / "agent.sqlite")
os.environ["CHROMA_DIR"] = str(VAR / "chroma")

from langchain_core.messages import HumanMessage  # noqa: E402
from langgraph.types import Command  # noqa: E402

from agent_rag.agent import tools as tools_mod  # noqa: E402
from agent_rag.agent.graph import build_graph  # noqa: E402
from agent_rag.db import seed_demo_balances, session  # noqa: E402


class _EmptyRetriever:
    """检索桩：不碰向量库，让模型只能走请假流程。"""

    @staticmethod
    def retrieve(_query: str, _k: int | None = None):
        return []


def balance(employee: str = "张三") -> int:
    with session() as conn:
        row = conn.execute(
            "SELECT days_left FROM leave_balances WHERE employee = ?", (employee,)
        ).fetchone()
    if row is None:
        raise AssertionError(f"没有 {employee} 的年假记录")
    return int(row["days_left"])


def main() -> int:
    tools_mod.set_retriever(_EmptyRetriever())  # type: ignore[arg-type]
    seed_demo_balances()

    start = balance()
    print(f"[1/4] 初始年假余额: {start} 天")
    if start < 2:
        print("❌ 演示数据不足，无法验证")
        return 1

    graph = build_graph()
    config = {"configurable": {"thread_id": "verify-hitl", "user": "张三"}}
    question = HumanMessage(content="帮我请 2 天病假，理由是去医院复查")

    print("[2/4] 发起请假请求……")
    state = graph.invoke({"messages": [question]}, config)
    interrupts = state.get("__interrupt__")
    if not interrupts:
        print("❌ 模型没有触发 apply_for_leave，本次无法验证（换个措辞再试）")
        print("   模型最后回复:", state["messages"][-1].content[:120])
        return 1

    payload = interrupts[0].value or {}
    print("      已挂起，待确认:", payload.get("summary"))

    pending = balance()
    if pending != start:
        print(f"❌ 挂起时余额变成 {pending}，说明工具在确认前就被执行了")
        return 1
    print(f"✅ 挂起时余额仍是 {pending} 天 —— 写操作确实没有执行")

    print("[3/4] 确认执行……")
    state = graph.invoke(Command(resume={"approved": True}), config)
    after = balance()
    if after != start - 2:
        print(f"❌ 确认后余额是 {after}，期望 {start - 2}")
        return 1
    print(f"✅ 确认后余额 {after} 天，且只扣了一次")
    print("      模型回复:", state["messages"][-1].content[:120])

    print("[4/4] 再发一次同样的请求……")
    state = graph.invoke({"messages": [question]}, config)
    while state.get("__interrupt__"):
        state = graph.invoke(Command(resume={"approved": True}), config)
    final = balance()
    if final != after:
        print(f"❌ 重复请求导致余额从 {after} 变成 {final}，幂等失效")
        return 1
    print(f"✅ 余额仍是 {final} 天，重复请求没有重复扣减")

    shutil.rmtree(VAR, ignore_errors=True)
    print("\n全部通过。")
    return 0


if __name__ == "__main__":
    sys.exit(main())
