from __future__ import annotations

from langchain_core.messages import AIMessage

from agent_rag.agent.graph import route_after_agent


def _ai(tool_name: str | None = None) -> AIMessage:
    if tool_name is None:
        return AIMessage(content="直接回答")
    return AIMessage(
        content="",
        tool_calls=[{"name": tool_name, "args": {}, "id": f"call-{tool_name}"}],
    )


def test_plain_answer_ends(tmp_settings):
    assert route_after_agent({"messages": [_ai()], "steps": 1}) == "end"


def test_readonly_tool_goes_straight_to_tools(tmp_settings):
    assert route_after_agent({"messages": [_ai("get_weather")], "steps": 1}) == "tools"


def test_sensitive_tool_requires_confirmation(tmp_settings):
    """写操作必须先过 confirm 节点，不能直接进 tools。"""
    assert route_after_agent({"messages": [_ai("apply_for_leave")], "steps": 1}) == "confirm"


def test_max_steps_breaks_the_loop(tmp_settings):
    """旧版没有步数上限，模型在工具间打转就会一直烧钱。"""
    assert route_after_agent({"messages": [_ai("get_weather")], "steps": 99}) == "end"
