from __future__ import annotations

from agent_rag.agent.tools import apply_for_leave
from agent_rag.db import seed_demo_balances


def _call(days: int, reason: str, user: str | None = "张三") -> str:
    configurable = {"user": user} if user else {}
    return apply_for_leave.invoke({"days": days, "reason": reason}, {"configurable": configurable})


def test_refuses_without_identity(tmp_settings):
    """员工身份不能由模型决定，拿不到身份就必须拒绝。"""
    result = _call(1, "事假", user=None)
    assert "身份" in result
    assert "拒绝" in result


def test_deducts_balance_and_is_idempotent(tmp_settings):
    seed_demo_balances()

    first = _call(2, "看病")
    assert "剩余年假 3 天" in first

    # 同一个人、同一天数、同一理由再调一次：不能重复扣
    second = _call(2, "看病")
    assert "重复" in second
    assert "剩余年假 3 天" in second


def test_insufficient_balance_does_not_poison_idempotency(tmp_settings):
    seed_demo_balances()

    assert "不足" in _call(9, "长假")
    # 余额不足属于业务拒绝，不能被记成"已提交过"
    assert "重复" not in _call(9, "长假")


def test_rejects_invalid_arguments(tmp_settings):
    seed_demo_balances()
    assert "1 到 30" in _call(0, "事假")
    assert "理由" in _call(1, "   ")
