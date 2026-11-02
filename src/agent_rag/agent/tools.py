"""工具层。三个硬约束，对应旧版 demo 的三个真实缺陷：

1. **身份来自请求上下文，不来自模型输出。**
   旧版 apply_for_leave(employee_name, ...) 让 LLM 决定给谁请假 —— 提示注入后
   谁都能给"张三"请假。这里 employee 从 RunnableConfig 里取，模型的 schema
   里根本没有这个参数，改不了。
2. **写操作幂等。**
   旧版人工确认前一停、确认后又 invoke 一次，年假被扣两次。这里先"占坑"再记账，
   同一个 (员工, 天数, 理由) 只会成功一次。
3. **失败返回可读文本，不抛裸异常。**
   tool 的异常会被序列化成给模型看的字符串，抛 traceback 等于把内部实现泄露进上下文。
"""

from __future__ import annotations

import hashlib
import logging
from datetime import UTC, datetime
from typing import Any

import httpx
from langchain_core.runnables import RunnableConfig
from langchain_core.tools import tool

from ..config import get_settings
from ..db import session
from ..retrieval.hybrid import HybridRetriever, format_context

log = logging.getLogger(__name__)

# 检索器在服务启动时注入，避免每次工具调用都重建 BM25 索引
_retriever: HybridRetriever | None = None


def set_retriever(retriever: HybridRetriever | None) -> None:
    global _retriever
    _retriever = retriever


def get_retriever() -> HybridRetriever:
    global _retriever
    if _retriever is None:
        _retriever = HybridRetriever()
    return _retriever


def _user_from_config(config: RunnableConfig | None) -> str | None:
    if not config:
        return None
    user = (config.get("configurable") or {}).get("user")
    return str(user) if user else None


def _now() -> str:
    return datetime.now(UTC).isoformat(timespec="seconds")


# --------------------------------------------------------------------------- #
# 1. 知识库检索
# --------------------------------------------------------------------------- #
@tool
def search_knowledge_base(query: str) -> str:
    """检索公司内部知识库，覆盖制度、流程、报销、人事、IT 支持等文档。

    凡是问题涉及公司内部信息，必须先调用本工具再回答。
    返回内容带有 [编号] 与出处，回答时请引用对应编号。
    """
    try:
        hits = get_retriever().retrieve(query)
    except Exception as exc:  # noqa: BLE001 - 工具边界必须兜住所有异常
        log.exception("知识库检索失败")
        return f"检索失败：{exc}。请告知用户稍后重试。"
    if not hits:
        return "知识库中没有检索到相关内容。请如实回答「没有查到」，不要编造。"
    return format_context(hits)


# --------------------------------------------------------------------------- #
# 2. 天气
# --------------------------------------------------------------------------- #
@tool
def get_weather(city: str) -> str:
    """查询指定城市的实时天气。仅用于天气类问题。"""
    settings = get_settings()
    api_key = settings.qweather_api_key.get_secret_value().strip()
    if not api_key or not settings.qweather_host:
        return "天气服务未配置（缺少 QWEATHER_API_KEY / QWEATHER_HOST），请告知用户无法查询。"

    clean_city = (
        city.strip().replace("市", "").replace("省", "").replace("区", "").replace("县", "")
    )
    headers = {"X-QW-Api-Key": api_key}

    try:
        with httpx.Client(timeout=10.0) as client:
            geo = client.get(
                f"https://{settings.qweather_host}/geo/v2/city/lookup",
                headers=headers,
                params={"location": clean_city, "number": 1},
            )
            geo.raise_for_status()
            geo_data = geo.json()
            locations = geo_data.get("location") or []
            if geo_data.get("code") != "200" or not locations:
                return f"找不到城市「{clean_city}」，请确认地名。"

            hit = locations[0]
            lat = float(hit["lat"])
            lon = float(hit["lon"])
            city_name = hit.get("name", clean_city)

            weather = client.get(
                f"https://{settings.qweather_host}/weather/v1/current/{lat:.2f}/{lon:.2f}",
                headers=headers,
            )
            weather.raise_for_status()
            data = weather.json()
    except httpx.HTTPError as exc:
        log.warning("天气接口调用失败: %s", exc)
        return f"天气接口暂时不可用（{type(exc).__name__}），请告知用户稍后重试。"

    if "condition" not in data:
        return f"{city_name} 的天气数据返回异常。"

    return (
        f"{city_name}当前{data['condition']['text']}，"
        f"气温 {data['temperature']['value']}℃，"
        f"体感 {data['feelsLike']['value']}℃，"
        f"湿度 {data['humidity'] * 100:.0f}%。"
    )


# --------------------------------------------------------------------------- #
# 3. 请假（写操作，敏感工具，需人工确认）
# --------------------------------------------------------------------------- #
@tool
def apply_for_leave(days: int, reason: str, config: RunnableConfig) -> str:
    """提交请假申请（会修改数据，需用户确认后执行）。

    Args:
        days: 请假天数，1-30 的整数。
        reason: 请假理由。
    """
    employee = _user_from_config(config)
    if not employee:
        return "无法确定当前用户身份，操作已拒绝。请提示用户先登录。"

    if not isinstance(days, int) or not 1 <= days <= 30:
        return "请假天数必须是 1 到 30 之间的整数。"

    reason = (reason or "").strip()
    if not reason:
        return "请假理由不能为空。"

    # 幂等键：同一员工 + 同一天数 + 同一理由，只允许成功一次
    key = hashlib.sha1(f"{employee}|{days}|{reason}".encode()).hexdigest()[:24]

    with session() as conn:
        # 先占坑（唯一主键），抢不到说明已经提交过
        claim = conn.execute(
            "INSERT OR IGNORE INTO tool_idempotency(key, result, created_at) VALUES (?, ?, ?)",
            (key, "pending", _now()),
        )
        if claim.rowcount == 0:
            row = conn.execute(
                "SELECT result FROM tool_idempotency WHERE key = ?", (key,)
            ).fetchone()
            previous = row["result"] if row else "已提交"
            return f"{previous}（重复请求，本次未重复扣减年假）"

        balance = conn.execute(
            "SELECT days_left FROM leave_balances WHERE employee = ?", (employee,)
        ).fetchone()
        if balance is None:
            # 占坑失败要回滚，否则下次同样的请求会被误判成"已提交过"
            conn.execute("DELETE FROM tool_idempotency WHERE key = ?", (key,))
            return f"系统中没有 {employee} 的年假记录，请联系 HR。"

        remaining = int(balance["days_left"]) - days
        if remaining < 0:
            conn.execute("DELETE FROM tool_idempotency WHERE key = ?", (key,))
            return f"{employee} 剩余年假 {balance['days_left']} 天，不足以请 {days} 天。"

        conn.execute(
            "UPDATE leave_balances SET days_left = ? WHERE employee = ?",
            (remaining, employee),
        )
        conn.execute(
            "INSERT INTO leave_requests(id, employee, days, reason, created_at) "
            "VALUES (?, ?, ?, ?, ?)",
            (hashlib.sha1(f"{key}|req".encode()).hexdigest()[:32], employee, days, reason, _now()),
        )
        result = f"已为 {employee} 提交 {days} 天请假（理由：{reason}），剩余年假 {remaining} 天。"
        conn.execute(
            "UPDATE tool_idempotency SET result = ? WHERE key = ?", (result, key)
        )

    log.info("请假写入成功: %s %s 天", employee, days)
    return result


ALL_TOOLS = [search_knowledge_base, get_weather, apply_for_leave]
SENSITIVE_TOOLS = {apply_for_leave.name}


def tool_names() -> list[str]:
    return [t.name for t in ALL_TOOLS]


def describe_sensitive_call(call: dict[str, Any]) -> str:
    args = call.get("args") or {}
    if call.get("name") == "apply_for_leave":
        return f"提交请假 {args.get('days')} 天（理由：{args.get('reason')}）"
    return str(call.get("name"))
