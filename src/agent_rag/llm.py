"""对话模型工厂。延迟构造，import 时不校验密钥、不发网络请求。"""

from __future__ import annotations

from functools import lru_cache

from langchain_deepseek import ChatDeepSeek

from .config import get_settings


@lru_cache(maxsize=4)
def get_chat_model(temperature: float | None = None) -> ChatDeepSeek:
    settings = get_settings()
    return ChatDeepSeek(
        model=settings.deepseek_model,
        api_key=settings.require_deepseek_key(),
        temperature=settings.llm_temperature if temperature is None else temperature,
        timeout=60,
        max_retries=2,
    )
