from __future__ import annotations

import pytest

from agent_rag.config import get_settings


@pytest.fixture(autouse=True)
def _isolated_caches():
    """每个测试都把全局缓存清掉，避免上一个测试的 settings / 连接串进来。"""
    from agent_rag.agent.graph import get_graph
    from agent_rag.llm import get_chat_model
    from agent_rag.retrieval.embed import get_embeddings
    from agent_rag.retrieval.store import get_store

    for fn in (get_graph, get_chat_model, get_embeddings, get_store, get_settings):
        fn.cache_clear()
    yield
    for fn in (get_graph, get_chat_model, get_embeddings, get_store, get_settings):
        fn.cache_clear()


@pytest.fixture()
def tmp_settings(tmp_path, monkeypatch):
    """把存储全部指到临时目录，测试不碰项目里的 var/ 和 data/。"""
    monkeypatch.setenv("SQLITE_PATH", str(tmp_path / "agent.sqlite"))
    monkeypatch.setenv("CHROMA_DIR", str(tmp_path / "chroma"))
    monkeypatch.setenv("DATA_DIR", str(tmp_path / "data"))
    monkeypatch.setenv("DEEPSEEK_API_KEY", "")  # 不读真实 .env
    get_settings.cache_clear()

    settings = get_settings()
    settings.ensure_dirs()
    return settings
