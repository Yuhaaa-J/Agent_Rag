from __future__ import annotations

import pytest

from agent_rag.config import get_settings


def test_defaults_are_anchored_to_project_root(tmp_settings):
    assert tmp_settings.chunk_size > 0
    assert tmp_settings.top_k >= 1
    # 相对路径必须被解析成绝对路径，否则换个工作目录就找不到库了
    assert tmp_settings.sqlite_path.is_absolute()
    assert tmp_settings.chroma_dir.is_absolute()


def test_missing_api_key_raises_only_when_used(tmp_settings):
    # 关键设计：import 阶段不校验密钥，否则单元测试和 CI 全部跑不了
    with pytest.raises(RuntimeError, match="DEEPSEEK_API_KEY"):
        tmp_settings.require_deepseek_key()


def test_env_overrides_defaults(monkeypatch, tmp_path):
    monkeypatch.setenv("TOP_K", "9")
    monkeypatch.setenv("USE_BM25", "false")
    get_settings.cache_clear()
    settings = get_settings()
    assert settings.top_k == 9
    assert settings.use_bm25 is False
