"""全部配置的唯一来源。代码里任何地方都不允许出现硬编码密钥。

优先级：环境变量 > .env 文件 > 这里的默认值。
"""

from __future__ import annotations

import logging
import os
from functools import lru_cache
from pathlib import Path

from pydantic import SecretStr, field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict

# src/agent_rag/config.py -> 向上三层是项目根目录
PROJECT_ROOT = Path(__file__).resolve().parents[2]

log = logging.getLogger(__name__)


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=(PROJECT_ROOT / ".env", ".env"),
        env_file_encoding="utf-8",
        extra="ignore",
        case_sensitive=False,
    )

    # --- LLM ---
    deepseek_api_key: SecretStr = SecretStr("")
    deepseek_model: str = "deepseek-chat"
    llm_temperature: float = 0.0

    # --- 天气工具（可选）---
    qweather_api_key: SecretStr = SecretStr("")
    qweather_host: str = ""

    # --- 向量化 ---
    embed_model: str = "BAAI/bge-small-zh-v1.5"
    hf_endpoint: str = "https://hf-mirror.com"
    hf_home: str | None = None

    # --- 切分与检索 ---
    chunk_size: int = 500
    chunk_overlap: int = 80
    top_k: int = 5
    use_bm25: bool = True
    rrf_k: int = 60

    # --- Agent ---
    max_agent_steps: int = 8

    # --- 存储 ---
    sqlite_path: Path = Path("var/agent.sqlite")
    chroma_dir: Path = Path("var/chroma")
    collection: str = "knowledge_base"
    data_dir: Path = Path("data/raw")
    temp_dir: Path = Path("var/tmp")

    # --- 服务 ---
    api_host: str = "127.0.0.1"
    api_port: int = 8000
    log_level: str = "INFO"

    @field_validator("sqlite_path", "chroma_dir", "data_dir", "temp_dir", mode="after")
    @classmethod
    def _anchor_to_project_root(cls, value: Path) -> Path:
        return value if value.is_absolute() else (PROJECT_ROOT / value)

    def require_deepseek_key(self) -> str:
        """只有真正要调模型时才校验，保证 import 和单元测试不需要密钥。"""
        key = self.deepseek_api_key.get_secret_value().strip()
        if not key or key.startswith("sk-xxx"):
            raise RuntimeError(
                "DEEPSEEK_API_KEY 未配置。复制 .env.example 为 .env 并填入真实 key。"
            )
        return key

    def ensure_dirs(self) -> None:
        for path in (self.chroma_dir, self.sqlite_path.parent, self.data_dir, self.temp_dir):
            path.mkdir(parents=True, exist_ok=True)


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    return Settings()


def ensure_writable_temp() -> Path | None:
    """让本进程有一个确定可写的临时目录，绕开系统 %TEMP% 的坑。

    背景：Chroma 1.x 的 Rust 层用 SQLite，SQLite 要建临时文件。系统临时目录一旦
    处于病态（权限被改乱、被安全软件拦住），报错只有一句
    「unable to open database file (code: 14)」，从字面完全看不出和 TEMP 有关，
    也有人说成"数据库损坏"——其实库是好的，Python 的 sqlite3 能正常打开。

    所以这里自管：var/tmp 能写，就把 TEMP / TMP / TMPDIR 指过去。
    必须在创建 Chroma 客户端之前调用，SQLite 是在打开库那一刻读这些变量的。
    """
    settings = get_settings()
    tmp = settings.temp_dir
    try:
        tmp.mkdir(parents=True, exist_ok=True)
        probe = tmp / f".probe-{os.getpid()}"
        probe.write_text("ok", encoding="utf-8")
        probe.unlink()
    except OSError as exc:
        log.warning("项目临时目录不可写（%s），回退系统临时目录", exc)
        return None

    for key in ("TEMP", "TMP", "TMPDIR"):
        os.environ[key] = str(tmp)
    log.debug("临时目录已指向 %s", tmp)
    return tmp
