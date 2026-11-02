"""SQLite：入库清单 + 请假余额 + 工具幂等表。

用 SQLite 而不是内存字典的唯一理由：**进程重启后事实必须还在**。
旧版 demo 的 `sessions_db = {}` 和 `EMPLOYEE_DATA = {}` 一重启就全丢，
这是 demo 和项目最本质的区别之一。
"""

from __future__ import annotations

import sqlite3
from collections.abc import Iterator
from contextlib import contextmanager
from pathlib import Path

from .config import get_settings

SCHEMA = """
CREATE TABLE IF NOT EXISTS documents (
    source      TEXT PRIMARY KEY,
    file_hash   TEXT NOT NULL,
    chunk_count INTEGER NOT NULL,
    indexed_at  TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS leave_balances (
    employee  TEXT PRIMARY KEY,
    days_left INTEGER NOT NULL
);

CREATE TABLE IF NOT EXISTS leave_requests (
    id         TEXT PRIMARY KEY,
    employee   TEXT NOT NULL,
    days       INTEGER NOT NULL,
    reason     TEXT NOT NULL,
    created_at TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS tool_idempotency (
    key        TEXT PRIMARY KEY,
    result     TEXT NOT NULL,
    created_at TEXT NOT NULL
);
"""


def connect(db_path: Path | None = None) -> sqlite3.Connection:
    path = Path(db_path) if db_path is not None else get_settings().sqlite_path
    path.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(path, check_same_thread=False)
    conn.row_factory = sqlite3.Row
    conn.executescript(SCHEMA)
    return conn


@contextmanager
def session(db_path: Path | None = None) -> Iterator[sqlite3.Connection]:
    conn = connect(db_path)
    try:
        yield conn
        conn.commit()
    finally:
        conn.close()


def seed_demo_balances(db_path: Path | None = None) -> None:
    """演示数据。真实项目里这里应该来自 HR 系统，而不是硬编码。"""
    with session(db_path) as conn:
        conn.executemany(
            "INSERT OR IGNORE INTO leave_balances(employee, days_left) VALUES (?, ?)",
            [("张三", 5), ("李四", 2)],
        )
