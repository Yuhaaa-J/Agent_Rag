"""入库流水线：幂等 + 可删除 + 可增量。

三条规则：
1. 文件内容 hash 没变 -> 整篇跳过，不重新向量化（省钱、省时间）；
2. 内容变了 -> 先按 source 删旧 chunk，再写新的（不会留下过期版本）；
3. 文件被删了 -> 对应 chunk 一起清掉（prune）。
"""

from __future__ import annotations

import hashlib
import logging
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path

from ..db import session
from ..retrieval.store import ChromaStore, get_store
from .loaders import iter_files, load_file
from .splitter import split_documents

log = logging.getLogger(__name__)


def file_digest(path: Path) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


@dataclass
class IngestStats:
    added: int = 0
    updated: int = 0
    skipped: int = 0
    removed: int = 0
    chunks: int = 0

    def summary(self) -> str:
        return (
            f"新增 {self.added} / 更新 {self.updated} / 跳过 {self.skipped} / "
            f"清理 {self.removed} 个文件，共写入 {self.chunks} 个 chunk"
        )


def _manifest() -> dict[str, str]:
    with session() as conn:
        rows = conn.execute("SELECT source, file_hash FROM documents").fetchall()
    return {row["source"]: row["file_hash"] for row in rows}


def _record(source: str, digest: str, chunk_count: int) -> None:
    now = datetime.now(UTC).isoformat(timespec="seconds")
    with session() as conn:
        conn.execute(
            "INSERT INTO documents(source, file_hash, chunk_count, indexed_at) "
            "VALUES (?, ?, ?, ?) "
            "ON CONFLICT(source) DO UPDATE SET "
            "file_hash=excluded.file_hash, chunk_count=excluded.chunk_count, "
            "indexed_at=excluded.indexed_at",
            (source, digest, chunk_count, now),
        )


def _forget(source: str) -> None:
    with session() as conn:
        conn.execute("DELETE FROM documents WHERE source = ?", (source,))


def ingest_paths(
    paths: list[Path],
    *,
    base: Path | None = None,
    reset: bool = False,
    prune: bool = False,
    store: ChromaStore | None = None,
    chunk_size: int | None = None,
    chunk_overlap: int | None = None,
) -> IngestStats:
    from ..config import get_settings

    settings = get_settings()
    store = store or get_store()
    size = chunk_size or settings.chunk_size
    overlap = chunk_overlap if chunk_overlap is not None else settings.chunk_overlap

    files: list[Path] = []
    for path in paths:
        files.extend(iter_files(Path(path)))

    if base is not None:
        root = Path(base)
    elif len(paths) == 1:
        root = Path(paths[0])
    else:
        root = Path.cwd()
    if root.is_file():
        root = root.parent

    stats = IngestStats()
    manifest = _manifest()
    seen: set[str] = set()

    for path in files:
        try:
            source = path.relative_to(root).as_posix()
        except ValueError:
            source = path.name
        seen.add(source)

        digest = file_digest(path)
        if not reset and manifest.get(source) == digest:
            stats.skipped += 1
            continue

        docs = load_file(path, root)
        if not docs:
            continue
        for doc in docs:
            doc.metadata["file_hash"] = digest

        chunks = split_documents(docs, chunk_size=size, chunk_overlap=overlap)
        removed = store.delete_source(source)
        if removed:
            log.debug("%s 删除旧 chunk %d 个", source, removed)

        added = store.add_chunks(
            ids=[c.metadata["chunk_id"] for c in chunks],
            texts=[c.page_content for c in chunks],
            metadatas=[c.metadata for c in chunks],
        )
        _record(source, digest, added)
        stats.chunks += added
        if source in manifest:
            stats.updated += 1
        else:
            stats.added += 1

    if prune:
        for source in set(manifest) - seen:
            store.delete_source(source)
            _forget(source)
            stats.removed += 1
            log.info("源文件已不存在，清理 chunk: %s", source)

    return stats


def delete_source(source: str, store: ChromaStore | None = None) -> int:
    store = store or get_store()
    removed = store.delete_source(source)
    _forget(source)
    return removed
