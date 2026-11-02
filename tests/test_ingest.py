from __future__ import annotations

from pathlib import Path

from agent_rag.ingest.pipeline import ingest_paths


class FakeStore:
    """用假向量库测入库逻辑，这样测试不需要下载 embedding 模型。"""

    def __init__(self) -> None:
        self.data: dict[str, list[str]] = {}

    def delete_source(self, source: str) -> int:
        return len(self.data.pop(source, []))

    def add_chunks(self, ids, texts, metadatas, embedder=None) -> int:
        source = metadatas[0]["source"]
        self.data.setdefault(source, []).extend(ids)
        return len(ids)

    def count(self) -> int:
        return sum(len(v) for v in self.data.values())


def _write(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")


def test_ingest_twice_does_not_duplicate(tmp_path, tmp_settings):
    src = tmp_path / "docs"
    _write(src / "a.md", "# 制度\n\n未满一年不享受年假。")

    store = FakeStore()
    first = ingest_paths([src], base=src, store=store)
    assert first.added == 1
    assert store.count() == first.chunks

    second = ingest_paths([src], base=src, store=store)
    assert second.skipped == 1
    assert second.added == 0
    assert store.count() == first.chunks, "第二次入库不能新增向量"


def test_changed_file_replaces_old_chunks(tmp_path, tmp_settings):
    src = tmp_path / "docs"
    doc = src / "a.md"
    _write(doc, "# 制度\n\n年假 5 天。")

    store = FakeStore()
    ingest_paths([src], base=src, store=store, chunk_size=100)
    old_ids = list(store.data["a.md"])

    _write(doc, "# 制度\n\n年假改成 10 天了，所有人注意。" * 3)
    stats = ingest_paths([src], base=src, store=store, chunk_size=100)

    assert stats.updated == 1
    new_ids = store.data["a.md"]
    assert set(old_ids).isdisjoint(new_ids), "旧版本的 chunk 必须被删掉"


def test_deleted_file_prunes_chunks(tmp_path, tmp_settings):
    src = tmp_path / "docs"
    _write(src / "a.md", "# A\n\n内容一。")
    _write(src / "b.md", "# B\n\n内容二。")

    store = FakeStore()
    ingest_paths([src], base=src, store=store)
    assert set(store.data) == {"a.md", "b.md"}

    (src / "b.md").unlink()
    stats = ingest_paths([src], base=src, store=store, prune=True)

    assert stats.removed == 1
    assert set(store.data) == {"a.md"}


def test_single_file_ingest_does_not_prune_others(tmp_path, tmp_settings):
    """单文件入库不能清理别的文档。

    CLI 里 prune=directory.is_dir()，这里模拟"只入库一个文件"的场景，
    锁死这个行为：否则用户往已有知识库里追加一个 PDF，会把库里其它文档全删掉。
    """
    src = tmp_path / "docs"
    _write(src / "a.md", "# A\n\n内容一。")
    _write(src / "b.md", "# B\n\n内容二。")

    store = FakeStore()
    ingest_paths([src], base=src, store=store)
    assert set(store.data) == {"a.md", "b.md"}

    ingest_paths([src / "b.md"], base=src, store=store, prune=False)
    assert set(store.data) == {"a.md", "b.md"}, "单文件入库误删了其它文档"
