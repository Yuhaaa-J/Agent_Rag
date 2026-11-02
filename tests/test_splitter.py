from __future__ import annotations

from langchain_core.documents import Document

from agent_rag.ingest.splitter import split_documents


def _doc(text: str, source: str = "rules.md") -> Document:
    return Document(page_content=text, metadata={"source": source, "page": 0})


def test_markdown_heading_is_carried_into_chunks():
    """切出来的块必须自带所属章节，否则"报销标准是多少"这种问题会丢上下文。"""
    body = "核心工作时间为上午 10:00 至下午 16:00。" * 8
    chunks = split_documents(
        [_doc("# 员工手册\n\n## 考勤制度\n\n" + body)], chunk_size=80, chunk_overlap=10
    )

    assert chunks
    assert any("考勤制度" in c.page_content for c in chunks)
    # heading 记录的是完整路径（一级 > 二级），比只记最后一级更有定位能力
    assert all(c.metadata["heading"] == "员工手册 > 考勤制度" for c in chunks)
    assert all(c.metadata["source"] == "rules.md" for c in chunks)


def test_chunk_ids_are_stable_and_unique():
    text = "# A\n\n第一段内容。" * 5 + "\n\n# B\n\n第二段内容。" * 5
    first = [c.metadata["chunk_id"] for c in split_documents([_doc(text)], 60, 10)]
    second = [c.metadata["chunk_id"] for c in split_documents([_doc(text)], 60, 10)]

    assert first == second, "同样的输入必须得到同样的 id，否则重复入库会产生重复向量"
    assert len(set(first)) == len(first)


def test_plain_text_without_headings_still_splits():
    chunks = split_documents([_doc("这是一段很长的普通文本。" * 40, source="a.txt")], 100, 20)
    assert len(chunks) > 1
    assert all(c.metadata["heading"] == "" for c in chunks)
    assert all(len(c.page_content) <= 150 for c in chunks)
