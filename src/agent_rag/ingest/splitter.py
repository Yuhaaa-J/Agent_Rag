"""结构化切分。

普通 RecursiveCharacterTextSplitter 的问题是：切出来的块没有"位置感"，
用户问"第三章报销标准是多少"时，块里可能只有数字没有"报销"两个字。

这里的处理：
1. Markdown 先按标题切成 section，记录 heading 路径；
2. 每个 chunk 的正文前面拼上 heading 路径，让块自带上下文；
3. chunk_id 由 (source, page, index, 内容) 决定 —— 同样的内容永远得到同样的 id，
   这是"重复入库不产生重复向量"的基础。
"""

from __future__ import annotations

import hashlib
import re

from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter

HEADING_RE = re.compile(r"^(#{1,6})\s+(.+?)\s*$")

# 中文优先的分隔符：先段落，再句号，最后才是字符
SEPARATORS = ["\n\n", "\n", "。", "！", "？", "；", "，", " ", ""]


def make_chunk_id(source: str, page: int, index: int, content: str) -> str:
    digest = hashlib.sha1(
        f"{source}|{page}|{index}|{content}".encode()
    ).hexdigest()
    return digest[:20]


def _markdown_sections(text: str) -> list[tuple[str, str]]:
    """按标题切 section，返回 [(heading 路径, 正文)]。"""
    stack: list[tuple[int, str]] = []
    buffer: list[str] = []
    sections: list[tuple[str, str]] = []

    def flush() -> None:
        body = "".join(buffer)
        if body.strip():
            sections.append((" > ".join(title for _, title in stack), body))

    for line in text.splitlines(keepends=True):
        match = HEADING_RE.match(line.rstrip("\n"))
        if match:
            flush()
            buffer = []
            level = len(match.group(1))
            while stack and stack[-1][0] >= level:
                stack.pop()
            stack.append((level, match.group(2).strip()))
        else:
            buffer.append(line)
    flush()
    return sections


def split_documents(
    docs: list[Document],
    chunk_size: int = 500,
    chunk_overlap: int = 80,
) -> list[Document]:
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        separators=SEPARATORS,
        keep_separator="end",
    )

    chunks: list[Document] = []
    for doc in docs:
        source = str(doc.metadata.get("source", ""))
        page = int(doc.metadata.get("page", 0))
        text = doc.page_content
        is_markdown = source.lower().endswith((".md", ".markdown"))

        sections = _markdown_sections(text) if is_markdown else [("", text)]

        index = 0
        for heading, body in sections:
            prefix = f"[{heading}]\n" if heading else ""
            for piece in splitter.split_text(body):
                content = f"{prefix}{piece}".strip()
                if not content:
                    continue
                chunks.append(
                    Document(
                        page_content=content,
                        metadata={
                            "source": source,
                            "page": page,
                            "heading": heading,
                            "chunk_index": index,
                            "chunk_id": make_chunk_id(source, page, index, content),
                            "file_hash": str(doc.metadata.get("file_hash", "")),
                        },
                    )
                )
                index += 1
    return chunks
