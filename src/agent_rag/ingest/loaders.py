"""文档加载。不用 langchain_community 的一堆 Loader，直接对接底层库：

- 好处 1：metadata（source / page）完全可控，检索结果能溯源；
- 好处 2：少一层抽象，出问题知道去哪看；
- 好处 3：不需要 pypdf / docx2txt 时不影响其余功能（延迟 import）。
"""

from __future__ import annotations

import logging
from pathlib import Path

from langchain_core.documents import Document

log = logging.getLogger(__name__)

SUPPORTED_SUFFIXES = {".md", ".markdown", ".txt", ".pdf", ".docx"}
SKIP_DIRS = {".git", ".venv", "venv", "__pycache__", "node_modules", "var", ".idea"}


def iter_files(root: Path) -> list[Path]:
    """递归列出可入库文件，顺序稳定（便于复现）。"""
    root = Path(root)
    if root.is_file():
        return [root] if root.suffix.lower() in SUPPORTED_SUFFIXES else []
    if not root.exists():
        raise FileNotFoundError(f"目录不存在: {root}")

    files: list[Path] = []
    for path in sorted(root.rglob("*")):
        if not path.is_file():
            continue
        if path.suffix.lower() not in SUPPORTED_SUFFIXES:
            continue
        rel_parts = path.relative_to(root).parts
        if any(part in SKIP_DIRS or part.startswith(".") for part in rel_parts):
            continue
        files.append(path)
    return files


def _read_text(path: Path) -> str:
    raw = path.read_bytes()
    for encoding in ("utf-8", "utf-8-sig", "gb18030"):
        try:
            return raw.decode(encoding)
        except UnicodeDecodeError:
            continue
    return raw.decode("utf-8", errors="ignore")


def load_file(path: Path, base: Path) -> list[Document]:
    """把单个文件读成 Document 列表。pdf 按页切，其余整篇。"""
    path = Path(path)
    base = Path(base)
    try:
        source = path.relative_to(base).as_posix()
    except ValueError:
        source = path.name

    suffix = path.suffix.lower()
    common = {"source": source, "suffix": suffix}

    if suffix in {".md", ".markdown", ".txt"}:
        text = _read_text(path)
        if not text.strip():
            log.warning("跳过空文件: %s", source)
            return []
        return [Document(page_content=text, metadata={**common, "page": 0})]

    if suffix == ".pdf":
        from pypdf import PdfReader

        reader = PdfReader(str(path))
        docs: list[Document] = []
        for page_no, page in enumerate(reader.pages, start=1):
            text = page.extract_text() or ""
            if text.strip():
                docs.append(
                    Document(page_content=text, metadata={**common, "page": page_no})
                )
        return docs

    if suffix == ".docx":
        import docx2txt

        text = docx2txt.process(str(path)) or ""
        if not text.strip():
            log.warning("跳过空文件: %s", source)
            return []
        return [Document(page_content=text, metadata={**common, "page": 0})]

    return []


def load_paths(paths: list[Path], base: Path) -> list[Document]:
    docs: list[Document] = []
    for path in paths:
        docs.extend(load_file(path, base))
    return docs
