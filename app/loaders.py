from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import pymupdf
from docx import Document

SUPPORTED_SUFFIXES = {".docx", ".pdf", ".txt"}


class DocumentLoadError(Exception):
    """Raised when a document cannot be read."""


class UnsupportedFileTypeError(DocumentLoadError):
    """Raised when a file type is not supported."""


class EmptyDocumentError(DocumentLoadError):
    """Raised when no usable text can be extracted."""


@dataclass(frozen=True)
class DocumentSection:
    text: str
    source: str
    file_type: str
    page: int | None = None
    paragraph: int | None = None


def load_document(path: str | Path) -> list[DocumentSection]:
    file_path = Path(path)
    if not file_path.is_file():
        raise DocumentLoadError(f"文件不存在或不是文件: {file_path}")

    suffix = file_path.suffix.lower()
    if suffix not in SUPPORTED_SUFFIXES:
        raise UnsupportedFileTypeError(f"不支持的文件类型: {suffix or '(无后缀)'}")

    try:
        if suffix == ".txt":
            sections = _load_txt(file_path)
        elif suffix == ".docx":
            sections = _load_docx(file_path)
        else:
            sections = _load_pdf(file_path)
    except DocumentLoadError:
        raise
    except Exception as exc:
        raise DocumentLoadError(f"读取 {file_path.name} 失败: {exc}") from exc

    if not sections:
        raise EmptyDocumentError(
            f"{file_path.name} 没有可用文字；如果是 PDF，可能是扫描件，需要 OCR。"
        )
    return sections


def _load_txt(path: Path) -> list[DocumentSection]:
    data = path.read_bytes()
    encodings = (
        ("utf-16",) if data.startswith((b"\xff\xfe", b"\xfe\xff")) else ("utf-8-sig", "gb18030")
    )

    text = None
    for encoding in encodings:
        try:
            text = data.decode(encoding)
            break
        except UnicodeDecodeError:
            continue
    if text is None:
        raise DocumentLoadError(f"{path.name} 编码无法识别，请转换为 UTF-8 或 GB18030")

    text = text.strip()
    if not text:
        return []
    return [DocumentSection(text=text, source=path.name, file_type="txt")]


def _load_docx(path: Path) -> list[DocumentSection]:
    document = Document(path)
    sections: list[DocumentSection] = []
    for index, paragraph in enumerate(document.paragraphs):
        text = paragraph.text.strip()
        if text:
            sections.append(
                DocumentSection(
                    text=text,
                    source=path.name,
                    file_type="docx",
                    paragraph=index,
                )
            )
    return sections


def _load_pdf(path: Path) -> list[DocumentSection]:
    sections: list[DocumentSection] = []
    with pymupdf.open(path) as document:
        if document.needs_pass:
            raise DocumentLoadError(f"{path.name} 需要密码，当前版本不支持加密 PDF")
        for page_index in range(document.page_count):
            page = document.load_page(page_index)
            text = page.get_text("text").strip()
            if text:
                sections.append(
                    DocumentSection(
                        text=text,
                        source=path.name,
                        file_type="pdf",
                        page=page_index + 1,
                    )
                )
    return sections
