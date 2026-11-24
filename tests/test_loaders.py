from __future__ import annotations

from pathlib import Path

import pymupdf
import pytest
from docx import Document

from app.loaders import (
    DocumentLoadError,
    EmptyDocumentError,
    UnsupportedFileTypeError,
    load_document,
)


@pytest.fixture
def txt_file(tmp_path: Path) -> Path:
    path = tmp_path / "sample.txt"
    path.write_text("第一段内容\n\n第二段内容", encoding="utf-8")
    return path


@pytest.fixture
def docx_file(tmp_path: Path) -> Path:
    path = tmp_path / "sample.docx"
    document = Document()
    document.add_heading("测试标题", level=1)
    document.add_paragraph("第一段")
    document.add_paragraph("")
    document.add_paragraph("第二段")
    document.save(path)
    return path


@pytest.fixture
def pdf_file(tmp_path: Path) -> Path:
    path = tmp_path / "sample.pdf"
    document = pymupdf.open()
    first_page = document.new_page()
    first_page.insert_text((72, 72), "Page one text")
    second_page = document.new_page()
    second_page.insert_text((72, 72), "Page two text")
    document.save(path)
    document.close()
    return path


def test_load_txt_returns_single_section_with_source(txt_file: Path) -> None:
    sections = load_document(txt_file)

    assert len(sections) == 1
    assert "第一段内容" in sections[0].text
    assert sections[0].source == "sample.txt"
    assert sections[0].file_type == "txt"
    assert sections[0].page is None
    assert sections[0].paragraph is None


def test_load_txt_supports_gb18030(tmp_path: Path) -> None:
    path = tmp_path / "gbk.txt"
    path.write_bytes("中文编码内容".encode("gb18030"))

    sections = load_document(path)

    assert sections[0].text == "中文编码内容"


def test_load_docx_returns_non_empty_paragraphs(docx_file: Path) -> None:
    sections = load_document(docx_file)

    assert [section.text for section in sections] == ["测试标题", "第一段", "第二段"]
    assert all(section.source == "sample.docx" for section in sections)
    assert all(section.file_type == "docx" for section in sections)
    assert all(section.page is None for section in sections)


def test_load_pdf_preserves_page_numbers(pdf_file: Path) -> None:
    sections = load_document(pdf_file)

    assert [section.page for section in sections] == [1, 2]
    assert "Page one text" in sections[0].text
    assert "Page two text" in sections[1].text
    assert all(section.file_type == "pdf" for section in sections)


def test_unsupported_suffix_raises(tmp_path: Path) -> None:
    path = tmp_path / "notes.md"
    path.write_text("not supported", encoding="utf-8")

    with pytest.raises(UnsupportedFileTypeError):
        load_document(path)


def test_missing_file_raises(tmp_path: Path) -> None:
    with pytest.raises(DocumentLoadError):
        load_document(tmp_path / "missing.txt")


def test_empty_txt_raises_empty_document_error(tmp_path: Path) -> None:
    path = tmp_path / "empty.txt"
    path.write_text(" \n\n ", encoding="utf-8")

    with pytest.raises(EmptyDocumentError):
        load_document(path)


def test_empty_pdf_raises_scan_hint(tmp_path: Path) -> None:
    path = tmp_path / "scanned.pdf"
    document = pymupdf.open()
    document.new_page()
    document.save(path)
    document.close()

    with pytest.raises(EmptyDocumentError, match="扫描"):
        load_document(path)
