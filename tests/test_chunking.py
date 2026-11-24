from __future__ import annotations

import pytest

from app.chunking import TextChunk, chunk_sections
from app.loaders import DocumentSection


def make_section(
    text: str,
    page: int | None = None,
    paragraph: int | None = None,
) -> DocumentSection:
    return DocumentSection(
        text=text,
        source="demo.txt",
        file_type="txt",
        page=page,
        paragraph=paragraph,
    )


def test_short_section_returns_one_chunk() -> None:
    chunks = chunk_sections([make_section("短文本")], chunk_size=100, overlap=20)

    assert len(chunks) == 1
    assert isinstance(chunks[0], TextChunk)
    assert chunks[0].text == "短文本"
    assert chunks[0].chunk_index == 0
    assert chunks[0].start_char == 0
    assert chunks[0].end_char == 3


def test_long_section_splits_with_overlap() -> None:
    text = "第一句话。" * 20

    chunks = chunk_sections([make_section(text)], chunk_size=30, overlap=5)

    assert len(chunks) > 1
    assert all(0 < len(chunk.text) <= 30 for chunk in chunks)
    assert all(
        chunks[index].start_char < chunks[index + 1].start_char for index in range(len(chunks) - 1)
    )
    assert any(
        chunks[index].end_char > chunks[index + 1].start_char for index in range(len(chunks) - 1)
    )


def test_prefers_sentence_boundary() -> None:
    text = "第一句内容。第二句内容。第三句内容。第四句内容。"

    chunks = chunk_sections([make_section(text)], chunk_size=20, overlap=2)

    assert len(chunks) > 1
    assert chunks[0].text.endswith("。")


def test_chunk_index_is_sequential() -> None:
    sections = [make_section("甲" * 80), make_section("乙" * 80)]

    chunks = chunk_sections(sections, chunk_size=40, overlap=8)

    assert [chunk.chunk_index for chunk in chunks] == list(range(len(chunks)))


def test_section_metadata_is_preserved() -> None:
    sections = [make_section("甲" * 80, page=3), make_section("乙" * 80, paragraph=2)]

    chunks = chunk_sections(sections, chunk_size=40, overlap=8)

    first_page_chunks = [chunk for chunk in chunks if chunk.page == 3]
    first_paragraph_chunks = [chunk for chunk in chunks if chunk.paragraph == 2]
    assert first_page_chunks
    assert first_paragraph_chunks
    assert all(chunk.source == "demo.txt" for chunk in chunks)
    assert all(chunk.file_type == "txt" for chunk in chunks)


def test_blank_section_is_skipped() -> None:
    chunks = chunk_sections([make_section("   \n  ")], chunk_size=100, overlap=20)

    assert chunks == []


@pytest.mark.parametrize(
    ("chunk_size", "overlap"),
    [(0, 0), (-1, 0), (100, -1), (100, 100), (100, 101)],
)
def test_invalid_parameters_raise_value_error(chunk_size: int, overlap: int) -> None:
    with pytest.raises(ValueError):
        chunk_sections([make_section("内容")], chunk_size=chunk_size, overlap=overlap)
