from __future__ import annotations

from pathlib import Path

import pytest

from app.chunking import TextChunk
from app.vector_store import VectorStore


def make_chunk(
    text: str,
    source: str = "a.txt",
    chunk_index: int = 0,
    page: int | None = None,
    paragraph: int | None = None,
) -> TextChunk:
    return TextChunk(
        text=text,
        source=source,
        file_type=Path(source).suffix.lstrip("."),
        page=page,
        paragraph=paragraph,
        chunk_index=chunk_index,
        start_char=0,
        end_char=len(text),
    )


def make_store(path: Path) -> VectorStore:
    return VectorStore(path=path, collection_name="test_knowledge")


def test_add_empty_chunks_does_not_write(tmp_path: Path) -> None:
    store = make_store(tmp_path / "chroma")

    assert store.add_chunks([], []) == 0
    assert store.count() == 0


def test_add_chunks_and_count(tmp_path: Path) -> None:
    store = make_store(tmp_path / "chroma")
    chunks = [make_chunk("第一段", chunk_index=0), make_chunk("第二段", chunk_index=1)]

    written = store.add_chunks(chunks, [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]])

    assert written == 2
    assert store.count() == 2


def test_add_chunks_rejects_mismatched_count(tmp_path: Path) -> None:
    store = make_store(tmp_path / "chroma")

    with pytest.raises(ValueError, match="数量"):
        store.add_chunks([make_chunk("第一段")], [])


def test_add_chunks_is_idempotent_for_same_chunks(tmp_path: Path) -> None:
    store = make_store(tmp_path / "chroma")
    chunks = [make_chunk("第一段", chunk_index=0), make_chunk("第二段", chunk_index=1)]
    embeddings = [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]]

    store.add_chunks(chunks, embeddings)
    store.add_chunks(chunks, embeddings)

    assert store.count() == 2


def test_data_persists_after_new_client(tmp_path: Path) -> None:
    path = tmp_path / "chroma"
    first = make_store(path)
    first.add_chunks([make_chunk("持久化内容")], [[1.0, 0.0, 0.0]])

    second = make_store(path)

    assert second.count() == 1
    assert second.source_counts() == {"a.txt": 1}


def test_delete_by_source_removes_only_matching_document(tmp_path: Path) -> None:
    store = make_store(tmp_path / "chroma")
    chunks = [
        make_chunk("A1", source="a.txt", chunk_index=0),
        make_chunk("A2", source="a.txt", chunk_index=1),
        make_chunk("B1", source="b.txt", chunk_index=0),
    ]
    store.add_chunks(
        chunks,
        [[1.0, 0.0, 0.0], [0.9, 0.1, 0.0], [0.0, 1.0, 0.0]],
    )

    deleted = store.delete_by_source("a.txt")

    assert deleted == 2
    assert store.count() == 1
    assert store.source_counts() == {"b.txt": 1}


def test_source_counts_groups_chunks_by_document(tmp_path: Path) -> None:
    store = make_store(tmp_path / "chroma")
    chunks = [
        make_chunk("A1", source="a.txt", chunk_index=0),
        make_chunk("A2", source="a.txt", chunk_index=1),
        make_chunk("B1", source="b.txt", chunk_index=0),
    ]
    store.add_chunks(
        chunks,
        [[1.0, 0.0, 0.0], [0.9, 0.1, 0.0], [0.0, 1.0, 0.0]],
    )

    assert store.source_counts() == {"a.txt": 2, "b.txt": 1}


def test_get_records_preserves_text_and_metadata(tmp_path: Path) -> None:
    store = make_store(tmp_path / "chroma")
    chunks = [
        make_chunk("PDF 内容", source="a.pdf", chunk_index=0, page=1),
        make_chunk("DOCX 内容", source="b.docx", chunk_index=1, paragraph=3),
    ]
    store.add_chunks(chunks, [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]])

    records = {record.text: record for record in store.get_records()}

    assert records["PDF 内容"].metadata["page"] == 1
    assert "paragraph" not in records["PDF 内容"].metadata
    assert records["DOCX 内容"].metadata["paragraph"] == 3
    assert "page" not in records["DOCX 内容"].metadata
    assert records["PDF 内容"].metadata["chunk_index"] == 0


def test_search_returns_cosine_similarity_semantics(tmp_path: Path) -> None:
    store = make_store(tmp_path / "chroma")
    chunks = [
        make_chunk("相同方向", source="same.txt", chunk_index=0),
        make_chunk("垂直方向", source="orthogonal.txt", chunk_index=0),
    ]
    store.add_chunks(chunks, [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]])

    results = store.search([1.0, 0.0, 0.0], top_k=2)

    assert [result.text for result in results] == ["相同方向", "垂直方向"]
    assert results[0].distance == pytest.approx(0.0, abs=1e-6)
    assert results[0].similarity == pytest.approx(1.0, abs=1e-6)
    assert results[1].distance == pytest.approx(1.0, abs=1e-6)
    assert results[1].similarity == pytest.approx(0.0, abs=1e-6)


def test_search_respects_top_k_and_preserves_metadata(tmp_path: Path) -> None:
    store = make_store(tmp_path / "chroma")
    chunks = [
        make_chunk("最相似", source="a.txt", chunk_index=0),
        make_chunk("次相似", source="b.txt", chunk_index=1),
        make_chunk("最不相似", source="c.txt", chunk_index=2),
    ]
    store.add_chunks(
        chunks,
        [[1.0, 0.0, 0.0], [0.8, 0.2, 0.0], [0.0, 1.0, 0.0]],
    )

    results = store.search([1.0, 0.0, 0.0], top_k=2)

    assert len(results) == 2
    assert results[0].text == "最相似"
    assert results[0].metadata["source"] == "a.txt"
    assert results[0].metadata["chunk_index"] == 0
    assert results[1].text == "次相似"


def test_search_returns_all_available_when_top_k_is_larger(tmp_path: Path) -> None:
    store = make_store(tmp_path / "chroma")
    store.add_chunks([make_chunk("唯一块")], [[1.0, 0.0, 0.0]])

    results = store.search([1.0, 0.0, 0.0], top_k=10)

    assert len(results) == 1


def test_search_empty_collection_returns_empty_list(tmp_path: Path) -> None:
    store = make_store(tmp_path / "chroma")

    assert store.search([1.0, 0.0, 0.0], top_k=5) == []


@pytest.mark.parametrize("top_k", [0, -1])
def test_search_rejects_invalid_top_k(tmp_path: Path, top_k: int) -> None:
    store = make_store(tmp_path / "chroma")

    with pytest.raises(ValueError, match="top_k"):
        store.search([1.0, 0.0, 0.0], top_k=top_k)


def test_search_rejects_empty_query_embedding(tmp_path: Path) -> None:
    store = make_store(tmp_path / "chroma")

    with pytest.raises(ValueError, match="查询向量"):
        store.search([], top_k=5)


def test_context_manager_releases_chroma_file_handles(tmp_path: Path) -> None:
    path = tmp_path / "chroma"
    with make_store(path) as store:
        store.add_chunks([make_chunk("需要释放句柄")], [[1.0, 0.0, 0.0]])

    moved = tmp_path / "chroma_moved"
    path.rename(moved)

    assert (moved / "chroma.sqlite3").is_file()
