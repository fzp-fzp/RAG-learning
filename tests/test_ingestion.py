from __future__ import annotations

from pathlib import Path

from app.chunking import TextChunk
from app.ingestion import ingest_documents
from app.vector_store import VectorStore


def make_store(path: Path) -> VectorStore:
    return VectorStore(path=path, collection_name="test_ingestion")


def fake_embedder(texts: list[str]) -> list[list[float]]:
    return [[1.0, float(len(text) + 1)] for text in texts]


def make_chunk(text: str, source: str, chunk_index: int = 0) -> TextChunk:
    return TextChunk(
        text=text,
        source=source,
        file_type=Path(source).suffix.lstrip("."),
        page=None,
        paragraph=None,
        chunk_index=chunk_index,
        start_char=0,
        end_char=len(text),
    )


def test_ingest_documents_writes_chunks_and_reports_totals(tmp_path: Path) -> None:
    document = tmp_path / "规则.txt"
    document.write_text("第一条规则。第二条规则。", encoding="utf-8")

    with make_store(tmp_path / "chroma") as store:
        report = ingest_documents(
            [document],
            store=store,
            chunk_size=100,
            overlap=10,
            embedder=fake_embedder,
        )

        assert report.ingested == [("规则.txt", 1)]
        assert report.failures == []
        assert report.total_chunks == 1
        assert report.source_counts == {"规则.txt": 1}
        assert store.get_records("规则.txt")[0].text == "第一条规则。第二条规则。"


def test_ingest_documents_replaces_old_source_chunks(tmp_path: Path) -> None:
    document = tmp_path / "规则.txt"
    document.write_text("旧内容。", encoding="utf-8")

    with make_store(tmp_path / "chroma") as store:
        store.add_chunks([make_chunk("旧内容。", "规则.txt")], [[1.0, 2.0]])
        document.write_text("新内容。", encoding="utf-8")

        report = ingest_documents(
            [document],
            store=store,
            chunk_size=100,
            overlap=10,
            embedder=fake_embedder,
        )

        assert report.total_chunks == 1
        assert [record.text for record in store.get_records("规则.txt")] == ["新内容。"]


def test_ingest_documents_keeps_old_index_when_embedding_fails(tmp_path: Path) -> None:
    document = tmp_path / "规则.txt"
    document.write_text("新内容。", encoding="utf-8")

    def fail_embedder(texts: list[str]) -> list[list[float]]:
        raise RuntimeError("embedding unavailable")

    with make_store(tmp_path / "chroma") as store:
        store.add_chunks([make_chunk("旧内容。", "规则.txt")], [[1.0, 2.0]])

        report = ingest_documents(
            [document],
            store=store,
            chunk_size=100,
            overlap=10,
            embedder=fail_embedder,
        )

        assert report.ingested == []
        assert report.failures == [("规则.txt", "embedding unavailable")]
        assert [record.text for record in store.get_records("规则.txt")] == ["旧内容。"]


def test_ingest_documents_continues_after_one_bad_file(tmp_path: Path) -> None:
    bad = tmp_path / "损坏.txt"
    bad.write_bytes(b"\xff\xff")
    good = tmp_path / "可用.txt"
    good.write_text("可用内容。", encoding="utf-8")

    with make_store(tmp_path / "chroma") as store:
        report = ingest_documents(
            [bad, good],
            store=store,
            chunk_size=100,
            overlap=10,
            embedder=fake_embedder,
        )

        assert [item[0] for item in report.ingested] == ["可用.txt"]
        assert [item[0] for item in report.failures] == ["损坏.txt"]
        assert report.total_chunks == 1


def test_ingest_documents_closes_store_it_creates(tmp_path: Path, monkeypatch) -> None:
    document = tmp_path / "规则.txt"
    document.write_text("内容。", encoding="utf-8")
    closed: list[bool] = []

    class FakeStore:
        def __init__(self) -> None:
            self.records: dict[str, str] = {}

        def delete_by_source(self, source: str) -> int:
            return 0

        def add_chunks(self, chunks, embeddings) -> int:
            self.records.update({chunk.source: chunk.text for chunk in chunks})
            return len(chunks)

        def count(self) -> int:
            return len(self.records)

        def source_counts(self) -> dict[str, int]:
            return {source: 1 for source in self.records}

        def close(self) -> None:
            closed.append(True)

    monkeypatch.setattr("app.ingestion.VectorStore", FakeStore)

    report = ingest_documents(
        [document],
        chunk_size=100,
        overlap=10,
        embedder=fake_embedder,
    )

    assert report.total_chunks == 1
    assert closed == [True]
