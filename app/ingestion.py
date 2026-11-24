from __future__ import annotations

from collections.abc import Callable, Sequence
from dataclasses import dataclass
from pathlib import Path

from app.chunking import chunk_sections
from app.config import CHUNK_OVERLAP, CHUNK_SIZE
from app.embeddings import embed_texts
from app.loaders import DocumentLoadError, load_document
from app.vector_store import VectorStore


@dataclass(frozen=True)
class IngestReport:
    ingested: list[tuple[str, int]]
    failures: list[tuple[str, str]]
    total_chunks: int
    source_counts: dict[str, int]


def ingest_documents(
    paths: Sequence[str | Path],
    *,
    store: VectorStore | None = None,
    chunk_size: int = CHUNK_SIZE,
    overlap: int = CHUNK_OVERLAP,
    embedder: Callable[[list[str]], list[list[float]]] = embed_texts,
) -> IngestReport:
    if chunk_size <= 0:
        raise ValueError("chunk_size 必须大于 0")
    if overlap < 0 or overlap >= chunk_size:
        raise ValueError("overlap 必须满足 0 <= overlap < chunk_size")

    owns_store = store is None
    actual_store = store or VectorStore()
    ingested: list[tuple[str, int]] = []
    failures: list[tuple[str, str]] = []

    try:
        for path_value in paths:
            path = Path(path_value)
            source = path.name
            try:
                sections = load_document(path)
                chunks = chunk_sections(sections, chunk_size=chunk_size, overlap=overlap)
                if not chunks:
                    raise DocumentLoadError(f"{source} 没有可索引文字")

                embeddings = embedder([chunk.text for chunk in chunks])
                actual_store.delete_by_source(source)
                written = actual_store.add_chunks(chunks, embeddings)
                ingested.append((source, written))
            except Exception as exc:
                failures.append((source, str(exc)))

        return IngestReport(
            ingested=ingested,
            failures=failures,
            total_chunks=actual_store.count(),
            source_counts=actual_store.source_counts(),
        )
    finally:
        if owns_store:
            actual_store.close()
