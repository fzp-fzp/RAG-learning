from __future__ import annotations

import hashlib
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import chromadb
from chromadb.config import Settings

from app.chunking import TextChunk
from app.config import CHROMA_DIR, COLLECTION_NAME


@dataclass(frozen=True)
class StoredChunk:
    id: str
    text: str
    metadata: dict[str, Any]


@dataclass(frozen=True)
class SearchResult:
    id: str
    text: str
    metadata: dict[str, Any]
    distance: float
    similarity: float


class VectorStore:
    def __init__(
        self,
        path: str | Path | None = None,
        collection_name: str | None = None,
    ) -> None:
        self.path = Path(path) if path is not None else CHROMA_DIR
        self.path.mkdir(parents=True, exist_ok=True)
        self.collection_name = collection_name or COLLECTION_NAME
        self._client = chromadb.PersistentClient(
            path=self.path,
            settings=Settings(anonymized_telemetry=False),
        )
        self._collection = self._client.get_or_create_collection(
            name=self.collection_name,
            configuration={"hnsw": {"space": "cosine"}},
            embedding_function=None,
        )

    def __enter__(self) -> VectorStore:
        return self

    def __exit__(self, exc_type: object, exc: object, traceback: object) -> None:
        self.close()

    def close(self) -> None:
        self._client.close()

    def add_chunks(self, chunks: list[TextChunk], embeddings: list[list[float]]) -> int:
        if len(chunks) != len(embeddings):
            raise ValueError(f"文本块与向量数量不匹配：{len(chunks)} != {len(embeddings)}")
        if not chunks:
            return 0

        dimensions = {len(embedding) for embedding in embeddings}
        if len(dimensions) != 1 or 0 in dimensions:
            raise ValueError("向量维度必须一致且非空")

        ids = [_chunk_id(chunk) for chunk in chunks]
        self._collection.upsert(
            ids=ids,
            documents=[chunk.text for chunk in chunks],
            embeddings=embeddings,
            metadatas=[_chunk_metadata(chunk) for chunk in chunks],
        )
        return len(chunks)

    def count(self) -> int:
        return self._collection.count()

    def delete_by_source(self, source: str) -> int:
        if not source.strip():
            raise ValueError("source 不能为空")
        before = self.count()
        self._collection.delete(where={"source": source})
        return before - self.count()

    def source_counts(self) -> dict[str, int]:
        result = self._collection.get(include=["metadatas"])
        counts: dict[str, int] = {}
        for metadata in result.get("metadatas") or []:
            source = metadata.get("source")
            if isinstance(source, str):
                counts[source] = counts.get(source, 0) + 1
        return dict(sorted(counts.items()))

    def get_records(self, source: str | None = None) -> list[StoredChunk]:
        arguments: dict[str, Any] = {"include": ["documents", "metadatas"]}
        if source is not None:
            arguments["where"] = {"source": source}
        result = self._collection.get(**arguments)

        records: list[StoredChunk] = []
        for record_id, text, metadata in zip(
            result.get("ids") or [],
            result.get("documents") or [],
            result.get("metadatas") or [],
            strict=True,
        ):
            records.append(
                StoredChunk(
                    id=record_id,
                    text=text or "",
                    metadata=dict(metadata or {}),
                )
            )
        return records

    def search(self, query_embedding: list[float], top_k: int = 5) -> list[SearchResult]:
        if top_k <= 0:
            raise ValueError("top_k 必须大于 0")
        if not query_embedding:
            raise ValueError("查询向量不能为空")

        count = self.count()
        if count == 0:
            return []

        result = self._collection.query(
            query_embeddings=[query_embedding],
            n_results=min(top_k, count),
            include=["documents", "metadatas", "distances"],
        )
        ids = (result.get("ids") or [[]])[0]
        documents = (result.get("documents") or [[]])[0]
        metadatas = (result.get("metadatas") or [[]])[0]
        distances = (result.get("distances") or [[]])[0]

        search_results: list[SearchResult] = []
        for record_id, text, metadata, distance in zip(
            ids,
            documents,
            metadatas,
            distances,
            strict=True,
        ):
            numeric_distance = float(distance)
            search_results.append(
                SearchResult(
                    id=record_id,
                    text=text or "",
                    metadata=dict(metadata or {}),
                    distance=numeric_distance,
                    similarity=1.0 - numeric_distance,
                )
            )
        return search_results


def _chunk_id(chunk: TextChunk) -> str:
    identity = f"{chunk.source}\0{chunk.chunk_index}"
    return hashlib.sha256(identity.encode("utf-8")).hexdigest()


def _chunk_metadata(chunk: TextChunk) -> dict[str, str | int | float | bool]:
    metadata: dict[str, str | int | float | bool] = {
        "source": chunk.source,
        "file_type": chunk.file_type,
        "chunk_index": chunk.chunk_index,
        "start_char": chunk.start_char,
        "end_char": chunk.end_char,
    }
    if chunk.page is not None:
        metadata["page"] = chunk.page
    if chunk.paragraph is not None:
        metadata["paragraph"] = chunk.paragraph
    return metadata
