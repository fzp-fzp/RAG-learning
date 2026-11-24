from __future__ import annotations

import pytest

from app.embeddings import EmbeddingError, cosine_similarity, embed_query, embed_texts


def test_embed_texts_returns_empty_list_for_no_input() -> None:
    assert embed_texts([]) == []


def test_embed_texts_sends_expected_request(monkeypatch: pytest.MonkeyPatch) -> None:
    calls: list[tuple[str, dict, float]] = []

    def fake_request(url: str, payload: dict, timeout: float = 180.0) -> dict:
        calls.append((url, payload, timeout))
        return {"embeddings": [[1.0, 0.0], [0.0, 1.0]]}

    monkeypatch.setattr("app.embeddings._request_json", fake_request)

    result = embed_texts(["第一段", "第二段"], model="test-model")

    assert result == [[1.0, 0.0], [0.0, 1.0]]
    assert calls[0][0].endswith("/api/embed")
    assert calls[0][1] == {"model": "test-model", "input": ["第一段", "第二段"]}


def test_embed_texts_rejects_blank_text() -> None:
    with pytest.raises(ValueError, match="空"):
        embed_texts(["有效内容", "   "])


def test_embed_texts_rejects_mismatched_count(monkeypatch: pytest.MonkeyPatch) -> None:
    def fake_request(url: str, payload: dict, timeout: float = 180.0) -> dict:
        return {"embeddings": [[1.0, 0.0]]}

    monkeypatch.setattr("app.embeddings._request_json", fake_request)

    with pytest.raises(EmbeddingError, match="数量"):
        embed_texts(["第一段", "第二段"])


def test_embed_query_returns_first_vector(monkeypatch: pytest.MonkeyPatch) -> None:
    def fake_request(url: str, payload: dict, timeout: float = 180.0) -> dict:
        return {"embeddings": [[0.1, 0.2, 0.3]]}

    monkeypatch.setattr("app.embeddings._request_json", fake_request)

    assert embed_query("问题") == [0.1, 0.2, 0.3]


def test_cosine_similarity_of_identical_vectors_is_one() -> None:
    assert cosine_similarity([1.0, 0.0], [1.0, 0.0]) == pytest.approx(1.0)


def test_cosine_similarity_of_orthogonal_vectors_is_zero() -> None:
    assert cosine_similarity([1.0, 0.0], [0.0, 1.0]) == pytest.approx(0.0)


def test_cosine_similarity_rejects_dimension_mismatch() -> None:
    with pytest.raises(ValueError, match="维度"):
        cosine_similarity([1.0, 0.0], [1.0, 0.0, 0.0])
