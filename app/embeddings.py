from __future__ import annotations

import math

import httpx

from app.config import EMBEDDING_MODEL, OLLAMA_BASE_URL


class EmbeddingError(Exception):
    """Raised when the embedding service cannot return usable vectors."""


def embed_texts(texts: list[str], model: str | None = None) -> list[list[float]]:
    if not texts:
        return []
    if any(not text.strip() for text in texts):
        raise ValueError("待向量化文本不能为空")

    selected_model = model or EMBEDDING_MODEL
    payload = {"model": selected_model, "input": texts}
    result = _request_json(f"{OLLAMA_BASE_URL}/api/embed", payload)

    embeddings = result.get("embeddings")
    if not isinstance(embeddings, list):
        raise EmbeddingError("Embedding 响应格式不正确：缺少 embeddings")
    if len(embeddings) != len(texts):
        raise EmbeddingError(
            f"Embedding 返回数量不匹配：请求 {len(texts)} 条，返回 {len(embeddings)} 条"
        )

    vectors: list[list[float]] = []
    for embedding in embeddings:
        if not isinstance(embedding, list) or not embedding:
            raise EmbeddingError("Embedding 响应包含空向量或错误格式")
        try:
            vectors.append([float(value) for value in embedding])
        except (TypeError, ValueError) as exc:
            raise EmbeddingError("Embedding 响应包含非数字向量值") from exc
    return vectors


def embed_query(text: str) -> list[float]:
    return embed_texts([text])[0]


def cosine_similarity(left: list[float], right: list[float]) -> float:
    if len(left) != len(right):
        raise ValueError(f"向量维度不一致：{len(left)} != {len(right)}")
    if not left:
        raise ValueError("向量不能为空")

    dot_product = sum(
        left_value * right_value for left_value, right_value in zip(left, right, strict=True)
    )
    left_norm = math.sqrt(sum(value * value for value in left))
    right_norm = math.sqrt(sum(value * value for value in right))
    if left_norm == 0 or right_norm == 0:
        raise ValueError("向量不能是零向量")
    return dot_product / (left_norm * right_norm)


def _request_json(url: str, payload: dict, timeout: float = 180.0) -> dict:
    try:
        with httpx.Client(timeout=timeout, trust_env=False) as client:
            response = client.post(url, json=payload)
            response.raise_for_status()
            result = response.json()
    except httpx.HTTPError as exc:
        raise EmbeddingError(f"Embedding 请求失败: {exc}") from exc
    except ValueError as exc:
        raise EmbeddingError(f"Embedding 响应不是有效 JSON: {exc}") from exc

    if not isinstance(result, dict):
        raise EmbeddingError("Embedding 响应格式不正确：顶层不是对象")
    return result
