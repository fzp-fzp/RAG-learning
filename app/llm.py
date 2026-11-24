from __future__ import annotations

import httpx

from app.config import (
    LLM_MODEL,
    LLM_NUM_PREDICT,
    LLM_TEMPERATURE,
    LLM_TIMEOUT,
    OLLAMA_BASE_URL,
)


class LLMError(Exception):
    """Raised when the local LLM service cannot return a usable answer."""


def build_chat_payload(
    messages: list[dict[str, str]],
    *,
    model: str | None = None,
    temperature: float | None = None,
    num_predict: int | None = None,
) -> dict:
    if not messages:
        raise ValueError("messages 不能为空")

    return {
        "model": model or LLM_MODEL,
        "messages": messages,
        "stream": False,
        "think": False,
        "options": {
            "temperature": LLM_TEMPERATURE if temperature is None else temperature,
            "num_predict": LLM_NUM_PREDICT if num_predict is None else num_predict,
        },
    }


def parse_chat_response(result: dict) -> str:
    message = result.get("message")
    if not isinstance(message, dict):
        raise LLMError("LLM 响应格式不正确：缺少 message")

    content = message.get("content")
    if not isinstance(content, str) or not content.strip():
        raise LLMError("LLM 返回空回答")
    return content.strip()


def chat(
    messages: list[dict[str, str]],
    *,
    model: str | None = None,
    temperature: float | None = None,
    num_predict: int | None = None,
    timeout: float = LLM_TIMEOUT,
) -> str:
    payload = build_chat_payload(
        messages,
        model=model,
        temperature=temperature,
        num_predict=num_predict,
    )
    result = _request_json(f"{OLLAMA_BASE_URL}/api/chat", payload, timeout)
    return parse_chat_response(result)


def _request_json(url: str, payload: dict, timeout: float) -> dict:
    try:
        with httpx.Client(timeout=timeout, trust_env=False) as client:
            response = client.post(url, json=payload)
            response.raise_for_status()
            result = response.json()
    except httpx.HTTPError as exc:
        raise LLMError(f"LLM 请求失败: {exc}") from exc
    except ValueError as exc:
        raise LLMError(f"LLM 响应不是有效 JSON: {exc}") from exc

    if not isinstance(result, dict):
        raise LLMError("LLM 响应格式不正确：顶层不是对象")
    return result
