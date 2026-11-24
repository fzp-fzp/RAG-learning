from __future__ import annotations

import pytest

import app.llm as llm


def test_build_chat_payload_disables_thinking_and_sets_options() -> None:
    messages = [{"role": "user", "content": "你好"}]

    payload = llm.build_chat_payload(messages, model="qwen3:8b", temperature=0.1, num_predict=128)

    assert payload["model"] == "qwen3:8b"
    assert payload["messages"] == messages
    assert payload["stream"] is False
    assert payload["think"] is False
    assert payload["options"] == {"temperature": 0.1, "num_predict": 128}


def test_build_chat_payload_rejects_empty_messages() -> None:
    with pytest.raises(ValueError, match="messages"):
        llm.build_chat_payload([])


def test_parse_chat_response_extracts_content() -> None:
    result = {"message": {"content": "  回答内容  "}}

    assert llm.parse_chat_response(result) == "回答内容"


def test_parse_chat_response_rejects_empty_content() -> None:
    with pytest.raises(llm.LLMError, match="空回答"):
        llm.parse_chat_response({"message": {"content": "   "}})


def test_chat_sends_payload_and_returns_content(monkeypatch: pytest.MonkeyPatch) -> None:
    captured: dict[str, object] = {}

    def fake_request(url: str, payload: dict, timeout: float) -> dict:
        captured["url"] = url
        captured["payload"] = payload
        captured["timeout"] = timeout
        return {"message": {"content": "模型回答"}}

    monkeypatch.setattr(llm, "_request_json", fake_request)

    answer = llm.chat([{"role": "user", "content": "你好"}], timeout=12.0)

    assert answer == "模型回答"
    assert captured["url"].endswith("/api/chat")
    assert captured["payload"]["think"] is False
    assert captured["timeout"] == 12.0


def test_chat_propagates_llm_error(monkeypatch: pytest.MonkeyPatch) -> None:
    def fake_request(url: str, payload: dict, timeout: float) -> dict:
        raise llm.LLMError("连接失败")

    monkeypatch.setattr(llm, "_request_json", fake_request)

    with pytest.raises(llm.LLMError, match="连接失败"):
        llm.chat([{"role": "user", "content": "你好"}])
