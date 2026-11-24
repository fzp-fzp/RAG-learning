from __future__ import annotations

import pytest

from app.rag import REFUSAL_TEXT, answer_question, build_context, build_messages
from app.vector_store import SearchResult


def make_result(
    text: str,
    source: str = "a.txt",
    chunk_index: int = 0,
    similarity: float = 0.9,
) -> SearchResult:
    return SearchResult(
        id=f"{source}:{chunk_index}",
        text=text,
        metadata={"source": source, "chunk_index": chunk_index},
        distance=1.0 - similarity,
        similarity=similarity,
    )


class FakeStore:
    def __init__(self, results: list[SearchResult]) -> None:
        self.results = results
        self.calls: list[tuple[list[float], int]] = []

    def search(self, query_embedding: list[float], top_k: int = 5) -> list[SearchResult]:
        self.calls.append((query_embedding, top_k))
        return self.results[:top_k]


def test_build_context_numbers_sources_and_preserves_text() -> None:
    results = [
        make_result("第一条规则", source="规则.txt", chunk_index=0),
        make_result("第二条规则", source="规则.txt", chunk_index=1),
    ]

    context, selected = build_context(results, max_chars=200)

    assert "[1]" in context
    assert "[2]" in context
    assert "规则.txt" in context
    assert "第一条规则" in context
    assert "第二条规则" in context
    assert [item.text for item in selected] == ["第一条规则", "第二条规则"]


def test_build_context_stops_before_exceeding_budget() -> None:
    results = [
        make_result("A" * 80),
        make_result("B" * 80),
    ]

    context, selected = build_context(results, max_chars=120)

    assert len(selected) == 1
    assert "A" in context
    assert "B" not in context


def test_build_context_truncates_first_block_when_needed() -> None:
    results = [make_result("X" * 100)]

    context, selected = build_context(results, max_chars=40)

    assert len(selected) == 1
    assert len(context) == 40


def test_build_messages_contains_rules_question_and_context() -> None:
    context, _ = build_context([make_result("资料内容")], max_chars=200)

    messages = build_messages("我该怎么做？", context)

    assert messages[0]["role"] == "system"
    assert "只根据" in messages[0]["content"]
    assert "资料不足" in messages[0]["content"]
    assert "我该怎么做？" in messages[1]["content"]
    assert "资料内容" in messages[1]["content"]


def test_answer_question_returns_refusal_without_results() -> None:
    store = FakeStore([])

    def fail_chat(messages: list[dict[str, str]]) -> str:
        raise AssertionError("没有资料时不应调用模型")

    result = answer_question(
        "知识库里有吗？",
        store=store,
        embedder=lambda text: [1.0, 0.0],
        chat_fn=fail_chat,
    )

    assert result.answer == REFUSAL_TEXT
    assert result.sources == []
    assert result.retrieved_count == 0


def test_answer_question_uses_context_and_returns_sources() -> None:
    store = FakeStore([make_result("补考最高 60 分", source="考试.txt", chunk_index=1)])
    captured: dict[str, object] = {}

    def fake_chat(messages: list[dict[str, str]]) -> str:
        captured["messages"] = messages
        return "补考最高按 60 分记入成绩单。[1]"

    result = answer_question(
        "补考成绩怎么算？",
        store=store,
        top_k=3,
        embedder=lambda text: [1.0, 0.0],
        chat_fn=fake_chat,
    )

    assert result.answer.startswith("补考最高")
    assert result.sources[0].text == "补考最高 60 分"
    assert result.retrieved_count == 1
    assert store.calls[0][1] == 3
    messages = captured["messages"]
    assert isinstance(messages, list)
    assert "补考成绩怎么算？" in messages[1]["content"]
    assert "补考最高 60 分" in messages[1]["content"]


def test_answer_question_rejects_empty_question() -> None:
    with pytest.raises(ValueError, match="问题"):
        answer_question(
            "   ",
            store=FakeStore([]),
            embedder=lambda text: [1.0, 0.0],
            chat_fn=lambda messages: "不应调用",
        )
