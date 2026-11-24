from __future__ import annotations

import pytest

from app.retrieval_eval import map_evidence_to_chunk_groups, score_question, summarize_metrics
from app.vector_store import SearchResult, StoredChunk


def make_record(record_id: str, text: str, source: str = "a.txt") -> StoredChunk:
    return StoredChunk(id=record_id, text=text, metadata={"source": source})


def make_result(result_id: str, text: str = "") -> SearchResult:
    return SearchResult(
        id=result_id,
        text=text,
        metadata={},
        distance=0.0,
        similarity=1.0,
    )


def test_map_evidence_to_chunk_groups_keeps_one_group_per_evidence() -> None:
    question = {
        "id": "q1",
        "should_answer": True,
        "evidence": [{"source": "a.txt", "contains": "关键规则"}],
    }
    records_by_source = {
        "a.txt": [
            make_record("c1", "前文 关键规则 后文"),
            make_record("c2", "另一个 关键规则 块"),
            make_record("c3", "无关内容"),
        ]
    }

    assert map_evidence_to_chunk_groups(question, records_by_source) == [{"c1", "c2"}]


def test_map_evidence_raises_when_answerable_evidence_is_missing() -> None:
    question = {
        "id": "q1",
        "should_answer": True,
        "evidence": [{"source": "a.txt", "contains": "不存在的规则"}],
    }

    with pytest.raises(ValueError, match="没有匹配"):
        map_evidence_to_chunk_groups(question, {"a.txt": [make_record("c1", "其他内容")]})


def test_score_question_hit_recall_precision_and_mrr() -> None:
    question = {
        "id": "q1",
        "should_answer": True,
        "relevant_chunk_groups": [["c2"], ["c3"]],
    }
    retrieved = [make_result("c1"), make_result("c2"), make_result("c3")]

    metrics = score_question(question, retrieved)

    assert metrics.hit == 1
    assert metrics.recall == pytest.approx(1.0)
    assert metrics.precision == pytest.approx(2 / 3)
    assert metrics.reciprocal_rank == pytest.approx(0.5)


def test_score_question_returns_zero_for_miss() -> None:
    question = {"id": "q1", "should_answer": True, "relevant_chunk_groups": [["c9"]]}
    retrieved = [make_result("c1"), make_result("c2")]

    metrics = score_question(question, retrieved)

    assert metrics.hit == 0
    assert metrics.recall == pytest.approx(0.0)
    assert metrics.precision == pytest.approx(0.0)
    assert metrics.reciprocal_rank == pytest.approx(0.0)


def test_summarize_metrics_averages_answerable_questions() -> None:
    metrics = [
        score_question(
            {"id": "q1", "should_answer": True, "relevant_chunk_groups": [["c2"]]},
            [make_result("c1"), make_result("c2")],
        ),
        score_question(
            {"id": "q2", "should_answer": True, "relevant_chunk_groups": [["c9"]]},
            [make_result("c1")],
        ),
    ]

    summary = summarize_metrics(metrics)

    assert summary["count"] == 2
    assert summary["hit"] == pytest.approx(0.5)
    assert summary["recall"] == pytest.approx(0.5)
    assert summary["precision"] == pytest.approx(0.25)
    assert summary["mrr"] == pytest.approx(0.25)


def test_overlapping_chunks_for_one_evidence_count_as_one_recall() -> None:
    question = {
        "id": "q1",
        "should_answer": True,
        "relevant_chunk_groups": [["c1", "c2"]],
    }

    metrics = score_question(question, [make_result("c1")])

    assert metrics.hit == 1
    assert metrics.recall == pytest.approx(1.0)
