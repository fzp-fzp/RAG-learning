from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from app.vector_store import SearchResult, StoredChunk


@dataclass(frozen=True)
class QuestionMetrics:
    question_id: str
    hit: int
    recall: float
    precision: float
    reciprocal_rank: float


def map_evidence_to_chunk_groups(
    question: dict[str, Any],
    records_by_source: dict[str, list[StoredChunk]],
) -> list[set[str]]:
    evidence_items = question.get("evidence") or []
    if not isinstance(evidence_items, list):
        raise ValueError(f"{question.get('id', '<unknown>')}: evidence 必须是列表")

    groups: list[set[str]] = []
    for evidence in evidence_items:
        if not isinstance(evidence, dict):
            raise ValueError(f"{question.get('id', '<unknown>')}: evidence 项必须是对象")
        source = evidence.get("source")
        contains = evidence.get("contains")
        if not isinstance(source, str) or not source:
            raise ValueError(f"{question.get('id', '<unknown>')}: evidence.source 不能为空")
        if not isinstance(contains, str) or not contains:
            raise ValueError(f"{question.get('id', '<unknown>')}: evidence.contains 不能为空")

        matches = {
            record.id for record in records_by_source.get(source, []) if contains in record.text
        }
        if not matches:
            raise ValueError(
                f"{question.get('id', '<unknown>')}: 证据没有匹配到任何文本块: {contains}"
            )
        groups.append(matches)

    if question.get("should_answer") and not groups:
        raise ValueError(f"{question.get('id', '<unknown>')}: 证据没有匹配到任何文本块")
    return groups


def score_question(question: dict[str, Any], retrieved: list[SearchResult]) -> QuestionMetrics:
    groups = [set(group) for group in question.get("relevant_chunk_groups") or []]
    if not groups:
        raise ValueError(f"{question.get('id', '<unknown>')}: 可回答问题的相关块不能为空")

    retrieved_ids = [result.id for result in retrieved]
    retrieved_set = set(retrieved_ids)
    relevant = set().union(*groups)
    matched_groups = [group for group in groups if group & retrieved_set]
    first_rank = next(
        (rank for rank, record_id in enumerate(retrieved_ids, start=1) if record_id in relevant),
        None,
    )

    return QuestionMetrics(
        question_id=str(question.get("id", "")),
        hit=1 if matched_groups else 0,
        recall=len(matched_groups) / len(groups),
        precision=len(retrieved_set & relevant) / len(retrieved_ids) if retrieved_ids else 0.0,
        reciprocal_rank=1.0 / first_rank if first_rank is not None else 0.0,
    )


def summarize_metrics(metrics: list[QuestionMetrics]) -> dict[str, float | int]:
    if not metrics:
        raise ValueError("没有可汇总的评估指标")

    count = len(metrics)
    return {
        "count": count,
        "hit": sum(metric.hit for metric in metrics) / count,
        "recall": sum(metric.recall for metric in metrics) / count,
        "precision": sum(metric.precision for metric in metrics) / count,
        "mrr": sum(metric.reciprocal_rank for metric in metrics) / count,
    }
