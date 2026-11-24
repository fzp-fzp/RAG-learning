from __future__ import annotations

import argparse
import json
import sys
import tempfile
import time
from pathlib import Path
from typing import Any

PROJECT_DIR = Path(__file__).resolve().parents[1]
if str(PROJECT_DIR) not in sys.path:
    sys.path.insert(0, str(PROJECT_DIR))

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")
if hasattr(sys.stderr, "reconfigure"):
    sys.stderr.reconfigure(encoding="utf-8")

from app.chunking import TextChunk, chunk_sections
from app.config import CHUNK_OVERLAP, CHUNK_SIZE, PROJECT_ROOT, TOP_K
from app.embeddings import embed_texts
from app.loaders import load_document
from app.retrieval_eval import (
    map_evidence_to_chunk_groups,
    score_question,
    summarize_metrics,
)
from app.vector_store import StoredChunk, VectorStore

DEFAULT_DATASET = PROJECT_ROOT / "data" / "eval" / "mianyang_synthetic_v1"


def load_chunks(
    dataset_dir: Path,
    documents: list[dict[str, Any]],
    chunk_size: int,
    overlap: int,
) -> dict[str, list[TextChunk]]:
    chunks_by_source: dict[str, list[TextChunk]] = {}
    for document in documents:
        path = dataset_dir / document["path"]
        sections = load_document(path)
        chunks = chunk_sections(sections, chunk_size=chunk_size, overlap=overlap)
        if not chunks:
            raise ValueError(f"{path.name}: 没有生成文本块")
        chunks_by_source[path.name] = chunks
    return chunks_by_source


def group_records_by_source(records: list[StoredChunk]) -> dict[str, list[StoredChunk]]:
    grouped: dict[str, list[StoredChunk]] = {}
    for record in records:
        source = record.metadata.get("source")
        if isinstance(source, str):
            grouped.setdefault(source, []).append(record)
    return grouped


def summarize_similarities(values: list[float]) -> dict[str, float | int]:
    if not values:
        return {"count": 0}
    return {
        "count": len(values),
        "mean": sum(values) / len(values),
        "min": min(values),
        "max": max(values),
    }


def build_result_item(
    question: dict[str, Any],
    results: list[Any],
    metrics: Any | None,
    relevant_groups: list[set[str]] | None,
) -> dict[str, Any]:
    top1_similarity = results[0].similarity if results else None
    item: dict[str, Any] = {
        "id": question["id"],
        "type": question["type"],
        "question": question["question"],
        "should_answer": question["should_answer"],
        "retrieved_ids": [result.id for result in results],
        "retrieved_sources": [result.metadata.get("source") for result in results],
        "top1_similarity": top1_similarity,
    }
    if metrics is None:
        item["metrics"] = None
        return item

    assert relevant_groups is not None
    item["relevant_chunk_groups"] = [sorted(group) for group in relevant_groups]
    item["relevant_chunk_ids"] = sorted(set().union(*relevant_groups))
    item["metrics"] = {
        "hit": metrics.hit,
        "recall": metrics.recall,
        "precision": metrics.precision,
        "reciprocal_rank": metrics.reciprocal_rank,
    }
    item["miss"] = metrics.hit == 0 or metrics.recall < 1.0
    return item


def main() -> int:
    parser = argparse.ArgumentParser(description="Evaluate retrieval on a local dataset.")
    parser.add_argument("--dataset", type=Path, default=DEFAULT_DATASET)
    parser.add_argument("--top-k", type=int, default=TOP_K)
    parser.add_argument("--chunk-size", type=int, default=CHUNK_SIZE)
    parser.add_argument("--overlap", type=int, default=CHUNK_OVERLAP)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()

    dataset_dir = args.dataset.resolve()
    questions_path = dataset_dir / "questions.json"
    output_path = args.output or dataset_dir / "eval_results.json"

    try:
        payload = json.loads(questions_path.read_text(encoding="utf-8"))
        documents = payload["documents"]
        questions = payload["questions"]
        chunks_by_source = load_chunks(
            dataset_dir,
            documents,
            chunk_size=args.chunk_size,
            overlap=args.overlap,
        )

        started = time.perf_counter()
        with (
            tempfile.TemporaryDirectory(prefix="rag_retrieval_eval_") as temp_dir,
            VectorStore(
                path=Path(temp_dir) / "chroma",
                collection_name="retrieval_eval",
            ) as store,
        ):
            for chunks in chunks_by_source.values():
                embeddings = embed_texts([chunk.text for chunk in chunks])
                store.add_chunks(chunks, embeddings)

            records_by_source = group_records_by_source(store.get_records())
            query_embeddings = embed_texts([question["question"] for question in questions])

            answerable_metrics = []
            answerable_top1: list[float] = []
            unanswerable_top1: list[float] = []
            per_question: list[dict[str, Any]] = []

            for question, query_embedding in zip(questions, query_embeddings, strict=True):
                results = store.search(query_embedding, top_k=args.top_k)
                if question["should_answer"]:
                    relevant_groups = map_evidence_to_chunk_groups(question, records_by_source)
                    question_for_score = {
                        **question,
                        "relevant_chunk_groups": [sorted(group) for group in relevant_groups],
                    }
                    metrics = score_question(question_for_score, results)
                    answerable_metrics.append(metrics)
                    if results:
                        answerable_top1.append(results[0].similarity)
                    per_question.append(
                        build_result_item(question, results, metrics, relevant_groups)
                    )
                else:
                    if results:
                        unanswerable_top1.append(results[0].similarity)
                    per_question.append(build_result_item(question, results, None, None))

        summary = summarize_metrics(answerable_metrics)
        failures = [
            item
            for item in per_question
            if item["should_answer"] and item["metrics"] is not None and item["miss"]
        ]
        elapsed = time.perf_counter() - started
        result = {
            "dataset": payload.get("dataset"),
            "fictional": payload.get("fictional"),
            "top_k": args.top_k,
            "chunk_size": args.chunk_size,
            "overlap": args.overlap,
            "documents": len(documents),
            "chunks": sum(len(chunks) for chunks in chunks_by_source.values()),
            "questions": len(questions),
            "answerable_questions": len(answerable_metrics),
            "unanswerable_questions": len(questions) - len(answerable_metrics),
            "elapsed_seconds": elapsed,
            "metrics": {
                f"hit@{args.top_k}": summary["hit"],
                f"recall@{args.top_k}": summary["recall"],
                f"precision@{args.top_k}": summary["precision"],
                f"mrr@{args.top_k}": summary["mrr"],
            },
            "answerable_top1_similarity": summarize_similarities(answerable_top1),
            "unanswerable_top1_similarity": summarize_similarities(unanswerable_top1),
            "failure_count": len(failures),
            "failures": failures,
            "per_question": per_question,
        }
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(
            json.dumps(result, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
    except (KeyError, OSError, ValueError, json.JSONDecodeError) as exc:
        print(f"评估失败: {exc}", file=sys.stderr)
        return 1

    print(f"Dataset: {dataset_dir}")
    print(f"Chunks: {result['chunks']}")
    print(f"Questions: {result['questions']} (answerable={result['answerable_questions']})")
    for name, value in result["metrics"].items():
        print(f"{name}: {value:.4f}")
    print(f"elapsed: {elapsed:.2f}s")
    print(f"failure_count: {result['failure_count']}")
    print(f"Results: {output_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
