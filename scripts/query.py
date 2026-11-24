from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

PROJECT_DIR = Path(__file__).resolve().parents[1]
if str(PROJECT_DIR) not in sys.path:
    sys.path.insert(0, str(PROJECT_DIR))

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")
if hasattr(sys.stderr, "reconfigure"):
    sys.stderr.reconfigure(encoding="utf-8")

from app.config import MAX_CONTEXT_CHARS, TOP_K
from app.embeddings import EmbeddingError, embed_query
from app.llm import LLMError
from app.rag import answer_question
from app.vector_store import SearchResult, VectorStore


def format_result(result: SearchResult, rank: int) -> str:
    metadata = result.metadata
    source = metadata.get("source", "(unknown)")
    chunk_index = metadata.get("chunk_index", "?")
    location = f"source={source} chunk={chunk_index}"
    if "page" in metadata:
        location += f" page={metadata['page']}"

    text = result.text.replace("\n", "\n    ")
    return (
        f"[{rank}] similarity={result.similarity:.4f} "
        f"distance={result.distance:.4f} {location}\n    {text}"
    )


def main() -> int:
    parser = argparse.ArgumentParser(description="Query the local ChromaDB knowledge base.")
    parser.add_argument("query", nargs="?", help="Question to search for.")
    parser.add_argument(
        "--retrieval-only",
        action="store_true",
        help="Only retrieve chunks; do not call Qwen.",
    )
    parser.add_argument("--top-k", type=int, default=TOP_K, help="Number of chunks to return.")
    parser.add_argument(
        "--max-context-chars",
        type=int,
        default=MAX_CONTEXT_CHARS,
        help="Maximum characters of retrieved context sent to Qwen.",
    )
    args = parser.parse_args()

    query = (args.query or "").strip()
    if not query:
        query = input("问题: ").strip()
    if not query:
        print("问题不能为空。", file=sys.stderr)
        return 2

    try:
        with VectorStore() as store:
            if store.count() == 0:
                print("知识库为空，请先运行 scripts/ingest.py。", file=sys.stderr)
                return 1

            if args.retrieval_only:
                query_embedding = embed_query(query)
                results = store.search(query_embedding, top_k=args.top_k)
            else:
                started = time.perf_counter()
                rag_answer = answer_question(
                    query,
                    store=store,
                    top_k=args.top_k,
                    max_context_chars=args.max_context_chars,
                )
                elapsed = time.perf_counter() - started
    except (EmbeddingError, LLMError, ValueError, OSError) as exc:
        print(f"查询失败: {exc}", file=sys.stderr)
        return 1

    print(f"问题: {query}")
    print(f"Top-K: {args.top_k}")
    if not args.retrieval_only:
        print(f"检索块数: {rag_answer.retrieved_count}")
        print(f"上下文长度: {rag_answer.context_chars} 字符")
        print(f"耗时: {elapsed:.2f}s")
        print("\n回答:")
        print(rag_answer.answer)
        print("\n来源:")
        if not rag_answer.sources:
            print("  (无)")
        for rank, source in enumerate(rag_answer.sources, start=1):
            metadata = source.metadata
            location = (
                f"source={metadata.get('source', '(unknown)')} "
                f"chunk={metadata.get('chunk_index', '?')}"
            )
            if "page" in metadata:
                location += f" page={metadata['page']}"
            print(f"  [{rank}] {location}")
        return 0

    if not results:
        print("没有检索到结果。")
        return 0

    for rank, result in enumerate(results, start=1):
        print(format_result(result, rank))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
