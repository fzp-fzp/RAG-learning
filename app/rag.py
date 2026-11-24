from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass

from app.config import MAX_CONTEXT_CHARS, TOP_K
from app.embeddings import embed_query
from app.llm import chat
from app.vector_store import SearchResult, VectorStore

REFUSAL_TEXT = "知识库中没有足够信息，无法根据现有资料回答。"

SYSTEM_PROMPT = """你是本地知识库问答助手。必须遵守以下规则：
1. 只根据用户提供的资料回答，不使用资料之外的知识补全。
2. 资料不足或没有相关资料时，直接回答“知识库中没有足够信息，无法根据现有资料回答。”，不要猜测。
3. 回答使用中文，尽量简洁。
4. 每条关键结论后用 [1]、[2] 标注资料编号；编号必须来自给定资料。
5. 不要编造来源、页码或引文。"""


@dataclass(frozen=True)
class RagAnswer:
    answer: str
    sources: list[SearchResult]
    retrieved_count: int
    context_chars: int


def build_context(
    results: list[SearchResult],
    max_chars: int = MAX_CONTEXT_CHARS,
) -> tuple[str, list[SearchResult]]:
    if max_chars <= 0:
        raise ValueError("max_chars 必须大于 0")

    lines: list[str] = []
    selected: list[SearchResult] = []
    for result in results:
        source = result.metadata.get("source", "(unknown)")
        chunk_index = result.metadata.get("chunk_index", "?")
        header = f"[{len(selected) + 1}] 来源：{source}；chunk：{chunk_index}"
        if "page" in result.metadata:
            header += f"；page：{result.metadata['page']}"
        block = f"{header}\n{result.text.strip()}"
        candidate = "\n\n".join([*lines, block])
        if len(candidate) <= max_chars:
            lines.append(block)
            selected.append(result)
            continue

        if not selected:
            truncated = block[:max_chars].rstrip()
            if truncated:
                lines.append(truncated)
                selected.append(result)
        break

    return "\n\n".join(lines), selected


def build_messages(question: str, context: str) -> list[dict[str, str]]:
    user_prompt = f"""资料：
{context}

问题：{question}

请根据以上资料回答。资料不足时明确说明，不要编造。"""
    return [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": user_prompt},
    ]


def answer_question(
    question: str,
    store: VectorStore,
    *,
    top_k: int = TOP_K,
    max_context_chars: int = MAX_CONTEXT_CHARS,
    embedder: Callable[[str], list[float]] = embed_query,
    chat_fn: Callable[[list[dict[str, str]]], str] = chat,
) -> RagAnswer:
    question = question.strip()
    if not question:
        raise ValueError("问题不能为空")

    query_embedding = embedder(question)
    results = store.search(query_embedding, top_k=top_k)
    context, selected = build_context(results, max_chars=max_context_chars)
    if not selected:
        return RagAnswer(
            answer=REFUSAL_TEXT,
            sources=[],
            retrieved_count=len(results),
            context_chars=0,
        )

    answer = chat_fn(build_messages(question, context)).strip()
    if not answer:
        raise ValueError("模型返回空回答")
    return RagAnswer(
        answer=answer,
        sources=selected,
        retrieved_count=len(results),
        context_chars=len(context),
    )
