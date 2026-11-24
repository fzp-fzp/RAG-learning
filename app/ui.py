from __future__ import annotations

import sys
import time
from collections.abc import Callable
from pathlib import Path
from typing import Any

PROJECT_DIR = Path(__file__).resolve().parents[1]
if str(PROJECT_DIR) not in sys.path:
    sys.path.insert(0, str(PROJECT_DIR))

import streamlit as st

from app.config import MAX_CONTEXT_CHARS, TOP_K, UPLOAD_DIR
from app.embeddings import EmbeddingError
from app.ingestion import IngestReport, ingest_documents
from app.llm import LLMError
from app.rag import RagAnswer, answer_question
from app.ui_helpers import UploadError, format_source_location, save_upload
from app.vector_store import SearchResult, VectorStore

AnswerFn = Callable[..., RagAnswer]
IngestFn = Callable[..., IngestReport]


def main(
    *,
    chroma_path: str | Path | None = None,
    collection_name: str | None = None,
    answer_fn: AnswerFn = answer_question,
    ingest_fn: IngestFn = ingest_documents,
) -> None:
    st.set_page_config(page_title="本地知识库助手", page_icon="📚", layout="wide")
    messages: list[dict[str, Any]] = st.session_state.setdefault("messages", [])
    notice = st.session_state.pop("kb_notice", None)

    try:
        store = VectorStore(path=chroma_path, collection_name=collection_name)
    except Exception as exc:
        st.title("本地知识库助手")
        st.error(f"无法打开知识库：{exc}")
        return

    try:
        with st.sidebar:
            st.header("知识库")
            if notice:
                level, message = notice
                getattr(st, level)(message)
            _render_uploader(store, ingest_fn)
            st.divider()
            _render_sources(store)

        st.title("本地知识库助手")
        st.caption("Qwen3-8B · BGE-M3 · ChromaDB · 全部在本机运行")
        _render_chat(store, messages, answer_fn)
    finally:
        store.close()


def _render_uploader(store: VectorStore, ingest_fn: IngestFn) -> None:
    st.caption("支持 Word、PDF、TXT。同名文件会替换旧索引。")
    uploaded_files = st.file_uploader(
        "选择文档",
        type=["docx", "pdf", "txt"],
        accept_multiple_files=True,
    )
    if st.button("导入到知识库", disabled=not uploaded_files, use_container_width=True):
        saved: dict[str, Path] = {}
        errors: list[str] = []
        for uploaded in uploaded_files or []:
            try:
                path = save_upload(uploaded.name, uploaded.getvalue(), upload_dir=UPLOAD_DIR)
            except UploadError as exc:
                errors.append(f"{uploaded.name}：{exc}")
            else:
                saved[path.name] = path

        for error in errors:
            st.error(error)
        if not saved:
            return

        with st.spinner("正在解析文档、切分文本并生成向量..."):
            try:
                report = ingest_fn(list(saved.values()), store=store)
            except (ValueError, OSError) as exc:
                st.error(f"导入失败：{exc}")
                return

        _show_ingest_report(report)


def _show_ingest_report(report: IngestReport) -> None:
    if report.ingested:
        details = "、".join(f"{source}（{chunks} 块）" for source, chunks in report.ingested)
        st.success(f"已导入：{details}")
    for source, error in report.failures:
        st.error(f"{source} 导入失败：{error}")


def _render_sources(store: VectorStore) -> None:
    st.subheader("已导入文档")
    counts = store.source_counts()
    if not counts:
        st.info("知识库还没有文档。上传文档并点击“导入到知识库”。")
        return

    st.caption(f"{len(counts)} 份文档 · {sum(counts.values())} 个文本块")
    for source, count in counts.items():
        name_column, action_column = st.columns([0.72, 0.28])
        with name_column:
            st.markdown(f"**{source}**")
            st.caption(f"{count} 个文本块")
        with action_column:
            if st.button("删除", key=f"delete:{source}", use_container_width=True):
                try:
                    deleted = store.delete_by_source(source)
                except Exception as exc:
                    st.error(f"删除失败：{exc}")
                else:
                    st.session_state["kb_notice"] = (
                        "success",
                        f"已删除 {source}（{deleted} 块）",
                    )
                    st.rerun()


def _render_chat(store: VectorStore, messages: list[dict[str, Any]], answer_fn: AnswerFn) -> None:
    st.subheader("对话")
    info_column, action_column = st.columns([0.72, 0.28])
    with info_column:
        st.caption(f"Top-K {TOP_K} · 上下文上限 {MAX_CONTEXT_CHARS} 字 · 回答只依据检索到的资料")
    with action_column:
        if st.button("清空聊天", disabled=not messages, use_container_width=True):
            messages.clear()
            st.rerun()

    if not messages:
        st.info("先导入文档，然后在下方提问。知识库为空时不会调用模型。")

    for message in messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])
            _render_message_sources(message.get("sources") or [])
            if message.get("meta"):
                st.caption(message["meta"])

    prompt = st.chat_input("根据知识库提问...")
    if not prompt:
        return

    messages.append({"role": "user", "content": prompt, "sources": []})
    if store.count() == 0:
        messages.append(
            {
                "role": "assistant",
                "content": "知识库为空，请先在左侧上传并导入文档。",
                "sources": [],
            }
        )
        st.rerun()

    started = time.perf_counter()
    with st.spinner("正在检索资料并生成回答..."):
        try:
            result = answer_fn(prompt, store=store)
        except (EmbeddingError, LLMError, ValueError, OSError) as exc:
            elapsed = time.perf_counter() - started
            messages.append(
                {
                    "role": "assistant",
                    "content": (
                        f"问答失败：{exc}\n\n"
                        "请确认 Ollama 已启动，且 `qwen3:8b` 与 `bge-m3-cpu` 可用。"
                    ),
                    "sources": [],
                    "meta": f"耗时 {elapsed:.2f}s",
                }
            )
        else:
            elapsed = time.perf_counter() - started
            messages.append(
                {
                    "role": "assistant",
                    "content": result.answer,
                    "sources": [_serialize_source(source) for source in result.sources],
                    "meta": (
                        f"检索 {result.retrieved_count} 块 · 使用 {len(result.sources)} 块 · "
                        f"上下文 {result.context_chars} 字 · 耗时 {elapsed:.2f}s"
                    ),
                }
            )
    st.rerun()


def _render_message_sources(sources: list[dict[str, Any]]) -> None:
    if not sources:
        return
    with st.expander(f"查看来源（{len(sources)} 条）"):
        for rank, source in enumerate(sources, start=1):
            similarity = source.get("similarity")
            heading = f"**[{rank}] {format_source_location(source)}**"
            if isinstance(similarity, (int, float)):
                heading += f" · 相似度 {similarity:.3f}"
            st.markdown(heading)
            st.text(source.get("text", ""))


def _serialize_source(source: SearchResult) -> dict[str, Any]:
    return {
        "source": source.metadata.get("source", "(unknown)"),
        "chunk_index": source.metadata.get("chunk_index", "?"),
        "page": source.metadata.get("page"),
        "similarity": source.similarity,
        "text": source.text,
    }


if __name__ == "__main__":
    main()
