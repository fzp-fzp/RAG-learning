from __future__ import annotations

import subprocess
import sys
from pathlib import Path

from streamlit.testing.v1 import AppTest

from app.chunking import TextChunk
from app.llm import LLMError
from app.rag import RagAnswer
from app.vector_store import SearchResult, VectorStore

UI_PATH = Path(__file__).resolve().parents[1] / "app" / "ui.py"
PROJECT_ROOT = UI_PATH.parents[1]


def test_ui_script_can_import_app_when_app_dir_is_first_on_path() -> None:
    code = f"""
import runpy
import sys

project_root = r"{PROJECT_ROOT}"
app_dir = r"{UI_PATH.parent}"
sys.path = [path for path in sys.path if path not in ("", project_root)]
sys.path.insert(0, app_dir)
runpy.run_path(r"{UI_PATH}", run_name="ui_import_test")
"""

    result = subprocess.run(
        [sys.executable, "-c", code],
        cwd=PROJECT_ROOT,
        capture_output=True,
        check=False,
    )

    assert result.returncode == 0, result.stderr.decode("utf-8", errors="replace")


def test_ui_renders_empty_knowledge_base(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.setattr("app.vector_store.CHROMA_DIR", tmp_path / "chroma")
    monkeypatch.setattr("app.vector_store.COLLECTION_NAME", "test_ui_empty")

    at = AppTest.from_file(UI_PATH).run()

    assert not at.exception
    assert at.title[0].value == "本地知识库助手"
    assert any("还没有文档" in item.value for item in at.info)


def test_ui_chat_appends_answer_and_sources(tmp_path: Path, monkeypatch) -> None:
    path = tmp_path / "chroma"
    chunk = TextChunk(
        text="补考最高按 60 分记入成绩单。",
        source="考试.txt",
        file_type="txt",
        page=None,
        paragraph=None,
        chunk_index=0,
        start_char=0,
        end_char=15,
    )
    with VectorStore(path=path, collection_name="test_ui_chat") as store:
        store.add_chunks([chunk], [[1.0, 0.0]])

    def fake_answer(question: str, store: VectorStore, **kwargs) -> RagAnswer:
        return RagAnswer(
            answer="补考最高按 60 分记入成绩单。[1]",
            sources=[
                SearchResult(
                    id="source-1",
                    text=chunk.text,
                    metadata={"source": chunk.source, "chunk_index": 0},
                    distance=0.1,
                    similarity=0.9,
                )
            ],
            retrieved_count=1,
            context_chars=20,
        )

    monkeypatch.setattr("app.vector_store.CHROMA_DIR", path)
    monkeypatch.setattr("app.vector_store.COLLECTION_NAME", "test_ui_chat")
    monkeypatch.setattr("app.rag.answer_question", fake_answer)

    at = AppTest.from_file(UI_PATH).run()
    at.chat_input[0].set_value("补考成绩怎么算？").run()

    assert not at.exception
    messages = at.session_state["messages"]
    assert [message["role"] for message in messages] == ["user", "assistant"]
    assert messages[0]["content"] == "补考成绩怎么算？"
    assert messages[1]["content"].startswith("补考最高")
    assert messages[1]["sources"][0]["source"] == "考试.txt"


def test_ui_shows_actionable_error_when_ollama_fails(tmp_path: Path, monkeypatch) -> None:
    path = tmp_path / "chroma"
    chunk = TextChunk(
        text="任意内容",
        source="测试.txt",
        file_type="txt",
        page=None,
        paragraph=None,
        chunk_index=0,
        start_char=0,
        end_char=4,
    )
    with VectorStore(path=path, collection_name="test_ui_error") as store:
        store.add_chunks([chunk], [[1.0, 0.0]])

    def fail_answer(question: str, store: VectorStore, **kwargs) -> RagAnswer:
        raise LLMError("connection refused")

    monkeypatch.setattr("app.vector_store.CHROMA_DIR", path)
    monkeypatch.setattr("app.vector_store.COLLECTION_NAME", "test_ui_error")
    monkeypatch.setattr("app.rag.answer_question", fail_answer)

    at = AppTest.from_file(UI_PATH).run()
    at.chat_input[0].set_value("测试问题").run()

    assert not at.exception
    message = at.session_state["messages"][-1]
    assert message["role"] == "assistant"
    assert "请确认 Ollama 已启动" in message["content"]
    assert message["sources"] == []


def test_ui_delete_source_removes_its_chunks(tmp_path: Path, monkeypatch) -> None:
    path = tmp_path / "chroma"
    chunk = TextChunk(
        text="待删除内容",
        source="待删除.txt",
        file_type="txt",
        page=None,
        paragraph=None,
        chunk_index=0,
        start_char=0,
        end_char=5,
    )
    with VectorStore(path=path, collection_name="test_ui_delete") as store:
        store.add_chunks([chunk], [[1.0, 0.0]])

    monkeypatch.setattr("app.vector_store.CHROMA_DIR", path)
    monkeypatch.setattr("app.vector_store.COLLECTION_NAME", "test_ui_delete")

    at = AppTest.from_file(UI_PATH).run()
    delete_button = next(button for button in at.button if button.label == "删除")
    delete_button.click().run()

    assert not at.exception
    with VectorStore(path=path, collection_name="test_ui_delete") as store:
        assert store.count() == 0
