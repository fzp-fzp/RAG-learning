from __future__ import annotations

import argparse
import json
import platform
import shutil
import subprocess
import sys
import time
from importlib import metadata
from pathlib import Path
from urllib.error import URLError
from urllib.request import ProxyHandler, Request, build_opener

PROJECT_DIR = Path(__file__).resolve().parents[1]
if str(PROJECT_DIR) not in sys.path:
    sys.path.insert(0, str(PROJECT_DIR))

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")
if hasattr(sys.stderr, "reconfigure"):
    sys.stderr.reconfigure(encoding="utf-8")

from app.config import (
    CHROMA_DIR,
    DATA_DIR,
    EMBEDDING_MODEL,
    LLM_MODEL,
    OLLAMA_BASE_URL,
    PROJECT_ROOT,
    UPLOAD_DIR,
)

OLLAMA_API = f"{OLLAMA_BASE_URL}/api"
OPENER = build_opener(ProxyHandler({}))


def request_json(path: str, payload: dict | None = None, timeout: float = 15.0) -> dict:
    data = None
    headers = {"Accept": "application/json"}
    if payload is not None:
        data = json.dumps(payload).encode("utf-8")
        headers["Content-Type"] = "application/json"

    request = Request(
        f"{OLLAMA_API}{path}",
        data=data,
        headers=headers,
        method="POST" if data else "GET",
    )
    with OPENER.open(request, timeout=timeout) as response:
        return json.loads(response.read().decode("utf-8"))


def print_header(title: str) -> None:
    print(f"\n[{title}]")


def has_model(models: list[str], name: str) -> bool:
    return name in models or f"{name}:latest" in models


def check_python() -> bool:
    print(f"Python: {platform.python_version()}")
    print(f"Executable: {sys.executable}")
    print(f"Project: {PROJECT_ROOT}")
    print(f"Platform: {platform.platform()}")
    return sys.version_info[:2] == (3, 12)


def check_packages() -> None:
    packages = [
        ("httpx", "httpx"),
        ("PyMuPDF", "fitz"),
        ("python-docx", "docx"),
        ("chromadb", "chromadb"),
        ("streamlit", "streamlit"),
        ("pytest", "pytest"),
    ]

    for distribution, module in packages:
        try:
            version = metadata.version(distribution)
        except metadata.PackageNotFoundError:
            print(f"{module:<22} not installed")
        else:
            print(f"{module:<22} {version}")


def check_ollama() -> tuple[bool, list[str]]:
    try:
        result = request_json("/tags")
    except (URLError, OSError, TimeoutError, json.JSONDecodeError) as exc:
        print(f"Ollama API: unreachable ({exc})")
        return False, []

    models = [item.get("name", "") for item in result.get("models", [])]
    print(f"Ollama API: {OLLAMA_BASE_URL}")
    print(f"Installed models: {', '.join(models) if models else 'none'}")

    found = has_model(models, LLM_MODEL)
    print(f"Required model: {LLM_MODEL} ({'found' if found else 'missing'})")
    return found, models


def check_model() -> bool:
    try:
        result = request_json("/show", {"model": LLM_MODEL}, timeout=30.0)
    except (URLError, OSError, TimeoutError, json.JSONDecodeError) as exc:
        print(f"Model metadata: unavailable ({exc})")
        return False

    details = result.get("details", {})
    model_info = result.get("model_info", {})
    capabilities = result.get("capabilities", [])
    context_length = next(
        (value for key, value in model_info.items() if key.endswith(".context_length")),
        "unknown",
    )

    print(f"Architecture: {details.get('family', 'unknown')}")
    print(f"Parameter size: {details.get('parameter_size', 'unknown')}")
    print(f"Quantization: {details.get('quantization_level', 'unknown')}")
    print(f"Max context: {context_length}")
    print(f"Capabilities: {', '.join(capabilities) if capabilities else 'unknown'}")
    return True


def check_gpu() -> bool:
    if shutil.which("nvidia-smi") is None:
        print("nvidia-smi not found")
        return False

    command = [
        "nvidia-smi",
        "--query-gpu=name,memory.total,memory.used,memory.free",
        "--format=csv,noheader,nounits",
    ]
    try:
        result = subprocess.run(command, capture_output=True, text=True, timeout=10, check=False)
    except (OSError, subprocess.SubprocessError) as exc:
        print(f"GPU query failed: {exc}")
        return False

    if result.returncode != 0:
        print(f"GPU query failed: {result.stderr.strip()}")
        return False

    print(f"GPU: {result.stdout.strip()}")
    return True


def check_paths() -> None:
    paths = [
        ("data", DATA_DIR),
        ("uploads", UPLOAD_DIR),
        ("chroma", CHROMA_DIR),
    ]
    for label, path in paths:
        state = "exists" if path.exists() else "not created yet"
        print(f"{label:<10} {path} ({state})")


def check_chat() -> bool:
    payload = {
        "model": LLM_MODEL,
        "messages": [{"role": "user", "content": "请只回复：模型可用"}],
        "stream": False,
        "think": False,
        "options": {"temperature": 0, "num_predict": 64},
    }
    started = time.perf_counter()
    try:
        result = request_json("/chat", payload, timeout=180.0)
    except (URLError, OSError, TimeoutError, json.JSONDecodeError) as exc:
        print(f"Chat request failed: {exc}")
        return False

    elapsed = time.perf_counter() - started
    message = result.get("message", {})
    answer = (message.get("content") or "").strip()
    thinking = (message.get("thinking") or "").strip()

    print(f"Latency: {elapsed:.2f}s")
    print(f"Answer: {answer or '(empty)'}")
    if thinking:
        print(f"Thinking chars: {len(thinking)}")
    return bool(answer)


def main() -> int:
    parser = argparse.ArgumentParser(description="Check the local RAG project environment.")
    parser.add_argument("--chat", action="store_true", help="Run a small Qwen chat request.")
    args = parser.parse_args()

    failures: list[str] = []

    print_header("Python")
    if not check_python():
        failures.append("Python is not 3.12")

    print_header("Packages")
    check_packages()

    print_header("Ollama")
    ollama_ok, models = check_ollama()
    if not ollama_ok:
        failures.append(f"Ollama model {LLM_MODEL} is unavailable")
    elif not check_model():
        failures.append("Model metadata is unavailable")

    embedding_found = has_model(models, EMBEDDING_MODEL)
    print(f"Embedding model: {EMBEDDING_MODEL} ({'found' if embedding_found else 'missing'})")
    if ollama_ok and not embedding_found:
        failures.append(f"Ollama embedding model {EMBEDDING_MODEL} is unavailable")

    print_header("GPU")
    if not check_gpu():
        failures.append("GPU query failed")

    print_header("Paths")
    check_paths()

    if args.chat:
        print_header("Chat")
        if not check_chat():
            failures.append("Chat check failed")

    print_header("Summary")
    if failures:
        for failure in failures:
            print(f"FAIL: {failure}")
        return 1

    print("OK: Environment checks passed")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
