from __future__ import annotations

from collections.abc import Mapping
from pathlib import Path
from typing import Any

from app.config import UPLOAD_DIR
from app.loaders import SUPPORTED_SUFFIXES

MAX_UPLOAD_BYTES = 50 * 1024 * 1024


class UploadError(ValueError):
    """Raised when an uploaded file cannot be accepted safely."""


def safe_upload_name(filename: str) -> str:
    name = Path(str(filename).replace("\\", "/")).name.strip()
    if not name or name in {".", ".."} or any(ord(char) < 32 for char in name):
        raise UploadError("文件名无效")

    suffix = Path(name).suffix.lower()
    if suffix not in SUPPORTED_SUFFIXES:
        supported = "、".join(sorted(SUPPORTED_SUFFIXES))
        raise UploadError(f"不支持的文件类型；仅支持 {supported}")
    return name


def save_upload(
    filename: str,
    data: bytes,
    *,
    upload_dir: str | Path = UPLOAD_DIR,
    max_bytes: int = MAX_UPLOAD_BYTES,
) -> Path:
    name = safe_upload_name(filename)
    if not data:
        raise UploadError("文件为空")
    if len(data) > max_bytes:
        raise UploadError(f"文件过大；上限为 {max_bytes // (1024 * 1024)} MB")

    directory = Path(upload_dir)
    directory.mkdir(parents=True, exist_ok=True)
    path = directory / name
    if path.resolve().parent != directory.resolve():
        raise UploadError("文件名无效")

    path.write_bytes(data)
    return path


def format_source_location(source: Mapping[str, Any]) -> str:
    location = f"{source.get('source', '(unknown)')} · chunk {source.get('chunk_index', '?')}"
    if source.get("page") is not None:
        location += f" · 第 {source['page']} 页"
    return location
