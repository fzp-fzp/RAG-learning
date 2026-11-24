from __future__ import annotations

from pathlib import Path

import pytest

from app.ui_helpers import UploadError, safe_upload_name, save_upload


def test_safe_upload_name_removes_directories() -> None:
    assert safe_upload_name(r"..\..\临时\规则.docx") == "规则.docx"
    assert safe_upload_name("/tmp/规则.pdf") == "规则.pdf"


@pytest.mark.parametrize("filename", ["", "   ", ".", ".."])
def test_safe_upload_name_rejects_empty_names(filename: str) -> None:
    with pytest.raises(UploadError, match="文件名"):
        safe_upload_name(filename)


def test_safe_upload_name_rejects_unsupported_suffix() -> None:
    with pytest.raises(UploadError, match="支持"):
        safe_upload_name("恶意.exe")


def test_safe_upload_name_rejects_control_characters() -> None:
    with pytest.raises(UploadError, match="文件名"):
        safe_upload_name("规则\x00.txt")


def test_save_upload_writes_file_inside_upload_dir(tmp_path: Path) -> None:
    path = save_upload("规则.txt", "测试内容".encode(), upload_dir=tmp_path)

    assert path == tmp_path / "规则.txt"
    assert path.read_text(encoding="utf-8") == "测试内容"


def test_save_upload_rejects_empty_file(tmp_path: Path) -> None:
    with pytest.raises(UploadError, match="空"):
        save_upload("规则.txt", b"", upload_dir=tmp_path)


def test_save_upload_rejects_oversized_file(tmp_path: Path) -> None:
    with pytest.raises(UploadError, match="过大"):
        save_upload("规则.txt", b"x" * 11, upload_dir=tmp_path, max_bytes=10)
