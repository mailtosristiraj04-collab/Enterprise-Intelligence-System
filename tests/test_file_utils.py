from pathlib import Path

import pytest

from backend.core.exceptions import FileTooLargeError, FileValidationError
from backend.utils.file_utils import save_file, validate_file


def test_validate_file_accepts_supported_file() -> None:
    validate_file("report.PDF", b"content", max_file_size_bytes=100, allowed_extensions={".pdf"})


def test_validate_file_rejects_unsupported_extension() -> None:
    with pytest.raises(FileValidationError):
        validate_file("report.exe", b"content", max_file_size_bytes=100, allowed_extensions={".pdf"})


def test_validate_file_rejects_large_file() -> None:
    with pytest.raises(FileTooLargeError):
        validate_file("report.pdf", b"12345", max_file_size_bytes=4, allowed_extensions={".pdf"})


def test_validate_file_rejects_mismatched_mime_type() -> None:
    with pytest.raises(FileValidationError):
        validate_file(
            "report.pdf",
            b"content",
            max_file_size_bytes=100,
            allowed_extensions={".pdf"},
            content_type="text/plain",
        )


def test_save_file_generates_safe_name(tmp_path: Path) -> None:
    saved = save_file(b"content", tmp_path, "../report?.pdf")

    assert saved.parent == tmp_path
    assert saved.suffix == ".pdf"
    assert saved.read_bytes() == b"content"
