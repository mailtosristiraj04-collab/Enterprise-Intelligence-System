"""File utility functions."""

import hashlib
import os
import re
import uuid
from pathlib import Path

from backend.core.exceptions import FileTooLargeError, FileValidationError, UnsupportedFileTypeError


def generate_safe_filename(original_filename: str) -> str:
    """Generate a safe filename preserving the extension.

    Args:
        original_filename: The original filename from the upload.

    Returns:
        A safe filename with UUID prefix and sanitized name.
    """
    path = Path(original_filename)
    extension = path.suffix.lower()

    stem = path.stem
    stem = os.path.basename(stem)
    stem = re.sub(r'[^\w\s\-.]', '', stem)
    stem = re.sub(r'\s+', '_', stem).strip('_')
    stem = stem[:100] if stem else "document"

    unique_id = uuid.uuid4().hex[:12]
    return f"{unique_id}_{stem}{extension}"


def compute_file_hash(file_content: bytes) -> str:
    """Compute SHA-256 hash of file content.

    Args:
        file_content: Raw file bytes.

    Returns:
        Hex digest of the SHA-256 hash.
    """
    return hashlib.sha256(file_content).hexdigest()


def get_file_extension(filename: str) -> str:
    """Get lowercase file extension.

    Args:
        filename: The filename.

    Returns:
        Lowercase extension including the dot (e.g., '.pdf').
    """
    return Path(filename).suffix.lower()


def format_file_size(size_bytes: int) -> str:
    """Format file size in human-readable format.

    Args:
        size_bytes: Size in bytes.

    Returns:
        Human-readable size string.
    """
    for unit in ['B', 'KB', 'MB', 'GB']:
        if size_bytes < 1024:
            return f"{size_bytes:.1f} {unit}"
        size_bytes /= 1024
    return f"{size_bytes:.1f} TB"


def estimate_tokens(text: str) -> int:
    """Rough token estimation (~4 chars per token for English).

    Args:
        text: Input text.

    Returns:
        Estimated token count.
    """
    return max(1, len(text) // 4)


def validate_file(
    filename: str,
    file_content: bytes,
    *,
    max_file_size_bytes: int,
    allowed_extensions: set[str],
    content_type: str | None = None,
) -> None:
    """Validate an uploaded file before it is written to disk."""
    if not filename or not Path(filename).name:
        raise FileValidationError("A filename is required.")
    if not file_content:
        raise FileValidationError("The uploaded file is empty.")
    extension = get_file_extension(filename)
    if extension not in {item.lower() for item in allowed_extensions}:
        raise UnsupportedFileTypeError(filename, extension or "[none]")
    expected_types = {
        ".pdf": {"application/pdf"},
        ".doc": {"application/msword", "application/octet-stream"},
        ".docx": {
            "application/vnd.openxmlformats-officedocument.wordprocessingml.document",
            "application/octet-stream",
        },
        ".txt": {"text/plain", "application/octet-stream"},
    }
    if content_type and content_type not in expected_types.get(extension, set()):
        raise FileValidationError(
            f"The MIME type '{content_type}' does not match '{extension}'."
        )
    if len(file_content) > max_file_size_bytes:
        raise FileTooLargeError(
            filename,
            len(file_content) / (1024 * 1024),
            max_file_size_bytes // (1024 * 1024),
        )


def save_file(file_content: bytes, directory: Path, filename: str) -> Path:
    """Write content using a generated safe filename and return its path."""
    directory.mkdir(parents=True, exist_ok=True)
    safe_filename = generate_safe_filename(filename)
    destination = (directory / safe_filename).resolve()
    if destination.parent != directory.resolve():
        raise FileValidationError("The destination path is outside the upload directory.")
    destination.write_bytes(file_content)
    return destination
