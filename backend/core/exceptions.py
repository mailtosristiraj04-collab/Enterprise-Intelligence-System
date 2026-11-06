"""Custom exception classes for the application."""

from typing import Any, Optional


class AppException(Exception):
    """Base application exception."""

    def __init__(
        self,
        message: str,
        status_code: int = 500,
        detail: Optional[Any] = None,
    ):
        self.message = message
        self.status_code = status_code
        self.detail = detail
        super().__init__(self.message)


class DocumentNotFoundError(AppException):
    """Raised when a document is not found."""

    def __init__(self, document_id: str):
        super().__init__(
            message=f"Document not found: {document_id}",
            status_code=404,
        )


class DocumentProcessingError(AppException):
    """Raised when document processing fails."""

    def __init__(self, message: str, detail: Optional[Any] = None):
        super().__init__(message=message, status_code=422, detail=detail)


class FileValidationError(AppException):
    """Raised when file validation fails."""

    def __init__(self, message: str):
        super().__init__(message=message, status_code=400)


class DuplicateDocumentError(AppException):
    """Raised when a document with the same content already exists."""

    def __init__(self, file_hash: str):
        super().__init__(
            message="A document with the same content has already been uploaded.",
            status_code=409,
            detail={"file_hash": file_hash},
        )


class UnsupportedFileTypeError(FileValidationError):
    """Raised for unsupported file types."""

    def __init__(self, filename: str, file_type: str):
        super().__init__(
            message=f"Unsupported file type '{file_type}' for file '{filename}'. "
            f"Supported types: PDF, DOC, DOCX, TXT"
        )


class FileTooLargeError(FileValidationError):
    """Raised when file exceeds size limit."""

    def __init__(self, filename: str, size_mb: float, max_mb: int):
        super().__init__(
            message=f"File '{filename}' ({size_mb:.1f} MB) exceeds maximum size of {max_mb} MB"
        )


class EmptyDocumentError(DocumentProcessingError):
    """Raised when a document contains no extractable text."""

    def __init__(self, filename: str):
        super().__init__(
            message=f"No extractable text found in '{filename}'. "
            f"The document may be empty, scanned (requiring OCR), or corrupted."
        )


class ExtractionDependencyError(DocumentProcessingError):
    """Raised when a required extraction dependency is missing."""

    def __init__(self, format_name: str, dependency: str, instructions: str):
        super().__init__(
            message=f"Cannot extract {format_name} files: {dependency} is not available.",
            detail={"instructions": instructions},
        )


class EmbeddingError(AppException):
    """Raised when embedding generation fails."""

    def __init__(self, message: str):
        super().__init__(message=message, status_code=500)


class VectorStoreError(AppException):
    """Raised when vector store operations fail."""

    def __init__(self, message: str):
        super().__init__(message=message, status_code=500)


class RetrievalError(AppException):
    """Raised when retrieval fails."""

    def __init__(self, message: str):
        super().__init__(message=message, status_code=500)


class LLMError(AppException):
    """Raised when LLM operations fail."""

    def __init__(self, message: str, detail: Optional[Any] = None):
        super().__init__(message=message, status_code=502, detail=detail)


class LLMRateLimitError(LLMError):
    """Raised when LLM rate limit is hit."""

    def __init__(self):
        super().__init__(
            message="LLM rate limit exceeded. Please try again in a moment.",
            detail={"retry_after_seconds": 60},
        )


class ConversationNotFoundError(AppException):
    """Raised when a conversation is not found."""

    def __init__(self, conversation_id: str):
        super().__init__(
            message=f"Conversation not found: {conversation_id}",
            status_code=404,
        )


class InvalidQueryError(AppException):
    """Raised for invalid queries."""

    def __init__(self, message: str):
        super().__init__(message=message, status_code=400)
