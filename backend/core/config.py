"""Centralized application configuration."""

from functools import lru_cache
from pathlib import Path

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """Application settings loaded from environment variables."""

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
        extra="ignore",
    )

    # Project root (two levels up from this file: core/config.py -> core -> backend -> project root)
    project_root: Path = Path(__file__).resolve().parent.parent.parent

    # Gemini API
    gemini_api_key: str = ""
    gemini_model: str = "gemini-2.0-flash"
    gemini_temperature: float = 0.2
    gemini_max_output_tokens: int = 2048

    # Embedding Model
    embedding_model: str = "sentence-transformers/all-MiniLM-L6-v2"
    embedding_dimension: int = 384
    embedding_batch_size: int = 64
    embedding_device: str = "cpu"
    vector_index_filename: str = "chunks.faiss"
    vector_mapping_filename: str = "chunks.mapping.json"

    # Reranker
    reranker_model: str = "cross-encoder/ms-marco-MiniLM-L-6-v2"
    reranker_enabled: bool = True

    # Chunking
    chunk_size: int = 500
    chunk_overlap: int = 100

    # Retrieval
    retrieval_top_k: int = 15
    rerank_top_k: int = 5
    rrf_constant: int = 60
    bm25_index_filename: str = "chunks.bm25.json"
    rerank_candidate_count: int = 30

    # Hybrid Search
    hybrid_search_enabled: bool = True
    bm25_weight: float = 0.3
    dense_weight: float = 0.7

    # File Upload
    max_file_size_mb: int = 50
    allowed_extensions: set[str] = {".pdf", ".doc", ".docx", ".txt"}

    # Storage Paths
    data_directory: str = "data"
    vector_store_directory: str = "data/vectorstore"
    upload_directory: str = "data/uploads"
    extracted_directory: str = "data/extracted"

    # Database
    metadata_database_url: str = "sqlite:///data/database/metadata.db"

    # Server
    backend_host: str = "0.0.0.0"
    backend_port: int = 8000

    # Logging
    log_level: str = "INFO"

    # Conversation
    max_conversation_history: int = 10
    max_question_length: int = 2000

    @property
    def max_file_size_bytes(self) -> int:
        """Maximum file size in bytes."""
        return self.max_file_size_mb * 1024 * 1024

    def get_data_path(self, subpath: str = "") -> Path:
        """Get absolute path within the data directory."""
        base = self.project_root / self.data_directory
        if subpath:
            return base / subpath
        return base

    def get_upload_path(self) -> Path:
        """Get absolute path for uploads."""
        return self.project_root / self.upload_directory

    def get_vectorstore_path(self) -> Path:
        """Get absolute path for vector store."""
        return self.project_root / self.vector_store_directory

    def get_extracted_path(self) -> Path:
        """Get absolute path for extracted text."""
        return self.project_root / self.extracted_directory

    def get_database_path(self) -> Path:
        """Get absolute path for the database file."""
        db_url = self.metadata_database_url
        if db_url.startswith("sqlite:///"):
            database_path = Path(db_url.removeprefix("sqlite:///"))
            if database_path.is_absolute():
                return database_path
            return self.project_root / database_path
        return Path(db_url)

    def get_database_url(self) -> str:
        """Get absolute database URL."""
        if not self.metadata_database_url.startswith("sqlite:///"):
            return self.metadata_database_url
        db_path = self.get_database_path().resolve()
        return f"sqlite:///{db_path.as_posix()}"

    def ensure_directories(self) -> None:
        """Create all required data directories."""
        dirs = [
            self.get_upload_path(),
            self.get_extracted_path(),
            self.get_vectorstore_path(),
            self.get_database_path().parent,
        ]
        for d in dirs:
            d.mkdir(parents=True, exist_ok=True)


@lru_cache()
def get_settings() -> Settings:
    """Get cached application settings."""
    settings = Settings()
    settings.ensure_directories()
    return settings
