from pathlib import Path

from backend.core.config import Settings
from backend.database.database import Database
from backend.database.models import Document


def test_database_creates_tables(tmp_path: Path, monkeypatch) -> None:
    settings = Settings(project_root=tmp_path)
    monkeypatch.setattr("backend.database.database.get_settings", lambda: settings)
    database = Database()
    database.create_tables()

    session = database.get_session()
    try:
        document = Document(
            filename="safe.pdf",
            original_filename="safe.pdf",
            file_type=".pdf",
            file_size=10,
            file_hash="a" * 64,
            stored_file_path=str(tmp_path / "safe.pdf"),
        )
        session.add(document)
        session.commit()
        assert session.get(Document, document.id) is not None
    finally:
        session.close()
