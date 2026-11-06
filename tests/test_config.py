from pathlib import Path

from backend.core.config import Settings


def test_settings_resolve_paths(tmp_path: Path) -> None:
    settings = Settings(project_root=tmp_path)

    assert settings.get_database_path() == tmp_path / "data/database/metadata.db"
    assert settings.get_upload_path() == tmp_path / "data/uploads"


def test_settings_create_directories(tmp_path: Path) -> None:
    settings = Settings(project_root=tmp_path)
    settings.ensure_directories()

    assert settings.get_upload_path().is_dir()
    assert settings.get_extracted_path().is_dir()
    assert settings.get_vectorstore_path().is_dir()
    assert settings.get_database_path().parent.is_dir()
