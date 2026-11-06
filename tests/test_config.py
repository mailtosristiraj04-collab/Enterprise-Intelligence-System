from pathlib import Path

import pytest

from backend.core.config import DEFAULT_GEMINI_MODEL, Settings


def test_default_gemini_model_is_current() -> None:
    settings = Settings(_env_file=None)

    assert settings.gemini_model == DEFAULT_GEMINI_MODEL == "gemini-3.8-flash"


def test_obsolete_gemini_override_is_replaced() -> None:
    with pytest.warns(UserWarning, match="obsolete model"):
        settings = Settings(_env_file=None, gemini_model="gemini-2.0-flash")

    assert settings.gemini_model == DEFAULT_GEMINI_MODEL


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
