"""Database connection and session management."""

from typing import Generator

from sqlalchemy import create_engine, event
from sqlalchemy.orm import sessionmaker, Session

from backend.core.config import get_settings
from backend.core.logging import get_logger
from backend.database.models import Base

logger = get_logger(__name__)


class Database:
    """Manages database connections and sessions."""

    def __init__(self) -> None:
        settings = get_settings()
        settings.ensure_directories()
        db_path = settings.get_database_path()
        db_path.parent.mkdir(parents=True, exist_ok=True)

        self.database_url = settings.get_database_url()

        self.engine = create_engine(
            self.database_url,
            echo=False,
            connect_args={"check_same_thread": False},
        )

        @event.listens_for(self.engine, "connect")
        def set_sqlite_pragma(dbapi_connection, _connection_record):
            cursor = dbapi_connection.cursor()
            cursor.execute("PRAGMA journal_mode=WAL")
            cursor.execute("PRAGMA foreign_keys=ON")
            cursor.close()

        self.SessionLocal = sessionmaker(
            autocommit=False,
            autoflush=False,
            bind=self.engine,
        )

    def create_tables(self) -> None:
        """Create all database tables."""
        Base.metadata.create_all(bind=self.engine)
        logger.info("database_tables_created")

    def get_session(self) -> Session:
        """Get a new database session."""
        return self.SessionLocal()


# Singleton
_db: Database | None = None


def get_database() -> Database:
    """Get the singleton database instance."""
    global _db
    if _db is None:
        _db = Database()
        _db.create_tables()
    return _db


def get_db_session() -> Generator[Session, None, None]:
    """Get a database session (for FastAPI dependency injection)."""
    db = get_database()
    session = db.get_session()
    try:
        yield session
        session.commit()
    except Exception:
        session.rollback()
        raise
    finally:
        session.close()
