import os
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker, DeclarativeBase


class Base(DeclarativeBase):
    pass


def ensure_parent_dir(db_url: str) -> None:
    """Create parent directory for SQLite database files."""
    if db_url.startswith("sqlite:///"):
        # sqlite:///./foo.db  → ./foo.db  (relative)
        # sqlite:////abs/foo.db → /abs/foo.db (absolute)
        path = db_url[len("sqlite:///"):]
        parent = os.path.dirname(path)
        if parent:
            os.makedirs(parent, exist_ok=True)


def make_engine(database_url: str):
    ensure_parent_dir(database_url)
    return create_engine(database_url, future=True)


def make_session_factory(engine):
    return sessionmaker(bind=engine, autoflush=False, autocommit=False, future=True)
