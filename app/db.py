# app/db.py
import os
from sqlalchemy import create_engine, event
from sqlalchemy.orm import sessionmaker, DeclarativeBase


class Base(DeclarativeBase):
    pass


def ensure_parent_dir(db_url: str) -> None:
    """Create parent directory for SQLite database files."""
    if db_url.startswith("sqlite:///"):
        path = db_url[len("sqlite:///"):]
        parent = os.path.dirname(path)
        if parent:
            os.makedirs(parent, exist_ok=True)


def make_engine(database_url: str):
    ensure_parent_dir(database_url)

    connect_args = {}
    if database_url.startswith("sqlite"):
        connect_args["check_same_thread"] = False

    eng = create_engine(
        database_url,
        future=True,
        connect_args=connect_args,
        pool_pre_ping=True,
    )

    # SQLite-specific: enable WAL mode and set a busy timeout so concurrent
    # writers wait instead of immediately raising "database is locked".
    if database_url.startswith("sqlite"):
        @event.listens_for(eng, "connect")
        def _set_sqlite_pragmas(dbapi_conn, connection_record):
            cursor = dbapi_conn.cursor()
            cursor.execute("PRAGMA journal_mode=WAL")
            cursor.execute("PRAGMA busy_timeout=5000")
            cursor.execute("PRAGMA synchronous=NORMAL")
            cursor.close()

    return eng


def make_session_factory(engine):
    return sessionmaker(bind=engine, autoflush=False, autocommit=False, future=True)
