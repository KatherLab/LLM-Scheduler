import os
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker, DeclarativeBase

# For simplicity this uses sync SQLAlchemy. It's plenty fast for a control plane.
class Base(DeclarativeBase):
    pass

def ensure_parent_dir(db_url: str) -> None:
    if db_url.startswith("sqlite:////"):
        path = db_url[len("sqlite:////"):]
        os.makedirs(os.path.dirname(path), exist_ok=True)

def make_engine(database_url: str):
    ensure_parent_dir(database_url)
    return create_engine(database_url, future=True)

def make_session_factory(engine):
    return sessionmaker(bind=engine, autoflush=False, autocommit=False, future=True)
