# app/dependencies.py
from .settings import settings
from .db import make_engine, make_session_factory, Base

engine = make_engine(settings.database_url)
SessionLocal = make_session_factory(engine)

def init_db():
    Base.metadata.create_all(bind=engine)
    # Migration: add vllm_version column to existing endpoints table
    try:
        from sqlalchemy import text
        with engine.connect() as conn:
            conn.execute(text("ALTER TABLE endpoints ADD COLUMN vllm_version VARCHAR(128)"))
            conn.commit()
    except Exception:
        pass  # Column already exists
