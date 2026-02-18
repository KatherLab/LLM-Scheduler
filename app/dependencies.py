# app/dependencies.py
from .settings import settings
from .db import make_engine, make_session_factory, Base

engine = make_engine(settings.database_url)
SessionLocal = make_session_factory(engine)

def init_db():
    Base.metadata.create_all(bind=engine)
