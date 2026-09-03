# app/dependencies.py
from .settings import settings
from .db import make_engine, make_session_factory, Base

engine = make_engine(settings.database_url)
SessionLocal = make_session_factory(engine)

def init_db():
    """Create tables. There is deliberately no migration system.

    Schema changes are rolled out by deleting the database and letting
    create_all rebuild it — the state here (leases, endpoints) is scheduling
    state, not records worth preserving across a redesign.
    """
    Base.metadata.create_all(bind=engine)
    _check_schema()


def _check_schema() -> None:
    """Fail loudly and usefully when an existing DB predates the models.

    `create_all` only creates missing *tables*; it never adds columns. So an
    older database starts fine and then dies mid-request with
    `no such column: ...`, which reads like a bug rather than the documented
    consequence of having no migration system.
    """
    from sqlalchemy import inspect

    inspector = inspect(engine)
    existing = set(inspector.get_table_names())
    drift: list[str] = []

    for table in Base.metadata.sorted_tables:
        if table.name not in existing:
            continue
        have = {c["name"] for c in inspector.get_columns(table.name)}
        missing = [c.name for c in table.columns if c.name not in have]
        if missing:
            drift.append(f"{table.name}: {', '.join(sorted(missing))}")

    if drift:
        raise RuntimeError(
            "Database schema is out of date — these columns are missing:\n  "
            + "\n  ".join(drift)
            + "\n\nThere is deliberately no migration system. Delete the database "
              "and let it be recreated:\n"
              "  local:  rm -f ./router.db\n"
              "  podman: podman compose down && podman volume rm llm-scheduler_scheduler-data\n"
              "  docker: docker compose down -v\n"
              "Bookings are scheduling state, not records worth migrating."
        )
