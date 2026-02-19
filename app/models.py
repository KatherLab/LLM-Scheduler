from __future__ import annotations

from datetime import datetime, timezone
from sqlalchemy.orm import Mapped, mapped_column
from sqlalchemy import String, Integer, DateTime, Text, Index, TypeDecorator

from .db import Base


def utc_now() -> datetime:
    return datetime.now(timezone.utc)


class TZDateTime(TypeDecorator):
    """A DateTime type that ensures UTC timezone on read, even with SQLite."""
    impl = DateTime
    cache_ok = True

    def process_bind_param(self, value, dialect):
        if value is not None:
            if value.tzinfo is None:
                value = value.replace(tzinfo=timezone.utc)
            else:
                value = value.astimezone(timezone.utc)
        return value

    def process_result_value(self, value, dialect):
        if value is not None:
            if value.tzinfo is None:
                value = value.replace(tzinfo=timezone.utc)
        return value


class Lease(Base):
    __tablename__ = "leases"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    model: Mapped[str] = mapped_column(String(256), index=True)
    requested_gpus: Mapped[int] = mapped_column(Integer)
    requested_tp: Mapped[int] = mapped_column(Integer)
    requested_port: Mapped[int] = mapped_column(Integer)
    slurm_job_id: Mapped[str | None] = mapped_column(String(64), index=True, nullable=True)

    owner: Mapped[str | None] = mapped_column(String(256), nullable=True)

    # UTC-aware timestamps — TZDateTime ensures awareness on read
    created_at: Mapped[datetime] = mapped_column(TZDateTime(), default=utc_now)
    begin_at: Mapped[datetime | None] = mapped_column(TZDateTime(), nullable=True)
    end_at: Mapped[datetime | None] = mapped_column(TZDateTime(), nullable=True)

    # PLANNED, SUBMITTED, RUNNING, CANCELED, ENDED, FAILED
    state: Mapped[str] = mapped_column(String(32), default="PLANNED")

    model_path: Mapped[str] = mapped_column(Text)
    tool_args: Mapped[str | None] = mapped_column(Text, nullable=True)
    extra_args: Mapped[str | None] = mapped_column(Text, nullable=True)
    reasoning_parser: Mapped[str | None] = mapped_column(String(128), nullable=True)
    gpu_memory_utilization: Mapped[str | None] = mapped_column(String(32), nullable=True)

    venv_activate: Mapped[str | None] = mapped_column(Text, nullable=True)

    # User-facing notes (who booked it, why, hints)
    notes: Mapped[str | None] = mapped_column(Text, nullable=True)

    # Retry tracking
    retry_count: Mapped[int] = mapped_column(
        Integer, default=0, server_default="0"
    )
    failed_at: Mapped[datetime | None] = mapped_column(
        TZDateTime(), nullable=True
    )

class Endpoint(Base):
    __tablename__ = "endpoints"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    model: Mapped[str] = mapped_column(String(256), index=True)
    host: Mapped[str] = mapped_column(String(256))
    port: Mapped[int] = mapped_column(Integer)
    slurm_job_id: Mapped[str] = mapped_column(String(64), index=True)

    state: Mapped[str] = mapped_column(String(32), default="STARTING")

    last_health_at: Mapped[datetime | None] = mapped_column(TZDateTime(), nullable=True)
    last_error: Mapped[str | None] = mapped_column(Text, nullable=True)

    created_at: Mapped[datetime] = mapped_column(TZDateTime(), default=utc_now)


Index("ix_endpoints_model_state", Endpoint.model, Endpoint.state)
