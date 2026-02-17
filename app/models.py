from __future__ import annotations

from datetime import datetime, timezone
from sqlalchemy.orm import Mapped, mapped_column
from sqlalchemy import String, Integer, DateTime, Text, Index
from .db import Base

def utc_now() -> datetime:
    return datetime.now(timezone.utc)

class Lease(Base):
    __tablename__ = "leases"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    model: Mapped[str] = mapped_column(String(256), index=True)
    requested_gpus: Mapped[int] = mapped_column(Integer)
    requested_tp: Mapped[int] = mapped_column(Integer)
    requested_port: Mapped[int] = mapped_column(Integer)
    slurm_job_id: Mapped[str | None] = mapped_column(String(64), index=True, nullable=True)

    owner: Mapped[str | None] = mapped_column(String(256), nullable=True)

    # UTC-aware timestamps
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=utc_now)
    begin_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True), nullable=True)
    end_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True), nullable=True)

    # PLANNED, SUBMITTED, RUNNING, CANCELED, ENDED, FAILED
    state: Mapped[str] = mapped_column(String(32), default="PLANNED")

    model_path: Mapped[str] = mapped_column(Text)
    tool_args: Mapped[str | None] = mapped_column(Text, nullable=True)
    extra_args: Mapped[str | None] = mapped_column(Text, nullable=True)
    reasoning_parser: Mapped[str | None] = mapped_column(String(128), nullable=True)
    gpu_memory_utilization: Mapped[str | None] = mapped_column(String(32), nullable=True)

    # NEW: optional venv activation script path
    venv_activate: Mapped[str | None] = mapped_column(Text, nullable=True)

class Endpoint(Base):
    __tablename__ = "endpoints"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    model: Mapped[str] = mapped_column(String(256), index=True)
    host: Mapped[str] = mapped_column(String(256))
    port: Mapped[int] = mapped_column(Integer)
    slurm_job_id: Mapped[str] = mapped_column(String(64), index=True)

    state: Mapped[str] = mapped_column(String(32), default="STARTING")  # STARTING, READY, FAILED, STOPPED

    last_health_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True), nullable=True)
    last_error: Mapped[str | None] = mapped_column(Text, nullable=True)

    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=utc_now)

Index("ix_endpoints_model_state", Endpoint.model, Endpoint.state)
