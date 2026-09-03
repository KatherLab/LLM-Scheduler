from __future__ import annotations

from datetime import datetime, timezone
from sqlalchemy.orm import Mapped, mapped_column
from sqlalchemy import Boolean, String, Integer, DateTime, Text, Index, TypeDecorator

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
    requested_cpus: Mapped[int | None] = mapped_column(Integer, nullable=True)
    requested_mem: Mapped[str | None] = mapped_column(String(32), nullable=True)
    requested_tp: Mapped[int] = mapped_column(Integer)
    requested_port: Mapped[int] = mapped_column(Integer)
    slurm_job_id: Mapped[str | None] = mapped_column(String(64), index=True, nullable=True)

    # Free-text display name, shown in the UI.
    owner: Mapped[str | None] = mapped_column(String(256), nullable=True)

    # ── Ownership & authorization ───────────────────────────────────────────
    # owner_sub is the stable LDAP uid and is what authz compares against;
    # owner_group lets a team co-own a deployment so "only the owner may edit"
    # does not break when someone is away.
    owner_sub: Mapped[str | None] = mapped_column(String(256), index=True, nullable=True)
    owner_group: Mapped[str | None] = mapped_column(String(256), nullable=True)
    # From the directory's `mail` attribute; Slurm job notifications prefer it
    # over SLURM_MAIL_USER so mail reaches whoever booked the model.
    owner_email: Mapped[str | None] = mapped_column(String(320), nullable=True)

    # Which pool this booking lives in — scopes operator permissions and
    # decides whether the start time is a promise or an estimate.
    pool: Mapped[str | None] = mapped_column(String(64), index=True, nullable=True)

    # ── Placement ───────────────────────────────────────────────────────────
    # The GRES type this booking is pinned to (gpu24/gpu48/gpu80/gpu96).
    # Submitting without it is not neutral: job_submit.lua appends `gpu24` to
    # any untyped --gres, so the job never reaches a larger card.
    gpu_class: Mapped[str | None] = mapped_column(String(32), index=True, nullable=True)
    # Node chosen by our planner. Only pinned via --nodelist on managed pools,
    # where the calendar is authoritative; on `slurm` pools Slurm decides.
    node: Mapped[str | None] = mapped_column(String(128), nullable=True)
    # Node the *user* explicitly asked for (by dropping on its row, or picking
    # it). Always honoured, on any pool — an explicit request beats our
    # placement — and kept separate so re-planning cannot silently drop it.
    pinned_node: Mapped[str | None] = mapped_column(String(128), nullable=True)
    # Runtime name from cluster.yaml; normally implied by gpu_class.
    runtime: Mapped[str | None] = mapped_column(String(64), nullable=True)

    # session -> fixed window, hard stop at end_at (benchmarks).
    # service -> no fixed end; renewed before TimeLimit expiry so a long-running
    #            model survives the cluster's MaxWall instead of fighting it.
    mode: Mapped[str] = mapped_column(String(16), default="session", server_default="session")
    # Co-located group: JSON list of CoTenant dicts. When set, this lease is a
    # GPU host running several vLLM servers inside one allocation, and `model`
    # is just the first of them.
    colocated_json: Mapped[str | None] = mapped_column(Text, nullable=True)
    # How many concurrent vLLM instances serve this model.
    replicas: Mapped[int] = mapped_column(Integer, default=1, server_default="1")
    # Set on the replacement lease during a rolling renew, pointing at the one
    # it will retire once it is READY.
    supersedes_id: Mapped[int | None] = mapped_column(Integer, index=True, nullable=True)
    # Timestamp at which draining began, for the handover timeout.
    draining_since: Mapped[datetime | None] = mapped_column(TZDateTime(), nullable=True)

    # Slurm's backfill estimate of when a queued job will actually start.
    # Only meaningful on `slurm` pools, where Slurm — not our calendar — owns
    # the decision. It moves as the queue changes, hence `estimate_updated_at`.
    estimated_start: Mapped[datetime | None] = mapped_column(TZDateTime(), nullable=True)
    estimate_updated_at: Mapped[datetime | None] = mapped_column(TZDateTime(), nullable=True)

    # Locked deployments are exempt from owner cancellation and from auto
    # cleanup: this is the "it is production, keep it up" flag, not merely a
    # permission bit.
    locked: Mapped[bool] = mapped_column(Boolean, default=False, server_default="0")
    locked_by: Mapped[str | None] = mapped_column(String(256), nullable=True)
    locked_reason: Mapped[str | None] = mapped_column(Text, nullable=True)

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
    env_json: Mapped[str | None] = mapped_column(Text, nullable=True)

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

    health_fail_count: Mapped[int] = mapped_column(
        Integer, default=0, server_default="0"
    )

    vllm_version: Mapped[str | None] = mapped_column(String(128), nullable=True)

Index("ix_endpoints_model_state", Endpoint.model, Endpoint.state)


class ImageBuild(Base):
    """One `apptainer build` run, which is itself a Slurm job.

    It has to be a job: Apptainer cannot cross-build, so an aarch64 image can
    only be produced on an aarch64 node. That makes the build the same kind of
    thing as a model launch — submitted, watched, and readable through the
    ordinary job logs.

    The row outlives the job because it is the only record of *why* a `.sif` on
    the shared filesystem exists and where it came from; the file itself
    carries no provenance.
    """

    __tablename__ = "image_builds"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)

    #: Target file name inside the images directory, e.g. `vllm-0.11.0-x86_64.sif`.
    image_name: Mapped[str] = mapped_column(String(256), index=True)
    #: Full source reference, e.g. `docker://vllm/vllm-openai:v0.11.0`.
    source_ref: Mapped[str] = mapped_column(Text)
    #: `x86_64` | `aarch64`. Decides which nodes may run the build.
    arch: Mapped[str] = mapped_column(String(32))

    partition: Mapped[str | None] = mapped_column(String(64), nullable=True)
    nodelist: Mapped[str | None] = mapped_column(String(256), nullable=True)

    # SUBMITTED, RUNNING, SUCCEEDED, FAILED, CANCELED
    state: Mapped[str] = mapped_column(String(32), default="SUBMITTED", index=True)
    slurm_job_id: Mapped[str | None] = mapped_column(String(64), nullable=True, index=True)

    #: LDAP uid of whoever asked for it — all jobs run as one service account,
    #: so this row is the only place the requester survives.
    requested_by: Mapped[str | None] = mapped_column(String(128), nullable=True)

    error: Mapped[str | None] = mapped_column(Text, nullable=True)
    size_bytes: Mapped[int | None] = mapped_column(Integer, nullable=True)

    created_at: Mapped[datetime] = mapped_column(TZDateTime(), default=utc_now)
    started_at: Mapped[datetime | None] = mapped_column(TZDateTime(), nullable=True)
    finished_at: Mapped[datetime | None] = mapped_column(TZDateTime(), nullable=True)

    @property
    def is_active(self) -> bool:
        return self.state in ("SUBMITTED", "RUNNING")
