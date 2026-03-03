# app/public_api.py
"""
Read-only public API for external schedule viewers.
Protected by SCHEDULE_API_KEY (separate from admin auth and VLLM_API_KEY).
"""
from __future__ import annotations

from datetime import datetime, timedelta, timezone

from fastapi import APIRouter, Depends
from sqlalchemy import select

from .auth import require_schedule_key
from .catalog import get_catalog
from .dependencies import SessionLocal
from .models import Endpoint, Lease
from .planner import compute_placements
from .schemas import (
    PublicLeaseInfo,
    PublicModelInfo,
    PublicScheduleResponse,
)
from .settings import settings
from .utils import ensure_utc

router = APIRouter(
    prefix="/api/v1/schedule",
    tags=["public-schedule"],
    dependencies=[Depends(require_schedule_key)],
)


def _now() -> datetime:
    return datetime.now(timezone.utc)


@router.get("/models", response_model=list[PublicModelInfo])
def list_models():
    """List all models in the catalog with their current availability."""
    catalog = get_catalog()
    with SessionLocal() as db:
        ready_models = set(
            db.execute(
                select(Endpoint.model).where(Endpoint.state == "READY")
            )
            .scalars()
            .all()
        )

    result = []
    for name, m in catalog.items():
        result.append(
            PublicModelInfo(
                name=name,
                gpus=m.gpus,
                tensor_parallel_size=m.tensor_parallel_size,
                tags=m.tags or [],
                notes=m.notes or "",
                ready=name in ready_models,
            )
        )
    result.sort(key=lambda x: x.name)
    return result


@router.get("", response_model=PublicScheduleResponse)
@router.get("/", response_model=PublicScheduleResponse)
def get_schedule():
    """
    Full schedule snapshot: all models + active/planned leases with GPU lane
    placements. Suitable for rendering an external timeline view.
    """
    now = _now()
    horizon_start = now - timedelta(hours=1)
    horizon_end = now + timedelta(hours=48)
    catalog = get_catalog()

    with SessionLocal() as db:
        ready_models = set(
            db.execute(
                select(Endpoint.model).where(Endpoint.state == "READY")
            )
            .scalars()
            .all()
        )

        leases = db.execute(
            select(Lease).order_by(Lease.id.desc())
        ).scalars().all()

        # Active-like leases for placement computation
        active_like = [
            l
            for l in leases
            if l.state in ("PLANNED", "SUBMITTED", "STARTING", "RUNNING")
            or (
                l.state == "FAILED"
                and l.end_at
                and ensure_utc(l.end_at) > now
            )
        ]

        placements = compute_placements(
            leases=active_like,
            total_gpus=settings.total_gpus,
            horizon_start=horizon_start,
            horizon_end=horizon_end,
        )

        # Build lease list (only active/planned — not historical)
        lease_infos: list[PublicLeaseInfo] = []
        for l in active_like:
            p = placements.get(l.id)
            lease_infos.append(
                PublicLeaseInfo(
                    id=l.id,
                    model=l.model,
                    state=l.state,
                    requested_gpus=l.requested_gpus,
                    begin_at=l.begin_at,
                    end_at=l.end_at,
                    notes=l.notes,
                    lane_start=p.lane_start if p else None,
                    lane_count=p.lane_count if p else None,
                    conflict=p.conflict if p else False,
                )
            )

    # Build model list
    model_infos: list[PublicModelInfo] = []
    for name, m in catalog.items():
        model_infos.append(
            PublicModelInfo(
                name=name,
                gpus=m.gpus,
                tensor_parallel_size=m.tensor_parallel_size,
                tags=m.tags or [],
                notes=m.notes or "",
                ready=name in ready_models,
            )
        )
    model_infos.sort(key=lambda x: x.name)

    return PublicScheduleResponse(
        now=now,
        total_gpus=settings.total_gpus,
        models=model_infos,
        leases=lease_infos,
    )


@router.get("/leases", response_model=list[PublicLeaseInfo])
def list_active_leases():
    """List only active/planned leases (no historical data)."""
    now = _now()

    with SessionLocal() as db:
        leases = db.execute(
            select(Lease).where(
                Lease.state.in_(
                    ["PLANNED", "SUBMITTED", "STARTING", "RUNNING"]
                )
            ).order_by(Lease.begin_at.asc())
        ).scalars().all()

        return [
            PublicLeaseInfo(
                id=l.id,
                model=l.model,
                state=l.state,
                requested_gpus=l.requested_gpus,
                begin_at=l.begin_at,
                end_at=l.end_at,
                notes=l.notes,
            )
            for l in leases
        ]
