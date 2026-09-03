# app/public_api.py
"""
Read-only public API for external schedule viewers.
Protected by SCHEDULE_API_KEY (separate from admin auth and VLLM_API_KEY).
"""
from __future__ import annotations

import asyncio
from datetime import datetime, timezone

from fastapi import APIRouter, Depends
from sqlalchemy import select

from .auth import require_schedule_key
from .catalog import get_catalog
from .dependencies import SessionLocal
from .metrics import get_metrics_summary
from .models import Endpoint, Lease
from . import inventory, scheduling
from .cluster import get_cluster
from .router_core import fetch_vllm_metrics
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

        # Same placement path as the admin dashboard, so the public view and
        # the internal one never disagree about where a booking sits.
        active_like = scheduling.active_leases(leases, now)
        placements, lanes = scheduling.plan(
            active_like, inventory.current(), get_cluster()
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
                    lane_start=scheduling.lane_index(lanes, p) if p else None,
                    lane_count=p.gpu_count if p else None,
                    conflict=p.conflict if p else False,
                    node=p.node if p else None,
                    gpu_class=l.gpu_class,
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
        total_gpus=sum(lane.gpu_count for lane in lanes),
        models=model_infos,
        leases=lease_infos,
    )


@router.get("/leases", response_model=list[PublicLeaseInfo])
def list_active_leases():
    """List only active/planned leases (no historical data)."""
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


# ── Metrics router ─────────────────────────────────────────────────────────

metrics_router = APIRouter(
    prefix="/api/v1",
    tags=["public-metrics"],
    dependencies=[Depends(require_schedule_key)],
)


@metrics_router.get("/metrics")
async def get_metrics():
    """
    Aggregated metrics snapshot for external dashboards.

    Combines proxy-level Prometheus metrics (latency, errors, health) with
    live per-instance vLLM stats (KV cache usage, throughput, requests, version)
    and cluster GPU utilization. Everything in one JSON response.

    Requires ``SCHEDULE_API_KEY`` via ``Authorization: Bearer <key>`` or ``?token=<key>``.
    """
    now = _now()

    # 1. Proxy-level aggregated metrics
    proxy_metrics = get_metrics_summary()

    # 2. Per-instance vLLM stats
    stats_data: list[dict] = []
    allocated_gpus = 0
    running_models: set[str] = set()

    with SessionLocal() as db:
        eps = db.execute(
            select(Endpoint).where(Endpoint.state.in_(["READY", "STARTING"]))
        ).scalars().all()

        for e in eps:
            uptime = None
            if e.created_at:
                uptime = (now - ensure_utc(e.created_at)).total_seconds()
            stats_data.append({
                "model": e.model,
                "host": e.host,
                "port": e.port,
                "state": e.state,
                "slurm_job_id": e.slurm_job_id,
                "uptime_seconds": uptime,
                "vllm_version": e.vllm_version,
            })
            if e.state == "READY":
                running_models.add(e.model)

        # Count GPUs allocated by active leases
        active_leases = db.execute(
            select(Lease).where(
                Lease.state.in_(["SUBMITTED", "STARTING", "RUNNING"])
            )
        ).scalars().all()
        allocated_gpus = sum(
            lease.requested_gpus for lease in active_leases if lease.requested_gpus
        )

    # Fetch live vLLM metrics in parallel (no DB session held)
    async def _enrich(s: dict) -> None:
        vm = await fetch_vllm_metrics(s["host"], s["port"])
        s["gpu_cache_usage"] = vm["gpu_cache_usage"]
        s["active_requests"] = vm["active_requests"]
        s["pending_requests"] = vm["pending_requests"]
        s["throughput_tps"] = vm["throughput_tps"]
        s["ttft_avg"] = vm["ttft_avg"]

    if stats_data:
        await asyncio.gather(*[_enrich(s) for s in stats_data])

    # 3. Assemble response
    return {
        "proxy": proxy_metrics,
        "instances": stats_data,
        "cluster": {
            "total_gpus": inventory.current().total_gpus or settings.total_gpus,
            "allocated_gpus": allocated_gpus,
            "running_models": sorted(running_models),
        },
    }
