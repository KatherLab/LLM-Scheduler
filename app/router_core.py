# app/router_core.py
from __future__ import annotations
import asyncio
from typing import Optional
from sqlalchemy.orm import Session
from sqlalchemy import select
import httpx

from .models import Endpoint

# ── Shared health-check client ──────────────────────────────────────────────
_health_client: httpx.AsyncClient | None = None
_health_client_lock = asyncio.Lock()


async def _get_health_client() -> httpx.AsyncClient:
    global _health_client
    if _health_client is not None and not _health_client.is_closed:
        return _health_client
    async with _health_client_lock:
        if _health_client is not None and not _health_client.is_closed:
            return _health_client
        _health_client = httpx.AsyncClient(
            timeout=httpx.Timeout(3.0, connect=2.0),
            limits=httpx.Limits(
                max_connections=50,
                max_keepalive_connections=20,
                keepalive_expiry=30,
            ),
            follow_redirects=False,
            http2=False,
        )
        return _health_client


async def close_health_client() -> None:
    global _health_client
    if _health_client is not None and not _health_client.is_closed:
        await _health_client.aclose()
        _health_client = None


def choose_ready_endpoint(db: Session, model: str) -> Optional[Endpoint]:
    """Pick the newest READY endpoint for a model."""
    eps = db.execute(
        select(Endpoint)
        .where(Endpoint.model == model, Endpoint.state == "READY")
        .order_by(Endpoint.id.desc())
    ).scalars().all()
    return eps[0] if eps else None


import time
import traceback

async def health_check_endpoint(
    host: str, port: int, timeout_s: float = 60.0
) -> tuple[bool, str | None]:
    """Check vLLM /health endpoint. Uses shared connection pool."""
    url = f"http://{host}:{port}/health"

    def dbg(msg: str) -> None:
        print(f"[health_check] {msg}")

    t0 = time.perf_counter()

    try:
        client = await _get_health_client()
        r = await client.get(url, timeout=timeout_s)

        if r.status_code == 200:
            return True, None

        dt_ms = (time.perf_counter() - t0) * 1000.0
        dbg(
            f"FAIL status={r.status_code} elapsed_ms={dt_ms:.1f} "
            f"url={url} body={r.text!r}"
        )
        return False, f"health status {r.status_code}"

    except Exception as e:
        dt_ms = (time.perf_counter() - t0) * 1000.0
        dbg(f"EXCEPTION after {dt_ms:.1f}ms url={url}: {type(e).__name__}: {e!r}")
        dbg("TRACEBACK:\n" + traceback.format_exc())
        return False, f"{type(e).__name__}: {e}"