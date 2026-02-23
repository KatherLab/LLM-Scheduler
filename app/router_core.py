# app/router_core.py
from __future__ import annotations
import asyncio
from typing import Optional
from sqlalchemy.orm import Session
from sqlalchemy import select
import httpx
import time

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


async def health_check_endpoint(
    host: str, port: int, timeout_s: float = 30.0
) -> tuple[bool, str | None]:
    """Check vLLM /health endpoint. Uses shared connection pool.

    Returns (ok, error_string_or_None).
    The error string now includes the type of failure (timeout, status code, etc.)
    so lifecycle logs can distinguish root causes.
    """
    url = f"http://{host}:{port}/health"
    t0 = time.perf_counter()

    try:
        client = await _get_health_client()
        # IMPORTANT: pass timeout_s as a per-request override so it actually
        # takes effect instead of being silently ignored by the client-level 3s default.
        r = await client.get(url, timeout=timeout_s)

        dt_ms = (time.perf_counter() - t0) * 1000.0

        if r.status_code == 200:
            return True, None

        return False, f"health status {r.status_code} (elapsed {dt_ms:.0f}ms, body={r.text[:200]!r})"

    except httpx.TimeoutException as e:
        dt_ms = (time.perf_counter() - t0) * 1000.0
        return False, f"TimeoutException after {dt_ms:.0f}ms: {type(e).__name__}: {e}"

    except httpx.RequestError as e:
        dt_ms = (time.perf_counter() - t0) * 1000.0
        return False, f"RequestError after {dt_ms:.0f}ms: {type(e).__name__}: {e}"

    except Exception as e:
        dt_ms = (time.perf_counter() - t0) * 1000.0
        return False, f"UnexpectedError after {dt_ms:.0f}ms: {type(e).__name__}: {e}"
