# app/router_core.py
from __future__ import annotations
import asyncio
import logging
import threading
import time
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
    """Pick the least-loaded READY endpoint for a model.

    This used to return the newest, which sends every request to one replica
    and leaves the others idle. Selection now uses in-flight requests plus
    vLLM's queue depth, and skips replicas being drained for a rolling restart.
    """
    from .loadbalancer import choose_least_loaded

    eps = db.execute(
        select(Endpoint)
        .where(Endpoint.model == model, Endpoint.state == "READY")
        .order_by(Endpoint.id.desc())
    ).scalars().all()
    return choose_least_loaded(eps)


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


# ── vLLM /metrics scraper ────────────────────────────────────────────────────

logger = logging.getLogger(__name__)

_VLLM_METRIC_KEYS = {
    "vllm:kv_cache_usage_perc": "gpu_cache_usage",
    "vllm:num_requests_running": "active_requests",
    "vllm:num_requests_waiting": "pending_requests",
    "vllm:generation_tokens_total": "gen_tokens_total",
    "vllm:time_to_first_token_seconds_count": "ttft_count",
    "vllm:time_to_first_token_seconds_sum": "ttft_sum",
}

# Counter delta cache for throughput rate computation.
# {host:port: (timestamp, gen_tokens_total)}
_vllm_counter_cache: dict[str, tuple[float, float]] = {}
_vllm_cache_lock = threading.Lock()

def _parse_vllm_metrics(text: str) -> dict[str, float | int | None]:
    """Extract key gauges from vLLM Prometheus /metrics output.

    Returns a dict with keys ``gpu_cache_usage``, ``active_requests``,
    ``pending_requests``, ``gen_tokens_total`` (or ``None`` if missing).
    """
    result: dict[str, float | int | None] = {
        "gpu_cache_usage": None,
        "active_requests": None,
        "pending_requests": None,
        "gen_tokens_total": None,
        "ttft_count": None,
        "ttft_sum": None,
    }
    for line in text.splitlines():
        line = line.strip()
        if line.startswith("#") or not line:
            continue
        # Extract metric name (before any { or whitespace)
        # Format: vllm:name{labels} value  or  vllm:name value
        name_part = line.split("{")[0].split()[0]
        mapped = _VLLM_METRIC_KEYS.get(name_part)
        if mapped is None:
            continue
        # Value is the last whitespace-separated token
        try:
            raw = float(line.rsplit(None, 1)[-1])
        except (ValueError, IndexError):
            continue
        if mapped in ("active_requests", "pending_requests"):
            result[mapped] = int(raw)
        else:
            result[mapped] = raw
    return result


async def fetch_vllm_metrics(
    host: str, port: int, timeout_s: float = 2.0
) -> dict[str, float | int | None]:
    """Fetch /metrics from a vLLM instance and extract key gauges + throughput.

    Returns a dict with keys ``gpu_cache_usage``, ``active_requests``,
    ``pending_requests``, ``throughput_tps`` (tokens/s, computed from counter
    deltas). Values may be ``None`` if the fetch fails or the metric is absent.
    """
    result: dict[str, float | int | None] = {
        "gpu_cache_usage": None,
        "active_requests": None,
        "pending_requests": None,
        "throughput_tps": None,
        "ttft_avg": None,
    }
    url = f"http://{host}:{port}/metrics"
    try:
        client = await _get_health_client()
        r = await client.get(url, timeout=timeout_s)
        if r.status_code == 200:
            parsed = _parse_vllm_metrics(r.text)
            result.update(parsed)

            # Compute TTFT average from vLLM's own histogram
            ttft_count = parsed.get("ttft_count")
            ttft_sum = parsed.get("ttft_sum")
            if ttft_count is not None and ttft_sum is not None and ttft_count > 0:
                result["ttft_avg"] = round(ttft_sum / ttft_count, 3)

            # Feed the routing signal: vLLM's own queue depth catches load we
            # cannot infer from our request counts (a replica still loading
            # weights, or one whose engine is backed up).
            from .loadbalancer import REGISTRY, LoadRegistry

            REGISTRY.record_scrape(
                LoadRegistry.key(host, port),
                pending=parsed.get("pending_requests"),
                running=parsed.get("active_requests"),
            )

            # Compute generation throughput from counter deltas
            gen_total = parsed.get("gen_tokens_total")
            if gen_total is not None:
                now = time.time()
                key = f"{host}:{port}"
                with _vllm_cache_lock:
                    prev = _vllm_counter_cache.get(key)
                    if prev is not None:
                        prev_time, prev_gen = prev
                        dt = now - prev_time
                        dg = gen_total - prev_gen
                        if dt >= 1.0 and dg >= 0:
                            result["throughput_tps"] = round(dg / dt, 1)
                        elif dg < 0:
                            # Counter reset (vLLM restart) — reset cache
                            _vllm_counter_cache[key] = (now, gen_total)
                            return result
                    _vllm_counter_cache[key] = (now, gen_total)
    except Exception as exc:
        logger.warning("fetch_vllm_metrics: failed to fetch %s: %s", url, exc)
    return result
