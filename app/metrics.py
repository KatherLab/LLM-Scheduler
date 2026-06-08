from __future__ import annotations

import time
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager

from prometheus_client import Counter, Gauge, Histogram, generate_latest, REGISTRY

# ── Upstream proxy metrics (tracked inside _ProxyResponse) ────────────

PROXY_REQUESTS_TOTAL = Counter(
    "llm_proxy_requests_total",
    "Total proxied requests to upstream vLLM instances",
    ["model", "endpoint", "status"],
)

PROXY_REQUEST_DURATION = Histogram(
    "llm_proxy_request_duration_seconds",
    "Proxied request latency in seconds",
    ["model", "endpoint"],
    buckets=(0.01, 0.025, 0.05, 0.1, 0.25, 0.5, 1.0, 2.5, 5.0, 10.0, 30.0, 60.0, 120.0, float("inf")),
)

PROXY_UPSTREAM_ERRORS = Counter(
    "llm_proxy_upstream_errors_total",
    "Upstream connection / timeout errors by type",
    ["model", "endpoint", "error_type"],
)

PROXY_ACTIVE = Gauge(
    "llm_proxy_active",
    "Currently in-flight proxied requests",
    ["model", "endpoint"],
)

PROXY_DOWNSTREAM_DISCONNECTS = Counter(
    "llm_proxy_downstream_disconnects_total",
    "Total downstream client disconnects during proxy",
)


# ── Upstream health (set by a background gauge worker) ────────────────

UPSTREAM_HEALTHY = Gauge(
    "llm_proxy_upstream_healthy",
    "Whether a model has a READY upstream endpoint (1=ready, 0=not ready)",
    ["model"],
)


# ── Helpers ───────────────────────────────────────────────────────────

def endpoint_label(url_path: str) -> str:
    """Derive a concise label from an upstream URL path.

    ``/v1/chat/completions`` → ``chat.completions``
    ``/v1/messages``        → ``messages``
    ``/metrics``            → ``metrics``
    """
    parts = url_path.strip("/").split("/")
    # Drop the version prefix (v1, v2, …) if present.
    if parts and parts[0].startswith("v"):
        parts = parts[1:]
    return ".".join(parts) if parts else url_path


@asynccontextmanager
async def track_proxy(upstream_url: str, model: str = "") -> AsyncIterator[dict]:
    """Context manager that records timing and status for a single proxy call.

    Usage inside ``_ProxyResponse.__call__``::

        async with track_proxy(self._upstream_url, model="gpt-4") as ctx:
            …               # on success:   ctx["status"] = resp.status_code
            …               # on exception: ctx["error"] = type(exc).__name__
    """
    t0 = time.perf_counter()
    ep = endpoint_label(upstream_url)
    model_label = model or ""
    PROXY_ACTIVE.labels(model=model_label, endpoint=ep).inc()

    ctx: dict = {}
    try:
        yield ctx
    finally:
        duration = time.perf_counter() - t0
        PROXY_ACTIVE.labels(model=model_label, endpoint=ep).dec()

        status = ctx.get("status", "502")
        error_type = ctx.get("error")

        if error_type:
            PROXY_UPSTREAM_ERRORS.labels(model=model_label, endpoint=ep, error_type=error_type).inc()
            PROXY_REQUESTS_TOTAL.labels(model=model_label, endpoint=ep, status=status).inc()
        else:
            PROXY_REQUESTS_TOTAL.labels(model=model_label, endpoint=ep, status=str(status)).inc()

        PROXY_REQUEST_DURATION.labels(model=model_label, endpoint=ep).observe(duration)


# ── Metrics summary (JSON-friendly) ───────────────────────────────────

_METRIC_PREFIX = "llm_proxy_"


def _is_ours(name: str) -> bool:
    """Check if a metric name belongs to this application."""
    return name.startswith(_METRIC_PREFIX)


def get_metrics_summary() -> dict:
    """Return a structured JSON-friendly summary of all proxy metrics.

    This iterates over the prometheus_client REGISTRY at call time, so it
    always reflects the latest counter / gauge / histogram values.
    """
    # ── Phase 1: collect all samples ───────────────────────────────────
    # Organised as {metric_family_name: [(labels_dict, value), ...]}
    samples: dict[str, list[tuple[dict, float]]] = {}

    for metric_family in REGISTRY.collect():
        if not _is_ours(metric_family.name):
            continue
        for sample in metric_family.samples:
            labels = dict(sample.labels) if sample.labels else {}
            samples.setdefault(sample.name, []).append((labels, sample.value))

    # ── Phase 2: build structured summary ──────────────────────────────

    # --- Requests total ---
    requests_total = 0
    requests_by_endpoint: dict[str, dict] = {}
    for labels, value in samples.get("llm_proxy_requests_total", []):
        ep = labels.get("endpoint", "?")
        status = labels.get("status", "?")
        requests_total += value
        ep_data = requests_by_endpoint.setdefault(ep, {"total": 0, "by_status": {}})
        ep_data["total"] += value
        ep_data["by_status"][status] = ep_data["by_status"].get(status, 0) + value

    # --- Active requests ---
    active_total = 0
    active_by_endpoint: dict[str, float] = {}
    for labels, value in samples.get("llm_proxy_active", []):
        ep = labels.get("endpoint", "?")
        active_total += value
        active_by_endpoint[ep] = active_by_endpoint.get(ep, 0) + value

    # --- Upstream errors ---
    errors_total = 0
    errors_by_endpoint: dict[str, dict] = {}
    for labels, value in samples.get("llm_proxy_upstream_errors_total", []):
        ep = labels.get("endpoint", "?")
        err_type = labels.get("error_type", "?")
        errors_total += value
        ep_data = errors_by_endpoint.setdefault(ep, {"total": 0, "by_type": {}})
        ep_data["total"] += value
        ep_data["by_type"][err_type] = ep_data["by_type"].get(err_type, 0) + value

    # --- Latency (histogram) ---
    # Collect bucket samples per endpoint
    latency_buckets: dict[str, list[tuple[float, float]]] = {}
    latency_counts: dict[str, float] = {}
    latency_sums: dict[str, float] = {}
    # Also collect by model
    latency_model_buckets: dict[str, list[tuple[float, float]]] = {}
    latency_model_counts: dict[str, float] = {}
    latency_model_sums: dict[str, float] = {}
    for name in samples:
        if not name.startswith("llm_proxy_request_duration_seconds"):
            continue
        for labels, value in samples[name]:
            ep = labels.get("endpoint", "?")
            model_label = labels.get("model", "") or ""
            if name.endswith("_bucket"):
                le_str = labels.get("le", "+Inf")
                le = float("inf") if le_str in ("+Inf", "+inf", "inf") else float(le_str)
                latency_buckets.setdefault(ep, []).append((le, value))
                latency_model_buckets.setdefault(model_label, []).append((le, value))
            elif name.endswith("_count"):
                latency_counts[ep] = value
                latency_model_counts[model_label] = latency_model_counts.get(model_label, 0) + value
            elif name.endswith("_sum"):
                latency_sums[ep] = value
                latency_model_sums[model_label] = latency_model_sums.get(model_label, 0) + value

    def _build_latency_dict(
        keys: list[str],
        buckets: dict[str, list],
        counts: dict[str, float],
        sums: dict[str, float],
    ) -> dict[str, dict]:
        result: dict[str, dict] = {}
        for key in keys:
            entry: dict = {
                "count": counts.get(key, 0),
                "sum": sums.get(key, 0),
            }
            if entry["count"]:
                entry["avg"] = round(entry["sum"] / entry["count"], 3)
            else:
                entry["avg"] = 0.0

            b = buckets.get(key, [])
            b.sort(key=lambda x: x[0])
            total = entry["count"]
            if total > 0 and b:
                percentile_targets = {
                    "p50": total * 0.50,
                    "p95": total * 0.95,
                    "p99": total * 0.99,
                }
                for pct_key, target in percentile_targets.items():
                    pct_value = _histogram_quantile(target, b)
                    entry[pct_key] = round(pct_value, 3) if pct_value is not None else None
            else:
                for k in ("p50", "p95", "p99"):
                    entry[k] = None

            result[key] = entry
        return result

    latency = _build_latency_dict(
        list(latency_counts.keys()), latency_buckets, latency_counts, latency_sums
    )
    latency_by_model = _build_latency_dict(
        list(latency_model_counts.keys()), latency_model_buckets, latency_model_counts, latency_model_sums
    )

    # --- Downstream disconnects ---
    disconnects = 0
    for _, value in samples.get("llm_proxy_downstream_disconnects_total", []):
        disconnects += value

    # --- Upstream health ---
    upstream_health: dict[str, bool] = {}
    for labels, value in samples.get("llm_proxy_upstream_healthy", []):
        model = labels.get("model", "?")
        upstream_health[model] = bool(value)

    return {
        "requests": {
            "total": requests_total,
            "by_endpoint": requests_by_endpoint,
        },
        "active_requests": {
            "total": active_total,
            "by_endpoint": active_by_endpoint,
        },
        "errors": {
            "total": errors_total,
            "by_endpoint": errors_by_endpoint,
        },
        "latency_by_model": latency_by_model,
        "latency": latency,
        "downstream_disconnects": disconnects,
        "upstream_health": upstream_health,
    }


def _histogram_quantile(target: float, buckets: list[tuple[float, float]]) -> float | None:
    """Compute a quantile value from sorted prometheus histogram buckets.

    *buckets* must be sorted ascending by the upper bound (le).
    *target* is the cumulative observation count for the desired quantile
             (e.g. total_count * 0.95 for p95).
    Returns the interpolated value, or ``None`` if no data.
    """
    if not buckets:
        return None

    for i, (le, cum) in enumerate(buckets):
        if cum >= target:
            if i == 0:
                return le / 2  # reasonable lower-bound guess
            prev_le, prev_cum = buckets[i - 1]
            if cum - prev_cum > 0:
                fraction = (target - prev_cum) / (cum - prev_cum)
                return prev_le + (le - prev_le) * fraction
            return le
    # Target beyond the highest bucket — extrapolate
    last_le, last_cum = buckets[-1]
    if last_cum > 0 and last_le != float("inf"):
        return last_le * (target / last_cum)
    return last_le if last_le != float("inf") else None
