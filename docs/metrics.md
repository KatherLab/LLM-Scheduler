# Metrics System

The proxy exposes metrics at two levels:

1. **Proxy-level Prometheus metrics** — counters, gauges, histograms tracked by the proxy itself
2. **Per-instance vLLM metrics** — live telemetry scraped from each running vLLM instance

---

## 1. Prometheus endpoint (`GET /metrics`)

```
GET /metrics                          → proxy's own Prometheus metrics (text format)
GET /metrics?model=<model-name>       → forward to that model's vLLM /metrics
```

### Proxy metrics (all prefixed with `llm_proxy_`)

| Metric | Type | Labels | Description |
|--------|------|--------|-------------|
| `llm_proxy_requests_total` | Counter | `model`, `endpoint`, `status` | Total proxied requests to upstream vLLM instances |
| `llm_proxy_request_duration_seconds` | Histogram | `model`, `endpoint` | Proxied request latency in seconds (buckets: 0.01–120s + +Inf) |
| `llm_proxy_upstream_errors_total` | Counter | `model`, `endpoint`, `error_type` | Upstream connection/timeout errors by type |
| `llm_proxy_active` | Gauge | `model`, `endpoint` | Currently in-flight proxied requests |
| `llm_proxy_downstream_disconnects_total` | Counter | _(none)_ | Total downstream client disconnects during proxy |
| `llm_proxy_upstream_healthy` | Gauge | `model` | Whether a model has a READY upstream endpoint (1=ready, 0=not ready) |

### Endpoint label derivation

The `endpoint` label is derived from the upstream URL path:

> `/v1/chat/completions` → `chat.completions`
> `/v1/messages` → `messages`
> `/metrics` → `metrics`
> `/v1/audio/transcriptions` → `audio.transcriptions`

---

## 2. Structured JSON summary (`GET /admin/metrics/summary`)

A pre-computed, JSON-friendly aggregation of all proxy metrics. Requires session auth.

### Response schema

```jsonc
{
  // ── Request counts ──────────────────────────────────────
  "requests": {
    "total": 1247,                     // lifetime proxied requests (all endpoints)
    "by_endpoint": {
      "chat.completions": {
        "total": 892,
        "by_status": {
          "200": 885,
          "500": 7
        }
      },
      "responses": {
        "total": 355,
        "by_status": {
          "200": 355
        }
      }
    }
  },

  // ── Currently in-flight requests ────────────────────────
  "active_requests": {
    "total": 3,                        // sum across all endpoints
    "by_endpoint": {
      "chat.completions": 2,
      "responses": 1
    }
  },

  // ── Upstream errors ─────────────────────────────────────
  "errors": {
    "total": 12,
    "by_endpoint": {
      "chat.completions": {
        "total": 12,
        "by_type": {
          "ConnectError": 8,
          "ReadTimeout": 4
        }
      }
    }
  },

  // ── Latency percentiles (by model) ──────────────────────
  "latency_by_model": {
    "llama-3-70b": {
      "avg": 2.4,       // mean latency in seconds
      "sum": 2140.8,    // total seconds across all requests
      "count": 892,     // number of requests sampled
      "p50": 1.8,       // median
      "p95": 8.2,       // 95th percentile
      "p99": 15.1       // 99th percentile
    }
  },

  // ── Latency percentiles (by endpoint, not just model) ──
  "latency": {
    "chat.completions": {
      "avg": 2.4, "sum": 2140.8, "count": 892,
      "p50": 1.8, "p95": 8.2, "p99": 15.1
    }
  },

  // ── Downstream disconnects ──────────────────────────────
  "downstream_disconnects": 5,

  // ── Model health ────────────────────────────────────────
  "upstream_health": {
    "llama-3-70b": true,
    "claude-3.5-sonnet": false
  }
}
```

### Notes on latency percentiles

- `p50`, `p95`, `p99` are computed server-side from the Prometheus histogram buckets using linear interpolation
- Fields are `null` when there are no samples (no requests proxied yet)
- `avg`, `sum`, `count` are always present (0 when no data)
- Extremely high values (NaN, Inf) are coerced to `null`

---

## 3. Per-instance live stats (part of `GET /admin/dashboard`)

The dashboard endpoint enriches each READY/STARTING endpoint with live data scraped from **that vLLM instance's own `/metrics`** endpoint.

### `endpoint_stats[]` array

```jsonc
[
  {
    "model": "llama-3-70b",
    "host": "node-01",
    "port": 8000,
    "state": "READY",                  // "READY" or "STARTING"
    "slurm_job_id": "123456",
    "last_health_at": "2026-06-08T10:00:00Z",
    "uptime_seconds": 11520,           // seconds since endpoint creation
    "vllm_version": "0.8.3",          // from vLLM /v1/models

    // vLLM /metrics derived (may be null if scrape fails):
    "gpu_cache_usage": 0.72,           // GPU KV cache utilization (0.0–1.0)
    "active_requests": 12,             // currently running requests
    "pending_requests": 2,             // waiting in queue
    "throughput_tps": 45.2            // generation tokens per second
  }
]
```

### Fetching logic (conceptual)

```python
# In the dashboard handler, for each READY/STARTING endpoint:
vm = await fetch_vllm_metrics(endpoint.host, endpoint.port)
# Returns: { gpu_cache_usage, active_requests, pending_requests, throughput_tps }
```

This call timeouts after 5 seconds and per-endpoint calls are parallelized with `asyncio.gather`.

---

## 4. Per-model health (`le` gauge)

The `llm_proxy_upstream_healthy{model="..."}` gauge is set by the `health_worker` background task in `app/main.py`. It runs every 60 seconds and marks each model as:

- `1` = has at least one READY Endpoint in the database
- `0` = no READY endpoint

This is the data backing `upstream_health` in the metrics summary.

---

## 5. Integration guide

### Minimal frontend render plan

Given the two data sources, a dashboard popover typically needs:

| Section | Data source | Endpoint |
|---------|-------------|----------|
| Active requests gauge + totals | `active_requests`, `requests`, `downstream_disconnects` | `/admin/metrics/summary` |
| Model health summary | `upstream_health` | `/admin/metrics/summary` |
| Per-instance cards (cache, throughput, etc.) | `endpoint_stats[]` | `/admin/dashboard` |
| Latency table | `latency_by_model` | `/admin/metrics/summary` |

### Polling

- The metrics summary is polled **every 60 seconds** alongside the dashboard refresh
- Errors are silently ignored (metrics are non-critical)

See [`app/ui/app.js`](../app/ui/app.js) function `renderMetricsPopover()` for the full frontend implementation (~140 lines).

---

## 6. Adding new metrics

To add a new metric:

1. Define a new `prometheus_client` metric at the module level in [`app/metrics.py`](../app/metrics.py) with the `llm_proxy_` prefix
2. Add recording calls (`.inc()`, `.observe()`, `.set()`) at the relevant points in `app/proxy.py`
3. The metric will automatically appear in:
   - The raw `GET /metrics` output
   - Can be added to `get_metrics_summary()` return dict for structured JSON access
