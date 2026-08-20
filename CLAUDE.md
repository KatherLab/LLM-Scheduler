# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Commands

```bash
# Install dependencies
uv sync

# Run development server
uv run uvicorn app.main:app --host 0.0.0.0 --port 9000 --reload

# Lint
uv run ruff check

# Format
uv run ruff format

# Run tests
uv run pytest

# Run with Docker
docker-compose up --build
```

## Tests

Tests live in `tests/` and use `pytest` with `pytest-asyncio`. Run with `uv run pytest`.

## Architecture Overview

FastAPI app managing GPU bookings and vLLM model serving on Slurm clusters. Acts as a shared GPU calendar + OpenAI-compatible API proxy.

### Key Files

| File | Purpose |
|------|---------|
| `app/main.py` | Entry point, runs 5 supervised background workers |
| `app/dependencies.py` | Module-level `engine`, `SessionLocal`, `init_db()` (auto-creates tables + runs migrations) |
| `app/db.py` | SQLAlchemy engine factory with SQLite WAL mode, busy timeout, parent dir creation |
| `app/settings.py` | All env vars via pydantic-settings |
| `app/schemas.py` | Pydantic request/response models (LeaseCreate, LeaseOut, DashboardResponse, etc.) |
| `app/models.py` | SQLAlchemy ORM: Lease and Endpoint tables with full field definitions |
| `app/admin.py` | Booking/lease CRUD, logs, dashboard API |
| `app/proxy.py` | OpenAI-compatible proxy (chat, responses, audio, models) with streaming |
| `app/planner.py` | Lane-based GPU allocation: conflict detection, ASAP search, contiguous GPU blocks |
| `app/slurm.py` | Slurm integration: sbatch, scancel, squeue, sacct, scontrol |
| `app/router_core.py` | Endpoint selection, health checks, vLLM metrics scraping |
| `app/public_api.py` | Read-only public API for external schedule viewers (SCHEDULE_API_KEY) |
| `app/metrics.py` | Prometheus metric definitions, `track_proxy()` context manager, `get_metrics_summary()` |
| `app/lifecycle_logger.py` | Rotating file logger for lifecycle events (separate from proxy logs) |
| `app/catalog.py` | Model catalog from YAML with auto-reload on file change, plus shared `defaults:` merging |
| `app/auth.py` | HMAC-signed cookie sessions |
| `app/ui/app.js` | ~1800-line vanilla JS frontend (no framework), SVG timeline with drag-and-drop |
| `app/ui/login.html` | Login page |
| `templates/vllm_job.sh` | Slurm job template — starts vLLM, registers back to router via HTTP POST |
| `docs/metrics.md` | Full documentation of proxy Prometheus metrics |
| `Dockerfile` | Multi-stage build, Slurm binaries bind-mounted from host |
| `docker-compose.yml` | Full compose with Slurm/munge volume mounts |

### App Startup (`app/main.py`)

- **Lifespan handler** (`@asynccontextmanager`): replaces deprecated `@app.on_event`
  1. Calls `init_db()` — creates tables, runs migrations (e.g. adding `vllm_version` column)
  2. Creates `httpx.AsyncClient` singleton shared across workers
  3. Launches all 5 background workers as supervised tasks via `_supervised()` wrapper
  4. On shutdown: cancels all workers, closes httpx client
- **`_supervised()`**: wraps each worker in a forever-loop; if a worker crashes, logs error and restarts after 2s delay; only exits on `asyncio.CancelledError`

### Background Workers (`app/main.py`, 3-phase pattern)

1. **Snapshot from DB** (fast, no I/O)
2. **External operations** outside DB session (HTTP calls, Slurm commands)
3. **Apply state changes** back to DB

| Worker | Interval | Purpose |
|--------|----------|---------|
| `health_worker` | 60s | Poll vLLM /health; STARTING→READY, READY→FAILED |
| `planned_submit_worker` | 5s | Submit Slurm jobs for planned bookings (120s lead time) |
| `endpoint_cleanup_worker` | 15s | Kill jobs for expired/canceled leases |
| `slurm_reconcile_worker` | 5s | Cross-reference DB with squeue/sacct |
| `retry_worker` | 10s | Retry FAILED leases (up to VLLM_MAX_RETRIES) |

### Lease State Machine

```
PLANNED → SUBMITTED → STARTING → RUNNING → ENDED
                              ↓
                           FAILED → RETRYING → SUBMITTED
```

### Lease Model Fields

Beyond core booking fields, the Lease ORM model (`app/models.py`) stores:
- `model_path`, `tool_args`, `extra_args`, `reasoning_parser`, `gpu_memory_utilization` — vLLM launch configuration
- `venv_activate`, `env_json` — per-lease virtualenv and environment overrides
- `retry_count`, `failed_at` — retry tracking
- `requested_cpus`, `requested_mem`, `requested_port` — resource requests
- `notes` — user-facing notes (who booked it, why)

### GPU Planner

- Lane-based allocation: GPUs split into contiguous blocks per model requirement
- Back-to-back bookings allowed with 30s tolerance
- ASAP mode finds earliest available slot across all lanes
- Conflict detection ensures no overlapping GPU assignments

### vLLM Registration Protocol

When a Slurm job starts (`templates/vllm_job.sh`):
1. Picks a random free port on the compute node
2. POSTs to `ROUTER_REGISTER_URL` with `slurm_job_id`, `model`, `host`, `port`, `vllm_version`
3. Retries registration up to 12 times (60s) in case the router is restarting
4. The router creates an Endpoint record (state=STARTING) via the admin `internal_router`
5. `health_worker` polls the vLLM /health until healthy, then transitions to READY

### Dependencies

- **httpx.AsyncClient**: Module-level singleton with connection pooling, shared across workers
- **SessionLocal**: SQLAlchemy sessions created per-request (not shared), SQLite with WAL mode, 5s busy timeout, synchronous=NORMAL
- **Settings**: pydantic-settings from `.env`

### Frontend

- **No framework** — vanilla JS with Tailwind CSS
- **Global state**: `DASH` object, `MODEL_MAP`, `METRICS_DATA`
- **Key features**: SVG timeline with zoom/pan/minimap, drag-and-drop from catalog, click-to-edit, dark/light mode
- **Refresh**: Polls `/admin/dashboard` every 60s

### Lifecycle Logger

`app/lifecycle_logger.py` writes structured lifecycle events to a separate rotating log file (`vllm_log_dir/lifecycle.log`, 50 MB, 5 backups):
- `log_health_check()` — health poll results with elapsed time, consecutive failures
- `log_state_transition()` — lease/endpoint state changes with reason
- `log_slurm_action()` — submit/cancel/extend/retry actions
Also logs to stderr at INFO level. Uses a dedicated `vllm_lifecycle` logger (non-propagating).

### Metrics System

`app/metrics.py` defines Prometheus metrics (all prefixed `llm_proxy_`):
- **Counters**: `requests_total`, `upstream_errors_total`, `downstream_disconnects_total`
- **Histogram**: `request_duration_seconds` (buckets 0.01–120s)
- **Gauge**: `active` (in-flight), `upstream_healthy` (per-model READY state)
- **`track_proxy()` context manager**: records timing/status for each proxy call
- **`get_metrics_summary()`**: JSON-friendly aggregation of all metrics (used by `/admin/metrics/summary`)
- Raw Prometheus at `GET /metrics`, forwarded vLLM metrics at `GET /metrics?model=<name>`
- Full reference: `docs/metrics.md`

## Configuration

Copy from `config/`:
- `config/example.env` → `.env`
- `config/models.example.yaml` → `config/models.yaml` (model catalog)

### Model Catalog `defaults:`

`config/models.yaml` may carry an optional top-level `defaults:` block with the same keys as a
model entry. It is merged into every model at load time (`load_catalog`), so cluster-wide flags
like `--enable-prompt-tokens-details` live in one place. Merge rules:
- Scalars (`venv_activate`, `cpus`, `mem`, `gpu_memory_utilization`, `reasoning_parser`, …) — per-model value wins
- `env` — dict merge, per-model key wins
- `extra_args` / `tool_args` — `merge_args()` prepends the defaults, dropping any default flag the model also sets (so `--max-num-batched-tokens` is overridden, not passed twice); `--flag=value` and `--flag value` forms are treated as the same flag
- `name`, `model_path`, `notes`, `tags` — always per-model

Defaults are resolved at catalog load, so they apply to leases created afterwards; existing leases
keep the args baked into their row. An explicit `extra_args`/`tool_args` on `POST /admin/leases`
replaces the merged catalog value entirely.

### All Environment Variables (`app/settings.py`)

| Variable | Default | Description |
|----------|---------|-------------|
| `ROUTER_HOST` | `0.0.0.0` | Listen address |
| `ROUTER_PORT` | `9000` | Listen port (must match docker-compose host-side mapping) |
| `PUBLIC_HOSTNAME` | `127.0.0.1` | Advertised hostname to Slurm jobs |
| `DATABASE_URL` | `sqlite:////var/lib/vllm-router/router.db` | SQLAlchemy DB URL |
| `AUTH_PASSWORD` | `changeme` | Login password for admin UI |
| `AUTH_SECRET_KEY` | `""` | HMAC signing key for sessions (auto-generated if empty) |
| `AUTH_SESSION_MAX_AGE_SECONDS` | `86400` | Session TTL |
| `SLURM_PARTITION` | — | Slurm partition |
| `SLURM_ACCOUNT` | — | Slurm account |
| `SLURM_QOS` | — | Slurm QoS |
| `SLURM_NODELIST` | — | Slurm node constraints |
| `SLURM_CPUS_PER_TASK` | `16` | CPUs requested per Slurm job |
| `SLURM_MAIL_USER` | — | Email for Slurm notifications |
| `SLURM_MAIL_TYPE` | `FAIL,END,TIME_LIMIT` | Slurm notification events |
| `VLLM_LOG_DIR` | `./logs` | Lifecycle log + Slurm job output directory |
| `SBATCH_TEMPLATE_PATH` | `/opt/vllm-swapper-router/templates/vllm_job.sh` | Path to Slurm job template |
| `TOTAL_GPUS` | `8` | Total available GPUs in the cluster |
| `SCHEDULER_SUBMIT_LEAD_SECONDS` | `120` | How early before begin_at to submit Slurm job |
| `VLLM_API_KEY` | `secret` | API key for vLLM instances |
| `SCHEDULE_API_KEY` | `""` | API key for public read-only endpoints |
| `ALLOW_ON_DEMAND_START` | `false` | Allow immediate job start without booking |
| `ON_DEMAND_MAX_WAIT_SECONDS` | `30` | Max wait for on-demand start |
| `VLLM_HEALTH_TIMEOUT_SECONDS` | `800` | Max wait for vLLM to become healthy |
| `VLLM_MAX_RETRIES` | `1` | Max retries for failed leases |
| `VLLM_RETRY_DELAY_SECONDS` | `10` | Delay before retry |

## API Endpoints

- **UI**: `GET /`, `GET /login`
- **Auth**: `POST /api/login`, `POST /api/logout`
- **Admin** (session auth):
  - CRUD on `/admin/leases`, `/admin/endpoints`
  - `/admin/dashboard` — timeline data + endpoint stats
  - `/admin/metrics/summary` — JSON metrics aggregation
  - `/admin/leases/{id}/logs` — Slurm job stdout/stderr
  - `/admin/leases/{id}/extend`, `/admin/leases/{id}/shorten`
- **Internal** (used by vLLM registration): `POST /admin/internal/endpoints/register` (no auth)
- **Public** (`SCHEDULE_API_KEY`): `GET /api/v1/schedule`, `/api/v1/schedule/models`, `/api/v1/schedule/leases`
- **Proxy** (OpenAI-compatible, no auth): `POST /v1/chat/completions`, `POST /v1/responses`, `POST /v1/messages`, `POST /v1/audio/transcriptions`, `POST /v1/audio/translations`, `GET /v1/models`
- **Health**: `GET /health`
- **Metrics**: `GET /metrics`, `GET /metrics?model=<name>`

## Docker Deployment

The Docker setup (`Dockerfile` + `docker-compose.yml`) bind-mounts Slurm binaries, libraries, and config from the host:
- `/usr/bin/sbatch`, `scancel`, `scontrol`, `squeue`, `sacct`
- `/usr/lib/.../slurm/` shared library
- `/etc/slurm`, `/etc/munge`, `/var/run/munge`
- Container runs at host UID/GID for Slurm submit permissions
- `extra_hosts` includes `host.docker.internal` for local testing
- Persistent data volume for SQLite, model catalog mounted as read-only
- Docker `HEALTHCHECK` hits `GET /health` every 30s
