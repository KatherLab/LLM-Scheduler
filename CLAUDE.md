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

# Run with Docker
docker-compose up --build
```

No test files exist yet — runtime validation relies on Pydantic schemas, structured lifecycle logging, and Prometheus metrics.

## Architecture Overview

FastAPI app managing GPU bookings and vLLM model serving on Slurm clusters. Acts as a shared GPU calendar + OpenAI-compatible API proxy.

### Key Files

| File | Purpose |
|------|---------|
| `app/main.py` | Entry point, runs 5 supervised background workers |
| `app/admin.py` | Booking/lease CRUD, logs, dashboard API |
| `app/proxy.py` | OpenAI-compatible proxy (chat, responses, audio, models) with streaming |
| `app/planner.py` | Lane-based GPU allocation: conflict detection, ASAP search, contiguous GPU blocks |
| `app/slurm.py` | Slurm integration: sbatch, scancel, squeue, sacct, scontrol |
| `app/router_core.py` | Endpoint selection, health checks, vLLM metrics scraping |
| `app/models.py` | SQLAlchemy ORM: Lease (PLANNED→SUBMITTED→STARTING→RUNNING→ENDED/FAILED) and Endpoint (STARTING→READY→STOPPED/FAILED) |
| `app/catalog.py` | Model catalog from YAML with auto-reload on file change |
| `app/auth.py` | HMAC-signed cookie sessions |
| `app/ui/app.js` | ~1800-line vanilla JS frontend (no framework), SVG timeline with drag-and-drop |

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

### GPU Planner

- Lane-based allocation: GPUs split into contiguous blocks per model requirement
- Back-to-back bookings allowed with 30s tolerance
- ASAP mode finds earliest available slot across all lanes
- Conflict detection ensures no overlapping GPU assignments

### Dependencies

- **httpx.AsyncClient**: Module-level singleton with connection pooling, shared across workers
- **SessionLocal**: SQLAlchemy sessions created per-request (not shared), SQLite with WAL mode and 5s busy timeout
- **Settings**: pydantic-settings from `.env`, notably `TOTAL_GPUS`, `PUBLIC_HOSTNAME`, `AUTH_PASSWORD`

### Frontend

- **No framework** — vanilla JS with Tailwind CSS
- **Global state**: `DASH` object, `MODEL_MAP`, `METRICS_DATA`
- **Key features**: SVG timeline with zoom/pan/minimap, drag-and-drop from catalog, click-to-edit, dark/light mode
- **Refresh**: Polls `/admin/dashboard` every 60s

## Configuration

Copy from `config/`:
- `config/example.env` → `.env`
- `config/models.example.yaml` → `config/models.yaml` (model catalog)

Key env vars: `TOTAL_GPUS` (default 8), `PUBLIC_HOSTNAME`, `AUTH_PASSWORD`, `DATABASE_URL`, `VLLM_API_KEY`, `SLURM_PARTITION`.

## API Endpoints

- **UI**: `GET /`, `GET /login`
- **Auth**: `POST /api/login`, `POST /api/logout`
- **Admin** (session auth): CRUD on `/admin/leases`, `/admin/endpoints`, `/admin/metrics/summary`
- **Public** (`SCHEDULE_API_KEY`): `GET /api/v1/schedule`, `/api/v1/schedule/models`, `/api/v1/schedule/leases`
- **Proxy** (OpenAI-compatible, no auth): `POST /v1/chat/completions`, `POST /v1/responses`, `POST /v1/messages`, `POST /v1/audio/transcriptions`, `POST /v1/audio/translations`, `GET /v1/models`
