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
| `app/main.py` | Entry point, runs 6 supervised background workers |
| `app/dependencies.py` | Module-level `engine`, `SessionLocal`, `init_db()` (creates tables; no migrations) |
| `app/db.py` | SQLAlchemy engine factory with SQLite WAL mode, busy timeout, parent dir creation |
| `app/settings.py` | All env vars via pydantic-settings |
| `app/schemas.py` | Pydantic request/response models (LeaseCreate, LeaseOut, DashboardResponse, etc.) |
| `app/models.py` | SQLAlchemy ORM: Lease and Endpoint tables with full field definitions |
| `app/admin.py` | Booking/lease CRUD, logs, dashboard API |
| `app/proxy.py` | OpenAI-compatible proxy (chat, responses, audio, models) with streaming |
| `app/backends/` | `ClusterBackend` abstraction — all scheduler access goes through here |
| `app/slurm.py` | Back-compat shim over `app/backends` (legacy function names/shapes) |
| `app/router_core.py` | Endpoint selection, health checks, vLLM metrics scraping |
| `app/loadbalancer.py` | Least-loaded routing across replicas; drain flags for handover |
| `app/renew.py` | Rolling renewal for `mode: service` |
| `app/tokens.py` | Slurm JWT acquisition + automatic renewal (static \| file \| command) |
| `app/leader.py` | Leader election (HA) + booking write serialization |
| `app/public_api.py` | Read-only public API for external schedule viewers (SCHEDULE_API_KEY) |
| `app/metrics.py` | Prometheus metric definitions, `track_proxy()` context manager, `get_metrics_summary()` |
| `app/lifecycle_logger.py` | Rotating file logger for lifecycle events (separate from proxy logs) |
| `app/images.py` | Apptainer images: list/delete on the shared FS, build as a Slurm job |
| `app/images_api.py` | Admin-only `/admin/images` routes |
| `app/cluster.py` | Cluster topology from `cluster.yaml`: GPU classes, runtimes, pools, quotas |
| `app/inventory.py` | Discovered node/foreign-job snapshot; replaces `TOTAL_GPUS` |
| `app/placement.py` | Per-`(node, gpu)` planner with foreign-job overlay (replaced `planner.py`) |
| `app/scheduling.py` | Bridge: ORM leases ↔ placer, lane offsets, legacy `TOTAL_GPUS` fallback |
| `app/catalog.py` | Model catalog from YAML with auto-reload on file change, plus shared `defaults:` merging |
| `app/auth.py` | HMAC-signed cookie sessions, login, `/api/me` |
| `app/identity.py` | Authentication: LDAP/FreeIPA bind, static provider, break-glass admin |
| `app/authz.py` | Authorization: roles, ownership, `can()` — the entire rule set |
| `app/ui/app.js` | ~1800-line vanilla JS frontend (no framework), SVG timeline with drag-and-drop |
| `app/ui/login.html` | Login page |
| `templates/vllm_job.sh` | Slurm job template — switches on `RUNTIME_KIND` (apptainer\|venv), starts vLLM, registers back |
| `templates/apptainer_build.sh` | Slurm job template — builds one `.sif` from a registry reference |
| `docs/metrics.md` | Full documentation of proxy Prometheus metrics |
| `docs/architecture-v2.md` | Multi-node / heterogeneous / multi-tenant design + phase plan |
| `Dockerfile` | Multi-stage build, Slurm binaries bind-mounted from host |
| `compose.yml` | Full compose with Slurm/munge volume mounts |

### App Startup (`app/main.py`)

- **Lifespan handler** (`@asynccontextmanager`): replaces deprecated `@app.on_event`
  1. Calls `init_db()` — creates tables (no migration system; schema changes mean deleting the DB)
  2. Creates `httpx.AsyncClient` singleton shared across workers
  3. Launches all 6 background workers as supervised tasks via `_supervised()` wrapper
  4. On shutdown: cancels all workers, closes httpx client
- **`_supervised()`**: wraps each worker in a forever-loop; if a worker crashes, logs error and restarts after 2s delay; only exits on `asyncio.CancelledError`

### Background Workers (`app/main.py`, 3-phase pattern)

1. **Snapshot from DB** (fast, no I/O)
2. **External operations** outside DB session (HTTP calls, Slurm commands)
3. **Apply state changes** back to DB

| Worker | Interval | Purpose |
|--------|----------|---------|
| `inventory_worker` | 60s | Refresh node + foreign-job snapshot (replaces `TOTAL_GPUS`) |
| `estimate_worker` | 45s | Refresh Slurm's backfill start estimate for queued jobs on `slurm` pools |
| `renew_worker` | 30s | Rolling handover for `mode: service` deployments |
| `health_worker` | 60s | Poll vLLM /health; STARTING→READY, READY→FAILED |
| `planned_submit_worker` | 5s | Submit Slurm jobs for planned bookings (120s lead time) |
| `endpoint_cleanup_worker` | 15s | Kill jobs for expired/canceled leases |
| `slurm_reconcile_worker` | 5s | Cross-reference DB with squeue/sacct |
| `retry_worker` | 10s | Retry FAILED leases (up to VLLM_MAX_RETRIES) |
| `image_build_worker` | 15s | Watch `apptainer build` jobs; verify the `.sif` actually appeared |

### Cluster Backend (`app/backends/`)

All batch-scheduler access goes through the `ClusterBackend` protocol
(`app/backends/base.py`). Select the implementation with `CLUSTER_BACKEND`.

| Module | Purpose |
|--------|---------|
| `types.py` | `JobSpec`, `JobState`, `ExitInfo`, `NodeInfo`, `ForeignJob`, capability constants |
| `base.py` | The `ClusterBackend` protocol |
| `slurm_parse.py` | Pure parsers (hostlist expansion, GRES, times) — no subprocess, fully unit-tested |
| `slurm_cli.py` | Subprocess implementation; needs Slurm binaries + munge on the app host |
| `slurm_rest.py` | **slurmrestd over JWT** — runs anywhere; verified on Slurm 25.05 / `v0.0.42` |
| `local.py` | In-memory fake — lets scheduling logic be tested without a cluster |

Backends advertise **capabilities** (`CAP_ACCOUNTING`, `CAP_TEST_ONLY`,
`CAP_NODE_DISCOVERY`, `CAP_FOREIGN_JOBS`, `CAP_RESERVATIONS`) probed at
construction. Callers check the capability rather than catching exceptions.

**slurmrestd (`CLUSTER_BACKEND=slurm_rest`)** removes the host dependencies
entirely: no binaries, no munge, no bind-mounts. Field shapes were taken from
the OpenAPI doc slurmrestd serves at `/openapi/v3`, not guessed — they shift
between API versions. On `v0.0.42`:

- numbers are wrapped `{"set":bool,"infinite":bool,"number":N}` — unset/infinite
  must map to `None`, **not 0** (0 reads as "ended in 1970")
- `job_state` and node `state` are *lists* of flags, joined with `+`
- `--gres` becomes `tres_per_node="gres/gpu:gpu48:2"` (there is no `gres` field)
- `time_limit` is **minutes**, `memory_per_node` is **megabytes**
- `mail_type` has no `TIME_LIMIT` — it is `TIME=100%`
- `gres_detail` gives **exact GPU indices** (`gpu:gpu96:1(IDX:2)`), so the
  placer blocks precisely those GPUs instead of assuming the first N
- REST takes the script *contents*, so the template needs no shared filesystem

`CAP_TEST_ONLY` is **absent** from the REST backend — `sbatch --test-only` has
no REST equivalent, so start estimates still need `SlurmCliBackend`.

**Auth failures are `ClusterUnavailableError`, never "empty cluster"** —
treating an expired token as "no nodes / all jobs dead" would cancel
everything. Note slurmrestd returns **HTTP 511**, not 401, and can also report
the failure in a 200 body's `errors` array; both are detected.

### Token renewal (`app/tokens.py`)

`scontrol token` defaults to a **30-minute** lifespan, so a static `SLURM_JWT`
becomes an outage at an unpredictable time. `SLURM_TOKEN_MODE`:

| Mode | Source | Notes |
|---|---|---|
| `static` | `SLURM_JWT` | Simplest; expires |
| `file` | `SLURM_TOKEN_FILE` | **Safest** — the app holds a token, never a credential that mints tokens |
| `command` | `SLURM_TOKEN_COMMAND`, or the `SLURM_TOKEN_SSH_*` recipe | Fully automatic |

Renewal is driven by the token's own `exp` claim, not an assumed lifespan, and
is both **proactive** (refresh `SLURM_TOKEN_REFRESH_MARGIN_SECONDS` before
expiry) and **reactive** (a 511 forces a refresh and exactly one retry). So a
token revoked early self-heals instead of pausing the scheduler.

The command is split with `shlex` and run **without a shell**, so a stray quote
cannot become injection. Key paths are redacted from logs.

**The SSH key can mint Slurm credentials.** Pin it to a forced command in the
remote `authorized_keys` so a stolen key cannot open a shell, and supply
`known_hosts` — an impersonated host could return a token pointing at its own
slurmrestd. See `secrets/README.md`.

**`sacct` is now optional.** Job exit reasons degrade through three sources:
sacct → `scontrol show job` (works from controller memory for `MinJobAge`, no
slurmdbd needed) → the job's stderr log. This is what removes the requirement to
run on the accounting host.

`ClusterUnavailableError` means the controller is unreachable and is distinct
from "the job is gone" — the reconciler must skip a cycle rather than concluding
every model died.

See `docs/architecture-v2.md` for the full multi-node / heterogeneous design.

### Cluster Topology (`app/cluster.py`, `config/cluster.yaml`)

`cluster.yaml` describes what exists; `models.yaml` describes what to run on it.

- **GPU classes** are the GRES types Slurm knows. On this cluster:
  `gpu24 / gpu48 / gpu80 / gpu96`. They must match `slurm.conf`, because the
  site's `job_submit.lua` validates and rewrites `--gres` against them.
- **Runtimes** (`apptainer` | `venv`) are selected *by GPU class*, so an
  aarch64 node gets an aarch64 image without any model naming one.
- **Pools** carry `scheduling: managed | slurm` — see below.
- **Quotas** resolve `default -> group -> per_pool`; admins are exempt.

**`TOTAL_GPUS` is derived, not configured.** `inventory_worker` refreshes a node
snapshot every 60s. A discovery failure keeps the previous snapshot rather than
reporting an empty cluster.

**Always send typed GRES.** `job_submit.lua` appends `gpu24` as a hard
constraint to any *untyped* `--gres=gpu:N`, so an untyped job silently never
reaches a larger card. `_gres_for()` emits `gpu:<class>:<n>`.

Nodes may hold **several GPU classes at once** (`europa` is
`{gpu24: 1, gpu48: 1}`), so `NodeInfo.gpus` is a tuple of `GpuGroup`, never a
single type.

### `managed` vs `slurm` pools

| | `managed` | `slurm` |
|---|---|---|
| Placement by | our planner | Slurm backfill |
| Submission | `--begin` + `--nodelist` + typed `--gres` | typed `--gres` only |
| Shown as | **confirmed** (solid) | **estimated** (ghost block) |

**Start estimates come from two places.** Before submission, `sbatch
--test-only` validates and reports a start time without queueing — but it has
**no slurmrestd equivalent**, so `ESTIMATE_BACKEND=slurm_cli` is needed
alongside a REST primary, and an off-cluster router legitimately has neither.
After submission, Slurm's backfill estimate arrives free via
`JobState.start_time` and `estimate_worker` writes it to
`Lease.estimated_start` — this works over REST.

`POST /admin/leases/preview` returns a `confidence` of `confirmed` |
`estimated` | `unknown` | `impossible`. **`unknown` is a real answer**: when
nothing can produce an estimate we say so rather than inventing a time.

The timeline draws an estimated booking at Slurm's estimate (not at the
requested time, which the user does not have), translucent and dashed, with a
tick marking the requested time so drift is visible. With no estimate it is
fainter still, because its position is a guess.

We deliberately do **not** create per-booking Slurm reservations: on dedicated
nodes nothing else can land there, and on shared nodes a reservation would take
capacity from other users while Slurm's backfill already answers "earliest
slot" better than we can.

### Catalog `variants`

`gpus`/`tp`/`gpu_memory_utilization` are a function of *where a model lands*,
not fixed properties. Layering is
`defaults -> model -> variants[gpu_class] -> per-lease override`; `extra_args`
goes through `merge_args()` at every layer. `requires:` filters candidate
classes; an explicit variant is itself a statement of support.

### GPU memory is configured in GB, not fractions

`--gpu-memory-utilization` is a fraction, but its *denominator varies*: on a
48 GB card `0.9` is ~43 GB; on a DGX Spark (GB10) the denominator is the whole
128 GB shared with the OS, so `0.9` would take ~115 GB and destabilise the node.

So the config states physical facts and the fraction is derived:

- `GpuClass.reserved_gb` — memory the host needs. `default_utilization` follows
  from it, and `usable_gb` is what models may actually take.
- `GpuClass.gpu_memory_utilization_max` — a belt-and-braces ceiling.
- `CatalogModel.memory_gb` — absolute budget, per model (and per variant).
- `LeaseCreate.memory_gb` — the UI's per-booking slider, seeded from the
  catalog default and bounded by the smallest eligible GPU class.

`cap_utilization()` always clamps into `(0, 1]`, so asking for more GB than the
card holds cannot produce a fraction above 1.

### Apptainer images (`app/images.py`, `app/images_api.py`)

**A build is a Slurm job, and has to be.** Apptainer cannot cross-build, so an
aarch64 `.sif` can only be produced on an aarch64 node. Architecture is already
known — it is a property of the GPU class in `cluster.yaml` — so
`build_targets()` answers "where can I build aarch64" from the discovered
inventory with no extra configuration. Where a partition holds more than one
architecture the job pins a node; where it is uniform it does not, so Slurm can
place the build wherever it fits.

Verified on this cluster: rootless `apptainer build docker://…` works as
`slurmweb` on both arches (no root, no fakeroot), and `vllm/vllm-openai` is
multi-arch, so one tag serves both.

The build requests **no GPU** — it needs CPU and disk, and holding a card idle
while it unpacks layers would be pure waste. That is also why
`build_sbatch_argv` now omits `--gres` entirely at 0 GPUs: `--gres=gpu:0` would
be rewritten by `job_submit.lua` into a hard `gpu24` constraint.

Two failure modes are designed against:

- **A half-written image is worse than none** — it looks like a working
  upgrade. So the job builds to `<name>.sif.building.<jobid>`, runs
  `apptainer exec … /bin/true` to prove the result executes on that
  architecture, and only then renames it into place (same directory, so the
  rename is atomic).
- **Apptainer's scratch defaults to `$HOME`**, which the service account does
  not have on every node, and `/tmp` is a RAM-backed tmpfs on some. So
  `APPTAINER_CACHEDIR`/`APPTAINER_TMPDIR` are set explicitly from
  `IMAGE_BUILD_SCRATCH`.

`image_build_worker` judges a finished build by **whether the `.sif` is
actually there**, not by the exit code alone — and says so explicitly when it
has no filesystem view to check with.

Listing and deleting are plain filesystem operations, so `APPTAINER_IMAGE_DIR`
must be mounted into the router for those. When it is not, the API returns the
reason rather than an empty list, which would read as "you have no images".

**Deleting an image `cluster.yaml` points at is refused** unless forced: the
file simply stops existing and every *future* job for that GPU class dies at
launch. Names and source references are validated against strict patterns —
both end up in a path or an argv on a shared filesystem.

Nothing here writes `cluster.yaml`. Which image a GPU class uses stays a
deliberate edit; the UI shows the path to paste.

### Co-location: several models on one GPU (`app/colocation.py`)

The cluster gives us no way to subdivide a GPU: **MPS is disabled**
(`job_submit.lua` rejects `mps:`), **MIG** is static and absent on GB10 (where
it *is* configured — `ganymede` → 8× `gpu24` — the scheduler already sees plain
GPUs and needs none of this), and **`gres/shard`** would need a `slurm.conf`
change.

So co-location happens *inside* one job: Slurm grants one GPU, the job runs
several vLLM servers on it, each on its own port, each registering as an
ordinary endpoint — so routing, health checks and metrics are unchanged.

`memory_gb` is **required** for every co-tenant: absolute budgets add up,
fractions of a shared card do not. The group is proven to fit before
submission, because a half-started group is worse than a refused booking (the
models that did start look healthy). Fractions are **floored**, never rounded.

Trade-offs, all deliberate: co-tenants share one Slurm job (they stop
together, though each is restarted individually inside it), and the GPU
time-slices between them — fine for embedding/reranking, but it will quietly
invalidate benchmark numbers.

### Placement (`app/placement.py`, `app/scheduling.py`)

Grid is per `(node, gpu_index)`; contiguity only matters *within* a node.
Foreign jobs (other users') are overlaid as immovable blocks — this is what
makes the calendar honest on a shared partition. Placement is **best-fit** to
avoid a 1-GPU booking breaking a node's only contiguous block.

`guard_gap_seconds` **requires** separation between bookings (the old
`OVERLAP_TOLERANCE` did the opposite — it permitted tighter packing). It
defaults to **0** so deployments without a `cluster.yaml` keep the historical
"A ends 18:00, B starts 18:00 is fine" behaviour; real gaps are opted into per
GPU class.

**Legacy fallback.** The old flat-lane model *is* "one node with N untyped
GPUs", so with no `cluster.yaml` and no discovered inventory,
`scheduling.effective_nodes()` synthesises exactly that from `TOTAL_GPUS` and
placement reproduces the retired `planner.py` byte-for-byte. Covered by
`tests/test_scheduling_bridge.py`.

**Lanes are a rendering concern.** Each node gets a stable `lane_offset`
(ordered by name), and `LeaseOut.lane_start = offset + gpu_start`. The flat
SVG timeline therefore works unchanged across nodes; `LeaseOut.node` /
`gpu_start` and `DashboardResponse.nodes` let it draw node boundaries and
labels. Drained nodes are excluded from the layout entirely.

### Replicas, routing and `mode: service`

`choose_ready_endpoint` is **least-loaded**, not newest. Two signals:
in-flight requests we dispatched (exact, free — this process *is* the proxy;
tracked inside `track_proxy`, keyed on `host:port` **not** the route label) and
vLLM's scraped queue depth (breaks ties, ignored when stale).

**`mode: session`** stops at `end_at`. **`mode: service`** is a chain of
bounded jobs — clusters cap wall time, so a month-long booking is not an
option. `renew_worker` does: submit replacement → wait for READY → drain old →
cancel old. The drain flag is what makes the handover zero-downtime, so this
depends on least-loaded routing. It also gives rolling vLLM upgrades free.

`DRAIN_TIMEOUT` exists so a hung stream cannot hold a GPU forever. Service
leases (like locked ones) are skipped by `endpoint_cleanup_worker` — they are
retired explicitly after handover, not reaped by `end_at`.

### High availability (`HA_ENABLED`)

Every instance serves proxy traffic; **only the leader runs the workers** —
two instances submitting for one booking would double-submit. The lock is a
heartbeat row (`locks` table), deliberately not consensus: a brief gap with no
leader is fine since every worker is periodic, but two leaders is not. A DB
error makes an instance stand down rather than keep a stale `is_leader`.

Requires a shared database, i.e. **Postgres**. `booking_lock()` serializes the
read-validate-insert booking path (`SELECT … FOR UPDATE` on Postgres; SQLite
serializes writes anyway) — without it two concurrent bookings can both pass
validation and double-book the same GPUs.

### Identity & Authorization

`AUTH_MODE=password` keeps the historical shared secret. `AUTH_MODE=ldap` binds
against FreeIPA and derives roles from group membership. **The break-glass local
admin (`AUTH_PASSWORD`) always works** — an IPA outage must not lock you out.

Roles: `ADMIN_GROUPS` → admin, `POOL_OPERATORS` (`pool:group`) → operator of that
pool, `USER_GROUPS` (empty = everyone) → may create bookings.

Ownership is `owner_sub` (LDAP uid) plus optional `owner_group`, so a team can
co-own a booking and it survives someone being on holiday. The whole rule is
`authz.can()`; endpoints call `_authorize()` and `LeaseOut` carries
`can_edit`/`can_cancel`/`can_lock` so the UI never reimplements it.

`locked` is a production flag, not just a permission bit: a locked lease refuses
owner cancellation **and** is skipped by `endpoint_cleanup_worker`.

There is deliberately no migration system — schema changes mean deleting the DB.

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
- `config/cluster.example.yaml` → `config/cluster.yaml` (cluster topology)
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
| `CLUSTER_BACKEND` | `slurm_cli` | Scheduler backend: `slurm_cli` or `local` (in-memory fake) |
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
| `VLLM_LOG_DIR` | `./logs` | Our own lifecycle log, written by this process |
| `JOB_LOG_DIR` | = `VLLM_LOG_DIR` | Slurm job stdout/stderr — a path on the **compute nodes** |
| `JOB_LOG_DIR_LOCAL` | = `JOB_LOG_DIR` | The same directory as this process sees it |
| `APPTAINER_IMAGE_DIR` | `""` | Shared `.sif` directory; empty disables image management |
| `IMAGE_BUILD_SCRATCH` | `<images>/../build-tmp` | Apptainer cache + tmpdir for builds |
| `IMAGE_BUILD_TEMPLATE_PATH` | `./templates/apptainer_build.sh` | Build job template |
| `IMAGE_BUILD_TIME_LIMIT` | `02:00:00` | Wall time for a build job |
| `IMAGE_BUILD_CPUS` | `8` | CPUs for a build job |
| `IMAGE_REGISTRY_USERNAME` / `_PASSWORD` | `""` | Optional private-registry credentials |
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
  - `/admin/images` (admin only) — list; `POST /admin/images/build`;
    `DELETE /admin/images/{name}`; `POST /admin/images/builds/{id}/cancel`;
    `GET /admin/images/builds/{id}/logs`
  - `/admin/leases/{id}/logs` — Slurm job stdout/stderr
  - `/admin/leases/{id}/extend`, `/admin/leases/{id}/shorten`
- **Internal** (used by vLLM registration): `POST /admin/internal/endpoints/register` (no auth)
- **Public** (`SCHEDULE_API_KEY`): `GET /api/v1/schedule`, `/api/v1/schedule/models`, `/api/v1/schedule/leases`
- **Proxy** (OpenAI-compatible, no auth): `POST /v1/chat/completions`, `POST /v1/responses`, `POST /v1/messages`, `POST /v1/audio/transcriptions`, `POST /v1/audio/translations`, `GET /v1/models`
- **Health**: `GET /health`
- **Metrics**: `GET /metrics`, `GET /metrics?model=<name>`

## Container Deployment

`Dockerfile` + `compose.yml`, built for `CLUSTER_BACKEND=slurm_rest`. Compared
to the old CLI setup this needs **no** Slurm binaries, munge socket,
`/etc/slurm`, bind-mounts, or host-matching UID — it runs unprivileged as
`scheduler` (1000:1000) with `no-new-privileges`.

What it *does* need is **network**, which is the real constraint:

| Direction | Why |
|---|---|
| container → slurmrestd `:6820` | all scheduler operations |
| container → compute nodes | health checks + proxying to vLLM's ephemeral ports |
| compute nodes → container | jobs POST to `ROUTER_REGISTER_URL` |

Also needed locally: `config/models.yaml`, `config/cluster.yaml`, and
`templates/vllm_job.sh` (read and inlined into the REST submit, so no shared
filesystem is required for it).

### Two log directories, because two machines write them

`VLLM_LOG_DIR` is ours (`lifecycle.log`, written by this process) and may be
container-local. `JOB_LOG_DIR` is Slurm's: it becomes `--output`/`--error` and
the job's cwd, and is resolved **by the compute node**. Passing a container
path there does not merely hide the logs — Slurm cannot open the output file
and the job dies at launch, with no log to say why. It must be a shared
filesystem, writable by the account the JWT belongs to, and it must already
exist (Slurm will not create it; we `makedirs` it at startup when we can see
it, and warn when we cannot).

Empty `JOB_LOG_DIR` falls back to `VLLM_LOG_DIR` — the historical behaviour,
correct only when the router runs on the cluster's own filesystem.

Job **log viewing** additionally needs that directory mounted here.
`JOB_LOG_DIR_LOCAL` covers the case where it lands under a different path;
mount it at the same path and it is unnecessary. Without any mount, log
viewing is empty and everything else still works.

The legacy CLI mounts are kept commented at the bottom of `compose.yml`; they
are only needed for `ESTIMATE_BACKEND=slurm_cli` (`sbatch --test-only`
previews, which have no REST equivalent).

### Local development on macOS + Podman

`compose.override.yml` exists because the Podman VM has only a default route
via gvproxy, so it cannot reach the cluster network — host VPN routes do not
propagate into the VM. An SSH tunnel on the Mac bridges it:

```bash
ssh -f -N -L 0.0.0.0:16820:localhost:6820 \
          -L 0.0.0.0:12222:localhost:22 <you>@<cluster-host>
podman compose up -d --build
```

The override then points `SLURM_REST_URL` and `SLURM_TOKEN_SSH_HOST` at
`host.containers.internal`. `secrets/known_hosts.container` holds the same host
keys relabelled for the tunnel's `host:port`, since known_hosts entries are
keyed by name. **None of this applies to a Linux VM on the cluster network.**
