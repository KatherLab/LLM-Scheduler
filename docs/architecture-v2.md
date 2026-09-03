# Architecture v2 — Multi-Node, Heterogeneous, Multi-Tenant

Status: **design agreed, implementation in progress**
Supersedes the implicit single-node design described in `CLAUDE.md`.

---

## 1. Why

The current app assumes the cluster is *one exclusive node*. That assumption is
load-bearing in four places, and each one breaks independently as we scale out:

| Assumption | Where | Breaks because |
|---|---|---|
| Cluster is a flat strip of `TOTAL_GPUS` | `settings.py`, `planner.py` | Contiguity is only meaningful *within* a node; heterogeneous VRAM is not representable |
| Lane placement is real | `planner.py:compute_placements` | Lanes are recomputed per render, never persisted, never told to Slurm — a drawing, not an allocation |
| The `leases` table is the whole world | `admin.py:_validate_no_conflicts` | On a shared partition, foreign jobs consume GPUs we cannot see |
| One partition / QoS / nodelist globally | `settings.py` | We need a dedicated LLM partition *and* a general one |
| `gpus`/`tp`/`gpu_memory_utilization` fixed per model | `catalog.py:CatalogModel` | The same model needs different values on H200 vs A100-40 vs GB10 |
| One `venv_activate` per model | `catalog.py` | x86_64 CUDA and aarch64 (DGX Spark) need different runtimes |
| Jobs always submit "now" (`begin=None`) | `admin.py:_submit_to_slurm` | The calendar makes a promise Slurm never agreed to |
| Newest READY endpoint wins | `router_core.py:choose_ready_endpoint` | No scale-out, no load balancing |

The root cause is that a `Lease` row is four things fused together: a calendar
booking, a resource allocation, a vLLM launch spec, and a Slurm job. That is a
1:1 mapping only on a single exclusive node.

## 2. Decisions taken

These were settled up front and constrain everything below.

| # | Decision | Consequence |
|---|---|---|
| 1 | **All Slurm jobs run as one service account** | Attribution lives in our DB, not in Slurm accounting. We stamp `--comment=user:<uid>,lease:<id>` so `squeue` stays traceable. |
| 2 | **Identity from FreeIPA/LDAP**, admin group + per-deployment ownership | Replaces the single shared `AUTH_PASSWORD`. Break-glass local admin retained. |
| 3 | **Single-node models only** (no multi-node TP) | Planner never needs cross-node blocks. **No NCCL/InfiniBand-in-Apptainer work.** Biggest scope reduction. |
| 4 | **Quotas configurable, admins exempt** | Policy layer evaluated at booking time, in the same transaction as conflict validation. |
| 5 | **One app instance; separate Slurm clusters get merged into one** | No `Cluster` parent entity. But the DB merge and instance consolidation is its own migration. |
| 6 | **`slurmrestd` can be deployed** | Primary backend is HTTP/JWT. CLI backend retained for gaps (notably `--test-only`). |
| 7 | **Shared filesystem across all nodes, incl. Spark** | No image/weight distribution problem. One `.sif` per arch on shared storage. |

### 2.1 The one place we deliberately do *not* use Slurm reservations

Reservations are a **guarantee** mechanism. We need a guarantee in neither
default path:

- **Dedicated LLM nodes** — they are already ours; nothing else can land there.
  Our calendar *is* the allocator. A per-booking reservation would duplicate
  that bookkeeping in a second place, require operator privileges, and create
  two sources of truth that can drift. Instead we make our allocation truthful:
  pin `--nodelist=<node>` + `--gres=gpu:<class>:<n>` to exactly what the planner
  chose, and submit with `--begin=<start>`.
  *(Optional, admin-created once: a standing reservation over the whole
  partition for the service account, to keep stray user jobs out.)*

- **General nodes** — the requirement is "earliest slot, whenever that is",
  which is explicitly *not* a guarantee. Slurm's backfill scheduler already
  solves this and has information we don't (the full queue). Reservations here
  would be strictly worse and would take capacity from other cluster users, since
  a reservation forces Slurm to drain nodes ahead of it.

Reservations remain a **rare, explicit escalation** ("Request guaranteed
window", admin-gated) for cases like a fixed demo slot. Not needed for v1.

---

## 3. Resource model

```
Pool          set of nodes + scheduling policy      (who decides placement)
 └ Node       hostname, gpu_class, gpu_count        (discovered, not configured)
GpuClass      vram, arch, default runtime, caps
Runtime       apptainer image | venv activate path
Deployment    a model instance; mode=service|session; N replicas
 └ Replica    one Slurm job
    └ Endpoint  a live vLLM URL          (exists today)
```

Splitting `Deployment` from the booking is what allows long-running services and
bounded benchmark sessions to coexist. Today "long-running" is emulated by
booking a huge window, which fights cluster `MaxWall` limits.

### 3.1 Pool scheduling modes — the central distinction

```yaml
pools:
  - name: llm-dedicated
    partition: llm
    scheduling: managed     # our calendar is truth → we can promise a slot
    operators: [llm-gpu-team]

  - name: general
    partition: general
    scheduling: slurm       # Slurm is truth → we queue and estimate, never promise
```

| | `managed` | `slurm` |
|---|---|---|
| Placement decided by | our planner | Slurm backfill |
| Submission | `--begin` + `--nodelist` + `--gres` | `--gres` only, submit immediately |
| Start time shown as | **confirmed** (solid block) | **estimated** (ghost block) |
| Conflict validation | full, against our grid | none — advisory preview only |
| Foreign jobs | shouldn't exist; surfaced as an alert if they do | expected; overlaid on the timeline |

This distinction must be visible in the UI. Presenting a `slurm`-pool booking as
confirmed is the single most misleading thing the app could do.

### 3.2 Drag-and-drop on `slurm` pools

Two Slurm features map onto the UX directly:

1. **`sbatch --test-only`** — validates and returns
   *"Job N to start at 2026-08-20T15:40:00 … on nodes gpu07"* **without
   submitting**. This is the drag-and-drop preview primitive. No privileges needed.
2. **`squeue --start` / `-o "%S"`** — Slurm's live estimate for a pending job.

Flow: drag → resolve spec → `--test-only` → show *"would start ≈ 15:40"* →
user confirms → real submit → poll `%S` and render a ghost block that firms up
as it approaches; solid once RUNNING.

Caveats to design around: `%S` is `N/A` until backfill has considered the job,
is recomputed only every `bf_interval` (~30s default), and moves as the queue
changes. UI language must say *est.*

---

## 4. Configuration

### 4.1 `config/cluster.yaml` (new)

```yaml
gpu_classes:
  h200:
    vram_gb: 141
    arch: x86_64
    runtime: x86-cuda
    guard_gap_seconds: 120        # see §6.2
  a100:
    vram_gb: 40
    arch: x86_64
    runtime: x86-cuda
  gb10:                            # DGX Spark
    vram_gb: 128
    arch: aarch64
    runtime: arm-spark
    unified_memory: true
    gpu_memory_utilization_max: 0.70   # hard cap, see §4.3
    guard_gap_seconds: 180

runtimes:
  x86-cuda:
    kind: apptainer
    image: /shared/sif/vllm-0.11.0-cu128.sif
    nv: true
    binds: ["/models:/models:ro", "/scratch"]
  arm-spark:
    kind: apptainer
    image: /shared/sif/vllm-0.11.0-arm64.sif
    nv: true
  legacy:
    kind: venv
    activate: /path/to/.venv/bin/activate

pools:
  - name: llm-dedicated
    partition: llm
    scheduling: managed
    operators: [llm-gpu-team]
    nodes: [gpu01, gpu02, gpu03]     # optional; else discovered from partition
  - name: general
    partition: general
    scheduling: slurm
```

Node inventory (hostname, GPU count, GPU type, state) is **discovered** via the
backend's `nodes()` call (`sinfo`/`scontrol show node`, or the slurmrestd
equivalent), then annotated with the `gpu_class` mapping. `TOTAL_GPUS` becomes
derived, not configured.

### 4.2 Catalog: `variants` per GPU class

The highest-value catalog change for a heterogeneous cluster. Replaces "gpus/tp
are fixed per model" with "gpus/tp are a function of where the model lands".

```yaml
defaults:
  extra_args: "--enable-prompt-tokens-details"
  env: {HF_HUB_OFFLINE: "1"}

models:
  - name: Qwen3-235B
    model_path: Qwen/Qwen3-235B
    requires: {min_vram_gb: 80}
    variants:
      h200: {gpus: 4, tp: 4, gpu_memory_utilization: 0.95}
      gb10: {gpus: 2, tp: 2, gpu_memory_utilization: 0.65,
             extra_args: "--max-model-len 32768"}
```

**Resolution order:** `defaults` → `model` → `variants[gpu_class]` → per-lease
override. The existing `merge_args()` in `catalog.py` already implements the
flag-override semantics; this adds one more layer to the same chain.

`requires:` filters which nodes are candidates at all. A model with no variant
for a class and no way to satisfy `requires` is simply not placeable there, and
the UI should grey out those pools.

### 4.3 `gpu_memory_utilization` capping

Effective value is `min(resolved_value, gpu_class.gpu_memory_utilization_max)`.

This one mechanism covers two cases: DGX Spark unified memory (where the
fraction is of memory shared with the OS, so an uncapped 0.95 will destabilise
the node) and ordinary "same model, smaller card" adjustment.

### 4.4 Quotas

```yaml
quotas:
  default:
    max_concurrent_gpus: 4
    max_gpu_hours_inflight: 96
    max_booking_horizon_days: 14
    max_booking_duration_hours: 48
  groups:
    llm-power-users: {max_concurrent_gpus: 8, max_booking_horizon_days: 30}
  per_pool:
    general: {max_booking_duration_hours: 168}   # best-effort anyway; looser
  admins: unlimited
```

Evaluated inside the booking transaction, alongside conflict validation.

---

## 5. Identity, authorization, ownership

### 5.1 Mechanism

Direct **LDAP simple bind against FreeIPA** for v1. This is the smallest change
to `app/auth.py`: HMAC cookie sessions stay exactly as they are; only the
password check is swapped and a group lookup is added (membership cached ~5 min).
OIDC later if Keycloak is put in front of IPA.

Two things not to skip:

- **Break-glass local admin** via the existing `AUTH_PASSWORD`. An IPA outage
  must not lock us out of our own scheduler. Flag its use visibly in the UI.
- **Per-user API tokens.** People running benchmarks will script bookings.

### 5.2 Roles

Derived from LDAP group membership:

| Group | Role | Can |
|---|---|---|
| `llm-admins` | admin | everything; lock/unlock; exceed quota; create reservations |
| `pools[].operators` | pool operator | manage any deployment **in that pool** |
| `llm-users` | user | create; manage own or own-group deployments |
| (none) | viewer | read-only schedule |

Stewardship is scoped **per pool**, not globally — the dedicated pool has
different caretakers than the general one.

### 5.3 Ownership

A deployment carries `owner_sub` (LDAP uid) **and** optional `owner_group`.
Pure individual ownership fails predictably: shared team models, someone on
vacation, a wedged deployment nobody but an admin can clean up. Group ownership
fixes all three without escalating to admin.

### 5.4 The whole authorization rule

One function, not a policy engine:

```python
def can(user, action, dep) -> bool:
    if user.is_admin:                                   return True
    if dep.locked and action in MUTATING:               return False
    if dep.pool in user.operator_pools:                 return True
    if user.sub == dep.owner_sub:                       return True
    if dep.owner_group and dep.owner_group in user.groups: return True
    return False
```

### 5.5 `locked` semantics

More than a permission flag — it is the "this is production" primitive:

- exempt from owner cancellation (admin/operator only)
- exempt from auto-cleanup by `endpoint_cleanup_worker`
- in `mode: service`, auto-renewed indefinitely (§7.2)

### 5.6 Service-account consequence

Since all jobs run as one Unix user, everything vLLM writes (HF cache, logs,
benchmark output) is owned by that account. **Per-lease output paths must be
namespaced by lease ID** so two users' runs cannot collide or read each other's
leftovers.

---

## 6. Planner

### 6.1 Grid

Occupancy becomes per `(node, gpu_index)` rather than a flat `TOTAL_GPUS` strip.

- Contiguity required only *within* a node (that is where NVLink/NCCL matters).
- Candidate nodes filtered by `requires:` and by `gpu_class` variant availability.
- Placement result is `(node, gpu_start, gpu_count)`.
- Decision 3 means `nodes` is always 1 — but the field is stored on the
  deployment defaulting to `1`, and the planner treats placement as
  `(node, gpu_range)` rather than assuming single-node implicitly, so multi-node
  needs no schema migration later.

**Foreign occupancy overlay:** for every pool, poll all users' jobs, extract
nodes + GRES + time-remaining, and insert them into the grid as immovable
blocks. Render greyed out. This is what makes the calendar honest on a shared
cluster.

### 6.2 Guard gap replaces `OVERLAP_TOLERANCE`

The current global 30 s tolerance (`planner.py:9`) is too tight once large
models and back-to-back `--nodelist`-pinned bookings are in play: weight loading
from shared storage takes minutes and teardown is not instant, so an overrun
cascades into the next booking on the same node. Becomes a per-`gpu_class`
(overridable per-model) `guard_gap_seconds`.

### 6.3 Booking race

`_validate_no_conflicts` (`admin.py:279`) is read → validate → insert with no
locking; two concurrent bookings can both pass. With multiple teams this will
occur. Fix: serialize the write path (`SELECT … FOR UPDATE` on Postgres) and
evaluate quota + conflict inside one transaction.

`_merge_same_model_if_applicable` (`admin.py:324`) silently absorbs one user's
booking into another's for the same model. At multi-team scale this needs
attribution at minimum, and should probably become opt-in.

---

## 7. Runtime, replicas, lifecycle

### 7.1 Apptainer

Adopted as a **runtime abstraction**, not the centre of the redesign.

Fixes: the arm64/x86 split (one image per arch, selected by node class — far
cleaner than a `venv_activate` per model per arch); reproducible pinned vLLM
versions per benchmark; no shared-FS venv startup latency.

Does *not* fix: scheduling, reservations, the `sacct` dependency, or the
planner. Those are orthogonal.

`templates/vllm_job.sh` becomes a thin launcher switching on `RUNTIME_KIND`,
keeping the venv path working so there is no flag day.

Build note: Apptainer cannot cross-build. The aarch64 `.sif` must be built on a
Spark node (or under emulation). Both images live on the shared FS.

### 7.2 Deployment modes

- **`mode: session`** — fixed window, hard stop at `end_at`. Current behaviour.
- **`mode: service`** — no fixed end. Before `TimeLimit` expiry, submit the
  replacement job, wait for READY, then drain and kill the old one. Zero
  downtime, lives within cluster wall limits, and gives rolling vLLM upgrades
  for free. Depends on §7.3.

### 7.3 Replicas and routing

`Deployment.replicas = N`; each replica is its own Slurm job + Endpoint.

`choose_ready_endpoint` (`router_core.py:46`) currently picks the *newest*
READY endpoint. It becomes **least-loaded** using
`num_requests_running + num_requests_waiting` — both already scraped by
`fetch_vllm_metrics`. Small change, immediately useful, and a prerequisite for
both scale-out and rolling renew.

Autoscaling is opt-in per deployment and only into `scheduling: slurm` pools,
where extra capacity is best-effort by definition.

---

## 8. Cluster backend abstraction

Everything touching Slurm goes behind one protocol with capability flags:

```python
class ClusterBackend(Protocol):
    capabilities: set[str]     # {"reservations", "accounting", "test_only", "json"}
    async def submit(spec: JobSpec) -> JobId
    async def cancel(job_id: JobId) -> None
    async def job_states(ids: list[JobId]) -> dict[JobId, JobStatus]
    async def job_exit_info(ids: list[JobId]) -> dict[JobId, ExitInfo | None]
    async def nodes() -> list[NodeInfo]
    async def foreign_jobs(pool: str) -> list[ForeignJob]
    async def estimate_start(spec: JobSpec) -> datetime | None   # --test-only
```

Three implementations:

- **`SlurmRestBackend`** (primary) — slurmrestd over JWT. No munge, no binary
  bind-mounts, no container UID gymnastics, structured JSON instead of `%T`/`|`
  parsing. **This is the answer to the "must run on the sacct host" problem.**
  Pin the API version (`v0.0.4x`) explicitly; the schema does shift between
  Slurm releases and that is a standing maintenance item.
- **`SlurmCliBackend`** — today's subprocess code, lifted as-is. Retained
  because `--test-only` has no clean REST equivalent in most versions (verify on
  the deployed version). Capability flags let the backend degrade per call.
- **`LocalBackend`** — in-memory fake. Worth it on its own: scheduling logic is
  currently untestable without a live Slurm.

### 8.1 Dropping the `sacct` dependency

`sacct` is used only to enrich a failure reason (`main.py:831`, `main.py:1103`).
Probe for it at startup and degrade gracefully:

1. `scontrol show job <id>` — works from controller memory for `MinJobAge`
   (default 300 s), needs no slurmdbd.
2. Tail the job's `.err` file — `_find_log_files` already locates it.

`accounting` becomes a capability, not a requirement.

---

## 9. Storage and availability

- **Alembic.** The current hand-rolled migration (`dependencies.py:11-17` — a
  bare `try/except: pass` around an `ALTER TABLE`) will not survive this schema
  growth. Add it *before* the schema changes, not after.
- **Postgres.** The SQLite-specific WAL/busy_timeout in `db.py` becomes
  conditional (it already is); the write path needs `FOR UPDATE` support (§6.3).
- **Leader election.** With one instance serving everyone, the proxy is a single
  point of failure for all inference. Run 2+ replicas where *all* serve proxy
  traffic but only the leader runs submit/cancel/reconcile. A `locks` table with
  a heartbeat row is sufficient and works on both backends.

---

## 10. Migration

### 10.1 Sequence the app migration and the cluster merge independently

Do **not** change app architecture and cluster topology at once — debugging the
intersection is miserable.

1. Stand up v2 against the big cluster (general pool + whatever dedicated nodes
   exist there).
2. Run it alongside the two existing single-node instances.
3. Move the two node sets into the big Slurm cluster **one at a time**.
4. Retire the old dashboards.

### 10.2 DB merge

Two SQLite DBs → one Postgres. Lease IDs collide. Migrate with ID remapping and
stamp every row with an `origin` field for provenance. Do this with Alembic in
place, not by hand.

---

## 11. Phases

| Phase | Contents |
|---|---|
| **0** | `ClusterBackend` (slurmrestd + CLI + local fake); optional `sacct`; Alembic; LDAP auth + RBAC + ownership/lock |
| **1** | `cluster.yaml` (gpu_classes/runtimes/pools); node discovery; catalog `variants`; runtime abstraction (venv \| apptainer); per-`(node, gpu)` planner; node-grouped UI; quotas |
| **2** | Foreign-job overlay; `managed`/`slurm` pool modes; `--test-only` preview + ghost blocks |
| **3** | Postgres + leader election; replicas + least-loaded routing; rolling renew for `mode: service`; cluster merge and instance consolidation |

Phase 0 is pure refactor plus an auth swap — no behavioural change to
scheduling, so it can ship to the existing single-node instances safely.
