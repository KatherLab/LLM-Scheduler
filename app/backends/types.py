"""Backend-neutral value types.

These deliberately mirror Slurm's vocabulary (it is the only backend we have)
while staying free of Slurm's *transport* — no `%T` format strings, no `|`
parsing, no subprocess assumptions. A slurmrestd backend produces the same
types from JSON.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime


# ── Capabilities ─────────────────────────────────────────────────────────────
# Backends advertise what they can actually do so callers degrade instead of
# crashing. The `accounting` capability is why the app no longer has to run on
# the sacct host.

CAP_ACCOUNTING = "accounting"        # sacct / job history after it leaves the queue
CAP_TEST_ONLY = "test_only"          # sbatch --test-only start-time estimate
CAP_RESERVATIONS = "reservations"    # scontrol create reservation
CAP_NODE_DISCOVERY = "node_discovery"  # sinfo / node inventory
CAP_FOREIGN_JOBS = "foreign_jobs"    # can see other users' jobs


@dataclass(frozen=True)
class JobSpec:
    """Everything needed to submit one vLLM job.

    `gres` overrides the plain `gpus` count when a GPU *type* matters
    (`gpu:h200:4`), which is how heterogeneous pools pin to a class.
    """

    job_name: str
    script_path: str
    gpus: int
    time_limit: str                      # "HH:MM:SS" or "D-HH:MM:SS"
    env: dict[str, str] = field(default_factory=dict)

    cpus: int = 16
    mem: str | None = None

    partition: str | None = None
    account: str | None = None
    qos: str | None = None
    nodelist: str | None = None
    gres: str | None = None
    reservation: str | None = None

    begin: datetime | None = None
    comment: str | None = None           # attribution: "user:<uid>,lease:<id>"

    log_dir: str = "./logs"
    mail_user: str | None = None
    mail_type: str | None = None


@dataclass(frozen=True)
class SubmitResult:
    job_id: str
    raw: str


@dataclass(frozen=True)
class JobState:
    """A job that is still known to the scheduler's queue.

    `job_states()` maps an id to `None` when the job has left the queue —
    callers must treat `None` as "gone", not as "unknown".
    """

    state: str                           # PENDING, RUNNING, COMPLETING, ...
    nodes: str | None = None             # compact hostlist, may need expanding
    start_time: datetime | None = None   # actual, or backfill estimate if PENDING

    @property
    def is_pending(self) -> bool:
        return self.state in ("PENDING", "CONFIGURING")

    @property
    def is_running(self) -> bool:
        return self.state in ("RUNNING", "COMPLETING")


@dataclass(frozen=True)
class ExitInfo:
    """Why a job left the queue. `source` records how we found out, because the
    fallbacks are less reliable than sacct and that matters when reading logs."""

    state: str
    exit_code: str = "?"
    source: str = "unknown"              # "sacct" | "scontrol" | "log"


@dataclass(frozen=True)
class GpuGroup:
    """A block of identical GPUs on one node.

    `gpu_class` is the GRES type Slurm knows (on this cluster: gpu24, gpu48,
    gpu80, gpu96), or None when the node declares untyped GPUs.
    """

    gpu_class: str | None
    count: int


@dataclass(frozen=True)
class NodeInfo:
    """A node may hold *several* GPU classes at once.

    This is not hypothetical: `europa` is configured as
    `{gpu24: 1, gpu48: 1}`, so a single `gpu_type` field would misreport it and
    the planner would place a gpu48 model on a 24 GB card.
    """

    name: str
    gpus: tuple[GpuGroup, ...] = ()
    features: tuple[str, ...] = ()       # node Features: rtx_6000_ada, l40, a100...
    partitions: tuple[str, ...] = ()
    state: str = "UNKNOWN"               # IDLE, ALLOCATED, MIXED, DOWN, DRAIN...
    cpus: int = 0
    mem_mb: int = 0

    @property
    def gpu_count(self) -> int:
        return sum(g.count for g in self.gpus)

    @property
    def gpu_classes(self) -> dict[str | None, int]:
        return {g.gpu_class: g.count for g in self.gpus}

    def count_of(self, gpu_class: str) -> int:
        """How many GPUs of one class this node has. 0 means "cannot host"."""
        return sum(g.count for g in self.gpus if g.gpu_class == gpu_class)

    @property
    def is_usable(self) -> bool:
        """DOWN/DRAIN/FAIL nodes must not receive placements."""
        s = self.state.upper().rstrip("*~#$@+")
        return not any(bad in s for bad in ("DOWN", "DRAIN", "FAIL", "INVAL", "MAINT"))


@dataclass(frozen=True)
class ForeignJob:
    """A job we do not own, occupying GPUs we might want.

    This is what makes the calendar honest on a shared partition.
    """

    job_id: str
    user: str
    state: str
    nodes: tuple[str, ...] = ()
    gpus: int = 0
    partition: str | None = None
    start_time: datetime | None = None
    end_time: datetime | None = None
    #: Exact GPU indices held, when the backend can tell us (slurmrestd reports
    #: `gres_detail` as `gpu:gpu48:2(IDX:0-1)`). Empty means "unknown", and the
    #: placer then falls back to assuming the first `gpus` slots are taken.
    gpu_indices: tuple[int, ...] = ()


@dataclass(frozen=True)
class StartEstimate:
    """Result of a dry-run submission (`sbatch --test-only`)."""

    start_time: datetime | None
    nodes: tuple[str, ...] = ()
    raw: str = ""


class ClusterUnavailableError(Exception):
    """The scheduler itself is unreachable.

    Distinct from "the job is gone": callers must skip a reconcile cycle rather
    than concluding every job has died.
    """


class ClusterCommandError(Exception):
    """A scheduler command failed for a job-specific reason."""
