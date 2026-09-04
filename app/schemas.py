from __future__ import annotations
from pydantic import BaseModel, Field
from datetime import datetime
from typing import Optional, Any

class LeaseCreate(BaseModel):
    model: str
    owner: Optional[str] = None
    # Co-own with a team so the booking survives the owner being away.
    # Must be a group the caller is actually in.
    owner_group: Optional[str] = None
    notes: Optional[str] = None
    begin_at: Optional[datetime] = None
    duration_seconds: int = Field(default=6*3600, ge=60)
    asap: bool = False  # NEW: find earliest available slot

    # session -> hard stop at end_at (benchmarks)
    # service -> renewed indefinitely; survives the cluster's MaxWall
    mode: str = "session"
    replicas: int = Field(default=1, ge=1, le=16)

    # Additional models to run on the *same* GPU, inside one Slurm job.
    # Every co-tenant (including `model`) must declare memory_gb in the catalog.
    colocate: list[str] = Field(default_factory=list)

    # Which pool to book in; None uses the legacy global Slurm settings.
    pool: Optional[str] = None
    # Force a GPU class instead of letting the smallest sufficient one win.
    gpu_class: Optional[str] = None
    # Pin to one specific node. Narrower than `gpu_class`: the job waits for
    # that machine rather than any card of the right type, which on a shared
    # partition can mean waiting considerably longer.
    node: Optional[str] = None
    # "Any GPU with at least this much memory." More useful than naming a class:
    # the user knows how big their model is, not which class is free. Combined
    # with asap=true, the class that can start soonest wins.
    min_memory_gb: Optional[float] = Field(default=None, gt=0.0)

    gpus: Optional[int] = Field(default=None, ge=1)
    tensor_parallel_size: Optional[int] = Field(default=None, ge=1)
    # Absolute GPU memory for this booking, in GB. Preferred over
    # gpu_memory_utilization: a fraction means different things on a 48 GB card
    # and a 128 GB Spark, GB does not. Capped by the GPU class.
    memory_gb: Optional[float] = Field(default=None, gt=0.0)
    gpu_memory_utilization: Optional[float] = Field(default=None, gt=0.0, le=1.0)
    extra_args: Optional[str] = None
    tool_args: Optional[str] = None
    reasoning_parser: Optional[str] = None
    # Apptainer image filename to launch (see GpuClassOut.available_images).
    # None picks the newest image matching the runtime at submit time.
    image: Optional[str] = None

class LeaseUpdate(BaseModel):
    begin_at: Optional[datetime] = None
    end_at: Optional[datetime] = None
    requested_gpus: Optional[int] = Field(default=None, ge=1)
    requested_tp: Optional[int] = Field(default=None, ge=1)
    notes: Optional[str] = None

class LeaseLockRequest(BaseModel):
    """Lock marks a deployment as production: exempt from owner cancellation
    and from auto-cleanup."""
    reason: Optional[str] = None
    # Also switch it to `service` mode, making it *permanent*: renewed before
    # every wall-time expiry and resurrected after a crash, node reboot or
    # cluster outage, until an admin unlocks it.
    permanent: bool = False


class LeaseOut(BaseModel):
    id: int
    model: str
    owner: Optional[str]
    owner_sub: Optional[str] = None
    owner_group: Optional[str] = None
    pool: Optional[str] = None
    locked: bool = False
    locked_by: Optional[str] = None
    locked_reason: Optional[str] = None
    # Server-evaluated permissions so the UI can hide controls it cannot use.
    can_edit: bool = False
    can_cancel: bool = False
    can_lock: bool = False
    notes: Optional[str] = None
    state: str
    slurm_job_id: Optional[str]
    host: str
    port: int
    requested_gpus: int
    requested_tp: int
    begin_at: Optional[datetime]
    end_at: Optional[datetime]
    created_at: datetime

    # Flat global lane index, for the existing timeline rendering.
    lane_start: Optional[int] = None
    lane_count: Optional[int] = None
    conflict: bool = False
    # Node-aware placement: lets the timeline group rows per node.
    node: Optional[str] = None
    gpu_start: Optional[int] = None
    gpu_class: Optional[str] = None
    pinned_node: Optional[str] = None
    # On a `slurm` pool the start time is Slurm's backfill estimate, not a
    # promise. The UI must render these differently — presenting an estimate as
    # confirmed is the most misleading thing this app could do.
    scheduling: str = "managed"
    estimated_start: Optional[datetime] = None
    estimate_updated_at: Optional[datetime] = None
    mode: str = "session"
    replicas: int = 1
    supersedes_id: Optional[int] = None
    # Models sharing this lease's GPU, if any (includes `model` itself).
    colocated: list[str] = Field(default_factory=list)
    # Apptainer image actually pinned/resolved for this lease, if any.
    image: Optional[str] = None

    @property
    def is_estimated(self) -> bool:
        return self.scheduling == "slurm"

class LeasePreviewRequest(BaseModel):
    """Ask when a booking *would* start, without creating it."""
    model: str
    pool: Optional[str] = None
    gpu_class: Optional[str] = None
    duration_seconds: int = Field(default=6*3600, ge=60)
    begin_at: Optional[datetime] = None

class LeasePreviewResponse(BaseModel):
    model: str
    pool: Optional[str] = None
    scheduling: str = "managed"
    gpu_class: Optional[str] = None
    gpus: int = 1
    # "confirmed" on a managed pool; "estimated" when Slurm decides; "unknown"
    # when we genuinely cannot tell.
    confidence: str = "confirmed"
    start_at: Optional[datetime] = None
    end_at: Optional[datetime] = None
    node: Optional[str] = None
    detail: str = ""

class NodeLaneOut(BaseModel):
    """One node's slice of the timeline."""
    name: str
    lane_offset: int
    gpu_count: int
    gpu_classes: list[list[Any]] = []
    state: str = "UNKNOWN"
    pool: Optional[str] = None
    # True when standing in for TOTAL_GPUS because no inventory was discovered.
    synthetic: bool = False
    # Node in an opt-in ("scale-out") pool. These sort to the end of the lane
    # strip, so the UI hides them by drawing fewer lanes.
    scale_out: bool = False

class ForeignJobOut(BaseModel):
    """Someone else's job occupying GPUs we can see but do not control."""
    job_id: str
    user: str
    state: str
    nodes: list[str] = []
    gpus: int = 0
    # Exact GPU indices held, when the backend can tell us (slurmrestd reports
    # them; the CLI backend cannot). Empty means "unknown" and the UI packs
    # sequentially instead.
    gpu_indices: list[int] = Field(default_factory=list)
    begin_at: Optional[datetime] = None
    end_at: Optional[datetime] = None

class LeaseExtend(BaseModel):
    duration_seconds: int = Field(..., ge=60)

class LeaseShortenRequest(BaseModel):
    """Shorten a running/submitted lease to a new end time."""
    new_end_at: datetime

class EndpointRegister(BaseModel):
    slurm_job_id: str
    model: str
    host: str
    port: int
    vllm_version: Optional[str] = None

class EndpointOut(BaseModel):
    id: int
    model: str
    host: str
    port: int
    slurm_job_id: str
    state: str
    last_health_at: Optional[datetime]
    last_error: Optional[str]
    created_at: datetime
    vllm_version: Optional[str] = None

class OpenAIModelsResponse(BaseModel):
    object: str = "list"
    data: list[dict[str, Any]]

class DashboardModel(BaseModel):
    id: str
    ready: bool
    meta: dict[str, Any]


class GpuClassOut(BaseModel):
    """What the UI needs to render a memory slider for a class."""
    name: str
    vram_gb: int = 0
    usable_gb: float = 0.0
    unified_memory: bool = False
    # Apptainer images this class's runtime can launch, newest first. Empty
    # for a venv runtime or when no image matches.
    available_images: list[str] = []

class EndpointStats(BaseModel):
    model: str
    host: str
    port: int
    state: str
    slurm_job_id: str
    last_health_at: Optional[datetime]
    # vLLM /metrics or /v1/models based stats
    active_requests: Optional[int] = None
    pending_requests: Optional[int] = None
    gpu_cache_usage: Optional[float] = None
    uptime_seconds: Optional[float] = None
    vllm_version: Optional[str] = None
    throughput_tps: Optional[float] = None

class DashboardResponse(BaseModel):
    now: datetime
    # Derived from discovered inventory, not the TOTAL_GPUS env var.
    total_gpus: int
    # GPUs in the pools shown by default — i.e. excluding scale-out. The
    # timeline's default height, so revealing scale-out capacity is purely
    # additive at the bottom of the strip.
    default_gpus: int = 0
    models: list[DashboardModel]
    leases: list[LeaseOut]
    endpoint_stats: list[EndpointStats] = []
    gpu_classes: list[GpuClassOut] = []
    nodes: list[NodeLaneOut] = []
    foreign_jobs: list[ForeignJobOut] = []
    # Set when node discovery failed, so the UI can say the view may be stale
    # instead of showing an empty cluster as though everything were free.
    inventory_error: Optional[str] = None

class LogResponse(BaseModel):
    slurm_job_id: str
    log_stdout: str
    log_stderr: str
    truncated: bool = False


# ── Apptainer images ─────────────────────────────────────────────────────────

class ImageOut(BaseModel):
    """A .sif on the shared filesystem."""
    name: str
    path: str
    size_bytes: int
    modified_at: datetime
    #: Non-empty means cluster.yaml points at this file and deleting it would
    #: break every future job for those GPU classes.
    used_by_runtimes: list[str] = []
    used_by_gpu_classes: list[str] = []
    can_delete: bool = True

class ImageBuildProgressOut(BaseModel):
    """What a running build is doing, read back out of its job log.

    Deliberately no percentage — the total download size lives in the registry
    manifest, which we never fetch. Absolute bytes are true; a percentage
    against a guessed denominator would not be.
    """
    phase: str
    label: str
    elapsed_seconds: Optional[int] = None
    downloaded_bytes: Optional[int] = None
    unpacked_bytes: Optional[int] = None
    image_bytes: Optional[int] = None
    bytes_per_second: Optional[float] = None
    last_line: str = ""
    #: mtime of the log the numbers came from, so the UI can say "nothing for
    #: 6 minutes" instead of showing a stale phase as if it were live.
    updated_at: Optional[datetime] = None

class ImageBuildOut(BaseModel):
    id: int
    image_name: str
    source_ref: str
    arch: str
    state: str
    partition: Optional[str] = None
    nodelist: Optional[str] = None
    slurm_job_id: Optional[str] = None
    requested_by: Optional[str] = None
    error: Optional[str] = None
    size_bytes: Optional[int] = None
    created_at: datetime
    started_at: Optional[datetime] = None
    finished_at: Optional[datetime] = None
    #: Only for builds still running, and only where the job log directory is
    #: readable from here. None means "no view", not "nothing happening".
    progress: Optional[ImageBuildProgressOut] = None

class BuildTargetOut(BaseModel):
    """Somewhere a build for one architecture can actually run."""
    arch: str
    partition: str
    pool: str
    nodes: list[str] = []

class ImagesResponse(BaseModel):
    images: list[ImageOut] = []
    builds: list[ImageBuildOut] = []
    targets: list[BuildTargetOut] = []
    image_dir: str = ""
    #: Set when the directory is not readable from here. The UI says so rather
    #: than showing an empty list, which would read as "there are no images".
    error: Optional[str] = None

class ImageBuildRequest(BaseModel):
    source_ref: str
    arch: str
    #: Defaults to a name derived from the reference and architecture.
    name: Optional[str] = None
    partition: Optional[str] = None
    #: Rebuild over an existing image of the same name.
    overwrite: bool = False

class PublicModelInfo(BaseModel):
    """Model from the catalog, with availability status."""
    name: str
    gpus: int
    tensor_parallel_size: int
    tags: list[str] = []
    notes: str = ""
    ready: bool = False  # True if currently running and healthy

class PublicLeaseInfo(BaseModel):
    """A scheduled booking (read-only view)."""
    id: int
    model: str
    state: str
    requested_gpus: int
    begin_at: Optional[datetime] = None
    end_at: Optional[datetime] = None
    notes: Optional[str] = None
    lane_start: Optional[int] = None
    lane_count: Optional[int] = None
    conflict: bool = False
    node: Optional[str] = None
    gpu_class: Optional[str] = None

class PublicScheduleResponse(BaseModel):
    """Full read-only schedule snapshot."""
    now: datetime
    total_gpus: int
    models: list[PublicModelInfo]
    leases: list[PublicLeaseInfo]
