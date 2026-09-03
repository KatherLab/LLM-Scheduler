"""Cluster topology: GPU classes, runtimes, pools and quotas.

Loaded from `config/cluster.yaml`, which describes *what exists* — as opposed to
`config/models.yaml`, which describes what we want to run on it.

Three concepts do most of the work:

  GpuClass  a GRES type Slurm knows (here: gpu24/gpu48/gpu80/gpu96). Carries the
            per-class caps that stop one model definition from being wrong on
            half the cluster.
  Runtime   how vLLM is launched — an Apptainer image or a venv. Selected by GPU
            class, which is how aarch64 nodes get an aarch64 image without every
            model having to know.
  Pool      a set of nodes plus *who decides placement*: `managed` (our calendar
            is the allocator, so we can promise a slot) or `slurm` (the backfill
            scheduler decides, so we can only estimate).

Reloads automatically when the file's mtime changes, matching `catalog.py`.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from threading import Lock
from typing import Any

import yaml

# Placement is ours to make (dedicated nodes) vs Slurm's (shared partition).
SCHEDULING_MANAGED = "managed"
SCHEDULING_SLURM = "slurm"
_SCHEDULING_MODES = {SCHEDULING_MANAGED, SCHEDULING_SLURM}

RUNTIME_APPTAINER = "apptainer"
RUNTIME_VENV = "venv"
_RUNTIME_KINDS = {RUNTIME_APPTAINER, RUNTIME_VENV}

# Time a node needs between two back-to-back bookings.
#
# The default is 0 deliberately: it reproduces the historical behaviour where
# "model A ends 18:00, model B starts 18:00" is allowed, so deployments without
# a cluster.yaml see no change. Real gaps are opted into per GPU class, because
# large weights on shared storage take minutes to load and teardown is not
# instant — without a gap, one overrun cascades into the next booking.
DEFAULT_GUARD_GAP_SECONDS = 0


class ClusterConfigError(Exception):
    """The cluster file is invalid. Raised with a message naming the key."""


@dataclass(frozen=True)
class GpuClass:
    name: str
    vram_gb: int = 0
    arch: str = "x86_64"
    runtime: str | None = None
    #: True when GPU memory is shared with the host (DGX Spark GB10). Changes
    #: what `--gpu-memory-utilization` means: the denominator is the whole
    #: system's RAM, most of which is not ours to take.
    unified_memory: bool = False
    #: Memory that must stay free for the OS, the page cache, and the loader
    #: process itself. On unified memory this is large and load-bearing; on a
    #: discrete card a couple of GB covers the CUDA context.
    reserved_gb: float = 0.0
    #: Hard ceiling on --gpu-memory-utilization, applied after everything else.
    gpu_memory_utilization_max: float | None = None
    guard_gap_seconds: int = DEFAULT_GUARD_GAP_SECONDS
    #: Node Features implying this class, for cross-checking discovery.
    models: tuple[str, ...] = ()

    @property
    def usable_gb(self) -> float:
        """Memory actually available to models on one GPU of this class."""
        usable = max(0.0, self.vram_gb - self.reserved_gb)
        if self.gpu_memory_utilization_max is not None:
            usable = min(usable, self.vram_gb * self.gpu_memory_utilization_max)
        return usable

    def utilization_for_gb(self, gb: float) -> float | None:
        """Convert an absolute memory budget into vLLM's fraction.

        Absolute GB is the unit that actually composes: two models sharing a
        card need their budgets to *add up*, and fractions of a shared pool do
        not. It is also the only stable way to express intent across classes —
        "40 GB" means the same thing on a 48 GB card and a 128 GB Spark, while
        "0.8" means wildly different things.
        """
        if not self.vram_gb:
            return None
        return self.cap_utilization(gb / self.vram_gb)

    def cap_utilization(self, requested: float | None) -> float | None:
        """Apply the class ceiling. Also handles "same model, smaller card".

        Always clamps into (0, 1]: a fraction above 1 is never valid, and
        asking for more GB than the card holds must not turn into one.
        """
        if requested is None:
            requested = self.default_utilization
        if requested is None:
            return None
        ceiling = 1.0
        if self.gpu_memory_utilization_max is not None:
            ceiling = min(ceiling, self.gpu_memory_utilization_max)
        return max(0.0, min(requested, ceiling))

    @property
    def default_utilization(self) -> float | None:
        """What to use when neither the model nor the request says.

        Derived from `reserved_gb` where set, so a class only has to state how
        much the host needs rather than hand-computing a fraction.
        """
        if self.gpu_memory_utilization_max is not None:
            return self.gpu_memory_utilization_max
        if self.reserved_gb and self.vram_gb:
            return max(0.0, (self.vram_gb - self.reserved_gb) / self.vram_gb)
        return None


@dataclass(frozen=True)
class Runtime:
    name: str
    kind: str = RUNTIME_VENV
    image: str | None = None          # apptainer .sif path
    activate: str | None = None       # venv activate script
    nv: bool = True                   # apptainer --nv
    binds: tuple[str, ...] = ()
    env: dict[str, str] = field(default_factory=dict)

    def as_job_env(self) -> dict[str, str]:
        """Variables the job template switches on."""
        env = {"RUNTIME_KIND": self.kind, "RUNTIME_NAME": self.name}
        if self.kind == RUNTIME_APPTAINER:
            env["APPTAINER_IMAGE"] = self.image or ""
            env["APPTAINER_NV"] = "1" if self.nv else "0"
            env["APPTAINER_BINDS"] = ",".join(self.binds)
        elif self.activate:
            env["VENV_ACTIVATE"] = self.activate
        env.update(self.env)
        return env


@dataclass(frozen=True)
class Pool:
    name: str
    partition: str
    scheduling: str = SCHEDULING_SLURM
    #: Extra capacity that is not offered by default. The pool is collapsed on
    #: the timeline and excluded from automatic GPU-class selection and
    #: placement; a booking only lands here when it names the pool or a node in
    #: it. This is for partitions we do not own — being *able* to scale out onto
    #: them should not make them the thing every booking silently lands on.
    scale_out: bool = False
    operators: tuple[str, ...] = ()
    #: Explicit node list; empty means "every node in the partition".
    nodes: tuple[str, ...] = ()
    #: Restrict to these GPU classes; empty means all.
    gpu_classes: tuple[str, ...] = ()
    account: str | None = None
    qos: str | None = None
    reservation: str | None = None

    @property
    def is_managed(self) -> bool:
        """True when our calendar is authoritative and a slot can be promised."""
        return self.scheduling == SCHEDULING_MANAGED

    def accepts_node(self, node_name: str) -> bool:
        return not self.nodes or node_name in self.nodes

    def accepts_class(self, gpu_class: str | None) -> bool:
        if not self.gpu_classes:
            return True
        return gpu_class is not None and gpu_class in self.gpu_classes


@dataclass(frozen=True)
class Quota:
    """`None` means unlimited on that axis."""

    max_concurrent_gpus: int | None = None
    max_gpu_hours_inflight: float | None = None
    max_booking_horizon_days: int | None = None
    max_booking_duration_hours: float | None = None

    def merged_with(self, other: "Quota") -> "Quota":
        """`other` wins where it sets a value."""
        return Quota(
            max_concurrent_gpus=_pick(other.max_concurrent_gpus, self.max_concurrent_gpus),
            max_gpu_hours_inflight=_pick(
                other.max_gpu_hours_inflight, self.max_gpu_hours_inflight),
            max_booking_horizon_days=_pick(
                other.max_booking_horizon_days, self.max_booking_horizon_days),
            max_booking_duration_hours=_pick(
                other.max_booking_duration_hours, self.max_booking_duration_hours),
        )


UNLIMITED = Quota()


def _pick(new, old):
    return old if new is None else new


@dataclass(frozen=True)
class ClusterConfig:
    gpu_classes: dict[str, GpuClass] = field(default_factory=dict)
    runtimes: dict[str, Runtime] = field(default_factory=dict)
    pools: dict[str, Pool] = field(default_factory=dict)
    default_quota: Quota = UNLIMITED
    group_quotas: dict[str, Quota] = field(default_factory=dict)
    pool_quotas: dict[str, Quota] = field(default_factory=dict)

    # ── lookups ─────────────────────────────────────────────────────────────
    def gpu_class(self, name: str | None) -> GpuClass | None:
        return self.gpu_classes.get(name) if name else None

    def pool(self, name: str | None) -> Pool | None:
        return self.pools.get(name) if name else None

    def is_scale_out(self, pool_name: str | None) -> bool:
        """True for a pool that is opt-in rather than part of the default view."""
        pool = self.pool(pool_name)
        return bool(pool and pool.scale_out)

    @property
    def has_scale_out(self) -> bool:
        return any(p.scale_out for p in self.pools.values())

    def pools_for_partition(self, partition: str) -> list[Pool]:
        return [p for p in self.pools.values() if p.partition == partition]

    def pool_for_node(self, node_name: str, partitions: tuple[str, ...]) -> Pool | None:
        """Which pool a discovered node belongs to.

        Explicit node lists win over partition-wide pools, so a few dedicated
        machines can be carved out of a shared partition.
        """
        candidates = [
            p for p in self.pools.values()
            if p.partition in partitions and p.accepts_node(node_name)
        ]
        if not candidates:
            return None
        candidates.sort(key=lambda p: (not p.nodes, p.name))
        return candidates[0]

    def runtime_for(
        self, gpu_class: str | None, override: str | None = None
    ) -> Runtime | None:
        """Model override wins; otherwise the GPU class decides.

        Class-driven selection is the point: an aarch64 node gets an aarch64
        image without every model entry having to name one.
        """
        if override:
            return self.runtimes.get(override)
        cls = self.gpu_class(gpu_class)
        if cls and cls.runtime:
            return self.runtimes.get(cls.runtime)
        return None

    def guard_gap_seconds(self, gpu_class: str | None) -> int:
        cls = self.gpu_class(gpu_class)
        return cls.guard_gap_seconds if cls else DEFAULT_GUARD_GAP_SECONDS

    def cap_utilization(self, gpu_class: str | None, requested: float | None):
        cls = self.gpu_class(gpu_class)
        return cls.cap_utilization(requested) if cls else requested

    # ── quotas ──────────────────────────────────────────────────────────────
    def quota_for(
        self, groups: frozenset[str] | set[str], pool: str | None = None,
        is_admin: bool = False,
    ) -> Quota:
        """default -> group overrides -> pool overrides. Admins are exempt."""
        if is_admin:
            return UNLIMITED
        quota = self.default_quota
        for group in sorted(groups):
            override = self.group_quotas.get(group)
            if override:
                quota = quota.merged_with(override)
        if pool and pool in self.pool_quotas:
            quota = quota.merged_with(self.pool_quotas[pool])
        return quota


# ── parsing ──────────────────────────────────────────────────────────────────

def _as_tuple(value: Any) -> tuple[str, ...]:
    if value is None:
        return ()
    if isinstance(value, str):
        return tuple(v.strip() for v in value.split(",") if v.strip())
    return tuple(str(v) for v in value)


def _parse_quota(data: dict | None) -> Quota:
    if not data:
        return UNLIMITED
    if isinstance(data, str) and data.strip().lower() == "unlimited":
        return UNLIMITED
    return Quota(
        max_concurrent_gpus=_int_or_none(data.get("max_concurrent_gpus")),
        max_gpu_hours_inflight=_float_or_none(data.get("max_gpu_hours_inflight")),
        max_booking_horizon_days=_int_or_none(data.get("max_booking_horizon_days")),
        max_booking_duration_hours=_float_or_none(data.get("max_booking_duration_hours")),
    )


def _int_or_none(v):
    return None if v is None else int(v)


def _float_or_none(v):
    return None if v is None else float(v)


def load_cluster(path: str) -> ClusterConfig:
    p = Path(path)
    data = yaml.safe_load(p.read_text(encoding="utf-8")) or {}

    runtimes: dict[str, Runtime] = {}
    for name, raw in (data.get("runtimes") or {}).items():
        raw = raw or {}
        kind = str(raw.get("kind", RUNTIME_VENV)).lower()
        if kind not in _RUNTIME_KINDS:
            raise ClusterConfigError(
                f"runtimes.{name}.kind must be one of {sorted(_RUNTIME_KINDS)}, got {kind!r}"
            )
        if kind == RUNTIME_APPTAINER and not raw.get("image"):
            raise ClusterConfigError(f"runtimes.{name}: apptainer runtime needs an 'image'")
        if kind == RUNTIME_VENV and not raw.get("activate"):
            raise ClusterConfigError(f"runtimes.{name}: venv runtime needs an 'activate'")
        runtimes[name] = Runtime(
            name=name,
            kind=kind,
            image=raw.get("image"),
            activate=raw.get("activate"),
            nv=bool(raw.get("nv", True)),
            binds=_as_tuple(raw.get("binds")),
            env={str(k): str(v) for k, v in (raw.get("env") or {}).items()},
        )

    gpu_classes: dict[str, GpuClass] = {}
    for name, raw in (data.get("gpu_classes") or {}).items():
        raw = raw or {}
        runtime_name = raw.get("runtime")
        if runtime_name and runtime_name not in runtimes:
            raise ClusterConfigError(
                f"gpu_classes.{name}.runtime={runtime_name!r} is not defined under runtimes"
            )
        util_max = raw.get("gpu_memory_utilization_max")
        if util_max is not None and not (0 < float(util_max) <= 1):
            raise ClusterConfigError(
                f"gpu_classes.{name}.gpu_memory_utilization_max must be in (0, 1]"
            )
        gpu_classes[name] = GpuClass(
            name=name,
            vram_gb=int(raw.get("vram_gb", 0) or 0),
            arch=str(raw.get("arch", "x86_64")),
            runtime=runtime_name,
            unified_memory=bool(raw.get("unified_memory", False)),
            reserved_gb=float(raw.get("reserved_gb", 0) or 0),
            gpu_memory_utilization_max=None if util_max is None else float(util_max),
            guard_gap_seconds=int(
                raw.get("guard_gap_seconds", DEFAULT_GUARD_GAP_SECONDS)
            ),
            models=_as_tuple(raw.get("models")),
        )

    pools: dict[str, Pool] = {}
    for raw in data.get("pools") or []:
        raw = raw or {}
        name = raw.get("name")
        if not name:
            raise ClusterConfigError("every entry under pools needs a 'name'")
        partition = raw.get("partition")
        if not partition:
            raise ClusterConfigError(f"pools.{name} needs a 'partition'")
        scheduling = str(raw.get("scheduling", SCHEDULING_SLURM)).lower()
        if scheduling not in _SCHEDULING_MODES:
            raise ClusterConfigError(
                f"pools.{name}.scheduling must be one of {sorted(_SCHEDULING_MODES)}, "
                f"got {scheduling!r}"
            )
        classes = _as_tuple(raw.get("gpu_classes"))
        for cls in classes:
            if cls not in gpu_classes:
                raise ClusterConfigError(
                    f"pools.{name}.gpu_classes references unknown class {cls!r}"
                )
        if name in pools:
            raise ClusterConfigError(f"duplicate pool name {name!r}")
        pools[name] = Pool(
            name=name,
            partition=str(partition),
            scheduling=scheduling,
            scale_out=bool(raw.get("scale_out", False)),
            operators=_as_tuple(raw.get("operators")),
            nodes=_as_tuple(raw.get("nodes")),
            gpu_classes=classes,
            account=raw.get("account"),
            qos=raw.get("qos"),
            reservation=raw.get("reservation"),
        )

    quotas = data.get("quotas") or {}
    pool_quotas = {
        k: _parse_quota(v) for k, v in (quotas.get("per_pool") or {}).items()
    }
    for pool_name in pool_quotas:
        if pool_name not in pools:
            raise ClusterConfigError(
                f"quotas.per_pool references unknown pool {pool_name!r}"
            )

    return ClusterConfig(
        gpu_classes=gpu_classes,
        runtimes=runtimes,
        pools=pools,
        default_quota=_parse_quota(quotas.get("default")),
        group_quotas={
            k: _parse_quota(v) for k, v in (quotas.get("groups") or {}).items()
        },
        pool_quotas=pool_quotas,
    )


# ── auto-reloading singleton (mirrors catalog.py) ────────────────────────────

_cache: ClusterConfig | None = None
_mtime: float = 0.0
_lock = Lock()
_PATH = "config/cluster.yaml"

#: Used when no cluster.yaml exists, so single-node deployments keep working
#: without one.
EMPTY = ClusterConfig()


def get_cluster(path: str = _PATH) -> ClusterConfig:
    """Return the cluster config, reloading when the file changes.

    A missing file yields an empty config rather than an error: the app still
    runs untyped-GRES on a single node, which is what it did before this
    existed.
    """
    global _cache, _mtime

    p = Path(path)
    try:
        current = p.stat().st_mtime
    except OSError:
        return _cache if _cache is not None else EMPTY

    if _cache is not None and current <= _mtime:
        return _cache

    with _lock:
        if _cache is not None and current <= _mtime:
            return _cache
        _cache = load_cluster(path)
        _mtime = current
        print(
            f"cluster: loaded {len(_cache.gpu_classes)} gpu classes, "
            f"{len(_cache.runtimes)} runtimes, {len(_cache.pools)} pools from {path}"
        )
        return _cache


def set_cluster(config: ClusterConfig | None) -> None:
    """Override the loaded config. For tests."""
    global _cache, _mtime
    _cache = config
    _mtime = float("inf") if config is not None else 0.0
