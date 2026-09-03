"""Discovered cluster inventory.

`TOTAL_GPUS` was a hand-maintained integer that could silently disagree with
reality. This replaces it with a periodically refreshed snapshot from the
backend, annotated with the pool each node belongs to.

The snapshot is cached rather than fetched per request: dashboard rendering and
booking validation both need it, and neither should shell out to `sinfo`.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone

from .backends import ClusterUnavailableError, get_backend
from .backends.types import CAP_FOREIGN_JOBS, CAP_NODE_DISCOVERY, ForeignJob, NodeInfo
from .cluster import ClusterConfig, Pool, get_cluster

logger = logging.getLogger(__name__)

#: How long a snapshot is served before it is considered stale. Node topology
#: changes rarely; node *state* (drain/down) changes more often.
DEFAULT_TTL = timedelta(seconds=120)


@dataclass(frozen=True)
class Inventory:
    nodes: tuple[NodeInfo, ...] = ()
    foreign_jobs: tuple[ForeignJob, ...] = ()
    fetched_at: datetime | None = None
    #: node name -> pool name
    node_pools: dict[str, str] = field(default_factory=dict)
    #: Set when discovery failed, so callers can say so instead of showing an
    #: empty cluster as though everything were busy.
    error: str | None = None

    @property
    def is_empty(self) -> bool:
        return not self.nodes

    @property
    def total_gpus(self) -> int:
        return sum(n.gpu_count for n in self.nodes)

    def usable_nodes(self) -> list[NodeInfo]:
        return [n for n in self.nodes if n.is_usable]

    def nodes_in_pool(self, pool: str | None) -> list[NodeInfo]:
        if pool is None:
            return self.usable_nodes()
        return [
            n for n in self.usable_nodes() if self.node_pools.get(n.name) == pool
        ]

    def gpu_class_counts(self) -> dict[str, int]:
        """Total GPUs per class across usable nodes — what the UI shows."""
        counts: dict[str, int] = {}
        for node in self.usable_nodes():
            for group in node.gpus:
                key = group.gpu_class or "untyped"
                counts[key] = counts.get(key, 0) + group.count
        return counts

    def classes_available(self) -> set[str]:
        return {c for c in self.gpu_class_counts() if c != "untyped"}

    def foreign_jobs_for(self, pool: Pool | None) -> list[ForeignJob]:
        """Foreign jobs touching a pool's nodes.

        On a `managed` pool there should be none — if there are, the calendar's
        promise is being broken by something outside our control and the UI
        should say so rather than quietly mis-plan.
        """
        if pool is None:
            return list(self.foreign_jobs)
        names = {n.name for n in self.nodes_in_pool(pool.name)}
        return [j for j in self.foreign_jobs if names.intersection(j.nodes)]


EMPTY = Inventory()

_cache: Inventory = EMPTY


def _annotate_pools(nodes, cluster: ClusterConfig) -> dict[str, str]:
    mapping: dict[str, str] = {}
    for node in nodes:
        pool = cluster.pool_for_node(node.name, node.partitions)
        if pool:
            mapping[node.name] = pool.name
    return mapping


async def refresh(cluster: ClusterConfig | None = None) -> Inventory:
    """Re-query the backend. Never raises — failures are recorded on the result.

    A discovery failure keeps the previous snapshot: an empty node list would
    make every booking look unplaceable, which is worse than slightly stale
    topology.
    """
    global _cache
    cluster = cluster if cluster is not None else get_cluster()
    backend = get_backend()
    now = datetime.now(timezone.utc)

    if CAP_NODE_DISCOVERY not in backend.capabilities:
        _cache = Inventory(
            fetched_at=now,
            error="backend cannot discover nodes (no sinfo); "
                  "falling back to configured TOTAL_GPUS",
        )
        return _cache

    try:
        nodes = tuple(await backend.nodes())
    except ClusterUnavailableError as exc:
        logger.warning("inventory: cluster unavailable, keeping previous snapshot: %s", exc)
        return _replace_error(str(exc))
    except Exception as exc:
        logger.error("inventory: node discovery failed: %s", exc)
        return _replace_error(str(exc))

    foreign: tuple[ForeignJob, ...] = ()
    if CAP_FOREIGN_JOBS in backend.capabilities:
        try:
            foreign = tuple(await backend.foreign_jobs())
        except Exception as exc:
            # Losing foreign visibility degrades planning but is not fatal;
            # it must be visible though, or the calendar silently over-promises.
            logger.warning("inventory: foreign job scan failed: %s", exc)

    _cache = Inventory(
        nodes=nodes,
        foreign_jobs=foreign,
        fetched_at=now,
        node_pools=_annotate_pools(nodes, cluster),
    )
    return _cache


def _replace_error(message: str) -> Inventory:
    global _cache
    _cache = Inventory(
        nodes=_cache.nodes,
        foreign_jobs=_cache.foreign_jobs,
        fetched_at=_cache.fetched_at,
        node_pools=_cache.node_pools,
        error=message,
    )
    return _cache


def current() -> Inventory:
    """The last snapshot. Cheap; safe to call per request."""
    return _cache


def is_stale(ttl: timedelta = DEFAULT_TTL) -> bool:
    if _cache.fetched_at is None:
        return True
    return datetime.now(timezone.utc) - _cache.fetched_at > ttl


def set_inventory(inv: Inventory | None) -> None:
    """Override the snapshot. For tests and development."""
    global _cache
    _cache = inv if inv is not None else EMPTY


# ── GPU class selection ──────────────────────────────────────────────────────

def eligible_gpu_classes(
    model,
    cluster: ClusterConfig,
    inv: Inventory,
    *,
    pool: Pool | None = None,
    min_memory_gb: float | None = None,
) -> list[str]:
    """Every GPU class this model could actually run on, smallest first.

    "At least X GB" is the useful way to ask for a GPU: the user knows how big
    their model is, not which of gpu24/48/80/96 happens to be free. Ordering by
    size means the default choice is the least wasteful one.

    Note the caller cannot turn this into a single Slurm job spanning several
    classes: this cluster's `job_submit.lua` rejects a constraint naming more
    than one class. Picking one is mandatory; this just picks it well.
    """
    from .catalog import resolve_variant

    out: list[tuple[float, str]] = []
    for name in inv.classes_available():
        gpu_class = cluster.gpu_class(name)
        if gpu_class is None:
            continue
        if pool is not None and not pool.accepts_class(name):
            continue
        if not model.supports_class(gpu_class):
            continue
        if min_memory_gb and gpu_class.usable_gb < min_memory_gb:
            continue
        needed = resolve_variant(model, name).gpus
        candidates = inv.nodes_in_pool(pool.name if pool else None)
        if not any(n.count_of(name) >= needed for n in candidates):
            continue
        out.append((gpu_class.vram_gb, name))

    return [name for _, name in sorted(out)]


def choose_gpu_class(
    model,
    cluster: ClusterConfig,
    inv: Inventory,
    *,
    pool: Pool | None = None,
    preferred: str | None = None,
) -> str | None:
    """Pick the GPU class a model should run on.

    Sending an *untyped* `--gres=gpu:N` is not neutral on this cluster:
    job_submit.lua appends `gpu24` as a hard constraint, so the job silently
    never lands on anything larger. Choosing a class explicitly is what avoids
    that.

    Smallest sufficient class wins, so a model that fits on 48 GB does not
    occupy a 96 GB card. Returns None when nothing matches, and the caller
    should refuse the booking rather than fall back to untyped.
    """
    available = inv.classes_available()
    if not available:
        return None

    def _usable(name: str) -> bool:
        gpu_class = cluster.gpu_class(name)
        if gpu_class is None:
            return False
        if pool is not None and not pool.accepts_class(name):
            return False
        if not model.supports_class(gpu_class):
            return False
        # There must be a node with enough of this class for the resolved size.
        from .catalog import resolve_variant
        needed = resolve_variant(model, name).gpus
        # Aggregate memory: "needs 8x H200" is not expressible per-GPU, and a
        # booking that cannot load is worth refusing before it queues.
        if not model.requires.fits_on(gpu_class, needed):
            return False
        candidates = inv.nodes_in_pool(pool.name if pool else None)
        return any(n.count_of(name) >= needed for n in candidates)

    if preferred:
        return preferred if _usable(preferred) else None

    ranked = sorted(
        (c for c in available if _usable(c)),
        key=lambda c: (cluster.gpu_class(c).vram_gb, c),
    )
    return ranked[0] if ranked else None
