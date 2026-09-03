"""Bridge between ORM leases and the node-aware placer.

`admin.py` deals in `Lease` rows and the UI deals in timeline lanes;
`placement.py` deals in neither. This module converts between them and owns the
one compatibility decision that makes the migration safe:

**The legacy flat-lane model is exactly "one node with N untyped GPUs."** So a
deployment with no `cluster.yaml` gets a synthetic node built from
`TOTAL_GPUS`, and placement produces byte-identical results to the old planner.
Real inventory simply replaces that synthetic node.

Lanes are then a rendering concern: each node is assigned a stable base offset
and a lane index is `offset + gpu_index`, so the existing flat timeline keeps
working while gaining node boundaries to draw.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timedelta

from .backends.types import ForeignJob, GpuGroup, NodeInfo
from .cluster import ClusterConfig, Pool
from .inventory import Inventory
from .placement import Demand, Placement, compute_placements, find_earliest_slot
from .settings import settings
from .utils import ensure_utc

SYNTHETIC_NODE = "cluster"

#: States whose bookings occupy GPUs.
ACTIVE_STATES = ("PLANNED", "SUBMITTED", "STARTING", "RUNNING")


@dataclass(frozen=True)
class NodeLane:
    """One node's slice of the timeline."""

    name: str
    lane_offset: int
    gpu_count: int
    gpu_classes: tuple[tuple[str | None, int], ...]
    state: str
    pool: str | None
    synthetic: bool = False
    #: Node in an opt-in pool. Sorted to the end of the strip so the UI can
    #: collapse it by simply drawing fewer lanes — lane indices of the default
    #: nodes stay put whether it is shown or not.
    scale_out: bool = False


def effective_nodes(inv: Inventory) -> list[NodeInfo]:
    """Discovered nodes, or one synthetic node standing in for TOTAL_GPUS.

    The fallback is what lets this ship to the existing single-node
    deployments unchanged: an untyped GPU group matches any demand, and
    contiguity within the one node is the old lane behaviour.
    """
    nodes = inv.usable_nodes()
    if nodes:
        return nodes
    return [NodeInfo(
        name=SYNTHETIC_NODE,
        gpus=(GpuGroup(gpu_class=None, count=max(1, settings.total_gpus)),),
        state="IDLE",
    )]


def node_lanes(
    nodes: list[NodeInfo], inv: Inventory, cluster: ClusterConfig | None = None
) -> list[NodeLane]:
    """Assign each node a stable lane offset, ordered by name.

    Stability matters: the timeline would jump around between refreshes if
    offsets shifted when a node drained.

    Scale-out nodes come last as a block. That is what lets the UI hide them by
    shortening the strip instead of remapping every lane index — and it keeps a
    default booking's lane the same whether or not they are shown.
    """
    def key(node: NodeInfo) -> tuple[int, str]:
        pool = inv.node_pools.get(node.name)
        scale_out = cluster.is_scale_out(pool) if cluster else False
        return (1 if scale_out else 0, node.name)

    lanes: list[NodeLane] = []
    offset = 0
    for node in sorted(nodes, key=key):
        pool = inv.node_pools.get(node.name)
        lanes.append(NodeLane(
            name=node.name,
            lane_offset=offset,
            gpu_count=node.gpu_count,
            gpu_classes=tuple((g.gpu_class, g.count) for g in node.gpus),
            state=node.state,
            pool=pool,
            synthetic=node.name == SYNTHETIC_NODE,
            scale_out=cluster.is_scale_out(pool) if cluster else False,
        ))
        offset += node.gpu_count
    return lanes


def allowed_nodes(
    pool_name: str | None, inv: Inventory, cluster: ClusterConfig
) -> tuple[str, ...]:
    """Nodes a booking with this pool may be placed on.

    Empty means "no restriction", which is what a cluster with no scale-out
    pools gets — including the legacy single-node fallback, whose placement must
    stay byte-identical.
    """
    if not cluster.has_scale_out:
        return ()
    nodes = (
        inv.nodes_in_pool(pool_name) if pool_name
        else inv.candidate_nodes(cluster)
    )
    return tuple(sorted(n.name for n in nodes))


def lane_index(lanes: list[NodeLane], placement: Placement) -> int | None:
    """Global lane index for the flat timeline, or None if unplaced."""
    if placement.node is None or placement.gpu_start is None:
        return None
    for lane in lanes:
        if lane.name == placement.node:
            return lane.lane_offset + placement.gpu_start
    return None


def lease_to_demand(
    lease, *, pinned: bool = False, nodes: tuple[str, ...] = ()
) -> Demand:
    """Convert a Lease row into a placement Demand.

    A running booking is pinned to the node it actually landed on, so
    re-planning cannot "move" a model that is already serving traffic. A user
    pin (`pinned_node`, which is what dropping onto a node's row sets) is also
    honoured: the job goes out with `--nodelist`, so drawing it anywhere else
    would show a machine it will not run on.
    """
    begin = ensure_utc(lease.begin_at) if lease.begin_at else ensure_utc(lease.created_at)
    end = ensure_utc(lease.end_at) if lease.end_at else begin + timedelta(hours=1)
    pin = lease.pinned_node or (
        lease.node if (pinned or lease.state in ("RUNNING", "STARTING")) else None
    )
    return Demand(
        lease_id=lease.id,
        gpus=max(1, lease.requested_gpus or 1),
        begin=begin,
        end=end,
        gpu_class=lease.gpu_class,
        nodes=nodes,
        pinned_node=pin if pin else None,
    )


def active_leases(leases, now: datetime) -> list:
    """Bookings that occupy GPUs: active states, plus FAILED ones still inside
    their window (they hold the reservation until they expire or retry)."""
    return [
        lease for lease in leases
        if lease.state in ACTIVE_STATES
        or (lease.state == "FAILED" and lease.end_at and ensure_utc(lease.end_at) > now)
    ]


def plan(
    leases,
    inv: Inventory,
    cluster: ClusterConfig,
    *,
    include_foreign: bool = True,
) -> tuple[dict[int, Placement], list[NodeLane]]:
    """Place every booking and return the lane layout to render them in.

    Each booking is confined to its own pool's nodes: without a pool that means
    the default pools only, so a booking nobody asked to scale out does not get
    placed on capacity the timeline hides.
    """
    nodes = effective_nodes(inv)
    foreign: list[ForeignJob] = list(inv.foreign_jobs) if include_foreign else []
    placements = compute_placements(
        [
            lease_to_demand(x, nodes=allowed_nodes(x.pool, inv, cluster))
            for x in leases
        ],
        nodes,
        cluster=cluster,
        foreign_jobs=foreign,
    )
    return placements, node_lanes(nodes, inv, cluster)


def earliest_start(
    lease,
    others,
    inv: Inventory,
    cluster: ClusterConfig,
    *,
    pool: Pool | None = None,
    search_end: datetime | None = None,
) -> datetime | None:
    """Earliest time this booking fits.

    Only meaningful for `managed` pools. On a `slurm` pool the backfill
    scheduler knows the queue and we do not — the honest answer there comes
    from the backend's `estimate_start()` (`sbatch --test-only`).
    """
    nodes = effective_nodes(inv)
    if pool is not None and pool.nodes:
        allowed = set(pool.nodes)
        nodes = [n for n in nodes if n.name in allowed]
    if not nodes:
        return None

    # An explicit `nodes:` list on the pool is a hard restriction; otherwise the
    # pool (or its absence) still decides which nodes are on offer.
    restrict = tuple(pool.nodes) if pool is not None and pool.nodes else \
        allowed_nodes(pool.name if pool else None, inv, cluster)
    demand = lease_to_demand(lease, nodes=restrict)

    return find_earliest_slot(
        demand,
        nodes,
        [
            lease_to_demand(x, nodes=allowed_nodes(x.pool, inv, cluster))
            for x in others
        ],
        cluster=cluster,
        foreign_jobs=list(inv.foreign_jobs),
        search_end=search_end,
    )


def describe_conflict(
    candidate, others, placements: dict[int, Placement], lanes: list[NodeLane]
) -> str:
    """Explain *why* a booking does not fit, naming what is in the way."""
    cand_begin = ensure_utc(candidate.begin_at) if candidate.begin_at else ensure_utc(candidate.created_at)
    cand_end = ensure_utc(candidate.end_at) if candidate.end_at else cand_begin + timedelta(hours=1)

    blockers: list[str] = []
    for other in others:
        if other.id == candidate.id or other.state not in ACTIVE_STATES:
            continue
        ob = ensure_utc(other.begin_at) if other.begin_at else ensure_utc(other.created_at)
        oe = ensure_utc(other.end_at) if other.end_at else ob + timedelta(hours=1)
        if cand_end <= ob or oe <= cand_begin:
            continue
        p = placements.get(other.id)
        if p and not p.conflict and p.node:
            where = p.node if not _is_synthetic(lanes, p.node) else f"GPU {p.gpu_start}"
            blockers.append(
                f"{other.model} on {where} ({ob.strftime('%H:%M')}–{oe.strftime('%H:%M')})"
            )

    want = f"{candidate.requested_gpus} × {candidate.gpu_class}" if candidate.gpu_class \
        else f"{candidate.requested_gpus} GPUs"

    if blockers:
        detail = f"Needs {want} but overlaps with {', '.join(blockers[:3])}"
        if len(blockers) > 3:
            detail += f" and {len(blockers) - 3} more"
        return detail
    return (
        f"No node has {want} free for this window. "
        "Contiguous GPUs are only available within a single node."
    )


def _is_synthetic(lanes: list[NodeLane], node: str) -> bool:
    return any(lane.name == node and lane.synthetic for lane in lanes)
