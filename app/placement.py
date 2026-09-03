"""Per-node GPU placement.

Replaces `planner.py`'s flat `TOTAL_GPUS` strip, which had three problems on a
real cluster:

* Contiguity across the whole cluster is meaningless — a block of GPUs only
  matters within one node, which is where NVLink and NCCL live.
* Heterogeneous VRAM is not representable, so a gpu96 model could be "placed"
  on 24 GB cards.
* Only our own bookings were visible, so on a shared partition the calendar was
  promising GPUs Slurm had already given away.

The grid here is keyed by `(node, gpu_index)`, filtered by GPU class, and
accepts an overlay of foreign jobs it cannot move.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timedelta
from typing import Iterable, Sequence

from .backends.types import ForeignJob, NodeInfo
from .cluster import DEFAULT_GUARD_GAP_SECONDS, ClusterConfig
from .utils import ensure_utc


@dataclass(frozen=True)
class Placement:
    """Where a booking sits. `node`/`gpu_start` are None when it does not fit."""

    lease_id: int
    node: str | None = None
    gpu_class: str | None = None
    gpu_start: int | None = None
    gpu_count: int = 0
    conflict: bool = False

    @property
    def gpu_indices(self) -> tuple[int, ...]:
        if self.gpu_start is None:
            return ()
        return tuple(range(self.gpu_start, self.gpu_start + self.gpu_count))


@dataclass
class _Slot:
    """One schedulable GPU: a node, an index within it, and its class."""

    node: str
    index: int
    gpu_class: str | None
    busy: list[tuple[datetime, datetime]] = field(default_factory=list)


@dataclass
class Demand:
    """What a booking needs. Decoupled from the ORM so this stays testable."""

    lease_id: int
    gpus: int
    begin: datetime
    end: datetime
    gpu_class: str | None = None
    #: Restrict to these nodes (a `managed` pool's node list).
    nodes: tuple[str, ...] = ()
    #: Pin to one node — used when re-placing an already-running job.
    pinned_node: str | None = None


def build_slots(
    nodes: Iterable[NodeInfo],
    *,
    allowed_nodes: Iterable[str] | None = None,
    gpu_class: str | None = None,
) -> list[_Slot]:
    """Expand node inventory into individual GPU slots.

    Drained and down nodes are excluded — placing a booking on one would
    produce a calendar entry that can never run.
    """
    allowed = set(allowed_nodes) if allowed_nodes else None
    slots: list[_Slot] = []
    for node in nodes:
        if not node.is_usable:
            continue
        if allowed is not None and node.name not in allowed:
            continue
        index = 0
        for group in node.gpus:
            for _ in range(group.count):
                if gpu_class is None or group.gpu_class == gpu_class:
                    slots.append(_Slot(node.name, index, group.gpu_class))
                index += 1
    return slots


def _overlaps(a0, a1, b0, b1, guard: timedelta) -> bool:
    """True unless the two intervals are separated by at least `guard`.

    Note this is the opposite of the old `OVERLAP_TOLERANCE`, which *permitted*
    near-touching bookings. A guard gap **requires** separation: it covers
    weight loading and teardown, so that one booking overrunning does not
    cascade into the next on the same node.
    """
    return not (a1 + guard <= b0 or b1 + guard <= a0)


def apply_foreign_jobs(
    slots: Sequence[_Slot],
    foreign: Iterable[ForeignJob],
    *,
    default_duration: timedelta = timedelta(hours=4),
) -> None:
    """Mark GPUs occupied by jobs we do not own.

    When the backend reports exact indices (`gres_detail` over slurmrestd) we
    block precisely those GPUs. Otherwise we consume the first free slots on
    each node — pessimistic in the right direction, since it never invents free
    capacity.

    Jobs with no end time get `default_duration` — a running job of unknown
    length should block bookings, not be silently ignored.
    """
    by_node: dict[str, list[_Slot]] = {}
    for slot in slots:
        by_node.setdefault(slot.node, []).append(slot)

    for job in foreign:
        if job.gpus <= 0:
            continue
        start = ensure_utc(job.start_time) if job.start_time else None
        end = ensure_utc(job.end_time) if job.end_time else None
        if start is None and end is None:
            continue
        if start is None:
            start = end - default_duration
        if end is None:
            end = start + default_duration

        for node_name in job.nodes:
            candidates = by_node.get(node_name)
            if not candidates:
                continue

            if job.gpu_indices:
                # Exact indices known — block precisely those GPUs, leaving the
                # rest of the node genuinely bookable.
                wanted = set(job.gpu_indices)
                for slot in candidates:
                    if slot.index in wanted:
                        slot.busy.append((start, end))
                continue

            taken = 0
            for slot in candidates:
                if taken >= job.gpus:
                    break
                slot.busy.append((start, end))
                taken += 1


def compute_placements(
    demands: Iterable[Demand],
    nodes: Sequence[NodeInfo],
    *,
    cluster: ClusterConfig | None = None,
    foreign_jobs: Iterable[ForeignJob] = (),
) -> dict[int, Placement]:
    """Assign each demand a contiguous GPU block on a single node.

    Demands are placed longest-first within a start time so that big models get
    a shot at a contiguous block before small ones fragment the node.
    """
    items = sorted(
        demands, key=lambda d: (d.begin, -(d.end - d.begin).total_seconds(), d.lease_id)
    )

    all_slots = build_slots(nodes)
    apply_foreign_jobs(all_slots, foreign_jobs)

    by_node: dict[str, list[_Slot]] = {}
    for slot in all_slots:
        by_node.setdefault(slot.node, []).append(slot)
    for slot_list in by_node.values():
        slot_list.sort(key=lambda s: s.index)

    placements: dict[int, Placement] = {}

    for demand in items:
        guard = timedelta(seconds=(
            cluster.guard_gap_seconds(demand.gpu_class)
            if cluster else DEFAULT_GUARD_GAP_SECONDS
        ))
        begin, end = ensure_utc(demand.begin), ensure_utc(demand.end)

        candidate_nodes = sorted(by_node)
        if demand.pinned_node:
            candidate_nodes = [demand.pinned_node] if demand.pinned_node in by_node else []
        elif demand.nodes:
            allowed = set(demand.nodes)
            candidate_nodes = [n for n in candidate_nodes if n in allowed]

        def _matching(node_name: str) -> list[_Slot]:
            return [
                s for s in by_node[node_name]
                if demand.gpu_class is None or s.gpu_class == demand.gpu_class
            ]

        def _free_count(node_name: str) -> int:
            return sum(
                1 for s in _matching(node_name)
                if all(not _overlaps(begin, end, b0, b1, guard) for (b0, b1) in s.busy)
            )

        # Best-fit: prefer the node with the least spare matching capacity, so a
        # 1-GPU booking does not break up a node's only contiguous block.
        candidate_nodes.sort(key=lambda n: (_free_count(n), n))

        chosen: tuple[str, list[_Slot]] | None = None
        for node_name in candidate_nodes:
            usable = _matching(node_name)
            if len(usable) < demand.gpus:
                continue
            # Contiguity is only meaningful within a node.
            for start in range(len(usable) - demand.gpus + 1):
                window = usable[start:start + demand.gpus]
                if window[-1].index - window[0].index != demand.gpus - 1:
                    continue  # not physically contiguous
                if all(
                    not _overlaps(begin, end, s0, s1, guard)
                    for slot in window for (s0, s1) in slot.busy
                ):
                    chosen = (node_name, window)
                    break
            if chosen:
                break

        if chosen is None:
            placements[demand.lease_id] = Placement(
                lease_id=demand.lease_id, gpu_count=demand.gpus,
                gpu_class=demand.gpu_class, conflict=True,
            )
            continue

        node_name, window = chosen
        for slot in window:
            slot.busy.append((begin, end))
        placements[demand.lease_id] = Placement(
            lease_id=demand.lease_id,
            node=node_name,
            gpu_class=window[0].gpu_class,
            gpu_start=window[0].index,
            gpu_count=demand.gpus,
            conflict=False,
        )

    return placements


def find_earliest_slot(
    demand: Demand,
    nodes: Sequence[NodeInfo],
    existing: Iterable[Demand],
    *,
    cluster: ClusterConfig | None = None,
    foreign_jobs: Iterable[ForeignJob] = (),
    search_end: datetime | None = None,
    step: timedelta = timedelta(minutes=15),
) -> datetime | None:
    """Earliest start at which `demand` fits.

    Candidate times are the boundaries of existing bookings plus a coarse grid,
    rather than a fine sweep: a booking can only become placeable when something
    else starts or ends.

    Only meaningful for `managed` pools. On a `slurm` pool the backfill
    scheduler knows the queue and we do not — use the backend's
    `estimate_start()` (`sbatch --test-only`) there instead.
    """
    duration = ensure_utc(demand.end) - ensure_utc(demand.begin)
    search_start = ensure_utc(demand.begin)
    horizon = ensure_utc(search_end) if search_end else search_start + timedelta(days=14)

    existing = list(existing)
    candidates: set[datetime] = {search_start}
    for other in existing:
        for t in (ensure_utc(other.begin), ensure_utc(other.end)):
            if search_start <= t <= horizon:
                candidates.add(t)

    t = search_start
    while t + duration <= horizon:
        candidates.add(t)
        t += step

    for candidate in sorted(candidates):
        if candidate + duration > horizon:
            break
        trial = Demand(
            lease_id=demand.lease_id, gpus=demand.gpus,
            begin=candidate, end=candidate + duration,
            gpu_class=demand.gpu_class, nodes=demand.nodes,
            pinned_node=demand.pinned_node,
        )
        placements = compute_placements(
            [*existing, trial], nodes, cluster=cluster, foreign_jobs=foreign_jobs
        )
        result = placements.get(demand.lease_id)
        if result and not result.conflict:
            return candidate
    return None
