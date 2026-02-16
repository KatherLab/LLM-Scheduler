from __future__ import annotations
from dataclasses import dataclass
from datetime import datetime, timedelta
from typing import Iterable, Optional

@dataclass
class Placement:
    lease_id: int
    lane_start: Optional[int]
    lane_count: Optional[int]
    conflict: bool

def _t(dt: Optional[datetime], fallback: datetime) -> datetime:
    return dt if dt is not None else fallback

def compute_placements(
    *,
    leases: Iterable,
    total_gpus: int,
    horizon_start: datetime,
    horizon_end: datetime,
) -> dict[int, Placement]:
    """
    Place leases onto GPU lanes for visualization.
    - We treat each GPU as a lane.
    - A lease with g GPUs occupies a contiguous block of g lanes.
    - First-fit in time order.
    - If a lease cannot be placed (overbooked), mark conflict=True.
    """
    # Occupancy: for each lane, list of (start,end) intervals
    occ: list[list[tuple[datetime, datetime]]] = [[] for _ in range(total_gpus)]

    # Normalize/clip intervals to horizon (still used for overlap checks)
    items = []
    for l in leases:
        begin = _t(l.begin_at, l.created_at)
        end = l.end_at or (begin + timedelta(hours=1))
        # ignore fully outside horizon
        if end <= horizon_start or begin >= horizon_end:
            continue
        items.append((begin, end, l))

    # sort by begin time, then longer first helps packing
    items.sort(key=lambda x: (x[0], -(x[1]-x[0]).total_seconds()))

    placements: dict[int, Placement] = {}

    def overlaps(a0: datetime, a1: datetime, b0: datetime, b1: datetime) -> bool:
        return not (a1 <= b0 or b1 <= a0)

    def block_free(lane_idx: int, begin: datetime, end: datetime) -> bool:
        for (s, e) in occ[lane_idx]:
            if overlaps(begin, end, s, e):
                return False
        return True

    for begin, end, l in items:
        g = max(1, int(getattr(l, "requested_gpus", 1) or 1))
        placed = False

        # find contiguous block
        for start_lane in range(0, total_gpus - g + 1):
            ok = True
            for lane in range(start_lane, start_lane + g):
                if not block_free(lane, begin, end):
                    ok = False
                    break
            if ok:
                for lane in range(start_lane, start_lane + g):
                    occ[lane].append((begin, end))
                placements[l.id] = Placement(
                    lease_id=l.id,
                    lane_start=start_lane,
                    lane_count=g,
                    conflict=False,
                )
                placed = True
                break

        if not placed:
            placements[l.id] = Placement(
                lease_id=l.id,
                lane_start=None,
                lane_count=g,
                conflict=True,
            )

    return placements
