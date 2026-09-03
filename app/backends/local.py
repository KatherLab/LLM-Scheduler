"""In-memory fake scheduler.

Exists so the scheduling logic can be tested at all — until now none of it could
run without a live Slurm. Also useful for developing the UI on a laptop.

Time is injected rather than read from the clock so tests are deterministic:
pass a `clock` callable and drive it yourself.
"""

from __future__ import annotations

from dataclasses import replace
from datetime import datetime, timedelta, timezone
from typing import Callable

from .types import (
    CAP_FOREIGN_JOBS,
    CAP_NODE_DISCOVERY,
    CAP_TEST_ONLY,
    ClusterUnavailableError,
    ExitInfo,
    ForeignJob,
    JobSpec,
    JobState,
    NodeInfo,
    StartEstimate,
    SubmitResult,
)


class LocalBackend:
    """A scheduler that runs nothing but behaves plausibly.

    Submitted jobs go straight to RUNNING unless `pending_for` is set, in which
    case they sit PENDING for that long — enough to exercise the ghost-block
    path without a cluster.
    """

    name = "local"

    def __init__(
        self,
        *,
        nodes: list[NodeInfo] | None = None,
        clock: Callable[[], datetime] | None = None,
        pending_for: timedelta = timedelta(0),
        unavailable: bool = False,
    ):
        self.capabilities = frozenset(
            {CAP_TEST_ONLY, CAP_NODE_DISCOVERY, CAP_FOREIGN_JOBS}
        )
        self._nodes = nodes or []
        self._clock = clock or (lambda: datetime.now(timezone.utc))
        self._pending_for = pending_for
        self.unavailable = unavailable

        self._next_id = 1000
        self._jobs: dict[str, dict] = {}
        self._exits: dict[str, ExitInfo] = {}
        self._foreign: list[ForeignJob] = []

        # Test introspection
        self.submitted: list[JobSpec] = []

    # ── test hooks ──────────────────────────────────────────────────────────
    def set_foreign_jobs(self, jobs: list[ForeignJob]) -> None:
        self._foreign = list(jobs)

    def finish_job(self, job_id: str, state: str = "COMPLETED", exit_code: str = "0:0") -> None:
        """Remove a job from the queue and record why it left."""
        self._jobs.pop(job_id, None)
        self._exits[job_id] = ExitInfo(state=state, exit_code=exit_code, source="local")

    def fail_job(self, job_id: str, state: str = "FAILED", exit_code: str = "1:0") -> None:
        self.finish_job(job_id, state=state, exit_code=exit_code)

    def forget_job(self, job_id: str) -> None:
        """Job vanished with no accounting record — the no-sacct case."""
        self._jobs.pop(job_id, None)
        self._exits.pop(job_id, None)

    def _guard(self) -> None:
        if self.unavailable:
            raise ClusterUnavailableError("LocalBackend: simulated controller outage")

    # ── ClusterBackend ──────────────────────────────────────────────────────
    async def submit(self, spec: JobSpec) -> SubmitResult:
        self._guard()
        job_id = str(self._next_id)
        self._next_id += 1
        now = self._clock()
        start = spec.begin or now
        if self._pending_for:
            start = max(start, now + self._pending_for)
        self._jobs[job_id] = {"spec": spec, "start": start, "submitted": now}
        self.submitted.append(spec)
        return SubmitResult(job_id=job_id, raw=job_id)

    async def cancel(self, job_id: str) -> None:
        self._guard()
        if job_id in self._jobs:
            self.finish_job(job_id, state="CANCELLED", exit_code="0:15")

    async def extend_time(self, job_id: str, new_time_limit: str) -> None:
        self._guard()
        job = self._jobs.get(job_id)
        if job:
            job["spec"] = replace(job["spec"], time_limit=new_time_limit)

    async def job_states(self, job_ids: list[str]) -> dict[str, JobState | None]:
        self._guard()
        now = self._clock()
        out: dict[str, JobState | None] = {}
        for jid in job_ids:
            job = self._jobs.get(jid)
            if job is None:
                out[jid] = None
                continue
            pending = job["start"] > now
            node = self._nodes[0].name if self._nodes else "localhost"
            out[jid] = JobState(
                state="PENDING" if pending else "RUNNING",
                nodes=None if pending else node,
                start_time=job["start"],
            )
        return out

    async def job_exit_info(self, job_ids: list[str]) -> dict[str, ExitInfo | None]:
        return {jid: self._exits.get(jid) for jid in job_ids}

    async def nodes(self) -> list[NodeInfo]:
        self._guard()
        return list(self._nodes)

    async def foreign_jobs(self, partition: str | None = None) -> list[ForeignJob]:
        self._guard()
        if partition is None:
            return list(self._foreign)
        return [j for j in self._foreign if j.partition == partition]

    async def estimate_start(self, spec: JobSpec) -> StartEstimate:
        self._guard()
        now = self._clock()
        start = max(spec.begin or now, now + self._pending_for)
        nodes = (self._nodes[0].name,) if self._nodes else ()
        return StartEstimate(
            start_time=start,
            nodes=nodes,
            raw=f"local: job would start at {start.isoformat()}",
        )
