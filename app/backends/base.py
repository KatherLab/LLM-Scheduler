"""The ClusterBackend protocol.

Every call into the batch scheduler goes through here. Three implementations:

  SlurmCliBackend    subprocess — needs binaries + munge on the app host
  SlurmRestBackend   slurmrestd over JWT — runs anywhere (Phase 0, later)
  LocalBackend       in-memory fake — makes scheduling logic testable at all

Callers must check `capabilities` before using an optional method rather than
catching exceptions; `NotImplementedError` here means "this backend never
supports it", not "it failed this time".
"""

from __future__ import annotations

from typing import Protocol, runtime_checkable

from .types import (
    ExitInfo,
    ForeignJob,
    JobSpec,
    JobState,
    NodeInfo,
    StartEstimate,
    SubmitResult,
)


@runtime_checkable
class ClusterBackend(Protocol):
    name: str
    capabilities: frozenset[str]

    async def submit(self, spec: JobSpec) -> SubmitResult:
        """Submit a job. Raises ClusterUnavailableError if the scheduler is down."""
        ...

    async def cancel(self, job_id: str) -> None:
        """Cancel a job. Cancelling an already-gone job is not an error."""
        ...

    async def extend_time(self, job_id: str, new_time_limit: str) -> None:
        ...

    async def retarget_gpu_class(self, job_id: str, gres: str, gpu_class: str) -> None:
        """Repoint a PENDING job at a different GPU class, in place."""
        ...

    async def job_states(self, job_ids: list[str]) -> dict[str, JobState | None]:
        """Current queue state. `None` means the job has left the queue.

        Raises ClusterUnavailableError rather than reporting every job as gone
        when the controller is unreachable.
        """
        ...

    async def job_exit_info(self, job_ids: list[str]) -> dict[str, ExitInfo | None]:
        """Why jobs left the queue. Best-effort: returns `None` per job when no
        source is available. Never raises for missing accounting."""
        ...

    async def nodes(self) -> list[NodeInfo]:
        """Node inventory. Requires CAP_NODE_DISCOVERY."""
        ...

    async def foreign_jobs(self, partition: str | None = None) -> list[ForeignJob]:
        """Jobs owned by other users. Requires CAP_FOREIGN_JOBS."""
        ...

    async def estimate_start(self, spec: JobSpec) -> StartEstimate:
        """Dry-run: when would this job start? Requires CAP_TEST_ONLY."""
        ...
