"""The `app.slurm` shim must keep `main.py`/`admin.py` working unchanged.

The delicate part is the sync entry points: `reconcile_on_startup()` calls them
from inside the FastAPI lifespan, so a loop is already running and a naive
`asyncio.run()` would raise.
"""

from __future__ import annotations

from datetime import datetime, timezone

import pytest

from app import slurm
from app.backends import LocalBackend, NodeInfo, set_backend
from app.backends.types import GpuGroup

NOW = datetime(2026, 8, 20, 12, 0, tzinfo=timezone.utc)


@pytest.fixture
def local():
    backend = LocalBackend(
        nodes=[NodeInfo(name="jupiter", gpus=(GpuGroup("gpu96", 4),), state="IDLE")],
        clock=lambda: NOW,
    )
    set_backend(backend)
    yield backend
    set_backend(None)


def _submit() -> str:
    return slurm.submit_vllm_job(
        template_path="/opt/templates/vllm_job.sh",
        job_name="vllm-test",
        gpus=2,
        time_limit="02:00:00",
        begin=None,
        env={"MODEL_PATH": "x"},
    ).job_id


def test_submit_returns_legacy_result_shape(local):
    res = slurm.submit_vllm_job(
        template_path="/opt/templates/vllm_job.sh",
        job_name="vllm-test",
        gpus=2,
        time_limit="02:00:00",
        begin=None,
        env={},
    )
    assert res.job_id and isinstance(res.raw, str)
    assert local.submitted[0].gpus == 2


def test_squeue_batch_returns_plain_state_strings(local):
    """Legacy shape: {job_id: "RUNNING"} — not a JobState object."""
    job_id = _submit()
    states = slurm.squeue_job_states_batch([job_id])
    assert states == {job_id: "RUNNING"}


def test_squeue_batch_maps_gone_jobs_to_none(local):
    job_id = _submit()
    local.forget_job(job_id)
    assert slurm.squeue_job_states_batch([job_id]) == {job_id: None}


def test_exit_info_returns_legacy_dict_shape(local):
    job_id = _submit()
    local.fail_job(job_id, state="OUT_OF_MEMORY", exit_code="0:125")

    info = slurm.sacct_job_exit_info_batch([job_id])[job_id]
    assert info["state"] == "OUT_OF_MEMORY"
    assert info["exit_code"] == "0:125"
    assert info["source"] == "local"


def test_unavailable_error_alias_is_still_catchable(local):
    """`except slurm.SlurmUnavailableError` appears in the reconcile worker."""
    job_id = _submit()
    local.unavailable = True
    with pytest.raises(slurm.SlurmUnavailableError):
        slurm.squeue_job_states_batch([job_id])


async def test_sync_helpers_work_inside_a_running_event_loop(local):
    """`reconcile_on_startup()` calls these from within the lifespan.

    LocalBackend has no `*_sync` methods, so this exercises the worker-thread
    fallback rather than the fast path.
    """
    job_id = _submit()
    assert slurm.squeue_job_states_batch([job_id]) == {job_id: "RUNNING"}


async def test_async_helpers(local):
    job_id = _submit()
    assert await slurm.async_squeue_job_state(job_id) == "RUNNING"

    await slurm.async_cancel(job_id)
    assert await slurm.async_squeue_job_state(job_id) is None

    info = await slurm.async_sacct_job_exit_info_batch([job_id])
    assert info[job_id]["state"] == "CANCELLED"


def test_empty_batch_is_a_no_op(local):
    assert slurm.squeue_job_states_batch([]) == {}
    assert slurm.sacct_job_exit_info_batch([]) == {}
