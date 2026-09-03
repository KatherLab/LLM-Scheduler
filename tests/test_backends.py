"""Tests for the ClusterBackend layer.

Covers the two things that were previously untestable without a live Slurm:
the sbatch argv we actually build, and the reconcile-relevant distinction
between "job is gone" and "controller is down".
"""

from __future__ import annotations

from datetime import datetime, timedelta, timezone

import pytest

from app.backends import (
    CAP_TEST_ONLY,
    ClusterUnavailableError,
    JobSpec,
    LocalBackend,
    NodeInfo,
    set_backend,
)
from app.backends.slurm_cli import SlurmCliBackend, build_sbatch_argv
from app.backends.types import CAP_ACCOUNTING, ForeignJob, GpuGroup


@pytest.fixture(autouse=True)
def _reset_backend():
    yield
    set_backend(None)


def _spec(**over) -> JobSpec:
    base = dict(
        job_name="vllm-test",
        script_path="/opt/templates/vllm_job.sh",
        gpus=2,
        time_limit="02:00:00",
        env={"MODEL_PATH": "Qwen/Qwen3-0.6B"},
        cpus=8,
        log_dir="/tmp/logs",
    )
    base.update(over)
    return JobSpec(**base)


# ── sbatch argv construction ─────────────────────────────────────────────────

def test_argv_defaults_to_untyped_gres():
    argv = build_sbatch_argv(_spec())
    assert "--gres=gpu:2" in argv


def test_argv_typed_gres_pins_gpu_class():
    """Heterogeneous pools need the GPU *type*, not just a count."""
    argv = build_sbatch_argv(_spec(gres="gpu:h200:4"))
    assert "--gres=gpu:h200:4" in argv
    assert not any(a == "--gres=gpu:2" for a in argv)


def test_argv_includes_begin_nodelist_and_reservation():
    """The three flags that make a `managed`-pool booking truthful."""
    begin = datetime(2026, 8, 20, 15, 40, tzinfo=timezone.utc)
    argv = build_sbatch_argv(_spec(begin=begin, nodelist="gpu03", reservation="llm-standing"))
    assert "--begin=2026-08-20T15:40:00" in argv
    assert "--nodelist=gpu03" in argv
    assert "--reservation=llm-standing" in argv


def test_argv_carries_attribution_comment():
    """All jobs run as one service account, so the requester survives only here."""
    argv = build_sbatch_argv(_spec(comment="user:jrichartz,lease:412"))
    assert "--comment=user:jrichartz,lease:412" in argv


def test_argv_omits_optional_flags_when_unset():
    argv = build_sbatch_argv(_spec())
    joined = " ".join(argv)
    for flag in ("--mem=", "--begin=", "--nodelist=", "--reservation=", "--comment=", "--mail-user="):
        assert flag not in joined


def test_argv_script_path_is_last():
    """sbatch takes the script as the final positional argument."""
    argv = build_sbatch_argv(_spec())
    assert argv[-1] == "/opt/templates/vllm_job.sh"


def test_estimate_reuses_the_real_submit_argv():
    """The preview must describe the job we would actually submit."""
    spec = _spec(gres="gpu:h200:4", nodelist="gpu03")
    submit_argv = build_sbatch_argv(spec)
    estimate_argv = list(submit_argv)
    estimate_argv.insert(1, "--test-only")
    assert estimate_argv[2:] == submit_argv[1:]


# ── capability probing ───────────────────────────────────────────────────────

def test_cli_backend_can_skip_probing():
    backend = SlurmCliBackend(probe=False)
    assert backend.capabilities == frozenset()


def test_missing_sacct_is_not_fatal(monkeypatch):
    """Losing sacct costs richer failure reasons, not the whole app — this is
    what unties the deployment from the accounting host."""
    import app.backends.slurm_cli as mod

    monkeypatch.setattr(mod.shutil, "which", lambda b: None if b == "sacct" else f"/usr/bin/{b}")
    backend = SlurmCliBackend()
    assert CAP_ACCOUNTING not in backend.capabilities
    assert CAP_TEST_ONLY in backend.capabilities


# ── LocalBackend ─────────────────────────────────────────────────────────────

NOW = datetime(2026, 8, 20, 12, 0, tzinfo=timezone.utc)


def _local(**kw) -> LocalBackend:
    return LocalBackend(
        nodes=[NodeInfo(name="jupiter", gpus=(GpuGroup("gpu96", 4),), state="IDLE")],
        clock=lambda: NOW,
        **kw,
    )


async def test_local_submit_then_running():
    backend = _local()
    res = await backend.submit(_spec())
    states = await backend.job_states([res.job_id])
    assert states[res.job_id].is_running


async def test_local_pending_job_reports_estimated_start():
    """The ghost-block path: PENDING with a start estimate."""
    backend = _local(pending_for=timedelta(minutes=30))
    res = await backend.submit(_spec())
    state = (await backend.job_states([res.job_id]))[res.job_id]
    assert state.is_pending
    assert state.start_time == NOW + timedelta(minutes=30)


async def test_local_gone_job_is_none_with_exit_info():
    backend = _local()
    res = await backend.submit(_spec())
    backend.fail_job(res.job_id, state="OUT_OF_MEMORY", exit_code="0:125")

    assert (await backend.job_states([res.job_id]))[res.job_id] is None
    info = (await backend.job_exit_info([res.job_id]))[res.job_id]
    assert info.state == "OUT_OF_MEMORY"


async def test_local_forgotten_job_has_no_exit_info():
    """The no-accounting case: gone from the queue, nothing to say about why."""
    backend = _local()
    res = await backend.submit(_spec())
    backend.forget_job(res.job_id)

    assert (await backend.job_states([res.job_id]))[res.job_id] is None
    assert (await backend.job_exit_info([res.job_id]))[res.job_id] is None


async def test_local_outage_raises_rather_than_reporting_jobs_gone():
    """The reconciler must skip a cycle, not conclude every model died."""
    backend = _local()
    res = await backend.submit(_spec())
    backend.unavailable = True

    with pytest.raises(ClusterUnavailableError):
        await backend.job_states([res.job_id])


async def test_local_cancel_records_cancellation():
    backend = _local()
    res = await backend.submit(_spec())
    await backend.cancel(res.job_id)

    assert (await backend.job_states([res.job_id]))[res.job_id] is None
    assert (await backend.job_exit_info([res.job_id]))[res.job_id].state == "CANCELLED"


async def test_local_foreign_jobs_filter_by_partition():
    backend = _local()
    backend.set_foreign_jobs([
        ForeignJob(job_id="1", user="alice", state="RUNNING", nodes=("gpu01",),
                   gpus=2, partition="general"),
        ForeignJob(job_id="2", user="bob", state="RUNNING", nodes=("gpu02",),
                   gpus=4, partition="llm"),
    ])
    assert len(await backend.foreign_jobs()) == 2
    assert [j.user for j in await backend.foreign_jobs("general")] == ["alice"]


async def test_local_unknown_job_ids_map_to_none():
    backend = _local()
    assert await backend.job_states(["does-not-exist"]) == {"does-not-exist": None}


# ── NodeInfo usability ───────────────────────────────────────────────────────

@pytest.mark.parametrize("state", ["DOWN", "DRAIN", "DRAINING", "down*", "MAINT", "FAIL"])
def test_unusable_node_states(state):
    """Drained nodes must not receive placements."""
    assert not NodeInfo(name="n", state=state).is_usable


@pytest.mark.parametrize("state", ["IDLE", "ALLOCATED", "MIXED", "idle~"])
def test_usable_node_states(state):
    assert NodeInfo(name="n", state=state).is_usable
