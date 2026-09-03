"""Two log directories, written by two different machines.

`VLLM_LOG_DIR` is ours; `JOB_LOG_DIR` becomes Slurm's `--output`/`--error` and
is resolved *by the compute node*. Sending a container path there is not a
cosmetic mistake — Slurm cannot open the output file and the job dies at
launch, leaving no log to explain itself. These tests pin down the split and
the fallback that keeps single-host deployments working.
"""

from __future__ import annotations

import inspect

from app.settings import Settings


def test_job_log_dir_falls_back_to_our_own():
    """No JOB_LOG_DIR means the historical single-directory behaviour."""
    s = Settings(_env_file=None, VLLM_LOG_DIR="/var/log/router")
    assert s.job_log_dir == "/var/log/router"
    assert s.job_log_dir_local == "/var/log/router"


def test_job_log_dir_is_independent_when_set():
    s = Settings(_env_file=None, VLLM_LOG_DIR="/app/logs", JOB_LOG_DIR="/mnt/shared/logs")
    assert s.vllm_log_dir == "/app/logs"
    assert s.job_log_dir == "/mnt/shared/logs"
    # Unset local view means "the nodes and we agree on the path".
    assert s.job_log_dir_local == "/mnt/shared/logs"


def test_local_view_can_differ_from_the_nodes_view():
    """A container bind may land the same directory somewhere else."""
    s = Settings(
        _env_file=None,
        VLLM_LOG_DIR="/app/logs",
        JOB_LOG_DIR="/mnt/shared/logs",
        JOB_LOG_DIR_LOCAL="/app/job-logs",
    )
    assert s.job_log_dir == "/mnt/shared/logs"
    assert s.job_log_dir_local == "/app/job-logs"


def test_submission_sends_the_cluster_path_not_ours():
    """The regression that motivated the split: `/app/logs` reaching sbatch."""
    import app.admin as admin

    src = inspect.getsource(admin._submit_to_slurm_from_snapshot)
    assert "log_dir=settings.job_log_dir" in src
    assert "settings.vllm_log_dir" not in src


def test_log_viewing_reads_the_local_view():
    import app.admin as admin

    src = inspect.getsource(admin._find_log_files)
    assert "settings.job_log_dir_local" in src


def test_startup_tolerates_an_unreachable_job_log_dir(monkeypatch, caplog):
    """Off-cluster routers cannot see it; that must warn, not crash."""
    import app.main as main

    monkeypatch.setattr(main.settings, "job_log_dir_local", "/definitely/not/writable")
    monkeypatch.setattr(main.settings, "job_log_dir", "/definitely/not/writable")

    def boom(*a, **kw):
        raise OSError("Read-only file system")

    monkeypatch.setattr(main.os, "makedirs", boom)
    with caplog.at_level("WARNING"):
        main._ensure_job_log_dir()
    assert "job log directory" in caplog.text


def test_startup_creates_the_job_log_dir_when_it_can(tmp_path, monkeypatch):
    import app.main as main

    target = tmp_path / "shared" / "logs"
    monkeypatch.setattr(main.settings, "job_log_dir_local", str(target))
    main._ensure_job_log_dir()
    assert target.is_dir()
