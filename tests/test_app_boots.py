"""The app actually imports.

This exists because a syntax error in `app/main.py` once shipped to a running
container while the whole suite stayed green — nothing imported the module that
wires everything together. Every other test exercises `app.admin` or a library
module directly.
"""

from __future__ import annotations

import ast
import pathlib

import pytest

APP_DIR = pathlib.Path(__file__).resolve().parent.parent / "app"


@pytest.mark.parametrize(
    "path", sorted(APP_DIR.rglob("*.py")), ids=lambda p: str(p.name)
)
def test_every_module_parses(path):
    ast.parse(path.read_text(encoding="utf-8"), filename=str(path))


def test_the_asgi_app_imports_and_has_its_routes():
    """Catches import-time errors the parse check cannot: bad names, circular
    imports, decorator failures."""
    from app.main import app

    paths = {r.path for r in app.routes}
    for expected in ("/health", "/v1/chat/completions", "/admin/dashboard"):
        assert expected in paths


def test_all_workers_are_defined():
    """A worker referenced in the lifespan but missing would only fail at
    startup, in production."""
    import app.main as main

    for name in (
        "inventory_worker", "estimate_worker", "health_worker",
        "planned_submit_worker", "endpoint_cleanup_worker",
        "slurm_reconcile_worker", "retry_worker", "renew_worker",
        "leader_worker", "image_build_worker",
    ):
        assert callable(getattr(main, name)), name


@pytest.mark.parametrize("name", ["vllm_job.sh", "apptainer_build.sh"])
def test_job_template_is_valid_bash(name):
    """The templates are shipped as-is to compute nodes; a syntax error there
    is only visible once a job has queued and started."""
    import subprocess

    template = APP_DIR.parent / "templates" / name
    proc = subprocess.run(["bash", "-n", str(template)], capture_output=True, text=True)
    assert proc.returncode == 0, proc.stderr
