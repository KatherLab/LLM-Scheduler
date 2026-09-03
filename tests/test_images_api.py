"""The images API, through the real routes.

Image management is admin-only for a concrete reason: a build writes gigabytes
to shared storage and takes cluster time, and a delete can break every future
job for a GPU class. Unenforced RBAC is the failure mode worth testing for, so
every endpoint is checked, not just the interesting one.
"""

from __future__ import annotations

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from app import images
from app.auth import current_user, require_auth
from app.authz import User
from app.backends.types import SubmitResult
from app.dependencies import SessionLocal, init_db
from app.images_api import router as images_router
from app.inventory import Inventory
from app.models import ImageBuild
from tests.test_images import CLUSTER, NODES

ADMIN = User(sub="root", display_name="Root", is_admin=True, is_user=True, via="ldap")
ALICE = User(sub="alice", display_name="Alice", is_user=True, via="ldap")


@pytest.fixture
def client(tmp_path, monkeypatch):
    d = tmp_path / "images"
    d.mkdir()
    monkeypatch.setattr(images.settings, "apptainer_image_dir", str(d))
    monkeypatch.setattr(images.settings, "image_build_template_path",
                        "templates/apptainer_build.sh")
    monkeypatch.setattr("app.images_api.get_cluster", lambda *a, **k: CLUSTER)
    monkeypatch.setattr(
        "app.images_api.inventory.current",
        lambda: Inventory(nodes=tuple(NODES)),
    )

    submitted: list = []

    class FakeBackend:
        async def submit(self, spec):
            submitted.append(spec)
            return SubmitResult(job_id="4242", raw="4242")

        async def cancel(self, job_id):
            submitted.append(("cancel", job_id))

    monkeypatch.setattr("app.images_api.get_backend", lambda: FakeBackend())

    init_db()
    app = FastAPI()
    app.include_router(images_router)

    state = {"user": ADMIN}
    app.dependency_overrides[current_user] = lambda: state["user"]
    app.dependency_overrides[require_auth] = lambda: {"sub": state["user"].sub}

    c = TestClient(app)
    c.act_as = lambda user: state.__setitem__("user", user)
    c.image_dir = d
    c.submitted = submitted
    yield c

    with SessionLocal() as db:
        for row in db.query(ImageBuild).all():
            db.delete(row)
        db.commit()


# ── Authorization ────────────────────────────────────────────────────────────

def test_every_endpoint_refuses_a_non_admin(client):
    client.act_as(ALICE)
    assert client.get("/admin/images").status_code == 403
    assert client.post("/admin/images/build", json={
        "source_ref": "vllm/vllm-openai:v0.11.0", "arch": "aarch64",
    }).status_code == 403
    assert client.delete("/admin/images/anything.sif").status_code == 403
    assert client.post("/admin/images/builds/1/cancel").status_code == 403
    assert client.get("/admin/images/builds/1/logs").status_code == 403


# ── Listing ──────────────────────────────────────────────────────────────────

def test_listing_reports_targets_and_files(client):
    (client.image_dir / "spare.sif").write_bytes(b"x" * 5)
    body = client.get("/admin/images").json()

    assert [i["name"] for i in body["images"]] == ["spare.sif"]
    assert {t["arch"] for t in body["targets"]} == {"aarch64", "x86_64"}
    assert body["error"] is None


def test_an_unreadable_directory_is_reported_not_hidden(client, monkeypatch, tmp_path):
    """Showing an empty list would read as 'you have no images'."""
    monkeypatch.setattr(images.settings, "apptainer_image_dir", str(tmp_path / "gone"))
    body = client.get("/admin/images").json()
    assert body["images"] == []
    assert "not readable" in body["error"]


# ── Building ─────────────────────────────────────────────────────────────────

def test_build_submits_a_job_and_records_it(client):
    r = client.post("/admin/images/build", json={
        "source_ref": "vllm/vllm-openai:v0.11.0", "arch": "aarch64",
    })
    assert r.status_code == 200, r.text
    body = r.json()
    assert body["image_name"] == "vllm-openai-v0.11.0-aarch64.sif"
    assert body["source_ref"] == "docker://vllm/vllm-openai:v0.11.0"
    assert body["slurm_job_id"] == "4242"
    assert body["state"] == "SUBMITTED"
    assert body["requested_by"] == "root"

    spec = client.submitted[0]
    assert spec.partition == "spark"     # the only aarch64 partition
    assert spec.gpus == 0


def test_a_bad_reference_never_reaches_the_scheduler(client):
    r = client.post("/admin/images/build", json={
        "source_ref": "vllm/vllm-openai:v1; rm -rf /", "arch": "aarch64",
    })
    assert r.status_code == 400
    assert client.submitted == []


def test_building_for_an_architecture_with_no_nodes_is_refused(client, monkeypatch):
    monkeypatch.setattr(
        "app.images_api.inventory.current",
        lambda: Inventory(nodes=(NODES[3],)),   # vega, x86_64 only
    )
    r = client.post("/admin/images/build", json={
        "source_ref": "vllm/vllm-openai:v0.11.0", "arch": "aarch64",
    })
    assert r.status_code == 400
    assert "cross-build" in r.json()["detail"]


def test_overwriting_an_existing_image_needs_saying_so(client):
    (client.image_dir / "taken.sif").write_bytes(b"x")
    payload = {
        "source_ref": "vllm/vllm-openai:v0.11.0",
        "arch": "aarch64", "name": "taken.sif",
    }
    assert client.post("/admin/images/build", json=payload).status_code == 409

    payload["overwrite"] = True
    assert client.post("/admin/images/build", json=payload).status_code == 200


def test_two_builds_of_the_same_name_cannot_race(client):
    """They would write the same file; the second is refused, not queued."""
    payload = {"source_ref": "vllm/vllm-openai:v0.11.0", "arch": "aarch64"}
    assert client.post("/admin/images/build", json=payload).status_code == 200
    r = client.post("/admin/images/build", json=payload)
    assert r.status_code == 409
    assert "already" in r.json()["detail"]


# ── Deleting ─────────────────────────────────────────────────────────────────

def test_delete_removes_an_unreferenced_image(client):
    (client.image_dir / "spare.sif").write_bytes(b"x")
    r = client.delete("/admin/images/spare.sif")
    assert r.status_code == 200
    assert not (client.image_dir / "spare.sif").exists()


def test_delete_is_refused_while_a_build_is_writing_that_name(client):
    (client.image_dir / "busy.sif").write_bytes(b"x")
    client.post("/admin/images/build", json={
        "source_ref": "vllm/vllm-openai:v0.11.0", "arch": "aarch64",
        "name": "busy.sif", "overwrite": True,
    })
    r = client.delete("/admin/images/busy.sif")
    assert r.status_code == 409
    assert (client.image_dir / "busy.sif").exists()


def test_delete_cannot_escape_the_images_directory(client):
    r = client.delete("/admin/images/..%2F..%2Fetc%2Fshadow.sif")
    assert r.status_code in (404, 409)
