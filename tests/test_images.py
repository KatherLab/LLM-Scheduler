"""Apptainer image management.

The two things worth being strict about: a build is a Slurm job on a node of
the matching architecture (Apptainer cannot cross-build, so getting this wrong
produces an image that fails much later on a different machine), and a name or
reference from a user must not be able to escape the images directory or the
argv of `apptainer build`.
"""

from __future__ import annotations

import os

import pytest

from app import images
from app.backends.types import GpuGroup, NodeInfo
from app.cluster import ClusterConfig, GpuClass, Pool, Runtime

CLUSTER = ClusterConfig(
    runtimes={
        "arm-vllm": Runtime(name="arm-vllm", kind="apptainer",
                            image="/shared/images/vllm-aarch64.sif"),
        "legacy": Runtime(name="legacy", kind="venv", activate="/opt/venv/bin/activate"),
    },
    gpu_classes={
        "gb10": GpuClass(name="gb10", vram_gb=119, arch="aarch64", runtime="arm-vllm"),
        "gpu141": GpuClass(name="gpu141", vram_gb=140, arch="x86_64", runtime="legacy"),
        "gpu48": GpuClass(name="gpu48", vram_gb=48, arch="x86_64", runtime="legacy"),
    },
    pools={
        "spark": Pool(name="spark", partition="spark", scheduling="managed"),
        "llm": Pool(name="llm", partition="llm", scheduling="managed"),
    },
)


def _node(name, gpu_class, partition, state="IDLE"):
    return NodeInfo(name=name, gpus=(GpuGroup(gpu_class, 1),),
                    partitions=(partition,), state=state)


NODES = [
    _node("spark-049c", "gb10", "spark"),
    _node("spark-c127", "gb10", "spark"),
    _node("sv-ekfz-cai-h200", "gpu141", "llm"),
    _node("vega", "gpu48", "llm"),
]


@pytest.fixture
def image_dir(tmp_path, monkeypatch):
    d = tmp_path / "images"
    d.mkdir()
    monkeypatch.setattr(images.settings, "apptainer_image_dir", str(d))
    return d


# ── Names and references ─────────────────────────────────────────────────────

@pytest.mark.parametrize("raw,expected", [
    ("vllm.sif", "vllm.sif"),
    ("vllm", "vllm.sif"),                     # suffix is added, not demanded
    ("vllm-0.11.0-x86_64.sif", "vllm-0.11.0-x86_64.sif"),
])
def test_valid_names(raw, expected):
    assert images.validate_name(raw) == expected


@pytest.mark.parametrize("bad", [
    "../escape.sif",
    "/etc/passwd.sif",
    "sub/dir/img.sif",
    ".hidden.sif",
    "",
    "img$(whoami).sif",
])
def test_names_that_must_be_refused(bad):
    with pytest.raises(images.ImageError):
        images.validate_name(bad)


@pytest.mark.parametrize("raw,expected", [
    ("vllm/vllm-openai:v0.11.0", "docker://vllm/vllm-openai:v0.11.0"),
    ("docker://nvcr.io/nvidia/vllm:25.09", "docker://nvcr.io/nvidia/vllm:25.09"),
    ("alpine", "docker://alpine"),
])
def test_refs_are_normalised_to_a_scheme(raw, expected):
    assert images.validate_ref(raw) == expected


@pytest.mark.parametrize("bad", [
    "vllm/vllm-openai:v1; rm -rf /",
    "vllm/vllm-openai --force",
    "$(id)",
    "img`whoami`",
    "a b",
    "",
])
def test_refs_that_could_reach_a_shell_are_refused(bad):
    with pytest.raises(images.ImageError):
        images.validate_ref(bad)


def test_suggested_name_carries_the_tag_and_arch():
    ref = images.validate_ref("vllm/vllm-openai:v0.11.0")
    assert images.suggest_name(ref, "x86_64") == "vllm-openai-v0.11.0-x86_64.sif"
    assert images.suggest_name(ref, "aarch64") == "vllm-openai-v0.11.0-aarch64.sif"


def test_untagged_source_suggests_latest():
    assert images.suggest_name("docker://alpine", "x86_64") == "alpine-latest-x86_64.sif"


def test_image_path_stays_inside_the_directory(image_dir):
    assert images.image_path("a.sif") == os.path.join(str(image_dir), "a.sif")
    with pytest.raises(images.ImageError):
        images.image_path("../../etc/shadow.sif")


# ── Listing and deleting ─────────────────────────────────────────────────────

def _cluster_pointing_at(path) -> ClusterConfig:
    """A copy of CLUSTER whose apptainer runtime uses `path` (Runtime is frozen)."""
    runtimes = dict(CLUSTER.runtimes)
    runtimes["arm-vllm"] = Runtime(name="arm-vllm", kind="apptainer", image=str(path))
    return ClusterConfig(
        runtimes=runtimes, gpu_classes=CLUSTER.gpu_classes, pools=CLUSTER.pools,
    )


def test_listing_marks_images_the_config_depends_on(image_dir):
    (image_dir / "vllm-aarch64.sif").write_bytes(b"x" * 10)
    (image_dir / "spare.sif").write_bytes(b"y" * 20)
    (image_dir / "notes.txt").write_text("ignored")

    cluster = _cluster_pointing_at(image_dir / "vllm-aarch64.sif")
    found = {i.name: i for i in images.list_images(cluster)}

    assert set(found) == {"vllm-aarch64.sif", "spare.sif"}   # .txt is not an image
    assert found["vllm-aarch64.sif"].used_by_runtimes == ("arm-vllm",)
    assert found["vllm-aarch64.sif"].used_by_gpu_classes == ("gb10",)
    assert found["spare.sif"].used_by_runtimes == ()


def test_a_missing_images_directory_is_an_error_not_an_empty_list(monkeypatch, tmp_path):
    monkeypatch.setattr(images.settings, "apptainer_image_dir", str(tmp_path / "nope"))
    with pytest.raises(images.ImageError, match="not readable"):
        images.list_images(CLUSTER)


def test_deleting_a_referenced_image_is_refused(image_dir):
    path = image_dir / "vllm-aarch64.sif"
    path.write_bytes(b"x")
    cluster = _cluster_pointing_at(path)

    with pytest.raises(images.ImageError, match="cluster.yaml"):
        images.delete_image(cluster, "vllm-aarch64.sif")
    assert path.exists()

    images.delete_image(cluster, "vllm-aarch64.sif", force=True)
    assert not path.exists()


def test_deleting_an_unreferenced_image_works(image_dir):
    path = image_dir / "spare.sif"
    path.write_bytes(b"x")
    images.delete_image(CLUSTER, "spare.sif")
    assert not path.exists()


def test_deleting_something_that_is_not_there_says_so(image_dir):
    with pytest.raises(images.ImageError, match="does not exist"):
        images.delete_image(CLUSTER, "ghost.sif")


# ── Where a build can run ────────────────────────────────────────────────────

def test_build_targets_come_from_the_gpu_class_architecture():
    targets = {(t.arch, t.partition): t for t in images.build_targets(CLUSTER, NODES)}
    assert set(targets) == {("aarch64", "spark"), ("x86_64", "llm")}
    assert targets[("aarch64", "spark")].nodes == ("spark-049c", "spark-c127")


def test_a_uniform_partition_is_not_pinned_to_one_node():
    """Slurm should be free to place the build wherever it fits."""
    target = images.resolve_target(CLUSTER, NODES, "aarch64", None)
    assert target.pin_node is None


def test_a_mixed_architecture_partition_pins_the_node():
    """Otherwise the build could land on the wrong arch and silently produce
    an image that fails on a different machine days later."""
    mixed = [
        _node("x86-box", "gpu48", "shared"),
        _node("arm-box", "gb10", "shared"),
    ]
    cluster = ClusterConfig(
        runtimes=CLUSTER.runtimes,
        gpu_classes=CLUSTER.gpu_classes,
        pools={"shared": Pool(name="shared", partition="shared")},
    )
    target = images.resolve_target(cluster, mixed, "aarch64", None)
    assert target.pin_node == "arm-box"


def test_drained_nodes_are_not_build_targets():
    drained = [_node("spark-049c", "gb10", "spark", state="DRAIN")]
    assert images.build_targets(CLUSTER, drained) == []


def test_an_architecture_nothing_can_build_is_refused():
    x86_only = [_node("vega", "gpu48", "llm")]
    with pytest.raises(images.ImageError, match="cross-build"):
        images.resolve_target(CLUSTER, x86_only, "aarch64", None)


def test_asking_for_the_wrong_partition_lists_the_right_ones():
    with pytest.raises(images.ImageError, match="spark"):
        images.resolve_target(CLUSTER, NODES, "aarch64", "llm")


# ── The submitted job ────────────────────────────────────────────────────────

def test_build_job_requests_no_gpu(image_dir):
    """A build needs CPU and disk. Asking for a card would queue it behind real
    work and then hold that card idle while it unpacks layers."""
    target = images.resolve_target(CLUSTER, NODES, "aarch64", None)
    spec = images.build_job_spec(
        name="vllm-new.sif",
        source_ref="docker://vllm/vllm-openai:v0.11.0",
        target=target,
    )
    assert spec.gpus == 0
    assert spec.gres is None
    assert spec.partition == "spark"


def test_build_job_carries_the_target_and_scratch_paths(image_dir, monkeypatch):
    monkeypatch.setattr(images.settings, "image_build_scratch", "/scratch/build")
    target = images.resolve_target(CLUSTER, NODES, "x86_64", None)
    spec = images.build_job_spec(
        name="vllm-new.sif",
        source_ref="docker://vllm/vllm-openai:v0.11.0",
        target=target,
        requested_by="alice",
    )
    assert spec.env["BUILD_TARGET"] == str(image_dir / "vllm-new.sif")
    assert spec.env["BUILD_EXPECTED_ARCH"] == "x86_64"
    # Apptainer defaults these under $HOME, which the service account does not
    # have on every node — the failure appears minutes into a build.
    assert spec.env["APPTAINER_CACHEDIR"].startswith("/scratch/build")
    assert spec.env["APPTAINER_TMPDIR"].startswith("/scratch/build")
    assert "user:alice" in spec.comment


def test_cpu_only_jobs_send_no_gres_flag():
    """`--gres=gpu:0` would be rewritten by job_submit.lua into a hard gpu24
    constraint, queueing a CPU-only build behind one specific card."""
    from app.backends.slurm_cli import build_sbatch_argv
    from app.backends.types import JobSpec

    argv = build_sbatch_argv(JobSpec(
        job_name="sif-build", script_path="/tmp/x.sh", gpus=0,
        time_limit="02:00:00", log_dir="/tmp/logs",
    ))
    assert not any(a.startswith("--gres") for a in argv)
