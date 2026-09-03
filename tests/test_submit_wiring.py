"""What actually reaches sbatch.

The headline case is the gpu24 pinning bug: this cluster's job_submit.lua
appends `gpu24` as a hard constraint to any *untyped* `--gres=gpu:N`, so a
model that needs a big card silently never reaches one. Typed GRES is the fix,
and these tests pin it down.
"""

from __future__ import annotations

from datetime import datetime, timezone

import pytest

from app.backends.slurm_cli import build_sbatch_argv
from app.backends.types import GpuGroup, JobSpec, NodeInfo
from app.catalog import CatalogModel, ModelRequirements
from app.cluster import ClusterConfig, GpuClass, Pool, Runtime
from app.inventory import Inventory, choose_gpu_class

NOW = datetime(2026, 8, 20, 12, 0, tzinfo=timezone.utc)

CLUSTER = ClusterConfig(
    runtimes={
        "x86-cuda": Runtime(name="x86-cuda", kind="apptainer",
                            image="/sif/vllm.sif", binds=("/models",)),
        "legacy": Runtime(name="legacy", kind="venv", activate="/opt/venv/bin/activate"),
        "arm": Runtime(name="arm", kind="apptainer", image="/sif/arm.sif"),
    },
    gpu_classes={
        "gpu24": GpuClass(name="gpu24", vram_gb=24, runtime="legacy"),
        "gpu48": GpuClass(name="gpu48", vram_gb=48, runtime="legacy"),
        "gpu80": GpuClass(name="gpu80", vram_gb=80, runtime="x86-cuda"),
        "gpu96": GpuClass(name="gpu96", vram_gb=96, runtime="x86-cuda"),
        "gb10": GpuClass(name="gb10", vram_gb=128, arch="aarch64", runtime="arm",
                         unified_memory=True, gpu_memory_utilization_max=0.70),
    },
    pools={
        "llm-dedicated": Pool(name="llm-dedicated", partition="gpu",
                              scheduling="managed", nodes=("jupiter",),
                              gpu_classes=("gpu96",)),
        "general": Pool(name="general", partition="gpu", scheduling="slurm"),
        "dgx": Pool(name="dgx", partition="dgx", scheduling="slurm",
                    gpu_classes=("gpu80",)),
    },
)


def _node(name, groups, partitions=("gpu",)):
    return NodeInfo(
        name=name,
        gpus=tuple(GpuGroup(c, n) for c, n in groups),
        partitions=partitions,
        state="idle",
    )


NODES = [
    _node("titan", [("gpu48", 2)]),
    _node("europa", [("gpu24", 1), ("gpu48", 1)]),
    _node("jupiter", [("gpu96", 4)]),
    _node("ganymede", [("gpu24", 8)]),
    _node("dgx", [("gpu80", 4)], partitions=("dgx",)),
]

INV = Inventory(
    nodes=tuple(NODES),
    fetched_at=NOW,
    node_pools={n.name: (CLUSTER.pool_for_node(n.name, n.partitions).name
                         if CLUSTER.pool_for_node(n.name, n.partitions) else "")
                for n in NODES},
)


def _model(name="m", gpus=1, tp=1, **kw):
    return CatalogModel(name=name, model_path="p", gpus=gpus,
                        tensor_parallel_size=tp, **kw)


# ── The gpu24 bug ────────────────────────────────────────────────────────────

def test_typed_gres_is_emitted_so_the_plugin_cannot_pin_us_to_gpu24():
    argv = build_sbatch_argv(JobSpec(
        job_name="vllm-x", script_path="/t.sh", gpus=4,
        time_limit="02:00:00", gres="gpu:gpu96:4",
    ))
    assert "--gres=gpu:gpu96:4" in argv
    assert "--gres=gpu:4" not in argv


def test_untyped_gres_is_what_the_bug_looks_like():
    """Documents the failure mode: this is what the app used to send, and it
    is what job_submit.lua rewrites to gpu24."""
    argv = build_sbatch_argv(JobSpec(
        job_name="vllm-x", script_path="/t.sh", gpus=4, time_limit="02:00:00",
    ))
    assert "--gres=gpu:4" in argv


# ── Class selection ──────────────────────────────────────────────────────────

def test_smallest_sufficient_class_wins():
    """A model that fits on 24 GB should not occupy a 96 GB card."""
    model = _model(gpus=1, requires=ModelRequirements(min_vram_gb=0))
    assert choose_gpu_class(model, CLUSTER, INV) == "gpu24"


def test_vram_requirement_excludes_small_classes():
    model = _model(gpus=1, requires=ModelRequirements(min_vram_gb=80))
    assert choose_gpu_class(model, CLUSTER, INV) == "gpu80"


def test_class_needing_more_gpus_than_any_node_has_is_rejected():
    """europa has 1×gpu48 and titan 2 — a 4-GPU gpu48 model fits nowhere."""
    model = _model(gpus=4, requires=ModelRequirements(gpu_classes=("gpu48",)))
    assert choose_gpu_class(model, CLUSTER, INV) is None


def test_pool_restriction_narrows_the_choice():
    model = _model(gpus=2, requires=ModelRequirements(min_vram_gb=0))
    chosen = choose_gpu_class(model, CLUSTER, INV, pool=CLUSTER.pools["llm-dedicated"])
    assert chosen == "gpu96"


def test_preferred_class_is_honoured_when_usable():
    model = _model(gpus=1, requires=ModelRequirements(min_vram_gb=0))
    assert choose_gpu_class(model, CLUSTER, INV, preferred="gpu96") == "gpu96"


def test_preferred_class_that_cannot_work_is_refused_not_silently_swapped():
    """Silently substituting a class would produce a booking the user did not
    ask for; refusing lets the API explain why."""
    model = _model(gpus=1, requires=ModelRequirements(min_vram_gb=80))
    assert choose_gpu_class(model, CLUSTER, INV, preferred="gpu24") is None


def test_no_inventory_means_no_class():
    model = _model(gpus=1)
    assert choose_gpu_class(model, CLUSTER, Inventory()) is None


def test_drained_node_capacity_does_not_count():
    drained = NodeInfo(name="jupiter", gpus=(GpuGroup("gpu96", 4),),
                       partitions=("gpu",), state="drain")
    inv = Inventory(nodes=(drained,), fetched_at=NOW)
    model = _model(gpus=4, requires=ModelRequirements(gpu_classes=("gpu96",)))
    assert choose_gpu_class(model, CLUSTER, inv) is None


# ── Runtime follows the class ────────────────────────────────────────────────

def test_runtime_is_derived_from_the_chosen_class():
    assert CLUSTER.runtime_for("gpu96").kind == "apptainer"
    assert CLUSTER.runtime_for("gpu24").kind == "venv"


def test_aarch64_class_selects_the_aarch64_image():
    """The whole reason runtime selection is class-driven."""
    assert CLUSTER.runtime_for("gb10").image == "/sif/arm.sif"


# ── Utilization capping ──────────────────────────────────────────────────────

def test_unified_memory_class_caps_a_greedy_request():
    assert CLUSTER.cap_utilization("gb10", 0.95) == 0.70


def test_normal_class_does_not_cap():
    assert CLUSTER.cap_utilization("gpu96", 0.95) == 0.95


# ── Managed vs slurm pool submission ─────────────────────────────────────────

@pytest.mark.parametrize("pool_name,should_pin", [
    ("llm-dedicated", True),   # our calendar allocates -> pin the node
    ("general", False),        # Slurm allocates -> pinning would fight backfill
])
def test_node_pinning_follows_the_scheduling_mode(pool_name, should_pin):
    pool = CLUSTER.pools[pool_name]
    nodelist = "jupiter" if (pool.is_managed) else None
    argv = build_sbatch_argv(JobSpec(
        job_name="vllm-x", script_path="/t.sh", gpus=2, time_limit="01:00:00",
        gres="gpu:gpu96:2", partition=pool.partition, nodelist=nodelist,
    ))
    assert ("--nodelist=jupiter" in argv) is should_pin
    assert f"--partition={pool.partition}" in argv


def test_attribution_comment_survives_the_service_account():
    argv = build_sbatch_argv(JobSpec(
        job_name="vllm-x", script_path="/t.sh", gpus=1, time_limit="01:00:00",
        comment="user:alice,lease:412",
    ))
    assert "--comment=user:alice,lease:412" in argv


def test_dgx_pool_targets_its_own_partition():
    """job_submit.lua rejects gpu80 outside partition 'dgx'."""
    pool = CLUSTER.pools["dgx"]
    argv = build_sbatch_argv(JobSpec(
        job_name="vllm-x", script_path="/t.sh", gpus=2, time_limit="01:00:00",
        gres="gpu:gpu80:2", partition=pool.partition,
    ))
    assert "--partition=dgx" in argv
    assert "--gres=gpu:gpu80:2" in argv
