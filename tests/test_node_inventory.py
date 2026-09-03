"""Node discovery against this cluster's real topology.

The layout is taken from the site's `job_submit.lua`, which is the authority on
what GPU classes exist and which nodes hold them. The case that drives the
design is `europa`: one node holding *two* GPU classes.
"""

from __future__ import annotations

import pytest

from app.backends.slurm_cli import SlurmCliBackend, _Cmd
from app.backends.slurm_parse import parse_gres, parse_gres_map
from app.backends.types import GpuGroup, NodeInfo


# ── GRES maps ────────────────────────────────────────────────────────────────

def test_gres_map_splits_mixed_classes_on_one_node():
    """europa = {gpu24: 1, gpu48: 1}. Collapsing this to a single type would
    let the planner place a gpu48 model on a 24 GB card."""
    assert parse_gres_map("gpu:gpu24:1,gpu:gpu48:1") == {"gpu24": 1, "gpu48": 1}


@pytest.mark.parametrize(
    "spec,expected",
    [
        ("gpu:gpu48:2", {"gpu48": 2}),            # titan
        ("gpu:gpu96:4", {"gpu96": 4}),            # jupiter
        ("gpu:gpu24:8", {"gpu24": 8}),            # ganymede (MIG slices)
        ("gpu:gpu80:4", {"gpu80": 4}),            # dgx
        ("gpu:gpu48:1(S:0-1)", {"gpu48": 1}),     # topology annotation stripped
        ("gpu:4", {None: 4}),                     # untyped
        ("(null)", {}),
        ("nic:1", {}),
    ],
)
def test_gres_map_real_node_configs(spec, expected):
    assert parse_gres_map(spec) == expected


def test_gres_map_sums_repeated_classes():
    assert parse_gres_map("gpu:gpu24:2,gpu:gpu24:3") == {"gpu24": 5}


def test_parse_gres_total_still_works_for_foreign_jobs():
    """Foreign-job occupancy only needs the total count."""
    assert parse_gres("gpu:gpu24:1,gpu:gpu48:1") == ("gpu24", 2)


# ── NodeInfo ─────────────────────────────────────────────────────────────────

EUROPA = NodeInfo(
    name="europa",
    gpus=(GpuGroup("gpu24", 1), GpuGroup("gpu48", 1)),
    features=("gpu24", "gpu48", "rtx_a5000", "l40"),
    partitions=("gpu",),
    state="MIXED",
)


def test_node_gpu_count_sums_all_classes():
    assert EUROPA.gpu_count == 2


def test_node_count_of_class():
    assert EUROPA.count_of("gpu24") == 1
    assert EUROPA.count_of("gpu48") == 1


def test_node_count_of_absent_class_is_zero():
    """0 means "cannot host" — the planner's filter depends on this."""
    assert EUROPA.count_of("gpu96") == 0
    assert EUROPA.count_of("gpu80") == 0


def test_node_class_map():
    assert EUROPA.gpu_classes == {"gpu24": 1, "gpu48": 1}


# ── sinfo parsing ────────────────────────────────────────────────────────────

# One line per (node, partition), as `sinfo -N` emits.
SINFO_OUTPUT = "\n".join([
    "titan|gpu*|64|515000|idle|gpu:gpu48:2|gpu48,rtx_6000_ada",
    "europa|gpu*|32|257000|mixed|gpu:gpu24:1,gpu:gpu48:1|gpu24,gpu48,rtx_a5000,l40",
    "jupiter|gpu*|96|1030000|allocated|gpu:gpu96:4|gpu96,rtx_pro_6000",
    "ganymede|gpu*|64|515000|idle|gpu:gpu24:8|gpu24,rtx_pro_6000,1g",
    "alien0|gpu*|16|64000|drain|gpu:gpu24:1|gpu24,rtx_3090",
    "dgx|dgx|128|2060000|idle|gpu:gpu80:4|gpu80,a100",
])


@pytest.fixture
def inventory(monkeypatch) -> dict[str, NodeInfo]:
    import app.backends.slurm_cli as mod

    monkeypatch.setattr(mod, "_run_soft", lambda *a, **k: _Cmd(0, SINFO_OUTPUT, ""))
    nodes = SlurmCliBackend(probe=False).nodes_sync()
    return {n.name: n for n in nodes}


def test_inventory_discovers_all_nodes(inventory):
    assert set(inventory) == {"titan", "europa", "jupiter", "ganymede", "alien0", "dgx"}


def test_inventory_preserves_mixed_classes(inventory):
    assert inventory["europa"].gpu_classes == {"gpu24": 1, "gpu48": 1}


def test_inventory_reads_features_as_gpu_models(inventory):
    """Classes come from Gres, models from Features — the planner needs both."""
    assert "rtx_6000_ada" in inventory["titan"].features
    assert "a100" in inventory["dgx"].features


def test_inventory_strips_default_partition_marker(inventory):
    """`gpu*` marks the default partition; the '*' is not part of the name."""
    assert inventory["titan"].partitions == ("gpu",)
    assert inventory["dgx"].partitions == ("dgx",)


def test_inventory_excludes_drained_nodes_from_placement(inventory):
    assert inventory["alien0"].is_usable is False
    assert inventory["titan"].is_usable is True


def test_inventory_total_gpus_is_derived_not_configured(inventory):
    """TOTAL_GPUS was a hand-maintained env var; it is now a sum over reality."""
    assert sum(n.gpu_count for n in inventory.values()) == 2 + 2 + 4 + 8 + 1 + 4


def test_inventory_gpu96_only_exists_on_jupiter(inventory):
    """Sanity-check the class filter the planner will rely on."""
    hosts = [n.name for n in inventory.values() if n.count_of("gpu96") > 0]
    assert hosts == ["jupiter"]


def test_inventory_gpu80_only_exists_in_the_dgx_partition(inventory):
    """job_submit.lua rejects gpu80 outside partition 'dgx'; our view agrees."""
    hosts = [n for n in inventory.values() if n.count_of("gpu80") > 0]
    assert [n.name for n in hosts] == ["dgx"]
    assert all("dgx" in n.partitions for n in hosts)
