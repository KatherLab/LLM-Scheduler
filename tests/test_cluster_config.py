"""cluster.yaml loading, validation, runtime selection and quotas."""

from __future__ import annotations

import textwrap

import pytest

from app.catalog import CatalogModel, ModelRequirements, load_catalog, resolve_variant
from app.cluster import (
    DEFAULT_GUARD_GAP_SECONDS,
    ClusterConfigError,
    GpuClass,
    Quota,
    load_cluster,
)

CLUSTER_YAML = textwrap.dedent("""
    runtimes:
      x86-cuda:
        kind: apptainer
        image: /shared/sif/vllm-0.11.0-cu128.sif
        nv: true
        binds: ["/models:/models:ro", "/scratch"]
      arm-spark:
        kind: apptainer
        image: /shared/sif/vllm-arm64.sif
      legacy-venv:
        kind: venv
        activate: /opt/venv/bin/activate

    gpu_classes:
      gpu24:
        vram_gb: 24
        runtime: legacy-venv
      gpu96:
        vram_gb: 96
        runtime: x86-cuda
        guard_gap_seconds: 180
      gb10:
        vram_gb: 128
        arch: aarch64
        runtime: arm-spark
        unified_memory: true
        gpu_memory_utilization_max: 0.70

    pools:
      - name: llm-dedicated
        partition: gpu
        scheduling: managed
        operators: [llm-gpu-team]
        nodes: [jupiter]
        gpu_classes: [gpu96]
      - name: general
        partition: gpu
        scheduling: slurm

    quotas:
      default:
        max_concurrent_gpus: 4
        max_booking_horizon_days: 14
      groups:
        llm-power-users:
          max_concurrent_gpus: 8
      per_pool:
        general:
          max_booking_duration_hours: 168
""")


@pytest.fixture
def cluster(tmp_path):
    path = tmp_path / "cluster.yaml"
    path.write_text(CLUSTER_YAML)
    return load_cluster(str(path))


def _write(tmp_path, body):
    path = tmp_path / "cluster.yaml"
    path.write_text(textwrap.dedent(body))
    return str(path)


# ── Parsing ──────────────────────────────────────────────────────────────────

def test_loads_classes_runtimes_and_pools(cluster):
    assert set(cluster.gpu_classes) == {"gpu24", "gpu96", "gb10"}
    assert set(cluster.runtimes) == {"x86-cuda", "arm-spark", "legacy-venv"}
    assert set(cluster.pools) == {"llm-dedicated", "general"}


def test_scheduling_mode_distinguishes_promise_from_estimate(cluster):
    assert cluster.pools["llm-dedicated"].is_managed
    assert not cluster.pools["general"].is_managed


def test_guard_gap_defaults_when_unset(cluster):
    assert cluster.guard_gap_seconds("gpu24") == DEFAULT_GUARD_GAP_SECONDS
    assert cluster.guard_gap_seconds("gpu96") == 180
    assert cluster.guard_gap_seconds("unknown") == DEFAULT_GUARD_GAP_SECONDS


# ── Validation ───────────────────────────────────────────────────────────────

def test_class_referencing_an_undefined_runtime_is_rejected(tmp_path):
    path = _write(tmp_path, """
        gpu_classes:
          gpu24: {vram_gb: 24, runtime: nope}
    """)
    with pytest.raises(ClusterConfigError, match="not defined under runtimes"):
        load_cluster(path)


def test_apptainer_runtime_without_an_image_is_rejected(tmp_path):
    path = _write(tmp_path, "runtimes:\n  r: {kind: apptainer}\n")
    with pytest.raises(ClusterConfigError, match="needs an 'image'"):
        load_cluster(path)


def test_venv_runtime_without_activate_is_rejected(tmp_path):
    path = _write(tmp_path, "runtimes:\n  r: {kind: venv}\n")
    with pytest.raises(ClusterConfigError, match="needs an 'activate'"):
        load_cluster(path)


def test_unknown_scheduling_mode_is_rejected(tmp_path):
    path = _write(tmp_path, """
        pools:
          - {name: p, partition: gpu, scheduling: magic}
    """)
    with pytest.raises(ClusterConfigError, match="scheduling must be one of"):
        load_cluster(path)


def test_pool_referencing_an_unknown_gpu_class_is_rejected(tmp_path):
    path = _write(tmp_path, """
        pools:
          - {name: p, partition: gpu, gpu_classes: [gpu999]}
    """)
    with pytest.raises(ClusterConfigError, match="unknown class"):
        load_cluster(path)


def test_out_of_range_utilization_cap_is_rejected(tmp_path):
    path = _write(tmp_path, "gpu_classes:\n  c: {vram_gb: 1, gpu_memory_utilization_max: 1.5}\n")
    with pytest.raises(ClusterConfigError, match="must be in"):
        load_cluster(path)


def test_empty_file_is_valid(tmp_path):
    """A missing/empty cluster.yaml must not break single-node deployments."""
    assert load_cluster(_write(tmp_path, "")).pools == {}


# ── Utilization capping ──────────────────────────────────────────────────────

def test_unified_memory_class_caps_utilization(cluster):
    """On DGX Spark the fraction is of memory shared with the OS, so an
    uncapped 0.95 would destabilise the node."""
    assert cluster.cap_utilization("gb10", 0.95) == 0.70


def test_cap_leaves_a_lower_request_alone(cluster):
    assert cluster.cap_utilization("gb10", 0.60) == 0.60


def test_uncapped_class_passes_the_request_through(cluster):
    assert cluster.cap_utilization("gpu96", 0.95) == 0.95


def test_unknown_class_passes_through(cluster):
    assert cluster.cap_utilization("nope", 0.95) == 0.95


# ── Runtime selection ────────────────────────────────────────────────────────

def test_runtime_follows_the_gpu_class(cluster):
    """aarch64 nodes get an aarch64 image without the model naming one."""
    assert cluster.runtime_for("gb10").image == "/shared/sif/vllm-arm64.sif"
    assert cluster.runtime_for("gpu96").name == "x86-cuda"


def test_model_runtime_override_wins(cluster):
    assert cluster.runtime_for("gpu96", override="legacy-venv").kind == "venv"


def test_apptainer_runtime_exports_job_env(cluster):
    """APPTAINER_IMAGE is resolved separately (images.resolve_runtime_image),
    since `image` is now a filename pattern, not a fixed path."""
    env = cluster.runtime_for("gpu96").as_job_env()
    assert env["RUNTIME_KIND"] == "apptainer"
    assert "APPTAINER_IMAGE" not in env
    assert env["APPTAINER_NV"] == "1"
    assert "/scratch" in env["APPTAINER_BINDS"]


def test_venv_runtime_still_exports_the_old_variable(cluster):
    """Keeps existing venv deployments working — there is no flag day."""
    env = cluster.runtime_for("gpu24").as_job_env()
    assert env["RUNTIME_KIND"] == "venv"
    assert env["VENV_ACTIVATE"] == "/opt/venv/bin/activate"


# ── Pools ────────────────────────────────────────────────────────────────────

def test_explicit_node_list_wins_over_a_partition_wide_pool(cluster):
    """Dedicated machines can be carved out of a shared partition."""
    assert cluster.pool_for_node("jupiter", ("gpu",)).name == "llm-dedicated"
    assert cluster.pool_for_node("titan", ("gpu",)).name == "general"


def test_node_in_no_configured_partition_has_no_pool(cluster):
    assert cluster.pool_for_node("mystery", ("other",)) is None


def test_pool_class_restriction(cluster):
    dedicated = cluster.pools["llm-dedicated"]
    assert dedicated.accepts_class("gpu96")
    assert not dedicated.accepts_class("gpu24")
    # No restriction configured means everything is allowed.
    assert cluster.pools["general"].accepts_class("gpu24")


# ── Quotas ───────────────────────────────────────────────────────────────────

def test_default_quota_applies(cluster):
    quota = cluster.quota_for(frozenset())
    assert quota.max_concurrent_gpus == 4
    assert quota.max_booking_horizon_days == 14


def test_group_override_raises_the_limit(cluster):
    quota = cluster.quota_for(frozenset({"llm-power-users"}))
    assert quota.max_concurrent_gpus == 8
    # Unset axes fall through to the default.
    assert quota.max_booking_horizon_days == 14


def test_pool_override_applies_on_top(cluster):
    quota = cluster.quota_for(frozenset(), pool="general")
    assert quota.max_booking_duration_hours == 168
    assert quota.max_concurrent_gpus == 4


def test_admins_are_exempt(cluster):
    assert cluster.quota_for(frozenset(), is_admin=True) == Quota()


def test_quota_merge_keeps_unset_axes():
    base = Quota(max_concurrent_gpus=4, max_booking_horizon_days=14)
    merged = base.merged_with(Quota(max_concurrent_gpus=8))
    assert merged.max_concurrent_gpus == 8
    assert merged.max_booking_horizon_days == 14


# ── Catalog variants ─────────────────────────────────────────────────────────

CATALOG_YAML = textwrap.dedent("""
    defaults:
      extra_args: "--enable-prompt-tokens-details"

    models:
      - name: Qwen3-235B
        model_path: Qwen/Qwen3-235B
        gpus: 4
        tensor_parallel_size: 4
        gpu_memory_utilization: 0.95
        requires:
          min_vram_gb: 48
        variants:
          gpu96:
            gpus: 2
            tp: 2
          gpu24:
            gpus: 8
            tp: 8
            gpu_memory_utilization: 0.90
            extra_args: "--max-model-len 32768"
""")


@pytest.fixture
def model(tmp_path):
    path = tmp_path / "models.yaml"
    path.write_text(CATALOG_YAML)
    return load_catalog(str(path))["Qwen3-235B"]


def test_variant_changes_gpus_and_tp(model):
    """The core of heterogeneity: the same weights need different topologies."""
    on96 = resolve_variant(model, "gpu96")
    assert (on96.gpus, on96.tensor_parallel_size) == (2, 2)

    on24 = resolve_variant(model, "gpu24")
    assert (on24.gpus, on24.tensor_parallel_size) == (8, 8)


def test_variant_overrides_utilization(model):
    assert resolve_variant(model, "gpu24").gpu_memory_utilization == 0.90
    assert resolve_variant(model, "gpu96").gpu_memory_utilization == 0.95


def test_variant_args_merge_with_defaults_not_duplicate(model):
    args = resolve_variant(model, "gpu24").extra_args
    assert "--enable-prompt-tokens-details" in args
    assert args.count("--max-model-len") == 1


def test_no_variant_leaves_the_model_untouched(model):
    assert resolve_variant(model, "gpu48") is model
    assert resolve_variant(model, None) is model


def test_requires_admits_a_large_enough_card(model):
    assert model.supports_class(GpuClass(name="gpu96", vram_gb=96))


def test_requires_rejects_too_small_a_card():
    m = CatalogModel(name="m", model_path="p", gpus=1, tensor_parallel_size=1,
                     requires=ModelRequirements(min_vram_gb=80))
    assert not m.supports_class(GpuClass(name="gpu24", vram_gb=24))
    assert m.supports_class(GpuClass(name="gpu80", vram_gb=80))


def test_an_explicit_variant_is_a_statement_of_support(model):
    """gpu24 is below min_vram_gb, but a variant says how to run there anyway."""
    assert model.requires.min_vram_gb == 48
    assert model.supports_class(GpuClass(name="gpu24", vram_gb=24))


def test_requires_can_filter_by_arch():
    m = CatalogModel(name="m", model_path="p", gpus=1, tensor_parallel_size=1,
                     requires=ModelRequirements(arch="aarch64"))
    assert m.supports_class(GpuClass(name="gb10", vram_gb=128, arch="aarch64"))
    assert not m.supports_class(GpuClass(name="gpu96", vram_gb=96, arch="x86_64"))


# ── Aggregate memory requirements (8x H200 case) ─────────────────────────────

H200 = GpuClass(name="gpu141", vram_gb=141, reserved_gb=3)
GPU96 = GpuClass(name="gpu96", vram_gb=96)


def _big(**kw):
    """A model that only fits across a full H200 node."""
    return CatalogModel(
        name="big", model_path="p", gpus=8, tensor_parallel_size=8,
        requires=ModelRequirements(min_vram_gb=140, min_total_vram_gb=1000, **kw),
    )


def test_per_gpu_floor_alone_cannot_express_needing_eight_of_them():
    """min_vram_gb is satisfied by a single H200 — which is the whole problem."""
    reqs = ModelRequirements(min_vram_gb=140)
    assert reqs.allows(H200)
    assert reqs.fits_on(H200, gpus=1)


def test_aggregate_floor_rejects_too_few_gpus():
    model = _big()
    assert not model.requires.fits_on(H200, gpus=2)
    assert not model.requires.fits_on(H200, gpus=4)


def test_aggregate_floor_accepts_a_full_node():
    assert _big().requires.fits_on(H200, gpus=8)


def test_aggregate_floor_uses_usable_not_raw_vram():
    """Reserved memory is not available to the model, so it must not count."""
    tight = ModelRequirements(min_total_vram_gb=141 * 8)
    assert not tight.fits_on(H200, gpus=8)      # 8 x 138 usable < 1128


def test_a_smaller_class_is_rejected_even_with_enough_cards():
    """Per-GPU floor still applies: 12x gpu96 has the total but not the shard."""
    model = _big()
    assert not model.requires.fits_on(GPU96, gpus=12)


def test_shortfall_names_the_gap():
    msg = _big().requires.shortfall(H200, gpus=4)
    assert "1000 GB" in msg and "4x gpu141" in msg


def test_shortfall_is_none_when_it_fits():
    assert _big().requires.shortfall(H200, gpus=8) is None


def test_shortfall_reports_the_per_gpu_gap_when_that_is_the_binding_one():
    reqs = ModelRequirements(min_vram_gb=140)
    assert "per GPU" in reqs.shortfall(GPU96, gpus=8)


def test_models_without_an_aggregate_requirement_are_unaffected():
    plain = ModelRequirements(min_vram_gb=24)
    assert plain.fits_on(GPU96, gpus=1)


def test_aggregate_requirement_parses_from_yaml(tmp_path):
    path = tmp_path / "models.yaml"
    path.write_text(textwrap.dedent("""
        models:
          - name: huge
            model_path: org/huge
            gpus: 8
            tensor_parallel_size: 8
            requires:
              min_vram_gb: 140
              min_total_vram_gb: 1000
    """))
    reqs = load_catalog(str(path))["huge"].requires
    assert reqs.min_vram_gb == 140
    assert reqs.min_total_vram_gb == 1000


def test_reserved_gb_is_parsed_from_yaml(tmp_path):
    """It was silently dropped once: the field existed on the dataclass but not
    in the loader, so a configured reservation had no effect and `usable_gb`
    quietly reported the whole card."""
    path = _write(tmp_path, """
        gpu_classes:
          gb10: {vram_gb: 119, reserved_gb: 24, unified_memory: true}
    """)
    cls = load_cluster(path).gpu_classes["gb10"]
    assert cls.reserved_gb == 24
    assert cls.usable_gb == 95
    assert cls.default_utilization == pytest.approx((119 - 24) / 119)


def test_reserved_gb_defaults_to_zero(tmp_path):
    path = _write(tmp_path, "gpu_classes:\n  gpu48: {vram_gb: 48}\n")
    assert load_cluster(path).gpu_classes["gpu48"].reserved_gb == 0
