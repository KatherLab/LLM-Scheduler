"""Several models on one GPU.

The cluster gives us no way to subdivide a GPU (MPS disabled, MIG static and
absent on GB10, gres/shard needs a slurm.conf change), so co-location happens
inside one Slurm job. The safety property is that a group is proven to fit
*before* submission — a half-started group is worse than a refused booking,
because the models that did start look healthy.
"""

from __future__ import annotations

import json

import pytest

from app import colocation
from app.catalog import CatalogModel, ModelRequirements
from app.cluster import GpuClass

GPU48 = GpuClass(name="gpu48", vram_gb=48)
GPU24 = GpuClass(name="gpu24", vram_gb=24)
# DGX Spark: RAM is VRAM, so most of it belongs to the host.
GB10 = GpuClass(name="gb10", vram_gb=128, arch="aarch64",
                unified_memory=True, reserved_gb=40, gpu_memory_utilization_max=0.70)


def _model(name, memory_gb=None, gpus=1, tp=1, **kw):
    return CatalogModel(name=name, model_path=f"org/{name}", gpus=gpus,
                        tensor_parallel_size=tp, memory_gb=memory_gb, **kw)


EMBED = _model("bge-m3", memory_gb=12)
RERANK = _model("bge-reranker-v2", memory_gb=8)
BIG = _model("qwen-32b", memory_gb=40)


# ── Fitting ──────────────────────────────────────────────────────────────────

def test_two_small_models_fit_on_a_48gb_card():
    tenants = colocation.resolve_group([EMBED, RERANK], GPU48)
    assert [t.model for t in tenants] == ["bge-m3", "bge-reranker-v2"]


def test_absolute_budgets_are_converted_to_vllm_fractions():
    """vLLM takes a fraction, but budgets are what compose."""
    tenants = colocation.resolve_group([EMBED, RERANK], GPU48)
    by_name = {t.model: t for t in tenants}
    assert by_name["bge-m3"].gpu_memory_utilization == pytest.approx(12 / 48)
    assert by_name["bge-reranker-v2"].gpu_memory_utilization == pytest.approx(8 / 48, abs=1e-4)


def test_fractions_are_floored_so_they_can_only_undershoot():
    """Rounding up would let the fractions sum past the budget we just proved
    fits."""
    tenants = colocation.resolve_group([EMBED, RERANK], GPU48)
    for t in tenants:
        assert t.gpu_memory_utilization <= t.memory_gb / GPU48.vram_gb


def test_fractions_sum_to_less_than_one():
    tenants = colocation.resolve_group([EMBED, RERANK], GPU48)
    assert sum(t.gpu_memory_utilization for t in tenants) < 1.0


def test_the_same_budget_yields_different_fractions_per_class():
    """Which is exactly why the config is written in GB, not fractions."""
    on48 = colocation.resolve_group([EMBED, RERANK], GPU48)[0]
    on24 = colocation.resolve_group([EMBED, RERANK], GPU24)[0]
    assert on48.gpu_memory_utilization != on24.gpu_memory_utilization


def test_a_group_that_does_not_fit_is_refused():
    with pytest.raises(colocation.ColocationError, match="only"):
        colocation.resolve_group([BIG, EMBED], GPU24)


def test_the_refusal_says_what_to_do_about_it():
    with pytest.raises(colocation.ColocationError) as exc:
        colocation.resolve_group([BIG, EMBED], GPU24)
    message = str(exc.value)
    assert "Drop one" in message and "memory_gb" in message


def test_headroom_is_reserved_on_a_shared_card():
    """Co-location is tighter than one model; a little slack avoids an OOM at
    the last replica's startup."""
    exact = _model("a", memory_gb=24)
    other = _model("b", memory_gb=24)
    with pytest.raises(colocation.ColocationError):
        colocation.resolve_group([exact, other], GPU48)   # 48 == vram, no headroom


def test_headroom_is_configurable():
    a, b = _model("a", memory_gb=24), _model("b", memory_gb=24)
    assert colocation.resolve_group([a, b], GPU48, headroom_gb=0)


# ── Unified memory ───────────────────────────────────────────────────────────

def test_spark_usable_memory_excludes_the_host_reservation():
    """128 GB shared with the OS is not 128 GB of model capacity."""
    assert GB10.usable_gb == pytest.approx(min(128 - 40, 128 * 0.70))


def test_a_group_sized_for_a_discrete_card_may_not_fit_a_spark():
    a, b = _model("a", memory_gb=45), _model("b", memory_gb=45)
    # 90 GB total: fine against raw 128 GB, refused against usable ~88 GB.
    with pytest.raises(colocation.ColocationError):
        colocation.resolve_group([a, b], GB10)


def test_a_correctly_sized_spark_group_fits():
    a, b = _model("a", memory_gb=40), _model("b", memory_gb=40)
    tenants = colocation.resolve_group([a, b], GB10)
    assert sum(t.memory_gb for t in tenants) == 80


# ── Rejections ───────────────────────────────────────────────────────────────

def test_a_single_model_is_not_a_group():
    with pytest.raises(colocation.ColocationError, match="at least two"):
        colocation.resolve_group([EMBED], GPU48)


def test_undeclared_memory_is_rejected():
    """Without an absolute budget the fractions cannot be made to add up."""
    vague = _model("mystery")          # no memory_gb
    with pytest.raises(colocation.ColocationError, match="memory_gb"):
        colocation.resolve_group([EMBED, vague], GPU48)


def test_multi_gpu_models_cannot_be_co_located():
    """A tensor-parallel model owns whole devices; sharing one would deadlock
    the collective."""
    tp2 = _model("big-tp", memory_gb=10, gpus=2, tp=2)
    with pytest.raises(colocation.ColocationError, match="single-GPU"):
        colocation.resolve_group([EMBED, tp2], GPU48)


def test_duplicate_models_are_rejected_in_favour_of_replicas():
    with pytest.raises(colocation.ColocationError, match="replicas"):
        colocation.resolve_group([EMBED, EMBED], GPU48)


def test_an_unknown_gpu_class_is_rejected():
    with pytest.raises(colocation.ColocationError, match="memory size"):
        colocation.resolve_group([EMBED, RERANK], None)


# ── Variants apply ───────────────────────────────────────────────────────────

def test_a_variant_can_resize_a_model_per_class():
    model = CatalogModel(
        name="tuned", model_path="p", gpus=1, tensor_parallel_size=1, memory_gb=30,
        variants={"gpu24": {"memory_gb": 10}},
    )
    on24 = colocation.resolve_group([model, RERANK], GPU24)
    assert next(t for t in on24 if t.model == "tuned").memory_gb == 10


def test_requirements_do_not_block_colocation_directly():
    """Class eligibility is decided before this; here we only check the fit."""
    picky = _model("picky", memory_gb=8, requires=ModelRequirements(min_vram_gb=80))
    assert colocation.resolve_group([EMBED, picky], GPU48)


# ── Serialization / job env ──────────────────────────────────────────────────

def test_round_trip_through_json():
    tenants = colocation.resolve_group([EMBED, RERANK], GPU48)
    assert colocation.decode(colocation.encode(tenants)) == tenants


def test_decode_tolerates_garbage():
    assert colocation.decode(None) == []
    assert colocation.decode("not json") == []
    assert colocation.decode('{"not": "a list"}') == []


def test_job_env_carries_the_group_as_one_blob():
    tenants = colocation.resolve_group([EMBED, RERANK], GPU48)
    env = colocation.job_env(tenants)

    assert env["COLOCATED_COUNT"] == "2"
    payload = json.loads(env["COLOCATED_MODELS"])
    assert {m["model"] for m in payload} == {"bge-m3", "bge-reranker-v2"}
    assert all("gpu_memory_utilization" in m for m in payload)


def test_no_group_means_no_env_and_the_single_model_path_is_used():
    assert colocation.job_env([]) == {}


def test_describe_is_human_readable():
    tenants = colocation.resolve_group([EMBED, RERANK], GPU48)
    assert colocation.describe(tenants) == "bge-m3 (12 GB) + bge-reranker-v2 (8 GB)"


def test_total_memory():
    tenants = colocation.resolve_group([EMBED, RERANK], GPU48)
    assert colocation.total_memory_gb(tenants) == 20


# ── GpuClass memory helpers ──────────────────────────────────────────────────

def test_utilization_for_gb_is_capped_by_the_class_ceiling():
    assert GB10.utilization_for_gb(120) == 0.70


def test_default_utilization_derives_from_the_reservation():
    """A class states how much the host needs; the fraction follows."""
    cls = GpuClass(name="c", vram_gb=100, reserved_gb=20)
    assert cls.default_utilization == pytest.approx(0.80)


def test_discrete_class_without_a_reservation_has_no_forced_default():
    assert GpuClass(name="c", vram_gb=48).default_utilization is None


# ── Serialised startup ───────────────────────────────────────────────────────

def test_job_env_preserves_group_order():
    """The template starts them in this order, one at a time: vLLM sizes its KV
    cache by profiling *free* GPU memory, so two instances initialising at once
    both measure the same free memory and mis-size."""
    a, b, c = _model("a", 8), _model("b", 6), _model("c", 8)
    tenants = colocation.resolve_group([a, b, c], GPU48)
    assert [t.model for t in tenants] == ["a", "b", "c"]

    payload = json.loads(colocation.job_env(tenants)["COLOCATED_MODELS"])
    assert [m["model"] for m in payload] == ["a", "b", "c"]


def test_three_small_models_fit_one_a6000():
    """The concrete case: embedding + reranker + OCR on one RTX A6000."""
    group = [_model("bge-m3", 10), _model("bge-reranker", 8), _model("got-ocr2", 10)]
    tenants = colocation.resolve_group(group, GPU48)

    assert len(tenants) == 3
    assert colocation.total_memory_gb(tenants) == 28
    # Comfortably inside the card, with room for CUDA contexts.
    assert sum(t.gpu_memory_utilization for t in tenants) < 0.7


def test_a_fourth_model_that_would_not_fit_is_refused():
    group = [_model("bge-m3", 10), _model("bge-reranker", 8),
             _model("got-ocr2", 10), _model("big", 24)]
    with pytest.raises(colocation.ColocationError, match="52 GB"):
        colocation.resolve_group(group, GPU48)
