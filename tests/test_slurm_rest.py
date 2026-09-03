"""The slurmrestd backend.

Fixtures are real response shapes captured from Slurm 25.05 /
`data_parser/v0.0.42`, because the wrapped-number and flag-list encodings are
the parts most likely to break silently on an API-version bump.
"""

from __future__ import annotations

from datetime import datetime, timedelta, timezone

import httpx
import pytest

from app.backends.slurm_rest import (
    SlurmRestBackend,
    _epoch,
    _flags,
    _mail_types,
    _mem_to_mb,
    _num,
    _time_limit_minutes,
    parse_gres_detail,
)
from app.backends.types import ClusterUnavailableError, JobSpec

TOKEN = "fake.jwt.token"
BASE = "http://titan:6820/slurm/v0.0.42"

# ── Recorded fixtures ────────────────────────────────────────────────────────

NODES_RESPONSE = {
    "nodes": [
        {"name": "europa", "state": ["MIXED"], "gres": "gpu:gpu24:1,gpu:gpu48:1",
         "features": ["gpu24", "gpu48", "rtx_a5000", "l40"], "partitions": ["gpu"],
         "cpus": 64, "real_memory": 257570},
        {"name": "jupiter", "state": ["MIXED"], "gres": "gpu:gpu96:4",
         "features": ["gpu96", "rtx_pro_6000"], "partitions": ["gpu"],
         "cpus": 48, "real_memory": 257095},
        {"name": "dgx", "state": ["MIXED"], "gres": "gpu:gpu80:4(S:0)",
         "features": ["gpu80", "a100"], "partitions": ["dgx"],
         "cpus": 128, "real_memory": 515610},
        {"name": "alien0", "state": ["IDLE", "DRAIN"], "gres": "gpu:gpu24:1(S:0)",
         "features": ["gpu24", "rtx_3090"], "partitions": ["gpu"],
         "cpus": 24, "real_memory": 64214},
    ],
    "errors": [], "warnings": [],
}

JOBS_RESPONSE = {
    "jobs": [
        {"job_id": 62377, "user_name": "florianstritzke", "job_state": ["RUNNING"],
         "partition": "gpu", "nodes": "jupiter",
         "gres_detail": ["gpu:gpu96:1(IDX:2)"],
         "start_time": {"set": True, "infinite": False, "number": 1787212927},
         "end_time": {"set": True, "infinite": False, "number": 1787385727},
         "exit_code": {"return_code": {"set": True, "infinite": False, "number": 0}}},
        {"job_id": 62332, "user_name": "florianstritzke", "job_state": ["PENDING"],
         "partition": "gpu", "nodes": "", "gres_detail": [],
         "tres_per_node": "gres/gpu:gpu48:1",
         "start_time": {"set": True, "infinite": False, "number": 0},
         "end_time": {"set": True, "infinite": False, "number": 0}},
        {"job_id": 62400, "user_name": "someone", "job_state": ["RUNNING"],
         "partition": "gpu", "nodes": "gpu[01-02]",
         "gres_detail": ["gpu:gpu48:2(IDX:0-1)", "gpu:gpu48:2(IDX:0-1)"],
         "start_time": {"set": True, "infinite": False, "number": 1787212927},
         "end_time": {"set": False, "infinite": True, "number": 0}},
    ],
    "errors": [], "warnings": [],
}


def _backend(handler, **kw) -> SlurmRestBackend:
    b = SlurmRestBackend(BASE, TOKEN, **kw)
    b._client = httpx.AsyncClient(
        transport=httpx.MockTransport(handler),
        headers={"X-SLURM-USER-TOKEN": TOKEN},
    )
    return b


def _json(payload, status=200):
    return lambda request: httpx.Response(status, json=payload)


# ── Wrapped scalars ──────────────────────────────────────────────────────────

def test_num_unwraps_a_set_value():
    assert _num({"set": True, "infinite": False, "number": 42}) == 42


def test_num_treats_unset_as_none_not_zero():
    """Zero would look like a job that ended in 1970."""
    assert _num({"set": False, "infinite": False, "number": 0}) is None


def test_num_treats_infinite_as_none():
    assert _num({"set": True, "infinite": True, "number": 0}) is None


def test_num_accepts_a_plain_int():
    assert _num(7) == 7


def test_epoch_converts_to_utc_aware():
    dt = _epoch({"set": True, "infinite": False, "number": 1787212927})
    assert dt.tzinfo is not None
    assert dt == datetime(2026, 8, 20, 8, 2, 7, tzinfo=timezone.utc)


def test_epoch_of_zero_is_none():
    assert _epoch({"set": True, "infinite": False, "number": 0}) is None


# ── State flag lists ─────────────────────────────────────────────────────────

def test_flags_joins_a_state_list():
    assert _flags(["MIXED", "DRAIN"]) == "MIXED+DRAIN"


def test_flags_passes_a_plain_string_through():
    assert _flags("RUNNING") == "RUNNING"


# ── gres_detail indices ──────────────────────────────────────────────────────

def test_gres_detail_extracts_class_count_and_index():
    assert parse_gres_detail(["gpu:gpu96:1(IDX:2)"]) == ("gpu96", 1, (2,))


def test_gres_detail_expands_an_index_range():
    assert parse_gres_detail(["gpu:gpu48:2(IDX:0-1)"]) == ("gpu48", 2, (0, 1))


def test_gres_detail_handles_comma_separated_indices():
    assert parse_gres_detail(["gpu:gpu24:3(IDX:0,2,5)"]) == ("gpu24", 3, (0, 2, 5))


def test_gres_detail_empty_is_zero():
    assert parse_gres_detail([]) == (None, 0, ())


# ── Unit conversions for submission ──────────────────────────────────────────

@pytest.mark.parametrize("text,minutes", [
    ("06:00:00", 360), ("00:30:00", 30), ("1-00:00:00", 1440),
    ("1-06:30:00", 1830), ("00:00:30", 1),
])
def test_time_limit_to_minutes(text, minutes):
    """REST takes minutes; the CLI took HH:MM:SS."""
    assert _time_limit_minutes(text) == minutes


@pytest.mark.parametrize("text,mb", [
    ("500G", 512000), ("64000", 64000), ("1T", 1048576), ("48G", 49152), (None, None),
])
def test_memory_to_megabytes(text, mb):
    assert _mem_to_mb(text) == mb


def test_mail_type_time_limit_is_mapped_to_the_rest_enum():
    """The REST enum has no TIME_LIMIT — it is TIME=100%. Sending the CLI
    spelling would be rejected."""
    assert _mail_types("FAIL,END,TIME_LIMIT") == ["FAIL", "END", "TIME=100%"]


def test_mail_type_drops_unknown_values_rather_than_failing_submission():
    assert _mail_types("FAIL,NONSENSE") == ["FAIL"]


def test_mail_type_none_is_empty():
    assert _mail_types("NONE") == []


# ── Job description ──────────────────────────────────────────────────────────

def _spec(**kw) -> JobSpec:
    base = dict(job_name="vllm-x", script_path="templates/vllm_job.sh", gpus=2,
                time_limit="06:00:00", cpus=16, log_dir="/tmp/logs",
                env={"MODEL_PATH": "Qwen/Q"})
    base.update(kw)
    return JobSpec(**base)


def test_gres_becomes_a_tres_per_node_string():
    """There is no `gres` field in the REST schema."""
    desc = _backend(_json({}))._job_desc(_spec(gres="gpu:gpu96:2"))
    assert desc["tres_per_node"] == "gres/gpu:gpu96:2"


def test_untyped_gpus_still_produce_a_tres_string():
    desc = _backend(_json({}))._job_desc(_spec())
    assert desc["tres_per_node"] == "gres/gpu:2"


def test_environment_is_never_empty():
    """slurmrestd rejects an empty environment, and the job does not inherit
    the caller's the way --export=ALL does."""
    desc = _backend(_json({}))._job_desc(_spec(env={}))
    assert desc["environment"]
    assert any(e.startswith("PATH=") for e in desc["environment"])


def test_nodelist_becomes_required_nodes_list():
    desc = _backend(_json({}))._job_desc(_spec(nodelist="gpu[01-02]"))
    assert desc["required_nodes"] == ["gpu01", "gpu02"]


def test_begin_time_is_an_epoch_struct():
    when = datetime(2026, 8, 23, 12, 0, tzinfo=timezone.utc)
    desc = _backend(_json({}))._job_desc(_spec(begin=when))
    assert desc["begin_time"] == {"set": True, "number": int(when.timestamp())}


def test_attribution_comment_is_carried():
    desc = _backend(_json({}))._job_desc(_spec(comment="user:alice,lease:412"))
    assert desc["comment"] == "user:alice,lease:412"


def test_mail_is_omitted_entirely_when_no_user_is_set():
    desc = _backend(_json({}))._job_desc(_spec())
    assert "mail_user" not in desc and "mail_type" not in desc


# ── Node discovery ───────────────────────────────────────────────────────────

async def test_nodes_parses_mixed_classes():
    nodes = {n.name: n for n in await _backend(_json(NODES_RESPONSE)).nodes()}
    assert nodes["europa"].gpu_classes == {"gpu24": 1, "gpu48": 1}
    assert nodes["jupiter"].gpu_classes == {"gpu96": 4}


async def test_nodes_join_state_flags_so_drain_stays_detectable():
    nodes = {n.name: n for n in await _backend(_json(NODES_RESPONSE)).nodes()}
    assert nodes["alien0"].state == "IDLE+DRAIN"
    assert nodes["alien0"].is_usable is False
    assert nodes["jupiter"].is_usable is True


async def test_nodes_carry_features_and_partitions():
    nodes = {n.name: n for n in await _backend(_json(NODES_RESPONSE)).nodes()}
    assert "rtx_pro_6000" in nodes["jupiter"].features
    assert nodes["dgx"].partitions == ("dgx",)


async def test_node_memory_and_cpus():
    nodes = {n.name: n for n in await _backend(_json(NODES_RESPONSE)).nodes()}
    assert nodes["dgx"].cpus == 128
    assert nodes["dgx"].mem_mb == 515610


# ── Jobs ─────────────────────────────────────────────────────────────────────

async def test_job_states_reports_the_primary_flag():
    states = await _backend(_json(JOBS_RESPONSE)).job_states(["62377", "62332"])
    assert states["62377"].state == "RUNNING"
    assert states["62332"].state == "PENDING"


async def test_job_states_maps_unknown_ids_to_none():
    states = await _backend(_json(JOBS_RESPONSE)).job_states(["999999"])
    assert states == {"999999": None}


async def test_foreign_jobs_carry_exact_gpu_indices():
    """Knowing which GPUs are held lets the planner pack around them."""
    jobs = {j.job_id: j for j in await _backend(_json(JOBS_RESPONSE)).foreign_jobs()}
    assert jobs["62377"].gpu_indices == (2,)
    assert jobs["62377"].nodes == ("jupiter",)


async def test_foreign_pending_job_falls_back_to_the_requested_tres():
    jobs = {j.job_id: j for j in await _backend(_json(JOBS_RESPONSE)).foreign_jobs()}
    assert jobs["62332"].gpus == 1
    assert jobs["62332"].nodes == ()      # nothing allocated yet


async def test_foreign_job_hostlist_is_expanded():
    jobs = {j.job_id: j for j in await _backend(_json(JOBS_RESPONSE)).foreign_jobs()}
    assert jobs["62400"].nodes == ("gpu01", "gpu02")


async def test_foreign_job_with_infinite_end_time_has_no_end():
    jobs = {j.job_id: j for j in await _backend(_json(JOBS_RESPONSE)).foreign_jobs()}
    assert jobs["62400"].end_time is None


async def test_foreign_jobs_filter_by_partition():
    jobs = await _backend(_json(JOBS_RESPONSE)).foreign_jobs(partition="dgx")
    assert jobs == []


# ── Failure modes ────────────────────────────────────────────────────────────

async def test_expired_token_reports_unavailable_not_empty_cluster():
    """401 must not look like "no nodes" — that would make every booking
    unplaceable and every job look dead."""
    b = _backend(lambda r: httpx.Response(401, json={"errors": ["invalid token"]}))
    with pytest.raises(ClusterUnavailableError, match="expired"):
        await b.nodes()


async def test_network_failure_is_unavailable_not_jobs_gone():
    def boom(request):
        raise httpx.ConnectError("connection refused", request=request)
    with pytest.raises(ClusterUnavailableError, match="unreachable"):
        await _backend(boom).job_states(["1"])


async def test_server_error_is_unavailable():
    b = _backend(lambda r: httpx.Response(503, text="upstream down"))
    with pytest.raises(ClusterUnavailableError):
        await b.nodes()


async def test_submit_without_a_job_id_raises():
    from app.backends.types import ClusterCommandError
    b = _backend(_json({"errors": [{"description": "Invalid partition"}]}))
    with pytest.raises(ClusterCommandError, match="Invalid partition"):
        await b.submit(_spec())


async def test_cancelling_a_finished_job_is_not_an_error():
    b = _backend(lambda r: httpx.Response(200, json={"errors": [{"description": "already done"}]}))
    await b.cancel("1")    # must not raise


async def test_estimate_start_is_explicitly_unsupported():
    """No REST equivalent for --test-only; callers fall back to the CLI."""
    from app.backends.types import CAP_TEST_ONLY
    b = _backend(_json({}))
    assert CAP_TEST_ONLY not in b.capabilities
    with pytest.raises(NotImplementedError):
        await b.estimate_start(_spec())


# ── Wiring ───────────────────────────────────────────────────────────────────

def test_missing_url_is_rejected_at_construction():
    with pytest.raises(ValueError, match="SLURM_REST_URL"):
        SlurmRestBackend("", TOKEN)


def test_missing_token_is_rejected_at_construction():
    with pytest.raises(ValueError, match="SLURM_JWT"):
        SlurmRestBackend(BASE, "")


def test_accounting_url_is_derived_from_the_base_url():
    b = SlurmRestBackend(BASE, TOKEN)
    assert b.db_url == "http://titan:6820/slurmdb/v0.0.42"


def test_username_header_is_sent_when_configured():
    """Without it, jobs run as the token owner — often root."""
    b = SlurmRestBackend(BASE, TOKEN, username="svc-llm")
    assert b._client.headers["X-SLURM-USER-NAME"] == "svc-llm"


def test_backend_is_selectable_by_name(monkeypatch):
    from app.backends import _build
    from app.settings import settings
    monkeypatch.setattr(settings, "slurm_rest_url", BASE)
    monkeypatch.setattr(settings, "slurm_jwt", TOKEN)
    assert _build("slurm_rest").name == "slurm-rest"


def test_unknown_backend_name_is_rejected():
    from app.backends import _build
    with pytest.raises(ValueError, match="Unknown CLUSTER_BACKEND"):
        _build("nonsense")


# ── Placement uses the exact indices ─────────────────────────────────────────

async def test_exact_indices_leave_the_rest_of_the_node_bookable():
    """The payoff for parsing gres_detail: a job on GPU 2 blocks only GPU 2."""
    from app.backends.types import GpuGroup, NodeInfo
    from app.placement import Demand, compute_placements

    now = datetime(2026, 8, 20, 12, 0, tzinfo=timezone.utc)
    foreign = await _backend(_json(JOBS_RESPONSE)).foreign_jobs()
    jupiter_job = next(j for j in foreign if j.job_id == "62377")

    nodes = [NodeInfo(name="jupiter", gpus=(GpuGroup("gpu96", 4),), state="idle")]
    demand = Demand(1, 2, now + timedelta(days=1), now + timedelta(days=1, hours=1),
                    gpu_class="gpu96")

    result = compute_placements([demand], nodes, foreign_jobs=[jupiter_job])

    # The foreign job holds index 2, so a contiguous pair fits at 0-1.
    assert not result[1].conflict
    assert result[1].gpu_indices == (0, 1)


# ── Token renewal ────────────────────────────────────────────────────────────

async def test_expired_token_is_renewed_and_the_request_retried():
    """A token can die before its `exp` claim (revoked, key rotated, clock
    skew). One reactive refresh beats pausing the scheduler."""
    from app.tokens import TokenProvider

    issued = []

    def mint():
        issued.append(f"token-{len(issued)}")
        return issued[-1]

    provider = TokenProvider(mint, name="test")
    seen: list[str] = []

    def handler(request):
        seen.append(request.headers["X-SLURM-USER-TOKEN"])
        if len(seen) == 1:
            # slurmrestd's actual response to a dead token.
            return httpx.Response(511, json={"errors": [
                {"error_number": 1007, "error": "Protocol authentication error"}
            ]})
        return httpx.Response(200, json=NODES_RESPONSE)

    b = SlurmRestBackend(BASE, provider)
    b._client = httpx.AsyncClient(transport=httpx.MockTransport(handler))

    nodes = await b.nodes()

    assert len(nodes) == 4          # the retry succeeded
    assert seen == ["token-0", "token-1"]   # and it used a fresh token


async def test_a_persistently_bad_token_gives_up_with_an_actionable_message():
    """Retrying forever would hide the real problem."""
    from app.tokens import TokenProvider

    calls = []

    def handler(request):
        calls.append(1)
        return httpx.Response(511, json={"errors": [
            {"error_number": 1007, "error": "Protocol authentication error"}
        ]})

    b = SlurmRestBackend(BASE, TokenProvider(lambda: "dud", name="test"))
    b._client = httpx.AsyncClient(transport=httpx.MockTransport(handler))

    with pytest.raises(ClusterUnavailableError, match="scontrol token"):
        await b.nodes()
    assert len(calls) == 2          # one retry, then stop


async def test_an_auth_error_in_a_200_body_is_still_an_auth_error():
    """slurmrestd can report the failure in the body rather than the status —
    mistaking that for "empty cluster" would unplace every booking."""
    from app.tokens import TokenProvider

    b = SlurmRestBackend(BASE, TokenProvider(lambda: "dud", name="test"))
    b._client = httpx.AsyncClient(transport=httpx.MockTransport(
        lambda r: httpx.Response(200, json={
            "nodes": [], "errors": [{"error_number": 1007, "error": "Protocol authentication error"}],
        })
    ))
    with pytest.raises(ClusterUnavailableError):
        await b.nodes()


async def test_a_token_that_cannot_be_minted_reports_unavailable():
    from app.tokens import TokenError, TokenProvider

    def broken():
        raise TokenError("ssh: connection refused")

    b = SlurmRestBackend(BASE, TokenProvider(broken, name="test"))
    b._client = httpx.AsyncClient(transport=httpx.MockTransport(
        lambda r: httpx.Response(200, json=NODES_RESPONSE)
    ))
    with pytest.raises(ClusterUnavailableError, match="connection refused"):
        await b.nodes()
