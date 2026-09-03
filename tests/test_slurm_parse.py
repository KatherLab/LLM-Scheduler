"""Tests for the pure Slurm output parsers.

These matter more than they look: every one of them is a place where a
misparse silently becomes a wrong scheduling decision (a node with 0 GPUs
disappears from the planner, a misread hostlist hides a foreign job).
"""

from __future__ import annotations

from datetime import datetime, timezone

import pytest

from app.backends.slurm_parse import (
    expand_hostlist,
    parse_gres,
    parse_mem_mb,
    parse_scontrol_kv,
    parse_slurm_time,
    parse_test_only,
)


# ── hostlists ────────────────────────────────────────────────────────────────

@pytest.mark.parametrize(
    "spec,expected",
    [
        ("", []),
        ("(null)", []),
        ("gpu01", ["gpu01"]),
        ("gpu01,gpu02", ["gpu01", "gpu02"]),
        ("gpu[01-03]", ["gpu01", "gpu02", "gpu03"]),
        ("gpu[01-03,07]", ["gpu01", "gpu02", "gpu03", "gpu07"]),
        ("gpu[1-3]", ["gpu1", "gpu2", "gpu3"]),
        ("gpu[08-10],node5", ["gpu08", "gpu09", "gpu10", "node5"]),
        ("node[1-2]suffix", ["node1suffix", "node2suffix"]),
    ],
)
def test_expand_hostlist(spec, expected):
    assert expand_hostlist(spec) == expected


def test_expand_hostlist_preserves_zero_padding():
    """Padding comes from the range's lower bound, matching Slurm itself."""
    assert expand_hostlist("gpu[008-010]") == ["gpu008", "gpu009", "gpu010"]


def test_expand_hostlist_ignores_commas_inside_brackets():
    """The naive `split(',')` this replaces would shatter `gpu[01,03]`."""
    assert expand_hostlist("gpu[01,03],spark01") == ["gpu01", "gpu03", "spark01"]


# ── GRES ─────────────────────────────────────────────────────────────────────

@pytest.mark.parametrize(
    "spec,expected",
    [
        (None, (None, 0)),
        ("(null)", (None, 0)),
        ("gpu:4", (None, 4)),
        ("gpu:h200:4", ("h200", 4)),
        ("gpu:h200:4(S:0-1)", ("h200", 4)),
        ("gres/gpu=8", (None, 8)),
        # slurmrestd reports job GRES requests in this form.
        ("gres/gpu:gpu48:1", ("gpu48", 1)),
        ("gres/gpu:2", (None, 2)),
        ("gres:gpu:2", (None, 2)),
        ("gpu:a100:8,nic:1", ("a100", 8)),
        ("nic:1", (None, 0)),
    ],
)
def test_parse_gres(spec, expected):
    assert parse_gres(spec) == expected


def test_parse_gres_untyped_count_is_not_mistaken_for_a_type():
    """`gpu:4` must yield count 4 with no type, never type='4'."""
    gpu_type, count = parse_gres("gpu:4")
    assert gpu_type is None
    assert count == 4


# ── times ────────────────────────────────────────────────────────────────────

def test_parse_slurm_time_iso():
    assert parse_slurm_time("2026-08-20T15:40:00") == datetime(
        2026, 8, 20, 15, 40, tzinfo=timezone.utc
    )


@pytest.mark.parametrize("value", ["N/A", "(null)", "Unknown", "", None])
def test_parse_slurm_time_nullish(value):
    assert parse_slurm_time(value) is None


def test_parse_slurm_time_is_timezone_aware():
    """Naive datetimes leaking into the planner would break every comparison."""
    parsed = parse_slurm_time("2026-08-20T15:40:00")
    assert parsed is not None and parsed.tzinfo is not None


# ── memory ───────────────────────────────────────────────────────────────────

@pytest.mark.parametrize(
    "spec,expected",
    [("64000", 64000), ("500G", 512000), ("1T", 1048576), ("(null)", 0), ("64000+", 64000)],
)
def test_parse_mem_mb(spec, expected):
    assert parse_mem_mb(spec) == expected


# ── sbatch --test-only ───────────────────────────────────────────────────────

def test_parse_test_only_extracts_start_and_nodes():
    out = (
        "sbatch: Job 1234 to start at 2026-08-20T15:40:00 using 4 processors "
        "on nodes gpu07 in partition general"
    )
    start, nodes = parse_test_only(out)
    assert start == datetime(2026, 8, 20, 15, 40, tzinfo=timezone.utc)
    assert nodes == ["gpu07"]


def test_parse_test_only_expands_node_ranges():
    out = "sbatch: Job 9 to start at 2026-08-20T15:40:00 using 8 processors on nodes gpu[01-02]"
    _, nodes = parse_test_only(out)
    assert nodes == ["gpu01", "gpu02"]


def test_parse_test_only_on_rejection_yields_no_estimate():
    out = "sbatch: error: Batch job submission failed: Requested node configuration is not available"
    start, nodes = parse_test_only(out)
    assert start is None
    assert nodes == []


# ── scontrol ─────────────────────────────────────────────────────────────────

def test_parse_scontrol_kv():
    line = "JobId=412 JobName=vllm-Qwen JobState=OUT_OF_MEMORY ExitCode=0:125 NodeList=gpu03"
    kv = parse_scontrol_kv(line)
    assert kv["JobId"] == "412"
    assert kv["JobState"] == "OUT_OF_MEMORY"
    assert kv["ExitCode"] == "0:125"


def test_parse_scontrol_kv_keeps_first_occurrence():
    """Slurm repeats some keys; the first is the allocation-level one."""
    kv = parse_scontrol_kv("JobId=1 JobState=RUNNING JobState=SOMETHING_ELSE")
    assert kv["JobState"] == "RUNNING"
