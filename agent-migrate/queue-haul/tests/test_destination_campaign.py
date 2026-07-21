"""Claim: campaign inputs preserve trace shape without content and reserve GPU time only for irreducible measurements.

Plausible wrong implementations: leak source text, invent timestamps, accept topical non-code chat,
overlap splits, sample only one context region, or spend GPU time deriving quantities available offline.
"""

import destination_campaign as campaign
import pytest
import io
import json
from datetime import datetime
from pathlib import Path
from types import SimpleNamespace
from urllib.error import HTTPError


def messages(secret, gap, tools=False):
    return {
        "messages": [
            {"role": "user", "content": secret, "timestamp": 0},
            {"role": "assistant", "content": "answer", "timestamp": 1},
            {"role": "tool" if tools else "user", "content": secret, "timestamp": gap},
            {"role": "assistant", "content": "done", "timestamp": gap + 1},
        ]
    }


def count(value):
    if isinstance(value, list):
        return sum(len(row["content"].split()) + 1 for row in value)
    return len(value.split())


def test_evidence_audit_measures_only_irreducible_cells():
    audit = campaign.audit_evidence()
    assert audit["gpu_measurements"] == [
        "continuation",
        "foreground_impact",
        "kv_correctness",
        "loaded_migration_slowdown",
        "service_envelopes",
    ]
    assert not {"wan_capacity", "kv_bytes", "trace_growth"} & set(
        audit["gpu_measurements"]
    )


def test_trace_normalization_discards_content_and_does_not_invent_time():
    trace = campaign.normalize_traces(
        [dict(id="one", **messages("DO_NOT_LEAK", 60))],
        "trace-commons/agent-traces",
        "abc",
        count,
    )
    agent = campaign.normalize_traces(
        [
            {
                "id": "two",
                "messages": [
                    {"role": "user", "content": "ALSO_SECRET"},
                    {"role": "assistant", "content": "ok"},
                ],
            }
        ],
        "nvidia/SWE-Hero-openhands-trajectories",
        "def",
        count,
    )
    assert "DO_NOT_LEAK" not in repr(trace) and "ALSO_SECRET" not in repr(agent)
    assert [r["input_tokens_total"] for r in trace] == [2, 6]
    assert [r["newly_append_tokens"] for r in trace] == [2, 4]
    assert agent[0]["time_s"] is None


def test_campaign_uses_exact_gpt_oss_reasoning_chat_template(monkeypatch):
    seen = []
    monkeypatch.setattr(campaign.testbed, "http_json", lambda *args: seen.append(args[-1]) or {"count": 3})
    counter = campaign.token_counter("host", 1, "model")
    assert counter([{"role": "user", "content": "x"}]) == 3
    assert seen[0]["chat_template_kwargs"] == {
        "reasoning_effort": "low", "enable_thinking": True,
    }


def test_manifest_split_is_disjoint_deterministic_and_context_stratified():
    trace_rows = []
    for i in range(24):
        row = dict(id=f"trace-{i}", **messages("x " * (i + 1), i + 2))
        trace_rows += campaign.normalize_traces(
            [row], "trace-commons/agent-traces", "abc", count
        )
    interactive_rows = []
    for i in range(24):
        row = dict(id=f"interactive-{i}", **messages("x " * (i + 1), i + 2))
        interactive_rows += campaign.normalize_traces(
            [row], "allenai/WildChat-1M", "ghi", count
        )
    agent_rows = []
    for i in range(24):
        row = dict(id=f"agent-{i}", **messages("x " * (i + 1), 2, True))
        agent_rows += campaign.normalize_traces(
            [row], "nvidia/SWE-Hero-openhands-trajectories", "def", count
        )
    rows = trace_rows + interactive_rows + agent_rows
    first, second = (
        campaign.build_manifests(rows, 7),
        campaign.build_manifests(list(reversed(rows)), 7),
    )
    assert first == second
    for splits in first["splits"].values():
        assert [len(splits[k]) for k in ("fit", "tune", "validation")] == [12, 6, 6]
        assert len(set().union(*map(set, splits.values()))) == 24


def test_campaign_budget_dependencies_and_loaded_grid_are_frozen():
    plan = campaign.make_plan()
    campaign.validate_plan(plan)
    assert sum(job["hours"] for job in plan["jobs"]) == 72
    assert plan["jobs"][-1]["conditional"]
    assert plan["migration"]["rho"] == [0, 0.5, 0.8, 0.95, "emergency_inside"]
    with pytest.raises(ValueError, match="budget"):
        campaign.validate_plan(dict(plan, gpu_pair_hour_budget=71))


def test_boundary_disagreement_hard_fails_without_four_of_five():
    assert campaign.boundary_decision(["feasible"] * 3) == "feasible"
    assert campaign.boundary_decision(["feasible"] * 4 + ["infeasible"]) == "feasible"
    with pytest.raises(ValueError, match="four-of-five"):
        campaign.boundary_decision(["feasible", "infeasible", "feasible"])


def test_profile_reduction_is_conservative_in_the_safe_direction():
    anchors = [
        {
            "metric": metric,
            "context_tokens": context,
            "run_id": run,
            "tokens_per_s": base + run,
        }
        for metric, base in (("prefill", 100), ("decode", 50))
        for context in (1000, 2000)
        for run in range(3)
    ]
    service = [
        {"mode": mode, "facet": 0, "run_id": run, "bound": base + run / 10}
        for mode, base in (("normal", 1), ("emergency", 2), ("stable", 3))
        for run in range(3)
    ]
    loaded = [
        {
            "method": method,
            "rho": rho,
            "run_id": run,
            "slowdown": 1 + rho + run / 10,
            "context_tokens": 16000 + 4000 * run,
            "bandwidth_bytes_per_s": 5e8 + 5e8 * run,
        }
        for method in ("replay", "kv_transfer")
        for rho in (0, 0.5, 0.95)
        for run in range(3)
    ]
    identity = {
        "compatibility": {
            "model": "m",
            "tokenizer": "t",
            "durable_log": "l",
            "kv_abi": "k",
        },
        "kv_capacity_tokens": 1000,
        "workload_prefill_fraction_range": [0, 1],
        "provenance": "runs",
    }
    result = campaign.reduce_profile(anchors, service, loaded, identity, [[1, 1]])[
        "profiles"
    ]
    assert (
        result["conservative"]["bounds"]["normal"]
        < result["central"]["bounds"]["normal"]
    )
    assert result["conservative"]["prefill"][1][0] < result["central"]["prefill"][1][0]
    assert (
        result["conservative"]["loaded"]["replay"]["slowdown"][0]
        > result["central"]["loaded"]["replay"]["slowdown"][0]
    )


def test_failed_gate_stops_the_dag():
    good = {
        "image_sha256": campaign.IMAGE_SHA256,
        "gpu_count": 2,
        "same_session_cache_hit": True,
        "cross_session_cache_hits": 0,
        "tokenizer_ok": True,
    }
    campaign.check_gate(good, "preflight")
    with pytest.raises(ValueError, match="cross_session_cache_hits"):
        campaign.check_gate(dict(good, cross_session_cache_hits=1), "preflight")


def test_revision_stable_dataset_fetch_records_exact_source(tmp_path):
    replies = iter(
        [
            {"sha": "revision"},
            {"splits": [{"config": "default", "split": "train"}]},
            {"rows": [{"row": {"id": 1}}], "num_rows_total": 1},
            {"sha": "revision"},
        ]
    )

    urls = []

    def open_url(url):
        urls.append(url)
        return io.BytesIO(json.dumps(next(replies)).encode())

    out = tmp_path / "rows.jsonl"
    metadata = campaign.fetch_dataset("owner/data", out, opener=open_url)
    assert metadata == {
        "dataset": "owner/data",
        "revision": "revision",
        "config": "default",
        "split": "train",
        "rows": 1,
        "source_rows": 1,
        "scanned_rows": 1,
        "sha256": campaign.hashlib.sha256(out.read_bytes()).hexdigest(),
    }
    assert urls[0] == "https://huggingface.co/api/datasets/owner/data"
    assert (
        campaign.fetch_dataset(
            "owner/data", out, opener=lambda _: io.BytesIO(b'{"sha":"revision"}')
        )
        == metadata
    )


def test_dataset_fetch_retries_throttling_but_not_other_http_errors(monkeypatch):
    replies = iter(
        [HTTPError("url", 429, "slow", {"Retry-After": "0"}, None), {"ok": True}]
    )

    def open_url(_url):
        reply = next(replies)
        if isinstance(reply, Exception):
            raise reply
        return io.BytesIO(json.dumps(reply).encode())

    monkeypatch.setattr(campaign.time, "sleep", lambda _: None)
    assert campaign._get_json("url", open_url) == {"ok": True}
    with pytest.raises(HTTPError):
        campaign._get_json(
            "url",
            lambda _: (_ for _ in ()).throw(
                HTTPError("url", 403, "forbidden", {}, None)
            ),
        )


def test_wildchat_filter_requires_multiturn_high_precision_code_evidence():
    base = {"language": "English", "toxic": False}
    assert campaign.wildchat_coding(
        {
            **base,
            "conversation": [
                {"role": "user", "content": "My Python function raises this traceback"},
                {"role": "assistant", "content": "Try this"},
                {"role": "user", "content": "The unit test still fails"},
            ],
        }
    )
    assert not campaign.wildchat_coding(
        {
            **base,
            "conversation": [
                {"role": "user", "content": "Plan my cooking class"},
                {"role": "assistant", "content": "Sure"},
                {"role": "user", "content": "Make it interactive"},
            ],
        }
    )


def test_nvidia_filter_requires_permissive_license_and_agent_tool_loop():
    row = {
        "license": "MIT",
        "trajectory": [
            {"role": "user", "content": "fix it"},
            {"role": "assistant", "content": "running tests"},
            {"role": "tool", "content": "passed"},
        ],
    }
    assert campaign.nvidia_agentic(row)
    assert not campaign.nvidia_agentic(dict(row, license="GPL-3.0"))
    assert not campaign.nvidia_agentic(dict(row, trajectory=row["trajectory"][:2]))


def test_streaming_sample_is_permutation_invariant_and_exact():
    rows = [
        {"id": i, "eligible": i % 2 == 0, "time": datetime(2020, 1, 1)}
        for i in range(20)
    ]
    first = campaign.stable_sample([rows[:10], rows[10:]], lambda r: r["eligible"], 4)
    second = campaign.stable_sample([list(reversed(rows))], lambda r: r["eligible"], 4)
    assert first == second and len(first) == 4 and isinstance(first[0]["time"], str)
    with pytest.raises(ValueError, match="need 11 eligible"):
        campaign.stable_sample([rows], lambda r: r["eligible"], 11)


def test_checksum_manifest_detects_changed_artifact(tmp_path):
    (tmp_path / "raw.jsonl").write_text("one\n")
    campaign.write_checksums(tmp_path)
    campaign.verify_checksums(tmp_path)
    (tmp_path / "raw.jsonl").write_text("two\n")
    with pytest.raises(ValueError, match="checksum"):
        campaign.verify_checksums(tmp_path)


def test_slurm_submission_uses_afterok_and_skips_reserve(tmp_path):
    plan = tmp_path / "plan.json"
    plan.write_text(json.dumps(campaign.make_plan()))
    job_dir = tmp_path / "jobs"
    job_dir.mkdir()
    for shard in range(1, 7):
        job = job_dir / f"shard-{shard}.sh"
        job.write_text("true\n")
        job.with_suffix(".sh.sha256").write_text(
            campaign.hashlib.sha256(job.read_bytes()).hexdigest() + "\n"
        )
    calls = []

    def run(command, **_kwargs):
        calls.append(command)
        return SimpleNamespace(stdout=f"{100 + len(calls)};cluster\n")

    ids = campaign.submit(plan, Path("campaign.sbatch"), job_dir, run=run)
    assert ids == {1: "101", 2: "102", 3: "103", 4: "104", 5: "105"}
    assert not any("QH_SHARD=6" in arg for call in calls for arg in call)
    assert "--dependency=afterok:101:102" in calls[2]
