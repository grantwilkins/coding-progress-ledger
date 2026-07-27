"""Claim: campaign inputs preserve trace shape without content and reserve GPU time only for irreducible measurements.

Plausible wrong implementations: leak source text, invent timestamps, accept topical non-code chat,
misread ShareGPT's human/gpt schema,
retain the obsolete 72-hour grid, overlap splits, or spend GPU time deriving quantities available offline.
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


def content_free_manifest(path):
    splits = {
        job: {name: [f"{job}-{name}-{i}" for i in range(n)]
              for name, n in (("fit", 12), ("tune", 6), ("validation", 6))}
        for job in campaign.JOB_CLASSES
    }
    value = {"manifest": {"schema": campaign.MANIFEST_SCHEMA, "splits": splits},
             "traces": [{"session_id": sid, "input_tokens_total": 256}
                        for split in splits.values() for ids in split.values()
                        for sid in ids]}
    path.write_text(json.dumps(value))
    return path


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
    monkeypatch.setattr(
        campaign.testbed,
        "http_json",
        lambda *args: seen.append(args[-1]) or {"count": 3},
    )
    counter = campaign.token_counter("host", 1, "model")
    assert counter([{"role": "user", "content": "x"}]) == 3
    assert seen[0]["chat_template_kwargs"] == {
        "reasoning_effort": "low",
        "enable_thinking": True,
    }


def test_local_tokenizer_counts_rendered_chat_and_raw_output():
    class Encoding:
        def __len__(self):
            return 2

        def __getitem__(self, key):
            assert key == "input_ids"
            return [1, 2, 3]

    class Tokenizer:
        def __call__(self, _text, add_special_tokens):
            assert not add_special_tokens
            return {"input_ids": [1, 2]}

        def apply_chat_template(self, messages, **kwargs):
            assert messages and kwargs == {
                "tokenize": True,
                "add_generation_prompt": True,
                "reasoning_effort": "low",
                "enable_thinking": True,
            }
            return Encoding()

    counter = campaign.tokenizer_counter(Tokenizer())
    assert counter("answer") == 2
    assert counter([{"role": "user", "content": "prompt"}]) == 3


def test_agent_native_tools_become_valid_shape_only_user_appends():
    messages = [
        {"role": "system", "content": "policy"},
        {"role": "user", "content": "fix"},
        {"role": "tool", "content": "output"},
        {"role": "assistant", "content": "done"},
    ]
    assert campaign.renderable(messages) == [
        {"role": "system", "content": "policy"},
        {"role": "user", "content": "fix\noutput"},
        {"role": "assistant", "content": "done"},
    ]


def test_assistant_event_without_a_prompt_is_rejected():
    rows = campaign.normalize_traces(
        [
            {
                "id": "one",
                "messages": [
                    {"role": "assistant", "content": "orphan"},
                    {"role": "user", "content": "prompt", "timestamp": 1},
                    {"role": "assistant", "content": "answer"},
                ],
            }
        ],
        "trace-commons/agent-traces",
        "revision",
        count,
    )
    assert len(rows) == 1 and rows[0]["turn"] == 1


def test_source_row_without_messages_contributes_no_turns():
    assert (
        campaign.normalize_traces(
            [{"id": "empty"}], "trace-commons/agent-traces", "revision", count
        )
        == []
    )


def test_wildchat_order_is_preserved_without_inventing_arrival_times():
    rows = campaign.normalize_traces(
        [
            {
                "id": "chat",
                "conversation": [
                    {"role": "user", "content": "first"},
                    {"role": "assistant", "content": "answer"},
                    {"role": "user", "content": "second"},
                    {"role": "assistant", "content": "answer"},
                ],
            }
        ],
        "allenai/WildChat-1M",
        "revision",
        count,
    )
    assert [r["time_s"] for r in rows] == [None, None]
    assert {r["job_class"] for r in campaign.classify(rows)} == {"interactive_coding"}


def test_sharegpt_schema_becomes_content_free_conversation_shapes():
    rows = campaign.normalize_traces(
        [{
            "id": "chat",
            "conversations": [
                {"from": "human", "value": "DO_NOT_LEAK"},
                {"from": "gpt", "value": "answer"},
                {"from": "human", "value": "follow up"},
                {"from": "gpt", "value": "done"},
            ],
        }],
        "anon8231489123/ShareGPT_Vicuna_unfiltered",
        "sha256:abc",
        count,
    )

    assert len(rows) == 2
    assert all(row["time_s"] is None for row in rows)
    assert "DO_NOT_LEAK" not in repr(rows)
    assert {row["job_class"] for row in campaign.classify(rows)} == {"conversation"}


def test_sharegpt_requires_the_pinned_revision_and_checksum(tmp_path, monkeypatch):
    path = tmp_path / "sharegpt.json"
    path.write_text("[]")
    monkeypatch.setattr(campaign, "file_hash", lambda _: campaign.SHAREGPT_SHA256)

    campaign.validate_sharegpt(path, campaign.SHAREGPT_REVISION)
    with pytest.raises(ValueError, match="checksum changed"):
        campaign.validate_sharegpt(path, "moving-main")


def test_normalized_source_cache_reuses_only_exact_key(tmp_path):
    calls = []

    def make():
        calls.append(1)
        return [{"turn": len(calls)}]

    path = tmp_path / "cache.json"
    assert campaign.cached_normalize(path, "a", make) == [{"turn": 1}]
    assert campaign.cached_normalize(path, "a", make) == [{"turn": 1}]
    assert campaign.cached_normalize(path, "b", make) == [{"turn": 2}]


def test_manifest_split_is_disjoint_deterministic_and_context_stratified():
    trace_rows = []
    for i in range(24):
        row = dict(id=f"trace-{i}", **messages("x " * (300 + i), i + 2))
        trace_rows += campaign.normalize_traces(
            [row], "trace-commons/agent-traces", "abc", count
        )
    interactive_rows = []
    for i in range(24):
        row = dict(id=f"interactive-{i}", **messages("x " * (300 + i), i + 2))
        interactive_rows += campaign.normalize_traces(
            [row], "allenai/WildChat-1M", "ghi", count
        )
    agent_rows = []
    for i in range(24):
        row = dict(id=f"agent-{i}", **messages("x " * (300 + i), 2, True))
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


def test_manifest_excludes_sessions_the_runner_cannot_use():
    rows = [
        {"session_id": f"trace-commons/agent-traces:{i}", "turn": 0,
         "input_tokens_total": 74 if i == 0 else 256 + i, "reset": False}
        for i in range(25)
    ] + [
        {"session_id": f"{source}:{i}", "turn": 0,
         "input_tokens_total": 256 + i, "reset": False}
        for source in ("allenai/WildChat-1M", "nvidia/SWE-Hero-openhands-trajectories")
        for i in range(24)
    ]
    manifest = campaign.build_manifests(rows)
    assert "trace-commons/agent-traces:0" not in set().union(
        *map(set, manifest["splits"]["coding"].values())
    )


def test_campaign_is_one_mandatory_job_and_no_obsolete_grid(tmp_path):
    plan = campaign.make_plan(content_free_manifest(tmp_path / "manifest.json"))
    campaign.validate_plan(plan)
    assert plan["job"] == {"name": "mandatory", "hours": 12}
    assert plan["gpu_pair_hour_budget"] == plan["reserve_pair_hour_limit"] == 12
    assert "jobs" not in plan
    assert plan["migration"]["rho"] == [0, 0.8, "emergency_inside"]
    with pytest.raises(ValueError, match="budget"):
        campaign.validate_plan(dict(plan, gpu_pair_hour_budget=72))


def test_boundary_decision_uses_odd_run_majority():
    assert campaign.boundary_decision(["feasible"] * 3) == "feasible"
    assert campaign.boundary_decision(["feasible"] * 4 + ["infeasible"]) == "feasible"
    assert campaign.boundary_decision(
        ["feasible", "infeasible", "feasible", "infeasible", "infeasible"]
    ) == "infeasible"


def test_acceptance_targets_only_failed_reserve_cells(tmp_path):
    service = [
        {"cell": "coding", "actual_bound": 1, "predicted_bound": 1.1,
         "actual_feasible": True, "predicted_feasible": True},
        {"cell": "agentic", "actual_bound": 1, "predicted_bound": 1.2,
         "actual_feasible": False, "predicted_feasible": True},
    ]
    loaded = [
        {"cell": "replay-high", "observed_s": 10, "predicted_s": 10,
         "correct": True},
        {"cell": "kv-heldout", "observed_s": 10, "predicted_s": 12,
         "correct": False},
    ]
    report = campaign.acceptance_report(service, loaded)
    assert not report["accepted"]
    assert campaign.reserve_tasks(report) == [
        {"phase": "service", "cell": "agentic", "reason": "facet_validation"},
        {"phase": "migration", "cell": "kv-heldout", "reason": "interaction"},
        {"phase": "migration", "cell": "kv-heldout", "reason": "correctness"},
    ]


def test_reserve_bundle_exists_only_after_a_failed_reduction(tmp_path):
    bundle = tmp_path / "mandatory"
    campaign.prepare(content_free_manifest(tmp_path / "manifest.json"), bundle)
    report = tmp_path / "report.json"
    report.write_text(json.dumps({"accepted": True}))
    assert campaign.prepare_reserve(report, bundle, tmp_path / "none") is None
    assert not (tmp_path / "none").exists()
    report.write_text(json.dumps({"boundary_disagreements": ["coding-normal"]}))
    source = json.loads((bundle / "plan.json").read_text())
    source["migration"]["rho"] = [.8, "emergency_inside"]
    (bundle / "plan.json").write_text(json.dumps(source))
    plan = campaign.prepare_reserve(report, bundle, tmp_path / "reserve")
    assert plan["reserve_tasks"] == [{
        "phase": "service", "cell": "coding-normal",
        "reason": "boundary_disagreement",
    }]
    assert plan["migration"]["rho"] == [0, .8, "emergency_inside"]
    campaign.verify_checksums(tmp_path / "reserve")


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
        {"mode": mode, "facet": 0, "run_id": run, "bound": base + run / 10,
         "outside": base + run / 10 + .5, "inside_decision": "feasible",
         "outside_decision": "infeasible", "cache_state": "private_prefix"}
        for mode, base in (("normal", 1), ("emergency", 2), ("stable", 3))
        for run in range(3)
    ]
    loaded = [
        {
            "method": method,
            "rho": rho,
            "run_id": run,
            "duration_factor": .5 * (1 + rho + run / 10),
            "achieved_rho": rho,
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
        result["conservative"]["loaded"]["replay"]["baseline_factor"]
        > result["central"]["loaded"]["replay"]["baseline_factor"]
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


def test_prepare_and_submit_use_one_immutable_job(tmp_path):
    bundle = tmp_path / "bundle"
    campaign.prepare(content_free_manifest(tmp_path / "manifest.json"), bundle)
    campaign.verify_checksums(bundle)
    plan, job, sbatch = bundle / "plan.json", bundle / "mandatory.sh", tmp_path / "campaign.sbatch"
    sbatch.write_text("#!/bin/sh\n")
    calls = []

    def run(command, **_kwargs):
        calls.append(command)
        return SimpleNamespace(stdout=f"{100 + len(calls)};cluster\n")

    assert campaign.submit(plan, sbatch, job,
                           tmp_path / "run", run=run) == "101"
    assert len(calls) == 1 and not any("dependency" in arg for arg in calls[0])
    assert any("QH_RUN_ROOT=" in arg for arg in calls[0])
    job.write_text("changed\n")
    with pytest.raises(ValueError, match="checksum"):
        campaign.submit(plan, sbatch, job, tmp_path / "run")


def test_destination_batch_isolates_shared_node_ports():
    text = Path(campaign.__file__).with_name("destination_campaign.sbatch").read_text()
    assert "QH_PORT_OFFSET=${QH_PORT_OFFSET:-$((SLURM_JOB_ID % 40000 + 1000))}" in text
