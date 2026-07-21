"""Claim: campaign inputs preserve trace shape without content and reserve GPU time only for irreducible measurements.

Plausible wrong implementations: leak source text, invent NVIDIA timestamps, overlap splits,
sample only one context region, or spend GPU time deriving quantities available offline.
"""

import destination_campaign as campaign


def messages(secret, gap, tools=False):
    return {"messages": [
        {"role": "user", "content": secret, "timestamp": 0},
        {"role": "assistant", "content": "answer", "timestamp": 1},
        {"role": "tool" if tools else "user", "content": secret, "timestamp": gap},
        {"role": "assistant", "content": "done", "timestamp": gap + 1},
    ]}


def count(value):
    if isinstance(value, list):
        return sum(len(row["content"].split()) + 1 for row in value)
    return len(value.split())


def test_evidence_audit_measures_only_irreducible_cells():
    audit = campaign.audit_evidence()
    assert audit["gpu_measurements"] == [
        "continuation", "foreground_impact", "kv_correctness",
        "loaded_migration_slowdown", "service_envelopes",
    ]
    assert not {"wan_capacity", "kv_bytes", "trace_growth"} & set(audit["gpu_measurements"])


def test_trace_normalization_discards_content_and_does_not_invent_time():
    trace = campaign.normalize_traces([dict(id="one", **messages("DO_NOT_LEAK", 60))],
                                      "trace-commons/agent-traces", "abc", count)
    agent = campaign.normalize_traces([{"id": "two", "messages": [
        {"role": "user", "content": "ALSO_SECRET"}, {"role": "assistant", "content": "ok"},
    ]}], "nvidia/SWE-Hero-openhands-trajectories", "def", count)
    assert "DO_NOT_LEAK" not in repr(trace) and "ALSO_SECRET" not in repr(agent)
    assert [r["input_tokens_total"] for r in trace] == [2, 6]
    assert [r["newly_append_tokens"] for r in trace] == [2, 4]
    assert agent[0]["time_s"] is None


def test_manifest_split_is_disjoint_deterministic_and_context_stratified():
    trace_rows = []
    for i in range(48):
        row = dict(id=f"trace-{i}", **messages("x " * (i + 1), i + 2))
        trace_rows += campaign.normalize_traces([row], "trace-commons/agent-traces", "abc", count)
    agent_rows = []
    for i in range(24):
        row = dict(id=f"agent-{i}", **messages("x " * (i + 1), 2, True))
        agent_rows += campaign.normalize_traces([row], "nvidia/SWE-Hero-openhands-trajectories", "def", count)
    rows = trace_rows + agent_rows
    first, second = campaign.build_manifests(rows, 7), campaign.build_manifests(list(reversed(rows)), 7)
    assert first == second
    for splits in first["splits"].values():
        assert [len(splits[k]) for k in ("fit", "tune", "validation")] == [12, 6, 6]
        assert len(set().union(*map(set, splits.values()))) == 24
