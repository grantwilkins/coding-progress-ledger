"""
Claim:
`MobilityEpisode` is the Workstream K input schema. It carries N
workflows (each pointing at a per-workflow manifest JSON), source +
destination sites, a per-state warmness map, and per-resource
capacities. JSON load/dump roundtrips byte-deterministically.

The `scenario_class` property labels the episode under A2's taxonomy:
single-source-evacuation if one source site, distributed-origin if many.
"""
from __future__ import annotations

import json
from pathlib import Path

import pytest

from vagrant_agent.episode import (
    MobilityEpisode,
    Workflow,
    dump_episode,
    linear_session_to_episode,
    load_episode,
)

REPO = Path(__file__).resolve().parent.parent


# ---------------------------------------------------------------------------
# Construction + invariants
# ---------------------------------------------------------------------------


def _make_workflow(wid: str = "wf_0", site: str | None = "phoenix") -> Workflow:
    return Workflow(workflow_id=wid, manifest_path=f"manifests/{wid}.json",
                    src_site=site, deadline_s=None)


def test_minimal_episode_constructs():
    ep = MobilityEpisode(
        episode_id="ep0",
        source_sites=("phoenix",),
        destination_sites=("seattle",),
        workflows=(_make_workflow(),),
    )
    assert ep.episode_id == "ep0"
    assert ep.scenario_class == "single_source_evacuation"


def test_distributed_origin_label():
    ep = MobilityEpisode(
        episode_id="dist",
        source_sites=("phoenix", "seattle"),
        destination_sites=("austin",),
        workflows=(_make_workflow("wf_a", "phoenix"),
                   _make_workflow("wf_b", "seattle")),
    )
    assert ep.scenario_class == "distributed_origin"


def test_empty_source_sites_rejected():
    with pytest.raises(ValueError, match="source_sites"):
        MobilityEpisode(episode_id="e", source_sites=(),
                        destination_sites=("seattle",),
                        workflows=(_make_workflow(),))


def test_empty_destination_sites_rejected():
    with pytest.raises(ValueError, match="destination_sites"):
        MobilityEpisode(episode_id="e", source_sites=("phoenix",),
                        destination_sites=(),
                        workflows=(_make_workflow(),))


def test_empty_workflows_rejected():
    with pytest.raises(ValueError, match="workflows"):
        MobilityEpisode(episode_id="e", source_sites=("phoenix",),
                        destination_sites=("seattle",), workflows=())


def test_duplicate_workflow_ids_rejected():
    with pytest.raises(ValueError, match="unique"):
        MobilityEpisode(
            episode_id="e", source_sites=("phoenix",),
            destination_sites=("seattle",),
            workflows=(_make_workflow("a"), _make_workflow("a")),
        )


# ---------------------------------------------------------------------------
# JSON roundtrip
# ---------------------------------------------------------------------------


def _round_trip(ep: MobilityEpisode, tmp_path: Path) -> MobilityEpisode:
    p = tmp_path / "ep.json"
    dump_episode(ep, p)
    return load_episode(p)


def test_minimal_roundtrip(tmp_path: Path):
    ep = MobilityEpisode(
        episode_id="ep0",
        source_sites=("phoenix",),
        destination_sites=("seattle",),
        workflows=(_make_workflow(),),
    )
    rt = _round_trip(ep, tmp_path)
    assert rt == ep


def test_full_roundtrip_with_warmness_and_capacities(tmp_path: Path):
    ep = MobilityEpisode(
        episode_id="ep1",
        source_sites=("phoenix",),
        destination_sites=("seattle", "austin"),
        workflows=(
            Workflow("wf_a", "manifests/wf_a.json", src_site="phoenix", deadline_s=10.0),
            Workflow("wf_b", "manifests/wf_b.json", src_site="phoenix", deadline_s=None),
        ),
        state_warmness={
            "system_prompt": ("phoenix", "seattle"),
            "issue_text_wf_a": ("phoenix",),
        },
        capacities={
            "links": {"phoenix-seattle": 5e9},
            "sites": {"seattle": {"prefill_tok_s": 30000}},
        },
        trigger_t_s=12.5,
        notes="canonical test episode",
    )
    rt = _round_trip(ep, tmp_path)
    assert rt == ep
    assert rt.state_warmness["system_prompt"] == ("phoenix", "seattle")
    assert rt.capacities["links"]["phoenix-seattle"] == 5e9
    assert rt.trigger_t_s == 12.5


def test_dump_is_byte_deterministic(tmp_path: Path):
    ep = MobilityEpisode(
        episode_id="ep_det",
        source_sites=("phoenix",),
        destination_sites=("seattle",),
        workflows=(_make_workflow("wf_a"),
                   _make_workflow("wf_b", "seattle")),
        state_warmness={"sb": ("seattle",), "sa": ("phoenix", "seattle")},
    )
    a = tmp_path / "a.json"
    b = tmp_path / "b.json"
    dump_episode(ep, a)
    dump_episode(ep, b)
    assert a.read_bytes() == b.read_bytes()


def test_state_warmness_keys_sorted_in_output(tmp_path: Path):
    """Determinism check: warmness dict iterates in sorted key order, so a
    Python 3.7+ insertion-ordered dict still produces a stable file."""
    ep = MobilityEpisode(
        episode_id="ep_sort", source_sites=("phoenix",),
        destination_sites=("seattle",),
        workflows=(_make_workflow(),),
        state_warmness={"z_state": ("phoenix",), "a_state": ("seattle",)},
    )
    p = tmp_path / "ep.json"
    dump_episode(ep, p)
    raw = json.loads(p.read_text())
    keys = list(raw["state_warmness"].keys())
    assert keys == sorted(keys)


def test_state_warmness_sites_sorted_in_output(tmp_path: Path):
    ep = MobilityEpisode(
        episode_id="ep_sort", source_sites=("phoenix",),
        destination_sites=("seattle",),
        workflows=(_make_workflow(),),
        state_warmness={"sx": ("seattle", "austin", "phoenix")},
    )
    p = tmp_path / "ep.json"
    dump_episode(ep, p)
    raw = json.loads(p.read_text())
    assert raw["state_warmness"]["sx"] == ["austin", "phoenix", "seattle"]


# ---------------------------------------------------------------------------
# linear_session_to_episode adapter
# ---------------------------------------------------------------------------


def test_linear_session_adapter_default_is_no_op():
    """Default args produce src=dst=phoenix; a degenerate but valid
    episode useful for K1 schema tests."""
    ep = linear_session_to_episode(
        trace_path="some/trace.jsonl",
        manifest_path="some/manifest.json",
    )
    assert len(ep.workflows) == 1
    assert ep.workflows[0].manifest_path == "some/manifest.json"
    assert ep.workflows[0].src_site == "phoenix"
    assert ep.source_sites == ("phoenix",)
    assert ep.destination_sites == ("phoenix",)
    assert ep.scenario_class == "single_source_evacuation"


def test_linear_session_adapter_distinct_src_dst():
    ep = linear_session_to_episode(
        trace_path="t",
        manifest_path="m",
        src_site="phoenix",
        target_site="seattle",
        episode_id="evac_test",
    )
    assert ep.episode_id == "evac_test"
    assert ep.source_sites == ("phoenix",)
    assert ep.destination_sites == ("seattle",)


def test_linear_session_adapter_records_trace_in_notes():
    ep = linear_session_to_episode(trace_path="my/traj.jsonl", manifest_path="my/m.json")
    assert "my/traj.jsonl" in ep.notes


def test_linear_session_adapter_roundtrips(tmp_path: Path):
    ep = linear_session_to_episode(
        trace_path=REPO / "examples" / "traces" / "toy_subagent_trace.jsonl",
        manifest_path=REPO / "examples" / "manifests" / "toy.json",
        episode_id="toy_lift",
    )
    rt = _round_trip(ep, tmp_path)
    assert rt.episode_id == "toy_lift"
    assert len(rt.workflows) == 1
