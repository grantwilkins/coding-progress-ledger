"""
Claim:
Stage 1c is a minimal controller proof: a tiny active-session fixture is mapped
through the existing Queue-Haul solver, produces both replay and KV actions, and
serializes those decisions into a manifest with hard acceptance checks.

Plausible wrong implementations:
- Bypass dispatch.solve or call it with the wrong signature.
- Let idle/cold sessions produce zero-cost moves.
- Produce a single-action proof fixture.
- Accept manifests without route-specific replay/KV evidence.
"""

from __future__ import annotations

import pytest

import stage1c_controller as c


def test_default_fixture_solves_to_replay_and_kv_mix():
    fixture = c.default_fixture()

    summary = c.plan_summary(fixture)
    actions = [s["action"] for s in summary["sessions"]]

    assert summary["solver"]["feasible"] is True
    assert summary["solver"]["shortfall_w"] == 0
    assert set(actions) == {"R", "S"}
    assert [s["dispatch_rank"] for s in summary["sessions"]] == list(range(len(actions)))
    assert summary["movement"]["lambda_src_bytes_per_s"] == 125_000_000.0


def test_population_hard_fails_non_active_sessions():
    fixture = c.default_fixture()
    fixture["sessions"][0]["state"] = "idle"

    with pytest.raises(ValueError, match="active"):
        c.build_population(fixture)


def test_plan_uses_fixture_costs_for_action_choice():
    fixture = c.default_fixture()

    rows = c.planned_sessions(fixture)
    by_id = {row["id"]: row["action"] for row in rows}

    assert by_id["r0"] == "R"
    assert by_id["k0"] == "S"


def test_manifest_check_requires_solver_mix_deadline_and_route_evidence():
    manifest = {
        "schema": c.SCHEMA,
        "solver": {"feasible": True, "shortfall_w": 0},
        "smoke2": {"acceptance": {"ok": True}},
        "sessions": [
            {"id": "r", "action": "R", "dispatch_rank": 0, "actual_start_s": 0.0, "actual_end_s": 1.0, "http_status": 200, "deadline_met": True, "proxy_delta": {"api/client_to_target": 123}},
            {"id": "s", "action": "S", "dispatch_rank": 1, "actual_start_s": 1.0, "actual_end_s": 2.0, "http_status": 200, "deadline_met": True, "proxy_delta": {"kv/target_to_client": 456}},
        ],
    }

    c.check_manifest(manifest)
    manifest["sessions"][1]["proxy_delta"] = {}
    with pytest.raises(ValueError, match="KV action"):
        c.check_manifest(manifest)
    manifest["sessions"][1]["proxy_delta"] = {"kv/target_to_client": 456}
    manifest["sessions"][1]["actual_start_s"] = 0.5
    with pytest.raises(ValueError, match="serial"):
        c.check_manifest(manifest)
