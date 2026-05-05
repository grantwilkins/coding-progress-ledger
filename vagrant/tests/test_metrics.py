from pathlib import Path

import pytest
from ledger_progress import from_jsonl

from vagrant_agent import build_manifest
from vagrant_agent.metrics import (
    cost_weighted_duplication_factor,
    repeated_prefix_fraction,
    state_layer_breakdown,
)
from vagrant_agent.policies import run_request_level_no_reuse, run_shared_state_aware
from vagrant_agent.profiles import load_bundle

REPO = Path(__file__).resolve().parent.parent


def _setup():
    m = build_manifest(from_jsonl(str(REPO / "examples" / "traces" / "toy_subagent_trace.jsonl")))
    b = load_bundle(REPO / "configs" / "model_profiles.yaml",
                    REPO / "configs" / "sites_2site.yaml", "compact_kv")
    return m, b


def test_d2_duplication_factor_is_one_in_single_component_case():
    """Within a single-component placement, every row has count=1."""
    m, b = _setup()
    plan = run_shared_state_aware(m, b, tau=1)
    assert cost_weighted_duplication_factor(plan) == pytest.approx(1.0)


def test_d1_duplication_factor_is_above_one():
    m, b = _setup()
    plan = run_request_level_no_reuse(m, b)
    assert cost_weighted_duplication_factor(plan) > 1.0


def test_mvp_gate_d2_strictly_lower_than_d1():
    """The MVP headline metric gate."""
    m, b = _setup()
    a = run_request_level_no_reuse(m, b)
    c = run_shared_state_aware(m, b, tau=1)
    assert cost_weighted_duplication_factor(c) < cost_weighted_duplication_factor(a)


def test_repeated_prefix_fraction_on_toy():
    m, _ = _setup()
    # Shared: system_prefix(200×4) + repo_context(8000×4) + workspace_AC(0×2) = 32_800
    # Total: shared + private (1500+12000+2000) = 32_800 + 15_500 = 48_300
    expected = 32_800 / 48_300
    assert repeated_prefix_fraction(m) == pytest.approx(expected, rel=1e-9)


def test_state_layer_breakdown_assigns_costs():
    m, b = _setup()
    plan = run_request_level_no_reuse(m, b)
    breakdown = state_layer_breakdown(plan, m)
    assert "prompt_context" in breakdown
    assert breakdown["prompt_context"] > 0
    # Workspace_AC moves phoenix -> seattle (S2 and S4 both placed at seattle in default toy);
    # cost is artifact_copy > 0.
    assert breakdown.get("workspace", 0.0) > 0.0


def test_duplication_factor_empty_plan_is_one():
    """A plan with no materializations (no required state) returns 1.0 by convention."""
    from vagrant_agent.policies import Plan
    plan = Plan(policy="empty", placements=[], materializations=[])
    assert cost_weighted_duplication_factor(plan) == 1.0
