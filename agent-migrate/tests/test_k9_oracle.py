from __future__ import annotations

"""
Claim:
K9 is an exact search over its declared restricted candidate space:
workflow_level destination, prompt mode, and workspace mode choices, all
evaluated by the K4 simulator.

Plausible wrong implementations:
- callers cannot override seeded warmness with an explicit empty map;
- candidate counts are reported from the first workflow even when they vary;
- unsupported state layers are silently dropped from oracle plans;
- artifact rows omit scenario metadata needed to compare oracle cells;
- tests imply a global oracle even though action order and per_state choices
  are outside candidate_space v1.
"""

from pathlib import Path

import pytest

from agent_migrate_agent.adapters.herd import HerdSpec, build_herd_episode
from agent_migrate_agent.k8_regime import RegimeCell, default_bundle, make_k8_budget
from agent_migrate_agent.k9_oracle import run_small_n_oracle
from agent_migrate_agent.warmness import WarmnessMap


REPO = Path(__file__).resolve().parent.parent


def test_k9_oracle_enumerates_small_instance_and_bounds_heuristics():
    """Claim: K9 is an exact simulator_backed oracle on small N, so its
    p50 can expose headroom over strong reuse or the mixed heuristic in a
    controlled cell."""
    bundle = default_bundle(REPO)
    episode, manifests = build_herd_episode(
        HerdSpec(
            n_workflows=3,
            workspace_bytes_distribution="medium",
            prompt_tokens_distribution="medium",
            warm_cache_fraction=0.0,
            home_asymmetry="all_same",
            seed=9001,
        ),
        source_sites=("phoenix",),
        destination_sites=("seattle", "austin"),
        episode_id="test_k9_oracle",
    )
    budget = make_k8_budget(RegimeCell(
        n_workflows=3,
        state_scale="medium",
        prefill_capacity="tight",
        link_gbps=5,
    ))
    result = run_small_n_oracle(episode, manifests, bundle, budget, max_workflows=3)
    assert result.n_workflows == 3
    assert result.min_candidates_per_workflow == 8
    assert result.max_candidates_per_workflow == 8
    assert result.enumerated_plans == 8 ** 3
    assert result.oracle_p50_resume_s <= result.strong_reuse_p50_resume_s
    assert result.oracle_p50_resume_s <= result.mixed_p50_resume_s


def test_k9_oracle_refuses_large_exact_search():
    bundle = default_bundle(REPO)
    episode, manifests = build_herd_episode(
        HerdSpec(n_workflows=4, seed=9002),
        source_sites=("phoenix",),
        destination_sites=("seattle", "austin"),
    )
    budget = make_k8_budget(RegimeCell(
        n_workflows=4,
        state_scale="tiny",
        prefill_capacity="loose",
        link_gbps=100,
    ))
    try:
        run_small_n_oracle(episode, manifests, bundle, budget, max_workflows=3)
    except ValueError as exc:
        assert "exponential" in str(exc)
    else:
        raise AssertionError("expected exact K9 oracle to reject oversized instance")


def test_k9_oracle_respects_explicit_empty_warmness_override():
    bundle = default_bundle(REPO)
    episode, manifests = build_herd_episode(
        HerdSpec(n_workflows=2, warm_cache_fraction=1.0, seed=9003),
        source_sites=("phoenix",),
        destination_sites=("seattle", "austin"),
    )
    budget = make_k8_budget(RegimeCell(
        n_workflows=2,
        state_scale="tiny",
        prefill_capacity="loose",
        link_gbps=100,
    ))
    seeded = run_small_n_oracle(episode, manifests, bundle, budget, max_workflows=2)
    cold = run_small_n_oracle(
        episode, manifests, bundle, budget, WarmnessMap.empty(), max_workflows=2,
    )
    assert seeded.oracle_p50_resume_s == 0.0
    assert cold.oracle_p50_resume_s > 0.0


def test_k9_oracle_rejects_unsupported_state_layers():
    bundle = default_bundle(REPO)
    episode, manifests = build_herd_episode(
        HerdSpec(n_workflows=1, seed=9004),
        source_sites=("phoenix",),
        destination_sites=("seattle",),
    )
    manifest = next(iter(manifests.values()))
    state = next(iter(manifest.state_objects.values()))
    state.layer = "semantic"
    budget = make_k8_budget(RegimeCell(
        n_workflows=1,
        state_scale="tiny",
        prefill_capacity="loose",
        link_gbps=100,
    ))
    with pytest.raises(ValueError, match="candidate_space v1"):
        run_small_n_oracle(episode, manifests, bundle, budget, max_workflows=1)
