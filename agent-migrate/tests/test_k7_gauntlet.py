"""
K7 — falsification gauntlet for the mobility_episode pivot.

Per K0 calibration writeup, the pivot to mobility episodes is
provisional until ALL THREE tests pass:

  T1 (simulator correctness check / capacity_free collapse):
    Under infinite capacity, mixed_min_pressure ≈ cache_reuse ≈ H1
    within numerical noise. Failure indicates a K4 simulator bug.

  T2 (prefill_stampede falsification of design intent):
    Under prefill cap only, replay_all stampedes, mixed_min_pressure
    diversifies, and mixed.p50 < replay_all.p50 by ≥10%.

  T3 (multi_resource bottleneck falsification of design intent):
    Under all three caps (network + prefill + workspace), mixed beats
    every fixed_mode policy by ≥10% on p50 time_to_resume.

Honest framing per audit_honesty critic: T1 is a tautology (correctness
check), and T2/T3 use procedurally_generated K6 fixtures, so the K7
pass is necessary but not sufficient for any external claim about real
production workloads. See `docs/K0_calibration.md` and
`docs/L1_calibration_paper_draft.md` for the broader framing.
"""
from __future__ import annotations

import math
from pathlib import Path

import pytest

from agent_migrate_agent.adapters.herd import HerdSpec, build_herd_episode
from agent_migrate_agent.episode import dump_episode
from agent_migrate_agent.fluid_sim import simulate_fluid
from agent_migrate_agent.profiles import load_bundle
from agent_migrate_agent.reconstitution import RECONSTITUTION_POLICIES
from agent_migrate_agent.resources import ResourceBudget
from agent_migrate_agent.warmness import WarmnessMap

REPO = Path(__file__).resolve().parent.parent
MODELS = REPO / "configs" / "model_profiles.yaml"
SITES_3 = REPO / "configs" / "sites_3site.yaml"
EPISODES_DIR = REPO / "examples" / "episodes"

# Number of workflows in the gauntlet.
N_HERD = 100


def _bundle():
    return load_bundle(MODELS, SITES_3, "compact_kv")


def _run_all_policies(episode, manifests, budget) -> dict[str, float]:
    """Run every K5 policy on (episode, manifests, budget), return
    dict[policy_name -> p50 time_to_resume]."""
    bundle = _bundle()
    out: dict[str, float] = {}
    for policy_name, policy_fn in RECONSTITUTION_POLICIES.items():
        if policy_name == "random_mode":
            plan = policy_fn(episode, manifests, bundle,
                             WarmnessMap.from_episode_seed(episode.state_warmness),
                             budget, seed=0)
        else:
            plan = policy_fn(episode, manifests, bundle,
                             WarmnessMap.from_episode_seed(episode.state_warmness),
                             budget)
        result = simulate_fluid(
            episode, manifests, plan, bundle,
            WarmnessMap.from_episode_seed(episode.state_warmness), budget,
        )
        out[policy_name] = result.p50_resume_s()
    return out


# ---------------------------------------------------------------------------
# T1 — Simulator correctness check (capacity_free collapse)
# ---------------------------------------------------------------------------


def _t1_episode_and_manifests():
    spec = HerdSpec(
        n_workflows=N_HERD,
        workspace_bytes_distribution="tiny",
        prompt_tokens_distribution="medium",
        warm_cache_fraction=0.0,
        home_asymmetry="balanced",  # distributed_origin
        seed=1,
    )
    episode, manifests = build_herd_episode(
        spec,
        source_sites=("phoenix", "seattle", "austin"),
        destination_sites=("phoenix", "seattle", "austin"),
        episode_id="gauntlet_t1_infinite_capacity",
    )
    EPISODES_DIR.mkdir(parents=True, exist_ok=True)
    dump_episode(episode, EPISODES_DIR / "gauntlet_t1_infinite_capacity.json")
    return episode, manifests


def test_t1_capacity_free_collapse():
    """T1: under math.inf capacity on every axis, all policies should
    converge to a near_zero makespan (everything is free, parallelism
    is unlimited). cache_reuse and mixed_min_pressure must agree on the
    p50 within numerical noise."""
    episode, manifests = _t1_episode_and_manifests()
    budget = ResourceBudget.infinite(["phoenix", "seattle", "austin"])
    results = _run_all_policies(episode, manifests, budget)

    # The "L3 vs L1" distinction must vanish. mixed_min_pressure ≈ cache_reuse.
    # Under truly infinite capacity, both should produce p50 = 0.0.
    assert results["mixed_min_pressure"] == pytest.approx(results["cache_reuse"], abs=1e-6), (
        f"T1 FAIL: mixed_min_pressure ({results['mixed_min_pressure']:.6e}) "
        f"!= cache_reuse ({results['cache_reuse']:.6e}) under infinite capacity. "
        f"Simulator may be smuggling in an effect (likely K3 cost or K4 share computation)."
    )
    # Under math.inf capacity, every action's wallclock should be 0
    # (network cost = 0/inf = 0, prefill = T/inf = 0, etc.).
    for policy_name, p50 in results.items():
        assert p50 == pytest.approx(0.0, abs=1e-6), (
            f"T1 FAIL: policy {policy_name!r} p50 = {p50:.6e} under infinite "
            f"capacity; should be ~0. Simulator bug in {policy_name}."
        )


# ---------------------------------------------------------------------------
# T2 — Prefill_stampede falsification
# ---------------------------------------------------------------------------


def _t2_episode_and_manifests():
    spec = HerdSpec(
        n_workflows=N_HERD,
        workspace_bytes_distribution="tiny",
        prompt_tokens_distribution="medium",  # ~10K tokens per workflow
        warm_cache_fraction=0.0,
        home_asymmetry="all_same",  # single_source_evacuation (per A2)
        seed=2,
    )
    episode, manifests = build_herd_episode(
        spec,
        source_sites=("phoenix",),
        destination_sites=("seattle",),
        episode_id="gauntlet_t2_prefill_only",
    )
    dump_episode(episode, EPISODES_DIR / "gauntlet_t2_prefill_only.json")
    return episode, manifests


def test_t2_prefill_stampede():
    """T2: under prefill cap of 30K tok/s at one site, infinite network
    and workspace, replay_all stampedes prefill (every workflow's prompt
    replays at the same site, dividing prefill capacity by N). kv_all
    bypasses prefill via wire transfer (free under infinite network).
    budget_aware planners can avoid replay stampede by choosing KV transfer
    when the wire is unconstrained."""
    episode, manifests = _t2_episode_and_manifests()
    budget = ResourceBudget(
        network_bps_per_link={
            tuple(sorted(["phoenix", "seattle"])): math.inf,
            tuple(sorted(["phoenix", "austin"])): math.inf,
            tuple(sorted(["seattle", "austin"])): math.inf,
        },
        prefill_tok_s_per_site={"phoenix": 30000.0, "seattle": 30000.0, "austin": 30000.0},
        workspace_hydrate_bps_per_site={"phoenix": math.inf, "seattle": math.inf, "austin": math.inf},
        kv_memory_bytes_per_site={"phoenix": math.inf, "seattle": math.inf, "austin": math.inf},
    )
    results = _run_all_policies(episode, manifests, budget)

    # Expectations:
    #   replay_all stampedes prefill at seattle. p50 should be high.
    #   kv_all bypasses prefill (network=inf). p50 should be near zero.
    #   mixed_min_pressure avoids prefill load. p50 should be < 90% of replay_all.
    assert results["replay_all"] > results["kv_all"] + 1e-3, (
        f"T2 FAIL: replay_all ({results['replay_all']:.4f}s) does not exceed "
        f"kv_all ({results['kv_all']:.4f}s). Prefill cap should differentiate "
        f"these policies; if not, K3 prefill_tokens isn't being applied or "
        f"K4 share computation is wrong."
    )
    assert results["mixed_min_pressure"] < 0.9 * results["replay_all"] - 1e-6, (
        f"T2 FAIL: mixed_min_pressure ({results['mixed_min_pressure']:.4f}s) does "
        f"not beat replay_all ({results['replay_all']:.4f}s) by at least 10%. "
            f"Budget_aware mode choice should avoid prefill load. If this fails, "
            f"either mixed's budget use is broken or replay_all isn't actually "
        f"stampeding (check that all workflows route to the same dst)."
    )


# ---------------------------------------------------------------------------
# T3 — Multi_resource bottleneck falsification
# ---------------------------------------------------------------------------


def _t3_episode_and_manifests():
    spec = HerdSpec(
        n_workflows=N_HERD,
        workspace_bytes_distribution="medium",  # ~500 MB workspaces
        prompt_tokens_distribution="medium",
        warm_cache_fraction=0.0,
        home_asymmetry="all_same",
        seed=3,
    )
    episode, manifests = build_herd_episode(
        spec,
        source_sites=("phoenix",),
        destination_sites=("seattle", "austin"),
        episode_id="gauntlet_t3_multi_resource",
    )
    dump_episode(episode, EPISODES_DIR / "gauntlet_t3_multi_resource.json")
    return episode, manifests


def test_t3_multi_resource_bottleneck():
    """T3: under network=5e9, prefill=30K tok/s, workspace_hydrate=1e9
    bps (matching the canonical sites_3site config but with stricter
    caps), every fixed_mode policy stampedes one resource. mixed_min_
    pressure spreads modes + destinations and should beat the worst
    fixed_mode by >=10%.

    Caveat per architectural critic: the win includes both mode choice and
    destination_load balancing. The project explicitly accepts that T3
    conflates those effects; isolating mode_mixing alone is a follow_on task."""
    episode, manifests = _t3_episode_and_manifests()
    budget = ResourceBudget(
        network_bps_per_link={
            tuple(sorted(["phoenix", "seattle"])): 5e9,
            tuple(sorted(["phoenix", "austin"])): 5e9,
            tuple(sorted(["seattle", "austin"])): 5e9,
        },
        prefill_tok_s_per_site={"phoenix": 30000.0, "seattle": 30000.0, "austin": 30000.0},
        workspace_hydrate_bps_per_site={"phoenix": 1e9, "seattle": 1e9, "austin": 1e9},
        kv_memory_bytes_per_site={"phoenix": math.inf, "seattle": 1.0e11, "austin": 1.0e11},
    )
    results = _run_all_policies(episode, manifests, budget)

    # mixed must beat every fixed_mode policy by 10%.
    fixed_mode_policies = ["replay_all", "kv_all", "cache_reuse", "workspace_sticky"]
    worst_fixed = min(results[p] for p in fixed_mode_policies)
    assert results["mixed_min_pressure"] < 0.9 * worst_fixed - 1e-6, (
        f"T3 FAIL: mixed_min_pressure ({results['mixed_min_pressure']:.4f}s) "
        f"does not beat best_fixed_mode ({worst_fixed:.4f}s) by at least 10%. "
        f"Per_policy results: {results}. If mixed is winning by destination-"
        f"load_balancing alone, that's expected per K0 — but the >=10% bar "
        f"should still be cleared on the canonical fixture."
    )

    # Sanity: mixed beats random_mode by a meaningful margin.
    assert results["mixed_min_pressure"] < results["random_mode"], (
        f"T3 FAIL: mixed_min_pressure ({results['mixed_min_pressure']:.4f}s) "
        f"does not beat random_mode ({results['random_mode']:.4f}s). The "
        f"diversification heuristic is no better than chance."
    )


# ---------------------------------------------------------------------------
# Summary fixture: emits CSV for K7 results writeup.
# ---------------------------------------------------------------------------


def test_emit_gauntlet_results_csv():
    """Run the three gauntlet fixtures and emit a summary CSV. Always
    runs (after individual tests). Writes to runs/k7_gauntlet/."""
    out_dir = REPO / "runs" / "k7_gauntlet"
    out_dir.mkdir(parents=True, exist_ok=True)
    csv_path = out_dir / "gauntlet_results.csv"

    rows = []
    for test_name, builder, budget_kind in (
        ("T1", _t1_episode_and_manifests,
         ResourceBudget.infinite(["phoenix", "seattle", "austin"])),
        ("T2", _t2_episode_and_manifests, "t2_prefill_only"),
        ("T3", _t3_episode_and_manifests, "t3_multi_resource"),
    ):
        if test_name == "T2":
            budget = ResourceBudget(
                network_bps_per_link={
                    tuple(sorted(["phoenix", "seattle"])): math.inf,
                    tuple(sorted(["phoenix", "austin"])): math.inf,
                    tuple(sorted(["seattle", "austin"])): math.inf,
                },
                prefill_tok_s_per_site={"phoenix": 30000.0, "seattle": 30000.0, "austin": 30000.0},
                workspace_hydrate_bps_per_site={"phoenix": math.inf, "seattle": math.inf, "austin": math.inf},
                kv_memory_bytes_per_site={"phoenix": math.inf, "seattle": 1.0e11, "austin": 1.0e11},
            )
        elif test_name == "T3":
            budget = ResourceBudget(
                network_bps_per_link={
                    tuple(sorted(["phoenix", "seattle"])): 5e9,
                    tuple(sorted(["phoenix", "austin"])): 5e9,
                    tuple(sorted(["seattle", "austin"])): 5e9,
                },
                prefill_tok_s_per_site={"phoenix": 30000.0, "seattle": 30000.0, "austin": 30000.0},
                workspace_hydrate_bps_per_site={"phoenix": 1e9, "seattle": 1e9, "austin": 1e9},
                kv_memory_bytes_per_site={"phoenix": math.inf, "seattle": math.inf, "austin": math.inf},
            )
        else:
            budget = budget_kind
        episode, manifests = builder()
        results = _run_all_policies(episode, manifests, budget)
        for policy, p50 in results.items():
            rows.append((test_name, policy, p50))

    with csv_path.open("w") as f:
        f.write("test,policy,p50_time_to_resume_s\n")
        for test_name, policy, p50 in rows:
            f.write(f"{test_name},{policy},{p50:.9g}\n")

    assert csv_path.exists()
    assert len(rows) == 3 * len(RECONSTITUTION_POLICIES)
