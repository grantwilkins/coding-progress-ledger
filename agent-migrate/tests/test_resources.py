"""
Claim:
ResourceCost is the per_action 5_tuple (network_bytes, prefill_tokens,
workspace_bytes, kv_resident_bytes, wallclock_s) that K3 produces and
K4 consumes. reconstitution_cost composes costs.materialize_cost for
wallclock_s and derives the four resource fields from state + mode +
bundle.

Per A3: warm cache hit at dst short_circuits to network=prefill=workspace=0.
kv_resident_bytes is preserved (state still occupies KV memory).

Per A4: wallclock_s is additive (no max(transfer, prefill) pipelining).
The four resource dims are tracked separately so a future M_workstream
can compute pipelined wall_clock without re_deriving anything.
"""
from __future__ import annotations

import math
from pathlib import Path

import pytest
from ledger_progress import from_jsonl

from agent_migrate_agent import build_manifest
from agent_migrate_agent.costs import (
    ARTIFACT_COPY,
    CONTEXT_REPLAY,
    KV_TRANSFER,
    TEXT_TRANSFER,
    bandwidth_crossover_bps,
    materialize_cost,
)
from agent_migrate_agent.manifest import StateObject
from agent_migrate_agent.profiles import load_bundle
from agent_migrate_agent.resources import (
    ResourceBudget,
    ResourceCost,
    WORKSPACE_HYDRATE,
    WARM_REUSE,
    crossover_parity_check,
    reconstitution_cost,
)
from agent_migrate_agent.warmness import WarmnessMap

REPO = Path(__file__).resolve().parent.parent
MODELS = REPO / "configs" / "model_profiles.yaml"
SITES = REPO / "configs" / "sites_2site.yaml"


def _bundle():
    return load_bundle(MODELS, SITES, "compact_kv")


def _prompt_state(tokens: int = 1000) -> StateObject:
    return StateObject(
        state_id="sx", content_hash="hx", layer="prompt_context",
        lifetime="shared", tokens=tokens, bytes=None,
    )


def _workspace_state(bytes_: int = 1_000_000_000) -> StateObject:
    return StateObject(
        state_id="ws", content_hash="hws", layer="workspace",
        lifetime="private", tokens=0, bytes=bytes_,
    )


# ---------------------------------------------------------------------------
# ResourceCost construction
# ---------------------------------------------------------------------------


def test_resource_cost_zero():
    z = ResourceCost.zero()
    assert (z.network_bytes, z.prefill_tokens, z.workspace_bytes,
            z.kv_resident_bytes, z.wallclock_s) == (0, 0, 0, 0, 0.0)


def test_resource_cost_addition():
    a = ResourceCost(1, 2, 3, 4, 5.0)
    b = ResourceCost(10, 20, 30, 40, 50.0)
    c = a + b
    assert c == ResourceCost(11, 22, 33, 44, 55.0)


# ---------------------------------------------------------------------------
# Per_mode resource consumption
# ---------------------------------------------------------------------------


def test_kv_transfer_charges_network_only_cross_site():
    s = _prompt_state(tokens=1000)
    b = _bundle()
    wm = WarmnessMap.empty()
    rc = reconstitution_cost(s, KV_TRANSFER, "phoenix", "seattle", b, wm)
    expected_bytes = 1000 * b.model.kv_bytes_per_token
    assert rc.network_bytes == expected_bytes
    assert rc.prefill_tokens == 0
    assert rc.workspace_bytes == 0
    assert rc.kv_resident_bytes == expected_bytes


def test_kv_transfer_no_network_same_site():
    """Same_site has no wire bytes for any mode; only prefill (CONTEXT_REPLAY) charges."""
    s = _prompt_state(tokens=1000)
    b = _bundle()
    wm = WarmnessMap.empty()
    # Note: KV_TRANSFER same_site is not actually allowed by costs.materialize_cost
    # for prompt_context; the existing code raises. Use CONTEXT_REPLAY for the same-
    # site test instead.
    rc = reconstitution_cost(s, CONTEXT_REPLAY, "phoenix", "phoenix", b, wm)
    assert rc.network_bytes == 0
    assert rc.prefill_tokens == 1000


def test_context_replay_charges_prefill_only():
    s = _prompt_state(tokens=2000)
    b = _bundle()
    rc = reconstitution_cost(s, CONTEXT_REPLAY, "phoenix", "seattle", b, WarmnessMap.empty())
    assert rc.network_bytes == 0
    assert rc.prefill_tokens == 2000
    assert rc.workspace_bytes == 0


def test_artifact_copy_charges_network_and_workspace():
    s = _workspace_state(bytes_=10_000_000)
    b = _bundle()
    rc = reconstitution_cost(s, ARTIFACT_COPY, "phoenix", "seattle", b, WarmnessMap.empty())
    assert rc.network_bytes == 10_000_000
    assert rc.workspace_bytes == 10_000_000
    assert rc.prefill_tokens == 0
    assert rc.kv_resident_bytes == 0


def test_text_transfer_only_network():
    s = StateObject(state_id="m", content_hash="h", layer="memory",
                    lifetime="persistent", tokens=0, bytes=2_000_000)
    b = _bundle()
    rc = reconstitution_cost(s, TEXT_TRANSFER, "phoenix", "seattle", b, WarmnessMap.empty())
    assert rc.network_bytes == 2_000_000
    assert rc.prefill_tokens == 0
    assert rc.workspace_bytes == 0


def test_workspace_hydrate_no_network_workspace_charged():
    s = _workspace_state(bytes_=5_000_000)
    b = _bundle()
    # K3 charges workspace_bytes regardless of src/dst.
    rc = reconstitution_cost(s, WORKSPACE_HYDRATE, "seattle", "seattle", b,
                              WarmnessMap.empty())
    assert rc.network_bytes == 0
    assert rc.workspace_bytes == 5_000_000


def test_workspace_hydrate_uses_bytes_per_second_units():
    """1 GB hydrated by a 1 GB/s site takes 1 second, not 8 seconds."""
    s = _workspace_state(bytes_=1_000_000_000)
    sites_3 = REPO / "configs" / "sites_3site.yaml"
    b = load_bundle(MODELS, sites_3, "compact_kv")
    rc = reconstitution_cost(s, WORKSPACE_HYDRATE, "phoenix", "phoenix", b,
                              WarmnessMap.empty())
    assert rc.wallclock_s == pytest.approx(1.0)


def test_same_site_artifact_copy_does_not_charge_workspace_hydrate():
    """Same_site workspace state is already local; local hydrate is a separate mode."""
    s = _workspace_state(bytes_=1_000_000_000)
    b = _bundle()
    rc = reconstitution_cost(s, ARTIFACT_COPY, "phoenix", "phoenix", b,
                              WarmnessMap.empty())
    assert rc.network_bytes == 0
    assert rc.workspace_bytes == 0
    assert rc.wallclock_s == 0.0


# ---------------------------------------------------------------------------
# Warm cache short_circuit (L1 dedup)
# ---------------------------------------------------------------------------


def test_warm_cache_short_circuits_to_zero_for_workspace():
    """Workspace state on a warm site costs 0 across all dims (no
    KV_resident contribution, since it's not a prompt_context state)."""
    s = _workspace_state(bytes_=1_000_000_000)
    b = _bundle()
    wm = WarmnessMap.from_dict({"ws": ["seattle"]})
    rc = reconstitution_cost(s, ARTIFACT_COPY, "phoenix", "seattle", b, wm)
    assert rc == ResourceCost.zero()


def test_warm_cache_preserves_kv_resident_for_prompt_context():
    """A prompt_context state warm at dst still occupies KV memory there.
    K3 must charge kv_resident_bytes so K4's kv_memory pressure
    accounting is correct."""
    s = _prompt_state(tokens=8000)
    b = _bundle()
    wm = WarmnessMap.from_dict({"sx": ["seattle"]})
    rc = reconstitution_cost(s, CONTEXT_REPLAY, "phoenix", "seattle", b, wm)
    assert rc.network_bytes == 0
    assert rc.prefill_tokens == 0
    assert rc.workspace_bytes == 0
    assert rc.kv_resident_bytes == 8000 * b.model.kv_bytes_per_token
    assert rc.wallclock_s == 0.0


def test_warm_reuse_on_cold_state_raises():
    """warm_reuse without a warm cache entry is a caller bug."""
    s = _prompt_state(tokens=1000)
    b = _bundle()
    with pytest.raises(ValueError, match="warm_reuse on cold"):
        reconstitution_cost(s, WARM_REUSE, "phoenix", "seattle", b, WarmnessMap.empty())


# ---------------------------------------------------------------------------
# wallclock_s parity with the existing materialize_cost
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("mode", [KV_TRANSFER, CONTEXT_REPLAY])
def test_wallclock_matches_materialize_cost_prompt(mode):
    s = _prompt_state(tokens=4000)
    b = _bundle()
    rc = reconstitution_cost(s, mode, "phoenix", "seattle", b, WarmnessMap.empty())
    assert rc.wallclock_s == pytest.approx(materialize_cost(s, mode, "phoenix", "seattle", b))


def test_wallclock_matches_materialize_cost_workspace():
    s = _workspace_state(bytes_=50_000_000)
    b = _bundle()
    rc = reconstitution_cost(s, ARTIFACT_COPY, "phoenix", "seattle", b, WarmnessMap.empty())
    assert rc.wallclock_s == pytest.approx(
        materialize_cost(s, ARTIFACT_COPY, "phoenix", "seattle", b)
    )


# ---------------------------------------------------------------------------
# Crossover parity with costs.bandwidth_crossover_bps
# ---------------------------------------------------------------------------


def test_crossover_parity_check_returns_consistent_value():
    s = _prompt_state(tokens=1000)
    b = _bundle()
    a, c = crossover_parity_check(s, b)
    assert a == pytest.approx(c)
    expected = bandwidth_crossover_bps(b.model.kv_bytes_per_token,
                                       max(sp.prefill_tok_s for sp in b.sites.values()))
    assert a == pytest.approx(expected)


# ---------------------------------------------------------------------------
# ResourceBudget construction
# ---------------------------------------------------------------------------


def test_resource_budget_from_bundle_2site():
    b = _bundle()
    budget = ResourceBudget.from_bundle(b)
    assert tuple(sorted(budget.network_bps_per_link.keys())) == (("phoenix", "seattle"),)
    assert budget.prefill_tok_s_per_site["phoenix"] == 30000
    assert budget.prefill_tok_s_per_site["seattle"] == 45000
    # 2_site MVP doesn't set workspace_hydrate_bps / kv_memory_bytes -> defaults to inf
    assert budget.workspace_hydrate_bps_per_site["phoenix"] == math.inf
    assert budget.kv_memory_bytes_per_site["phoenix"] == math.inf


def test_resource_budget_infinite():
    budget = ResourceBudget.infinite(["phoenix", "seattle", "austin"])
    for v in budget.prefill_tok_s_per_site.values():
        assert v == math.inf
    for v in budget.workspace_hydrate_bps_per_site.values():
        assert v == math.inf
    for v in budget.kv_memory_bytes_per_site.values():
        assert v == math.inf
    # All_pairs links present.
    assert (("austin", "phoenix")) in budget.network_bps_per_link
    assert (("austin", "seattle")) in budget.network_bps_per_link
    assert (("phoenix", "seattle")) in budget.network_bps_per_link


def test_resource_budget_3site_yaml_picks_up_capacities():
    """The new sites_3site.yaml sets workspace_hydrate_bps and
    kv_memory_bytes explicitly; from_bundle should pick them up."""
    sites_3 = REPO / "configs" / "sites_3site.yaml"
    b = load_bundle(MODELS, sites_3, "compact_kv")
    budget = ResourceBudget.from_bundle(b)
    assert budget.prefill_tok_s_per_site["austin"] == 22500
    assert budget.workspace_hydrate_bps_per_site["austin"] == 4_000_000_000
    assert budget.kv_memory_bytes_per_site["austin"] == 50_000_000_000
    # Phoenix (older 2_site config has no capacity fields) -> inf in 2_site,
    # but explicit value in 3_site:
    assert budget.workspace_hydrate_bps_per_site["phoenix"] == 1_000_000_000


# ---------------------------------------------------------------------------
# Resource conservation invariant (K3 + manifest)
# ---------------------------------------------------------------------------


def test_resource_conservation_total_network_bytes():
    """Sum of network_bytes across all reconstitution actions for a
    plan equals manifest's transferred_byte budget. We approximate by
    summing per_state at a single (src, dst) pair under one mode."""
    m = build_manifest(from_jsonl(str(REPO / "examples" / "traces" / "h2_multi_session_swe.jsonl")))
    b = _bundle()
    wm = WarmnessMap.empty()

    total_kv_bytes = 0
    for state in m.state_objects.values():
        if state.layer == "prompt_context":
            rc = reconstitution_cost(state, KV_TRANSFER, "phoenix", "seattle", b, wm)
            total_kv_bytes += rc.network_bytes

    # KV bytes scale with total prompt_context tokens.
    expected = sum(s.tokens * b.model.kv_bytes_per_token
                   for s in m.state_objects.values() if s.layer == "prompt_context")
    assert total_kv_bytes == expected
