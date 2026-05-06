"""Tests for src/vagrant_agent/workloads.py (W1, W2, W3).

Claim:
W1/W2/W3 are workload anchors that (a) carry an S1-taxonomy state-layer
breakdown with mobility classes, (b) deterministically build a
MobilityEpisode + manifests sized by their breakdown, and (c) are
classifiable into one of the four regimes by replaying through K4.

The load-bearing structural invariants:

  - W1's workspace bytes = base_repo + dep_cache + build_artifacts +
    uncommitted_diff (test_logs are discarded; tool_output is prompt).
  - W2's workspace bytes = retrieved_documents + cleaned_intermediates,
    with `globally_available` layers reflected as initial warmness on
    the shared system_prompt at every destination.
  - W3 has K subagents per workflow that all consume the same
    `shared_task_<wid>` state — without this, cache_reuse cannot
    collapse the shared work and the "reuse regime" hypothesis is
    structurally unfounded.

Plausible wrong implementations:

  - Workspace byte aggregator picks the wrong subset of layers
    (e.g., test_logs counted, uncommitted_diff dropped, or simply
    `total_bytes_per_workflow` used as a shortcut).
  - W3 builds K subagent nodes without reading the shared state, so
    cache_reuse degenerates to per-(state, site) work that scales like
    K * private_size (would silently flip the regime hypothesis).
  - n_workflows scaling: total_bytes_per_layer multiplies by N+1 or
    forgets to multiply at all (right-formula-wrong-level mistake).
  - classify_regime check order swallows a workspace bottleneck:
    e.g., asks `if bottleneck == network` first and never reaches
    workspace branch, mislabeling state_locality as reuse.
  - mobility_class typos accepted silently (e.g., "globaly_available"
    in the layer table); breakdown then reports nonsense regime.
  - Determinism: build_episode on the same seed produces a different
    episode_id or different state_warmness shape.
"""
from __future__ import annotations

import math
from pathlib import Path

import pytest

from vagrant_agent.k8_regime import default_bundle, make_k8_budget, RegimeCell
from vagrant_agent.reconstitution import cache_reuse, mixed_min_pressure
from vagrant_agent.fluid_sim import simulate_fluid
from vagrant_agent.warmness import WarmnessMap
from vagrant_agent.workloads import (
    ANCHORS,
    MOBILITY_CLASSES,
    REGIMES,
    StateLayer,
    W1_LARGE_REPO_CODING,
    W2_DATA_RAG_HEAVY,
    W3_MULTI_AGENT_FANOUT,
    WorkloadAnchor,
    classify_regime,
)


REPO = Path(__file__).resolve().parents[1]


# ---------------------------------------------------------------------------
# Taxonomy invariants
# ---------------------------------------------------------------------------


def test_mobility_class_typo_rejected():
    """A typo'd mobility class would silently miscategorize a layer."""
    with pytest.raises(ValueError):
        StateLayer(
            name="x", bytes_per_workflow=1,
            mobility_class="globaly_available",   # missing 'l'
        )


def test_anchor_rejects_unknown_regime():
    with pytest.raises(ValueError):
        WorkloadAnchor(
            name="bad", description="",
            state_layers=(StateLayer("x", 1, "must_move"),),
            regime_hypothesis="latency",          # not in REGIMES
            builder=lambda *a, **k: None,        # not invoked
        )


def test_registry_lists_three_anchors_with_distinct_regimes():
    assert set(ANCHORS) == {
        "w1_large_repo_coding",
        "w2_data_rag_heavy",
        "w3_multi_agent_fanout",
    }
    # All hypothesized regimes are valid.
    for anchor in ANCHORS.values():
        assert anchor.regime_hypothesis in REGIMES


# ---------------------------------------------------------------------------
# Byte accounting (the right-formula-wrong-level family)
# ---------------------------------------------------------------------------


def test_must_move_bytes_only_count_must_move_layers():
    """Sum is over must_move only — globally_available and discardable
    layers must NOT contribute, even though they are physically large."""
    anchor = W1_LARGE_REPO_CODING
    must_move = anchor.must_move_bytes_per_workflow()
    expected = sum(
        layer.bytes_per_workflow for layer in anchor.state_layers
        if layer.mobility_class == "must_move"
    )
    assert must_move == expected
    # Sanity: must_move strictly less than total (W1 has at least one
    # globally_available layer, the base repo).
    assert must_move < anchor.total_bytes_per_workflow()


def test_total_bytes_per_layer_scales_linearly_in_n_workflows():
    """n -> 2n must double every layer total. A common bug is forgetting
    the multiplication or using N+1 / N-1 boundaries."""
    anchor = W2_DATA_RAG_HEAVY
    one = anchor.total_bytes_per_layer(1)
    two = anchor.total_bytes_per_layer(2)
    seven = anchor.total_bytes_per_layer(7)
    for layer_name, byte_count in one.items():
        assert two[layer_name] == 2 * byte_count, layer_name
        assert seven[layer_name] == 7 * byte_count, layer_name


def test_n_workflows_zero_rejected_in_total_layer_query():
    with pytest.raises(ValueError):
        W1_LARGE_REPO_CODING.total_bytes_per_layer(0)


# ---------------------------------------------------------------------------
# W1 — workspace-byte composition (the load-bearing aggregation rule)
# ---------------------------------------------------------------------------


def test_w1_workspace_bytes_exclude_discardable_and_prompt_layers():
    """The workspace state.bytes must equal base_repo + dep_cache +
    build_artifacts + uncommitted_diff — and must NOT include test_logs
    (can_be_discarded) or tool_output_context (prompt-context)."""
    episode, manifests = W1_LARGE_REPO_CODING.build_episode(
        n_workflows=2, seed=1,
    )

    layers_by_name = {layer.name: layer for layer in W1_LARGE_REPO_CODING.state_layers}
    expected_workspace = sum(
        layers_by_name[name].bytes_per_workflow
        for name in (
            "base_repo_checkout",
            "dependency_cache",
            "build_artifacts",
            "uncommitted_diff",
        )
    )
    forbidden_in_workspace = {
        layers_by_name["test_logs"].bytes_per_workflow,
        layers_by_name["tool_output_context"].bytes_per_workflow,
    }
    workspace_states_seen = 0
    for manifest in manifests.values():
        for state in manifest.state_objects.values():
            if state.layer != "workspace":
                continue
            workspace_states_seen += 1
            assert state.bytes == expected_workspace, (
                f"workspace bytes {state.bytes} != expected {expected_workspace}"
            )
            # Must not equal test_logs or tool_output alone — would mean
            # the aggregator picked the wrong subset.
            assert state.bytes not in forbidden_in_workspace
    assert workspace_states_seen >= 2, "expected >=1 workspace state per workflow"


# ---------------------------------------------------------------------------
# W2 — globally_available layers materialize as initial warmness
# ---------------------------------------------------------------------------


def test_w2_globally_available_layers_present_as_warm_state_objects():
    """globally_available layers must enter the manifest as real state
    objects with initial warmness at every destination — otherwise the
    `globally_available` claim is unfalsifiable and a buggy policy that
    ignores warmness would not be caught.
    """
    destinations = ("seattle", "austin")
    sources = ("phoenix",)
    episode, manifests = W2_DATA_RAG_HEAVY.build_episode(
        n_workflows=3, source_sites=sources, destination_sites=destinations, seed=2,
    )
    expected_warm_sites = set(destinations) | set(sources)
    for sid in ("global_data_bundle", "global_vector_index"):
        assert sid in episode.state_warmness, sid
        assert set(episode.state_warmness[sid]) == expected_warm_sites
    layers_by_name = {l.name: l for l in W2_DATA_RAG_HEAVY.state_layers}
    expected_bytes = {
        "global_data_bundle": layers_by_name["base_data_bundle"].bytes_per_workflow,
        "global_vector_index": layers_by_name["vector_index_shards"].bytes_per_workflow,
    }
    for manifest in manifests.values():
        for sid, expected in expected_bytes.items():
            assert sid in manifest.state_objects, (
                f"workflow {manifest.workflow_id} missing globally_available "
                f"state {sid!r} — globally_available claim is paper-only"
            )
            state = manifest.state_objects[sid]
            assert state.bytes == expected, (
                f"{sid} bytes drift: got {state.bytes}, expected {expected}"
            )


def test_w2_workspace_bytes_equals_must_move_artifact_sum():
    """W2's workspace bytes = retrieved_documents + cleaned_intermediates +
    generated_plots; prompt_summaries rides as prompt-context tokens, not
    workspace. Wrong aggregator could include prompt_summaries and inflate
    the workspace, OR drop generated_plots and undercount it."""
    episode, manifests = W2_DATA_RAG_HEAVY.build_episode(n_workflows=2, seed=2)
    layers_by_name = {l.name: l for l in W2_DATA_RAG_HEAVY.state_layers}
    expected = (
        layers_by_name["retrieved_documents"].bytes_per_workflow
        + layers_by_name["cleaned_intermediates"].bytes_per_workflow
        + layers_by_name["generated_plots"].bytes_per_workflow
    )
    forbidden = layers_by_name["prompt_summaries"].bytes_per_workflow
    for manifest in manifests.values():
        # The herd's per-workflow workspace state holds the must_move
        # artifact bytes; the globally_available states have their own,
        # different bytes. We check the per-workflow workspace_<wid>.
        for sid, state in manifest.state_objects.items():
            if state.layer != "workspace":
                continue
            if sid.startswith("workspace_"):
                assert state.bytes == expected
                assert state.bytes != forbidden


def test_w2_anchor_matches_state_locality_hypothesis_under_slow_link():
    """Hypothesis-match regression: a slow-link cell must surface W2 as
    state_locality. Anchor drift would silently flip this."""
    bundle = default_bundle(REPO)
    cell = RegimeCell(
        n_workflows=4, state_scale="medium", prefill_capacity="loose",
        link_gbps=1,
    )
    classification = classify_regime(
        W2_DATA_RAG_HEAVY, bundle, make_k8_budget(cell),
        n_workflows=4, seed=12,
    )
    assert classification.matches_hypothesis, (
        f"W2 anchor regime drifted: hypothesized "
        f"{W2_DATA_RAG_HEAVY.regime_hypothesis!r}, "
        f"observed {classification.observed_regime!r} "
        f"(bottleneck={classification.dominant_bottleneck!r})"
    )


# ---------------------------------------------------------------------------
# W3 — shared-state structure (the reuse hypothesis is structurally founded)
# ---------------------------------------------------------------------------


def test_w3_shared_task_state_is_required_by_all_subagent_nodes():
    """W3's reuse hypothesis depends on subagent nodes ALL consuming the
    same `shared_task_<wid>` state. If they don't, cache_reuse cannot
    collapse the shared work and a 'reuse' verdict would be vacuous.
    """
    episode, manifests = W3_MULTI_AGENT_FANOUT.build_episode(n_workflows=2, seed=3)
    for wid, manifest in manifests.items():
        shared_state_id = f"shared_task_{wid}"
        assert shared_state_id in manifest.state_objects
        subagent_nodes = [
            n for n in manifest.nodes.values() if n.node_type == "subagent"
        ]
        assert len(subagent_nodes) >= 2, (
            f"W3 reuse hypothesis only meaningful with >=2 subagents; "
            f"got {len(subagent_nodes)} in {wid}"
        )
        for node in subagent_nodes:
            assert shared_state_id in node.required_state, (
                f"subagent {node.node_id} does not require shared state "
                f"{shared_state_id}; cache_reuse cannot collapse it"
            )


def test_w3_private_transcripts_are_distinct_per_subagent():
    """K subagents must have K *distinct* private transcript state ids.
    A wrong implementation that reuses a single per-workflow private
    state would silently invalidate the per-subagent isolation claim
    and would also make this anchor's grouping question vacuous."""
    episode, manifests = W3_MULTI_AGENT_FANOUT.build_episode(n_workflows=2, seed=3)
    for wid, manifest in manifests.items():
        private_ids = {
            sid for sid in manifest.state_objects
            if sid.startswith(f"private_{wid}_")
        }
        assert len(private_ids) >= 2, (
            f"workflow {wid} has only {len(private_ids)} private "
            f"transcripts; W3 requires K distinct per-subagent transcripts"
        )


def test_w3_reviewer_node_completes_fanin():
    """The fanin half of fanout/fanin must exist as a reviewer node that
    parents on the planner and reads every subagent's private transcript
    plus the merge buffer state. Without it, `merge_review_buffer` is
    paper-only."""
    _, manifests = W3_MULTI_AGENT_FANOUT.build_episode(n_workflows=2, seed=3)
    for wid, manifest in manifests.items():
        reviewer_id = f"reviewer_{wid}"
        merge_sid = f"merge_{wid}"
        assert reviewer_id in manifest.nodes, f"missing {reviewer_id}"
        reviewer = manifest.nodes[reviewer_id]
        assert reviewer.parent_node_id == f"planner_{wid}", (
            f"reviewer must parent on planner; got {reviewer.parent_node_id}"
        )
        assert merge_sid in reviewer.required_state
        assert merge_sid in manifest.state_objects, (
            "merge_review_buffer paper-only — no merge state in manifest"
        )
        # reviewer must read every subagent's private transcript (fanin)
        private_ids = [
            sid for sid in manifest.state_objects
            if sid.startswith(f"private_{wid}_")
        ]
        for sid in private_ids:
            assert sid in reviewer.required_state, (
                f"reviewer skips {sid}; fanin is incomplete"
            )


def test_w3_global_dep_cache_present_and_warm_at_every_site():
    """globally_available dependency_cache must be a real state object
    with warmness across every (source ∪ destination) site. A buggy
    policy that ignores warmness would otherwise surface 200 MB of
    cross-site transfer per workflow."""
    sources, destinations = ("phoenix",), ("seattle", "austin")
    episode, manifests = W3_MULTI_AGENT_FANOUT.build_episode(
        n_workflows=2, source_sites=sources, destination_sites=destinations, seed=3,
    )
    expected_warm = set(sources) | set(destinations)
    assert "global_dep_cache" in episode.state_warmness
    assert set(episode.state_warmness["global_dep_cache"]) == expected_warm
    for manifest in manifests.values():
        assert "global_dep_cache" in manifest.state_objects
        assert manifest.state_objects["global_dep_cache"].bytes == 200_000_000


def test_w3_anchor_matches_landing_pressure_hypothesis_under_canonical_cell():
    """W3's `landing_pressure` hypothesis must hold under a canonical
    cell — N workflows × (K+2) llm_calls compete for prefill. Anchor
    drift in token sizes or subagent count would silently flip the
    regime; this regression test catches that.
    """
    bundle = default_bundle(REPO)
    cell = RegimeCell(
        n_workflows=4, state_scale="tiny", prefill_capacity="loose",
        link_gbps=25,
    )
    classification = classify_regime(
        W3_MULTI_AGENT_FANOUT, bundle, make_k8_budget(cell),
        n_workflows=4, seed=13,
    )
    assert classification.matches_hypothesis, (
        f"W3 anchor regime drifted: hypothesized "
        f"{W3_MULTI_AGENT_FANOUT.regime_hypothesis!r}, "
        f"observed {classification.observed_regime!r} "
        f"(bottleneck={classification.dominant_bottleneck!r}, "
        f"gap={classification.mixed_vs_strong_gap_frac:.3f})"
    )


def test_w3_strong_reuse_emits_one_action_for_shared_state_per_workflow():
    """Within one workflow, K subagent nodes all read the same
    shared_task_<wid> state. The reuse hypothesis demands cache_reuse
    emits ONE action for that state (not K — once per subagent
    consumer). A wrong implementation that iterates over node-level
    `required_state` instead of state-object-level uniqueness would
    emit K times, which is the structural failure that would invalidate
    the reuse hypothesis on this anchor.
    """
    episode, manifests = W3_MULTI_AGENT_FANOUT.build_episode(
        n_workflows=2,
        source_sites=("phoenix",),
        destination_sites=("seattle",),
        seed=3,
    )
    bundle = default_bundle(REPO)
    cell = RegimeCell(
        n_workflows=2, state_scale="tiny", prefill_capacity="loose",
        link_gbps=100,
    )
    budget = make_k8_budget(cell)
    warmness = WarmnessMap.from_episode_seed(episode.state_warmness)
    plan = cache_reuse(episode, manifests, bundle, warmness, budget)

    for wid, actions in plan.items():
        shared_id = f"shared_task_{wid}"
        shared_actions = [a for a in actions if a.state_id == shared_id]
        assert len(shared_actions) == 1, (
            f"cache_reuse must emit exactly one action for {shared_id}; "
            f"got {len(shared_actions)} (would imply per-subagent duplication)"
        )


# ---------------------------------------------------------------------------
# Episode construction is deterministic
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("anchor_name", ["w1_large_repo_coding", "w2_data_rag_heavy"])
def test_episode_is_deterministic_in_seed(anchor_name):
    anchor = ANCHORS[anchor_name]
    a, _ = anchor.build_episode(n_workflows=4, seed=11)
    b, _ = anchor.build_episode(n_workflows=4, seed=11)
    assert a.episode_id == b.episode_id
    assert a.state_warmness == b.state_warmness
    assert tuple(w.workflow_id for w in a.workflows) == tuple(
        w.workflow_id for w in b.workflows
    )


# ---------------------------------------------------------------------------
# classify_regime — direction tests on the threshold
# ---------------------------------------------------------------------------


def test_classify_regime_state_locality_when_artifact_bytes_dominate():
    """Direction test: with a slow link and a workspace-heavy anchor
    (W1), strong reuse uses ARTIFACT_COPY for the workspace, which
    saturates the network. The classifier must label that as
    `state_locality` (large workspace/artifact transfer is the
    binding cost), not as `landing_pressure` or `reuse`.
    """
    anchor = W1_LARGE_REPO_CODING
    bundle = default_bundle(REPO)
    # Slow 1 Gbps link forces network bottleneck for the ~1 GB workspace.
    cell = RegimeCell(
        n_workflows=4, state_scale="medium", prefill_capacity="loose",
        link_gbps=1,
    )
    classification = classify_regime(
        anchor, bundle, make_k8_budget(cell), n_workflows=4, seed=7,
    )
    assert classification.dominant_bottleneck in {"network", "workspace"}
    assert classification.observed_regime == "state_locality"


def test_w1_anchor_matches_state_locality_hypothesis_under_slow_link():
    """The W1 anchor's hypothesis is `state_locality`; under a slow link
    that hypothesis must hold, otherwise the anchor is miscalibrated.
    Catches an unintentional drift in W1's bytes_per_workflow values
    that would silently flip the anchor's regime."""
    bundle = default_bundle(REPO)
    cell = RegimeCell(
        n_workflows=4, state_scale="medium", prefill_capacity="loose",
        link_gbps=1,
    )
    classification = classify_regime(
        W1_LARGE_REPO_CODING, bundle, make_k8_budget(cell),
        n_workflows=4, seed=11,
    )
    assert classification.matches_hypothesis, (
        f"W1 anchor regime drifted: hypothesized "
        f"{W1_LARGE_REPO_CODING.regime_hypothesis!r}, "
        f"observed {classification.observed_regime!r} "
        f"(bottleneck={classification.dominant_bottleneck!r})"
    )


def test_classify_regime_landing_pressure_when_prefill_dominates():
    """Direction test: tight prefill capacity + a slow enough link that
    cache_reuse's min-cost picker chooses CONTEXT_REPLAY (not KV
    transfer) for the prompt-context states, then K4 saturates prefill
    under N concurrent workflows. classify_regime must label this
    `landing_pressure`. (At 100 Gbps the picker would prefer KV transfer
    and the bottleneck would shift to `network` instead — that boundary
    is exactly what the test guards.)
    """
    anchor = W3_MULTI_AGENT_FANOUT
    bundle = default_bundle(REPO)
    cell = RegimeCell(
        n_workflows=8, state_scale="tiny", prefill_capacity="tight",
        link_gbps=1,
    )
    classification = classify_regime(
        anchor, bundle, make_k8_budget(cell), n_workflows=8, seed=5,
    )
    assert classification.dominant_bottleneck == "prefill"
    assert classification.observed_regime == "landing_pressure"


def test_classify_regime_reuse_when_no_resource_is_pressed():
    """Direction test: with infinite-everywhere budget and a small W3
    fixture, no resource is bound; the gap should be ~0 and the regime
    should be `reuse`."""
    anchor = W3_MULTI_AGENT_FANOUT
    bundle = default_bundle(REPO)
    # Build a budget with infinite capacity in every dimension. K4
    # accepts math.inf via the existing make_k8_budget shape.
    cell = RegimeCell(
        n_workflows=2, state_scale="tiny", prefill_capacity="loose",
        link_gbps=100,
    )
    base = make_k8_budget(cell)
    inf_budget = type(base)(
        network_bps_per_link={k: math.inf for k in base.network_bps_per_link},
        prefill_tok_s_per_site={k: math.inf for k in base.prefill_tok_s_per_site},
        workspace_hydrate_bps_per_site={
            k: math.inf for k in base.workspace_hydrate_bps_per_site
        },
        kv_memory_bytes_per_site=base.kv_memory_bytes_per_site,
    )
    classification = classify_regime(
        anchor, bundle, inf_budget, n_workflows=2, seed=4,
    )
    # With infinite capacity, mixed cannot beat strong by >threshold.
    assert classification.mixed_vs_strong_gap_frac < 0.10
    assert classification.observed_regime == "reuse"
