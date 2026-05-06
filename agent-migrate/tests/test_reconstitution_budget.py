"""
Claim:
K5 policies that make cost or pressure decisions must plan against the same
ResourceBudget that K4 will simulate, not only the static ProfileBundle.

Plausible wrong implementations:
- `mixed_min_pressure` ignores the explicit ResourceBudget and keeps using
  site/link capacities from the YAML profile.
- The policy uses the right resource dimensions but the wrong unit for one
  axis, so changing a budget does not flip the selected mode.
- Optional budget plumbing works in tests but older callers without a budget
  no longer receive a valid plan.
"""
from __future__ import annotations

import math
from pathlib import Path

from agent_migrate_agent.episode import MobilityEpisode, Workflow
from agent_migrate_agent.manifest import ServingGroupManifest, StateObject, WorkNode
from agent_migrate_agent.profiles import load_bundle
from agent_migrate_agent.reconstitution import mixed_min_pressure
from agent_migrate_agent.resources import ResourceBudget
from agent_migrate_agent.warmness import WarmnessMap

REPO = Path(__file__).resolve().parent.parent
MODELS = REPO / "configs" / "model_profiles.yaml"
SITES_3 = REPO / "configs" / "sites_3site.yaml"


def _one_prompt_episode():
    state = StateObject(
        state_id="prompt",
        content_hash="hash_prompt",
        layer="prompt_context",
        lifetime="shared",
        tokens=10_000,
        bytes=None,
    )
    node = WorkNode(
        node_id="n",
        node_type="llm_call",
        parent_node_id=None,
        workflow_id="wf",
        label=None,
        status="complete",
        required_state=["prompt"],
        produced_state=[],
    )
    manifest = ServingGroupManifest(
        workflow_id="wf",
        root_task="test",
        nodes={"n": node},
        state_objects={"prompt": state},
        edges=[],
    )
    episode = MobilityEpisode(
        episode_id="budget_flip",
        source_sites=("phoenix",),
        destination_sites=("seattle",),
        workflows=(Workflow("wf", "<inline>", src_site="phoenix"),),
    )
    return episode, {"wf": manifest}


def test_mixed_min_pressure_uses_explicit_budget_to_choose_prompt_mode():
    bundle = load_bundle(MODELS, SITES_3, "compact_kv")
    episode, manifests = _one_prompt_episode()

    prefill_tight_network_free = ResourceBudget(
        network_bps_per_link={("phoenix", "seattle"): math.inf},
        prefill_tok_s_per_site={"phoenix": math.inf, "seattle": 1000.0},
        workspace_hydrate_bps_per_site={"phoenix": math.inf, "seattle": math.inf},
        kv_memory_bytes_per_site={"phoenix": math.inf, "seattle": math.inf},
    )
    plan = mixed_min_pressure(
        episode, manifests, bundle, WarmnessMap.empty(), prefill_tight_network_free,
    )
    assert plan["wf"][0].mode == "kv_transfer"

    network_tight_prefill_free = ResourceBudget(
        network_bps_per_link={("phoenix", "seattle"): 1.0},
        prefill_tok_s_per_site={"phoenix": math.inf, "seattle": math.inf},
        workspace_hydrate_bps_per_site={"phoenix": math.inf, "seattle": math.inf},
        kv_memory_bytes_per_site={"phoenix": math.inf, "seattle": math.inf},
    )
    plan = mixed_min_pressure(
        episode, manifests, bundle, WarmnessMap.empty(), network_tight_prefill_free,
    )
    assert plan["wf"][0].mode == "context_replay"

