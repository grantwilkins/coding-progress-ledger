from __future__ import annotations

"""
Claim:
`strong_site_reuse` is a paper_facing alias for the existing K_level
`cache_reuse` strong baseline, not a new policy with different behavior.

Plausible wrong implementations:
- the alias calls a weaker no_reuse or fixed_mode policy;
- the alias is omitted from the registry used by scripts;
- the alias returns the same costs only for cold states but loses warm reuse;
- docs drift and describe a request_level no_reuse baseline as "strong".
"""

from pathlib import Path

from agent_migrate_agent.k8_regime import RegimeCell, default_bundle, make_k8_budget, make_k8_episode
from agent_migrate_agent.reconstitution import (
    RECONSTITUTION_POLICIES,
    cache_reuse,
    run_reconstitution_policy,
    strong_site_reuse,
)
from agent_migrate_agent.warmness import WarmnessMap


REPO = Path(__file__).resolve().parent.parent


def _fingerprint(plan):
    return {
        workflow_id: [
            (a.state_id, a.mode, a.src_site, a.dst_site, a.reason)
            for a in actions
        ]
        for workflow_id, actions in plan.items()
    }


def test_strong_site_reuse_is_registry_alias_for_cache_reuse():
    """A wrong alias that calls replay_all/kv_all/min_cost would differ on
    either destination choice, mode choice, or warm_hit handling."""
    bundle = default_bundle(REPO)
    cell = RegimeCell(10, "medium", "tight", 5, seed=8301)
    episode, manifests = make_k8_episode(cell)
    budget = make_k8_budget(cell)
    warmness = WarmnessMap.from_episode_seed({
        "system_prompt": ("seattle",),
    })

    expected = cache_reuse(episode, manifests, bundle, warmness, budget)
    direct = strong_site_reuse(episode, manifests, bundle, warmness, budget)
    registered = run_reconstitution_policy(
        "strong_site_reuse", episode, manifests, bundle, warmness, budget,
    )

    assert RECONSTITUTION_POLICIES["strong_site_reuse"] is strong_site_reuse
    assert _fingerprint(direct) == _fingerprint(expected)
    assert _fingerprint(registered) == _fingerprint(expected)


def test_strong_site_reuse_baseline_doc_preserves_contract():
    text = (REPO / "docs" / "strong_site_reuse_baseline.md").read_text()
    assert "not a strawman" in text
    assert "choose the cheapest available materialization mode" in text
    assert "avoid paying twice" in text
    assert "no_reuse" in text
