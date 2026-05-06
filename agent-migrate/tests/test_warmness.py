"""
Claim:
WarmnessMap is a (state_id, site) -> WarmnessEntry index that K3
uses to short_circuit reconstitution costs and K4 mutates as the
simulator advances. It is frozen_dataclass; mutators return new
instances.

Per A3: warmness must be keyed at (state_id, site) granularity so K3
enforces L1_style dedup. Per CLAUDE.md hard rule: only K4 mutates a
warmness map (this is enforced by convention, not the type system —
the with_added/with_evicted/lru_evict methods are the documented
entry points).
"""
from __future__ import annotations

from pathlib import Path

from ledger_progress import from_jsonl

from agent_migrate_agent import build_manifest
from agent_migrate_agent.warmness import WarmnessEntry, WarmnessMap

REPO = Path(__file__).resolve().parent.parent


def test_empty_map_has_nothing_warm():
    wm = WarmnessMap.empty()
    assert len(wm) == 0
    assert not wm.is_warm("any_state", "any_site")


def test_from_dict_constructs_entries():
    wm = WarmnessMap.from_dict({
        "system_prompt": ["phoenix", "seattle"],
        "issue_text_a": ["phoenix"],
    })
    assert wm.is_warm("system_prompt", "phoenix")
    assert wm.is_warm("system_prompt", "seattle")
    assert wm.is_warm("issue_text_a", "phoenix")
    assert not wm.is_warm("issue_text_a", "seattle")


def test_from_episode_seed_works():
    wm = WarmnessMap.from_episode_seed({"sx": ("phoenix",)})
    assert wm.is_warm("sx", "phoenix")
    assert not wm.is_warm("sx", "seattle")


def test_states_warm_at_and_sites_warm_for():
    wm = WarmnessMap.from_dict({
        "a": ["phoenix", "seattle"],
        "b": ["phoenix"],
        "c": ["seattle", "austin"],
    })
    assert wm.states_warm_at("phoenix") == {"a", "b"}
    assert wm.states_warm_at("seattle") == {"a", "c"}
    assert wm.states_warm_at("austin") == {"c"}
    assert wm.sites_warm_for("a") == {"phoenix", "seattle"}


def test_fraction_warm_zero_one_half():
    """All_warm = 1.0; none_warm = 0.0; half_warm = 0.5 etc."""
    m = build_manifest(from_jsonl(str(REPO / "examples" / "traces" / "toy_subagent_trace.jsonl")))
    n = len(m.state_objects)

    empty = WarmnessMap.empty()
    assert empty.fraction_warm(m, "phoenix") == 0.0

    all_at_phx = WarmnessMap.from_dict({sid: ["phoenix"] for sid in m.state_objects})
    assert all_at_phx.fraction_warm(m, "phoenix") == 1.0
    assert all_at_phx.fraction_warm(m, "seattle") == 0.0

    half_keys = sorted(m.state_objects)[:n // 2]
    half_warm = WarmnessMap.from_dict({sid: ["phoenix"] for sid in half_keys})
    expected = (n // 2) / n
    assert abs(half_warm.fraction_warm(m, "phoenix") - expected) < 1e-9


def test_with_added_returns_new_instance_with_entry():
    wm = WarmnessMap.empty()
    wm2 = wm.with_added("sx", "phoenix", mode="kv_transfer")
    assert not wm.is_warm("sx", "phoenix")    # original unchanged
    assert wm2.is_warm("sx", "phoenix")
    e = wm2.get("sx", "phoenix")
    assert e is not None
    assert e.mode == "kv_transfer"
    assert e.age_s == 0.0


def test_with_evicted_removes_entry():
    wm = WarmnessMap.from_dict({"sx": ["phoenix", "seattle"]})
    wm2 = wm.with_evicted("sx", "phoenix")
    assert not wm2.is_warm("sx", "phoenix")
    assert wm2.is_warm("sx", "seattle")  # other entry unaffected
    assert wm.is_warm("sx", "phoenix")    # original unchanged


def test_with_aged_increments_all_entries():
    wm = WarmnessMap.from_dict({"a": ["phoenix"], "b": ["seattle"]})
    aged = wm.with_aged(2.5)
    for entry in aged.entries.values():
        assert entry.age_s == 2.5
    # original unchanged
    for entry in wm.entries.values():
        assert entry.age_s == 0.0


def test_lru_evict_picks_oldest_at_site():
    wm = (WarmnessMap.empty()
          .with_added("a", "phoenix", "kv_transfer", age_s=10.0)
          .with_added("b", "phoenix", "kv_transfer", age_s=5.0)
          .with_added("c", "phoenix", "kv_transfer", age_s=20.0))
    new_wm, evicted = wm.lru_evict("phoenix")
    assert evicted == "c"  # oldest
    assert not new_wm.is_warm("c", "phoenix")
    assert new_wm.is_warm("a", "phoenix")
    assert new_wm.is_warm("b", "phoenix")


def test_lru_evict_empty_site_returns_none():
    wm = WarmnessMap.from_dict({"x": ["phoenix"]})
    new_wm, evicted = wm.lru_evict("seattle")
    assert evicted is None
    assert new_wm == wm


def test_lru_evict_only_affects_target_site():
    wm = (WarmnessMap.empty()
          .with_added("a", "phoenix", "kv_transfer", age_s=10.0)
          .with_added("b", "seattle", "kv_transfer", age_s=20.0))
    new_wm, evicted = wm.lru_evict("phoenix")
    assert evicted == "a"
    assert new_wm.is_warm("b", "seattle")  # untouched


def test_warmness_entry_carries_mode_and_pressure():
    e = WarmnessEntry(state_id="x", site="phoenix", mode="kv_transfer",
                      age_s=5.0, eviction_pressure=0.7)
    assert e.eviction_pressure == 0.7
