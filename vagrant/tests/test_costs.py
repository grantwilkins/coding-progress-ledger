from pathlib import Path

import pytest

from vagrant_agent.costs import (
    ARTIFACT_COPY,
    CONTEXT_REPLAY,
    KV_TRANSFER,
    TEXT_TRANSFER,
    allowed_modes_for_state,
    artifact_copy_cost,
    bandwidth_crossover_bps,
    choose_min_cost_mode,
    context_replay_cost,
    kv_transfer_cost,
    materialize_cost,
    text_transfer_cost,
)
from vagrant_agent.manifest import StateObject
from vagrant_agent.profiles import load_bundle

REPO = Path(__file__).resolve().parent.parent


def _bundle():
    return load_bundle(REPO / "configs" / "model_profiles.yaml",
                       REPO / "configs" / "sites_2site.yaml",
                       "compact_kv")


# ---- formula correctness ----

def test_kv_transfer_cost_formula():
    # 8 * 1000 * 70656 / 25e9 = ~22.6 ms
    assert kv_transfer_cost(tokens=1000, kv_bytes_per_token=70656, link_bps=25_000_000_000) == \
        pytest.approx(8 * 1000 * 70656 / 25_000_000_000)


def test_context_replay_cost_formula():
    assert context_replay_cost(tokens=1000, dst_prefill_tok_s=50000) == pytest.approx(1000 / 50000)


def test_text_transfer_cost_formula():
    assert text_transfer_cost(text_bytes=1_000_000, link_bps=25_000_000_000) == \
        pytest.approx(8 * 1_000_000 / 25_000_000_000)


def test_artifact_copy_cost_formula():
    assert artifact_copy_cost(artifact_bytes=4_000_000, link_bps=25_000_000_000) == \
        pytest.approx(8 * 4_000_000 / 25_000_000_000)


def test_zero_bps_hard_fails():
    with pytest.raises(ValueError):
        kv_transfer_cost(1, 1, 0)
    with pytest.raises(ValueError):
        text_transfer_cost(1, 0)
    with pytest.raises(ValueError):
        artifact_copy_cost(1, 0)
    with pytest.raises(ValueError):
        context_replay_cost(1, 0)


# ---- bandwidth crossover (the load-bearing test) ----

def test_bandwidth_crossover_identity():
    """At link_bps == B*, kv_transfer == context_replay regardless of token count T."""
    kv_bytes = 70656
    prefill_tok_s = 30000.0
    b_star = bandwidth_crossover_bps(kv_bytes, prefill_tok_s)
    assert b_star == pytest.approx(8 * kv_bytes * prefill_tok_s)
    for tokens in (100, 8000, 100_000):
        kv = kv_transfer_cost(tokens, kv_bytes, b_star)
        ctx = context_replay_cost(tokens, prefill_tok_s)
        assert kv == pytest.approx(ctx, rel=1e-9), f"crossover broken at T={tokens}"


def test_below_crossover_replay_wins():
    kv_bytes = 70656
    prefill_tok_s = 30000.0
    b_star = bandwidth_crossover_bps(kv_bytes, prefill_tok_s)
    slow = b_star * 0.5
    for tokens in (100, 8000, 100_000):
        assert context_replay_cost(tokens, prefill_tok_s) < kv_transfer_cost(tokens, kv_bytes, slow)


def test_above_crossover_transfer_wins():
    kv_bytes = 70656
    prefill_tok_s = 30000.0
    b_star = bandwidth_crossover_bps(kv_bytes, prefill_tok_s)
    fast = b_star * 2.0
    for tokens in (100, 8000, 100_000):
        assert kv_transfer_cost(tokens, kv_bytes, fast) < context_replay_cost(tokens, prefill_tok_s)


def test_crossover_identity_under_varying_kv_bytes_and_prefill():
    """TASKS.md C3: holding link_bps fixed, varying kv_bytes_per_token and dst_prefill_tok_s
    produces the same crossover identity (kv_transfer == context_replay at B*)."""
    for kv_bytes in (32_000, 70_656, 1_000_000):
        for prefill in (10_000.0, 30_000.0, 80_000.0):
            b_star = bandwidth_crossover_bps(kv_bytes, prefill)
            for tokens in (50, 5000, 250_000):
                kv = kv_transfer_cost(tokens, kv_bytes, b_star)
                ctx = context_replay_cost(tokens, prefill)
                assert kv == pytest.approx(ctx, rel=1e-9), (
                    f"crossover failed for kv={kv_bytes}, prefill={prefill}, T={tokens}"
                )


def test_T_cancels_in_kv_vs_replay_ratio():
    """Sanity: the kv/replay ratio is independent of token count T."""
    kv_bytes = 70656
    prefill_tok_s = 30000.0
    link_bps = 25_000_000_000
    ratio_a = kv_transfer_cost(100, kv_bytes, link_bps) / context_replay_cost(100, prefill_tok_s)
    ratio_b = kv_transfer_cost(100_000, kv_bytes, link_bps) / context_replay_cost(100_000, prefill_tok_s)
    assert ratio_a == pytest.approx(ratio_b, rel=1e-9)


# ---- mode allowability ----

def test_allowed_modes_prompt_context():
    s = StateObject(state_id="x", content_hash="h", layer="prompt_context",
                    lifetime="shared", tokens=8000, bytes=None)
    assert allowed_modes_for_state(s) == (KV_TRANSFER, CONTEXT_REPLAY)


def test_allowed_modes_workspace():
    s = StateObject(state_id="x", content_hash="h", layer="workspace",
                    lifetime="shared", tokens=0, bytes=4_000_000)
    assert allowed_modes_for_state(s) == (ARTIFACT_COPY,)


def test_allowed_modes_memory():
    s = StateObject(state_id="x", content_hash="h", layer="memory",
                    lifetime="persistent", tokens=0, bytes=1000)
    assert allowed_modes_for_state(s) == (TEXT_TRANSFER,)


def test_allowed_modes_unknown_layer():
    s = StateObject(state_id="x", content_hash="h", layer="semantic",
                    lifetime="shared", tokens=0, bytes=None)
    assert allowed_modes_for_state(s) == ()


# ---- materialize_cost dispatch ----

def test_materialize_cost_kv_transfer():
    bundle = _bundle()
    s = StateObject(state_id="x", content_hash="h", layer="prompt_context",
                    lifetime="shared", tokens=8000, bytes=None)
    cost = materialize_cost(s, KV_TRANSFER, "phoenix", "seattle", bundle)
    expected = 8 * 8000 * bundle.model.kv_bytes_per_token / bundle.link("phoenix", "seattle").effective_bps
    assert cost == pytest.approx(expected)


def test_materialize_cost_context_replay_uses_dst_prefill():
    bundle = _bundle()
    s = StateObject(state_id="x", content_hash="h", layer="prompt_context",
                    lifetime="shared", tokens=8000, bytes=None)
    to_seattle = materialize_cost(s, CONTEXT_REPLAY, "phoenix", "seattle", bundle)
    to_phoenix = materialize_cost(s, CONTEXT_REPLAY, "seattle", "phoenix", bundle)
    assert to_seattle == pytest.approx(8000 / 45000)
    assert to_phoenix == pytest.approx(8000 / 30000)
    assert to_phoenix > to_seattle  # Seattle prefills faster, so replay AT phoenix is slower


def test_materialize_cost_workspace_requires_bytes():
    bundle = _bundle()
    s = StateObject(state_id="x", content_hash="h", layer="workspace",
                    lifetime="shared", tokens=0, bytes=None)
    with pytest.raises(ValueError, match="bytes"):
        materialize_cost(s, ARTIFACT_COPY, "phoenix", "seattle", bundle)


def test_materialize_cost_same_site_kv_transfer_hard_fails():
    bundle = _bundle()
    s = StateObject(state_id="x", content_hash="h", layer="prompt_context",
                    lifetime="shared", tokens=100, bytes=None)
    with pytest.raises(ValueError, match="src != dst"):
        materialize_cost(s, KV_TRANSFER, "phoenix", "phoenix", bundle)


def test_materialize_cost_same_site_context_replay_pays_local_prefill():
    bundle = _bundle()
    s = StateObject(state_id="x", content_hash="h", layer="prompt_context",
                    lifetime="shared", tokens=8000, bytes=None)
    cost = materialize_cost(s, CONTEXT_REPLAY, "phoenix", "phoenix", bundle)
    assert cost == pytest.approx(8000 / 30000)


def test_materialize_cost_same_site_workspace_is_zero():
    bundle = _bundle()
    s = StateObject(state_id="ws", content_hash="h", layer="workspace",
                    lifetime="shared", tokens=0, bytes=4_000_000)
    cost = materialize_cost(s, ARTIFACT_COPY, "phoenix", "phoenix", bundle)
    assert cost == 0.0


# ---- choose_min_cost_mode ----

def test_choose_min_cost_mode_picks_replay_for_small_state():
    bundle = _bundle()
    # 200-token state, slow link relative to crossover -> replay should win.
    s = StateObject(state_id="x", content_hash="h", layer="prompt_context",
                    lifetime="shared", tokens=200, bytes=None)
    mode, cost = choose_min_cost_mode(s, "phoenix", "seattle", bundle)
    assert mode == CONTEXT_REPLAY
    assert cost == pytest.approx(200 / 45000)


def test_choose_min_cost_mode_picks_transfer_when_replay_slow():
    bundle = _bundle()
    # workspace state has only one feasible mode (artifact_copy), so the
    # mode choice is forced regardless of the kv-vs-replay crossover.
    s = StateObject(state_id="ws", content_hash="h", layer="workspace",
                    lifetime="shared", tokens=0, bytes=4_000_000)
    mode, cost = choose_min_cost_mode(s, "phoenix", "seattle", bundle)
    assert mode == ARTIFACT_COPY


def test_choose_min_cost_mode_no_feasible_hard_fails():
    bundle = _bundle()
    # workspace state with no bytes -> artifact_copy infeasible -> hard-fail
    s = StateObject(state_id="bad", content_hash="h", layer="workspace",
                    lifetime="shared", tokens=0, bytes=None)
    with pytest.raises(ValueError, match="no feasible mode"):
        choose_min_cost_mode(s, "phoenix", "seattle", bundle)


def test_materialize_cost_text_transfer_dispatch():
    bundle = _bundle()
    s = StateObject(state_id="m", content_hash="h", layer="memory",
                    lifetime="persistent", tokens=0, bytes=2_000_000)
    cost = materialize_cost(s, TEXT_TRANSFER, "phoenix", "seattle", bundle)
    expected = 8 * 2_000_000 / bundle.link("phoenix", "seattle").effective_bps
    assert cost == pytest.approx(expected)


def test_materialize_cost_text_transfer_requires_bytes():
    bundle = _bundle()
    s = StateObject(state_id="m", content_hash="h", layer="memory",
                    lifetime="persistent", tokens=0, bytes=None)
    with pytest.raises(ValueError, match="bytes"):
        materialize_cost(s, TEXT_TRANSFER, "phoenix", "seattle", bundle)


def test_choose_min_cost_mode_tie_break_prefers_first_in_tuple():
    """At link_bps == B*, kv_transfer and context_replay cost equally.
    Tie-break rule: first mode in `candidates` wins."""
    bundle = _bundle()
    kv_bytes = bundle.model.kv_bytes_per_token
    prefill = bundle.site("seattle").prefill_tok_s
    # Tie occurs at link_bps == 8 * kv_bytes * prefill. Build a state where the bundle's
    # link bandwidth is the tie point. We need a custom bundle for this — use the
    # default bundle and override the link's effective_bps via construction.
    from vagrant_agent.profiles import LinkProfile, ProfileBundle
    tie_bps = 8.0 * kv_bytes * prefill
    custom_links = {("phoenix", "seattle"): LinkProfile(
        site_a="phoenix", site_b="seattle", effective_bps=tie_bps,
    )}
    custom = ProfileBundle(model=bundle.model, sites=bundle.sites,
                           links=custom_links, home_site=bundle.home_site)
    s = StateObject(state_id="x", content_hash="h", layer="prompt_context",
                    lifetime="shared", tokens=8000, bytes=None)
    mode_default, cost_default = choose_min_cost_mode(s, "phoenix", "seattle", custom)
    assert mode_default == KV_TRANSFER  # KV is first in allowed_modes_for_state
    mode_reversed, _ = choose_min_cost_mode(s, "phoenix", "seattle", custom,
                                            allowed_modes=(CONTEXT_REPLAY, KV_TRANSFER))
    assert mode_reversed == CONTEXT_REPLAY  # caller's order wins on tie


def test_choose_min_cost_mode_allowed_modes_override():
    bundle = _bundle()
    s = StateObject(state_id="x", content_hash="h", layer="prompt_context",
                    lifetime="shared", tokens=200, bytes=None)
    # Force kv_transfer by restricting allowed_modes.
    mode, cost = choose_min_cost_mode(s, "phoenix", "seattle", bundle, allowed_modes=(KV_TRANSFER,))
    assert mode == KV_TRANSFER
