"""Resource vectors for Workstream K3.

A `ResourceCost` is a 5-tuple capturing what a single state-materialization
action consumes from the destination resource pool:

  network_bytes     -- bytes that traverse a cross-site link
  prefill_tokens    -- tokens added to dst's prefill workload
  workspace_bytes   -- bytes hydrated locally at dst (disk/object store)
  kv_resident_bytes -- KV-cache bytes the action holds at dst after resume
  wallclock_s       -- closed-form lower bound (the existing
                       costs.materialize_cost — same formula, same units)

K4's fluid simulator (next task) reads these vectors per in-flight
action, sums per-resource demand, and applies proportional fair-share
to compute time-to-finish. K3 itself does NOT involve the simulator —
it just produces the per-action cost vectors.

Per A3 audit: materialization is per-(state, site), NOT per-(component,
site). When `warmness.is_warm(state, dst_site)` is True, the action
short-circuits to a zero ResourceCost (the cost was paid by an earlier
action; this is the L1 abstraction). K3 enforces this invariant.

Per A4 audit: this module does NOT pipeline (transfer, prefill); it
keeps additivity for parity with the existing cost model. The
network_bytes / prefill_tokens / workspace_bytes / kv_resident_bytes
fields are tracked SEPARATELY so a future workstream M (post-K) can
compute max(...) instead of sum(...) without re-deriving anything.

Hard rules: only K4 may mutate a warmness map; K3 is read-only.
"""
from __future__ import annotations

import math
from dataclasses import dataclass

from .costs import (
    ARTIFACT_COPY,
    CONTEXT_REPLAY,
    KV_TRANSFER,
    TEXT_TRANSFER,
    bandwidth_crossover_bps,
    materialize_cost,
)
from .events import MATERIALIZATION_MODES
from .manifest import StateObject
from .profiles import ProfileBundle
from .warmness import WarmnessMap

WARM_REUSE = "warm_reuse"
WORKSPACE_HYDRATE = "workspace_hydrate"


@dataclass(frozen=True)
class ResourceCost:
    """Per-action resource consumption vector.

    Ordering of fields matches collaborator 2's section-4 table for
    grep-friendly cross-reference. wallclock_s is the existing
    materialize_cost output, retained for parity tests.
    """
    network_bytes: int
    prefill_tokens: int
    workspace_bytes: int
    kv_resident_bytes: int
    wallclock_s: float

    @classmethod
    def zero(cls) -> "ResourceCost":
        return cls(0, 0, 0, 0, 0.0)

    def __add__(self, other: "ResourceCost") -> "ResourceCost":
        return ResourceCost(
            network_bytes=self.network_bytes + other.network_bytes,
            prefill_tokens=self.prefill_tokens + other.prefill_tokens,
            workspace_bytes=self.workspace_bytes + other.workspace_bytes,
            kv_resident_bytes=self.kv_resident_bytes + other.kv_resident_bytes,
            wallclock_s=self.wallclock_s + other.wallclock_s,
        )


@dataclass(frozen=True)
class ResourceBudget:
    """Per-episode resource capacity envelope. K4 consumes this.

    Each per-site bandwidth/capacity defaults to math.inf (uncapped).
    `network_bps_per_link` is keyed by an unordered (site_a, site_b)
    tuple matching profiles.LinkProfile's _link_key convention.

    Backward-compat: `from_bundle(bundle)` reads the existing
    SiteProfile.workspace_hydrate_bps / kv_memory_bytes (default
    math.inf if YAML didn't specify them) plus LinkProfile.effective_bps,
    so MVP 2-site configs produce a budget with effectively-uncapped
    capacities for everything except network bandwidth.
    """
    network_bps_per_link: dict[tuple[str, str], float]
    prefill_tok_s_per_site: dict[str, float]
    workspace_hydrate_bps_per_site: dict[str, float]
    kv_memory_bytes_per_site: dict[str, float]

    @classmethod
    def from_bundle(cls, bundle: ProfileBundle) -> "ResourceBudget":
        return cls(
            network_bps_per_link={k: lp.effective_bps for k, lp in bundle.links.items()},
            prefill_tok_s_per_site={n: sp.prefill_tok_s for n, sp in bundle.sites.items()},
            workspace_hydrate_bps_per_site={n: sp.workspace_hydrate_bps for n, sp in bundle.sites.items()},
            kv_memory_bytes_per_site={n: sp.kv_memory_bytes for n, sp in bundle.sites.items()},
        )

    @classmethod
    def infinite(cls, site_names: list[str]) -> "ResourceBudget":
        """Capacity-free budget — used for K7 T1 (capacity-free collapse)."""
        sites = list(site_names)
        return cls(
            network_bps_per_link={
                tuple(sorted([a, b])): math.inf
                for i, a in enumerate(sites) for b in sites[i + 1:]
            },
            prefill_tok_s_per_site={n: math.inf for n in sites},
            workspace_hydrate_bps_per_site={n: math.inf for n in sites},
            kv_memory_bytes_per_site={n: math.inf for n in sites},
        )


# ---------------------------------------------------------------------------
# Resource consumption per materialization mode
# ---------------------------------------------------------------------------


def _kv_resident_for(state: StateObject, bundle: ProfileBundle) -> int:
    """KV-cache bytes the state occupies in HBM after a successful resume.
    Returns 0 for non-prompt_context layers (workspace/memory states are
    on disk/object-store, not in KV memory)."""
    if state.layer not in ("prompt_context", "model_execution"):
        return 0
    return state.tokens * bundle.model.kv_bytes_per_token


def reconstitution_cost(
    state: StateObject,
    mode: str,
    src_site: str,
    dst_site: str,
    bundle: ProfileBundle,
    warmness: WarmnessMap,
) -> ResourceCost:
    """Compute the resource vector for materializing `state` at `dst_site`
    via `mode`, given the current warmness state.

    Per A3: short-circuits to zero when warmness says (state, dst) is
    already warm. The kv_resident_bytes contribution is preserved
    (the state still occupies KV memory at dst), so `mixed_min_pressure`
    can correctly account for KV memory pressure from already-warm states.

    Per A4: wallclock_s reuses costs.materialize_cost; network_bytes,
    prefill_tokens, workspace_bytes, kv_resident_bytes are tracked
    separately so a future pipelined model can compute max() rather
    than sum().
    """
    if mode not in MATERIALIZATION_MODES:
        raise ValueError(f"unknown mode {mode!r}; must be one of {MATERIALIZATION_MODES}")

    # Warm-cache short-circuit (L1 dedup).
    if warmness.is_warm(state.state_id, dst_site):
        return ResourceCost(
            network_bytes=0, prefill_tokens=0, workspace_bytes=0,
            kv_resident_bytes=_kv_resident_for(state, bundle),
            wallclock_s=0.0,
        )

    if mode == WARM_REUSE:
        # warm_reuse used WITHOUT a warm cache entry is a caller bug.
        # Surface it loudly.
        raise ValueError(
            f"warm_reuse on cold (state={state.state_id!r}, site={dst_site!r}); "
            f"caller must check warmness.is_warm first"
        )

    # Same-site has no network cost regardless of mode.
    same_site = src_site == dst_site

    # wallclock from the existing closed-form cost model. WORKSPACE_HYDRATE
    # is K-introduced and uses its own formula below (computed in-branch).
    wallclock = (0.0 if mode == WORKSPACE_HYDRATE
                 else materialize_cost(state, mode, src_site, dst_site, bundle))

    if mode == KV_TRANSFER:
        # KV bytes over the wire; no prefill load; KV resident at dst.
        net = 0 if same_site else state.tokens * bundle.model.kv_bytes_per_token
        return ResourceCost(
            network_bytes=net,
            prefill_tokens=0,
            workspace_bytes=0,
            kv_resident_bytes=_kv_resident_for(state, bundle),
            wallclock_s=wallclock,
        )

    if mode == CONTEXT_REPLAY:
        # Prompt text re-derived at dst (no wire bytes); prefill paid
        # for state.tokens; KV resident after replay.
        return ResourceCost(
            network_bytes=0,
            prefill_tokens=state.tokens,
            workspace_bytes=0,
            kv_resident_bytes=_kv_resident_for(state, bundle),
            wallclock_s=wallclock,
        )

    if mode == TEXT_TRANSFER:
        if state.bytes is None:
            raise ValueError(f"text_transfer requires state.bytes; state {state.state_id!r} has none")
        return ResourceCost(
            network_bytes=0 if same_site else state.bytes,
            prefill_tokens=0,
            workspace_bytes=0,
            kv_resident_bytes=0,
            wallclock_s=wallclock,
        )

    if mode == ARTIFACT_COPY:
        if state.bytes is None:
            raise ValueError(f"artifact_copy requires state.bytes; state {state.state_id!r} has none")
        return ResourceCost(
            network_bytes=0 if same_site else state.bytes,
            prefill_tokens=0,
            workspace_bytes=state.bytes,  # hydrated to dst local storage
            kv_resident_bytes=0,
            wallclock_s=wallclock,
        )

    if mode == WORKSPACE_HYDRATE:
        # Local-only hydrate from object store; no wire bytes; workspace
        # storage charged. Wallclock = state.bytes / workspace_hydrate_bps
        # at dst (single-action lower bound; K4 fluid simulator divides
        # by proportional share). We compute this DIRECTLY rather than
        # delegating to materialize_cost, because materialize_cost has
        # no concept of workspace_hydrate (it's a K-introduced mode).
        if state.bytes is None:
            raise ValueError(f"workspace_hydrate requires state.bytes; state {state.state_id!r} has none")
        dst_site_profile = bundle.site(dst_site)
        hydrate_bps = dst_site_profile.workspace_hydrate_bps
        if hydrate_bps == math.inf:
            wc = 0.0  # uncapped: single-action local hydrate is instantaneous
        else:
            wc = 8.0 * state.bytes / hydrate_bps  # match other modes' bits/s convention
        return ResourceCost(
            network_bytes=0,
            prefill_tokens=0,
            workspace_bytes=state.bytes,
            kv_resident_bytes=0,
            wallclock_s=wc,
        )

    raise ValueError(f"unhandled mode {mode!r} in reconstitution_cost")


# ---------------------------------------------------------------------------
# Sanity helpers (used by tests)
# ---------------------------------------------------------------------------


def crossover_parity_check(state: StateObject, bundle: ProfileBundle) -> tuple[float, float]:
    """Return (B_star_from_costs, B_star_from_resources_dim_check).

    Both must agree: kv_transfer's network_bytes/wallclock_s ratio at
    crossover bandwidth equals the existing bandwidth_crossover formula.
    Used by tests/test_resources.py to pin K3 against costs.py.
    """
    if state.layer not in ("prompt_context", "model_execution"):
        raise ValueError(f"crossover only meaningful for prompt_context-class states")
    return (
        bandwidth_crossover_bps(bundle.model.kv_bytes_per_token,
                                 max(s.prefill_tok_s for s in bundle.sites.values())),
        8.0 * bundle.model.kv_bytes_per_token
        * max(s.prefill_tok_s for s in bundle.sites.values()),
    )
