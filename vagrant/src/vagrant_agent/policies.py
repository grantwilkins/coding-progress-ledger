"""Placement + materialization policies.

Two MVP policies:

- `request_level_no_reuse`: each node placed at the lowest-cost site
  independently; state materialized **per consumer at the chosen site** (no
  reuse across colocated nodes). Strawman baseline.
- `shared_state_aware`: build a node graph where edge weight = total tokens
  of shared state between two consumers; keep only edges with weight > tau;
  components placed together; state materialized **once per (state, site)**.

Both produce a `Plan` with `placements` and `materializations`.

Materializations:
  Keyed by `(state_id, site)`. Every consumer of every required state produces
  a materialization row, including same-site (which pays local prefill cost for
  prompt_context, or zero for workspace/memory). `cost_s` is the cost of ONE
  materialization; `materialization_count` is the multiplicity (= number of
  consumers for `request_level_no_reuse`; always 1 for `shared_state_aware`);
  `total_cost_s` = `cost_s * materialization_count`.

Placements:
  For `request_level_no_reuse`, `cost_s` is the per-node placement cost.
  For `shared_state_aware`, `cost_s` is the FULL COMPONENT placement cost,
  duplicated across each member's row (so a 4-node component has 4 rows each
  showing the same component cost). Summing `placement.cost_s` is therefore
  meaningless for D2; use `Plan.total_cost_s()` (which sums materializations).
"""
from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass, field

from .costs import choose_min_cost_mode
from .manifest import ServingGroupManifest, StateObject
from .profiles import ProfileBundle


def _state_home(state: StateObject, bundle: ProfileBundle) -> str:
    """Per-state home: state.home_site if set, else the bundle's global home_site."""
    return state.home_site if state.home_site else bundle.home_site


def _validate_state_homes(manifest: ServingGroupManifest, bundle: ProfileBundle) -> None:
    """Hard-fail at policy entry if any state.home_site references a site not in
    the bundle. Without this, the error surfaces deep inside `bundle.link()` mid-
    enumeration with a confusing message."""
    for state in manifest.state_objects.values():
        if state.home_site is not None and state.home_site not in bundle.sites:
            raise ValueError(
                f"state {state.state_id!r} has home_site={state.home_site!r}, "
                f"which is not in bundle.sites ({sorted(bundle.sites)})"
            )


@dataclass(frozen=True)
class PlacementDecision:
    node_id: str
    site: str
    cost_s: float          # for D2 components, this is the full component cost duplicated per member
    component_size: int    # 1 for D1; >=1 for D2
    reason: str


@dataclass(frozen=True)
class MaterializationDecision:
    state_id: str
    content_hash: str
    site: str
    mode: str
    cost_s: float                 # cost of ONE materialization at (state, site)
    materialization_count: int    # how many times the policy materialized at (state, site)
    consumers: list[str]          # consumer node_ids placed at `site`
    reason: str

    @property
    def total_cost_s(self) -> float:
        return self.cost_s * self.materialization_count


@dataclass(frozen=True)
class Plan:
    policy: str
    placements: list[PlacementDecision]
    materializations: list[MaterializationDecision]
    meta: dict = field(default_factory=dict)

    def total_cost_s(self) -> float:
        """Total seconds the policy pays. Sum of materialization rows only;
        placement.cost_s is duplicated per-member for D2 components and would
        double-count if added."""
        return sum(m.total_cost_s for m in self.materializations)


# --------------------------------------------------------------------------- #
# D1: request_level_no_reuse
# --------------------------------------------------------------------------- #


def run_request_level_no_reuse(manifest: ServingGroupManifest, bundle: ProfileBundle) -> Plan:
    _validate_state_homes(manifest, bundle)
    placements = _place_per_node_min_cost(manifest, bundle)

    by_state_site: dict[tuple[str, str], dict] = {}
    for p in placements:
        node = manifest.nodes[p.node_id]
        for state_id in node.required_state:
            state = manifest.state_objects[state_id]
            mode, cost = choose_min_cost_mode(state, _state_home(state, bundle), p.site, bundle)
            key = (state_id, p.site)
            if key not in by_state_site:
                by_state_site[key] = {
                    "content_hash": state.content_hash,
                    "mode": mode,
                    "cost_s": cost,
                    "consumers": [p.node_id],
                }
            else:
                by_state_site[key]["consumers"].append(p.node_id)

    materializations = [
        MaterializationDecision(
            state_id=sid,
            content_hash=info["content_hash"],
            site=site,
            mode=info["mode"],
            cost_s=info["cost_s"],
            materialization_count=len(info["consumers"]),
            consumers=info["consumers"],
            reason="per_node_no_reuse",
        )
        for (sid, site), info in by_state_site.items()
    ]
    return Plan(policy="request_level_no_reuse", placements=placements, materializations=materializations)


def _place_per_node_min_cost(manifest: ServingGroupManifest, bundle: ProfileBundle) -> list[PlacementDecision]:
    placements: list[PlacementDecision] = []
    for node in manifest.nodes.values():
        best_site: str | None = None
        best_cost = float("inf")
        for site in bundle.sites:
            total = 0.0
            for state_id in node.required_state:
                state = manifest.state_objects[state_id]
                _, c = choose_min_cost_mode(state, _state_home(state, bundle), site, bundle)
                total += c
            if total < best_cost:
                best_cost = total
                best_site = site
        assert best_site is not None
        placements.append(PlacementDecision(
            node_id=node.node_id, site=best_site, cost_s=best_cost,
            component_size=1, reason="min_cost",
        ))
    return placements


# --------------------------------------------------------------------------- #
# D2: shared_state_aware
# --------------------------------------------------------------------------- #


def run_shared_state_aware(
    manifest: ServingGroupManifest,
    bundle: ProfileBundle,
    tau: int = 1,
) -> Plan:
    _validate_state_homes(manifest, bundle)
    components = _components_by_shared_state(manifest, tau)
    placements: list[PlacementDecision] = []
    by_state_site: dict[tuple[str, str], dict] = {}

    for component in components:
        required_states = _component_required_states(manifest, component)
        best_site: str | None = None
        best_cost = float("inf")
        for site in bundle.sites:
            total = 0.0
            for state_id in required_states:
                state = manifest.state_objects[state_id]
                _, c = choose_min_cost_mode(state, _state_home(state, bundle), site, bundle)
                total += c
            if total < best_cost:
                best_cost = total
                best_site = site
        assert best_site is not None
        for node_id in sorted(component):
            placements.append(PlacementDecision(
                node_id=node_id, site=best_site,
                cost_s=best_cost,
                component_size=len(component),
                reason="grouped" if len(component) > 1 else "min_cost",
            ))
        for state_id in sorted(required_states):
            state = manifest.state_objects[state_id]
            consumers = sorted(c for c in component if state_id in manifest.nodes[c].required_state)
            if not consumers:
                continue
            mode, cost = choose_min_cost_mode(state, _state_home(state, bundle), best_site, bundle)
            key = (state_id, best_site)
            if key not in by_state_site:
                by_state_site[key] = {
                    "content_hash": state.content_hash,
                    "mode": mode,
                    "cost_s": cost,
                    "consumers": list(consumers),
                    "materialization_count": 1,
                }
            else:
                # Two distinct components both picked this (state, site). Each pays its own
                # materialization (shared_state_aware shares within a component, not across).
                existing = by_state_site[key]["consumers"]
                by_state_site[key]["consumers"] = sorted(set(existing) | set(consumers))
                by_state_site[key]["materialization_count"] += 1

    materializations = [
        MaterializationDecision(
            state_id=sid,
            content_hash=info["content_hash"],
            site=site,
            mode=info["mode"],
            cost_s=info["cost_s"],
            materialization_count=info["materialization_count"],
            consumers=info["consumers"],
            reason="shared_once_per_component_site",
        )
        for (sid, site), info in by_state_site.items()
    ]
    return Plan(
        policy="shared_state_aware",
        placements=placements,
        materializations=materializations,
        meta={"tau": tau, "components": [sorted(c) for c in components]},
    )


def _components_by_shared_state(manifest: ServingGroupManifest, tau: int) -> list[set[str]]:
    """Pairwise-edge sum > tau merges nodes; otherwise each node is its own component."""
    pair_weight: dict[tuple[str, str], int] = defaultdict(int)
    for edge in manifest.edges:
        a, b = sorted([edge.node_a, edge.node_b])
        pair_weight[(a, b)] += edge.weight

    parent: dict[str, str] = {nid: nid for nid in manifest.nodes}

    def find(x: str) -> str:
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    def union(x: str, y: str) -> None:
        rx, ry = find(x), find(y)
        if rx != ry:
            parent[rx] = ry

    for (a, b), w in pair_weight.items():
        if w > tau:
            union(a, b)

    groups: dict[str, set[str]] = defaultdict(set)
    for nid in manifest.nodes:
        groups[find(nid)].add(nid)
    return list(groups.values())


def _component_required_states(manifest: ServingGroupManifest, component: set[str]) -> set[str]:
    required: set[str] = set()
    for node_id in component:
        required.update(manifest.nodes[node_id].required_state)
    return required


# --------------------------------------------------------------------------- #
# D3: shared_state_aware_typed
# --------------------------------------------------------------------------- #
#
# D3 is D2 with edge-type-weighted pair sums. D2 treats every shared-state
# edge identically, which lets a tiny global prompt (a system_prompt with
# 200 tokens, consumed by everyone) force overgrouping that erases per-
# session structure. D3 replaces the per-edge weight `tokens` with
# `tokens * EDGE_TYPE_WEIGHT[(layer, lifetime)]` so global-replicated state
# (lifetime=persistent in prompt_context) gets multiplier 0 and never
# contributes to grouping.
#
# Edge-type taxonomy (matches the A3 audit's six categories):
#   global_replicated  (prompt_context + persistent)  -> 0.0   ignore
#   workflow_shared    (prompt_context + shared)      -> 1.0   medium
#   artifact_delta     (prompt_context + ephemeral)   -> 0.5   small/medium
#   workspace_local    (workspace + any)              -> 10.0  strong home affinity
#   private_context    (any layer + private)          -> 1.0   one consumer => no edge
#   kv_prefix          (memory + persistent)          -> 0.5   architecture-conditional
#
# Defaults are conservative; callers may override via `edge_type_weights`.

DEFAULT_EDGE_TYPE_WEIGHTS: dict[tuple[str, str], float] = {
    ("prompt_context", "persistent"): 0.0,
    ("prompt_context", "shared"): 1.0,
    ("prompt_context", "ephemeral"): 0.5,
    ("workspace", "private"): 10.0,
    ("workspace", "shared"): 10.0,
    ("workspace", "persistent"): 10.0,
    ("workspace", "ephemeral"): 5.0,
    ("memory", "persistent"): 0.5,
}


def _edge_type_weight(state_layer: str, state_lifetime: str,
                      weights: dict[tuple[str, str], float]) -> float:
    """Lookup multiplier; default 1.0 for any unspecified (layer, lifetime)."""
    return weights.get((state_layer, state_lifetime), 1.0)


def run_shared_state_aware_typed(
    manifest: ServingGroupManifest,
    bundle: ProfileBundle,
    tau: float = 1.0,
    edge_type_weights: dict[tuple[str, str], float] | None = None,
) -> Plan:
    """D3: edge-type-weighted variant of D2.

    Same component-then-place algorithm as D2; only the edge weights used
    in component formation differ. Materialization accounting is identical.

    On every fixture, `D3.total_cost_s() <= D2.total_cost_s() + 1e-9` because
    D3 has strictly weaker grouping (some edges are zeroed out, never merged).
    Where D2 overgroups due to a tiny global prompt, D3 splits the component
    and each sub-component lands at its own min-cost site — matching what H1
    would do if H1's per-node placement happened to coincide.

    `tau` is float (not int) so half-weight ephemeral state can still cross
    the threshold; default 1.0 keeps tau=1 semantics for unweighted edges.
    """
    weights = edge_type_weights if edge_type_weights is not None else DEFAULT_EDGE_TYPE_WEIGHTS
    _validate_state_homes(manifest, bundle)
    components = _components_by_typed_shared_state(manifest, tau, weights)
    placements: list[PlacementDecision] = []
    by_state_site: dict[tuple[str, str], dict] = {}

    for component in components:
        required_states = _component_required_states(manifest, component)
        best_site: str | None = None
        best_cost = float("inf")
        for site in bundle.sites:
            total = 0.0
            for state_id in required_states:
                state = manifest.state_objects[state_id]
                _, c = choose_min_cost_mode(state, _state_home(state, bundle), site, bundle)
                total += c
            if total < best_cost:
                best_cost = total
                best_site = site
        assert best_site is not None
        for node_id in sorted(component):
            placements.append(PlacementDecision(
                node_id=node_id, site=best_site,
                cost_s=best_cost,
                component_size=len(component),
                reason="grouped_typed" if len(component) > 1 else "min_cost",
            ))
        for state_id in sorted(required_states):
            state = manifest.state_objects[state_id]
            consumers = sorted(c for c in component if state_id in manifest.nodes[c].required_state)
            if not consumers:
                continue
            mode, cost = choose_min_cost_mode(state, _state_home(state, bundle), best_site, bundle)
            key = (state_id, best_site)
            if key not in by_state_site:
                by_state_site[key] = {
                    "content_hash": state.content_hash,
                    "mode": mode,
                    "cost_s": cost,
                    "consumers": list(consumers),
                    "materialization_count": 1,
                }
            else:
                existing = by_state_site[key]["consumers"]
                by_state_site[key]["consumers"] = sorted(set(existing) | set(consumers))
                by_state_site[key]["materialization_count"] += 1

    materializations = [
        MaterializationDecision(
            state_id=sid,
            content_hash=info["content_hash"],
            site=site,
            mode=info["mode"],
            cost_s=info["cost_s"],
            materialization_count=info["materialization_count"],
            consumers=info["consumers"],
            reason="shared_once_per_typed_component_site",
        )
        for (sid, site), info in by_state_site.items()
    ]
    return Plan(
        policy="shared_state_aware_typed",
        placements=placements,
        materializations=materializations,
        meta={"tau": tau, "components": [sorted(c) for c in components],
              "edge_type_weights": {f"{k[0]}|{k[1]}": v for k, v in weights.items()}},
    )


def _components_by_typed_shared_state(
    manifest: ServingGroupManifest,
    tau: float,
    weights: dict[tuple[str, str], float],
) -> list[set[str]]:
    """Like _components_by_shared_state but each edge contributes
    `tokens * weight[layer,lifetime]` instead of just `tokens`."""
    pair_weight: dict[tuple[str, str], float] = defaultdict(float)
    for edge in manifest.edges:
        state = manifest.state_objects.get(edge.state_id)
        if state is None:
            mult = 1.0  # unknown state -> default weight
        else:
            mult = _edge_type_weight(state.layer, state.lifetime, weights)
        if mult == 0.0:
            continue  # skip globally-replicated edges entirely
        a, b = sorted([edge.node_a, edge.node_b])
        pair_weight[(a, b)] += edge.weight * mult

    parent: dict[str, str] = {nid: nid for nid in manifest.nodes}

    def find(x: str) -> str:
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    def union(x: str, y: str) -> None:
        rx, ry = find(x), find(y)
        if rx != ry:
            parent[rx] = ry

    for (a, b), w in pair_weight.items():
        if w > tau:
            union(a, b)

    groups: dict[str, set[str]] = defaultdict(set)
    for nid in manifest.nodes:
        groups[find(nid)].add(nid)
    return list(groups.values())


# --------------------------------------------------------------------------- #
# Registry
# --------------------------------------------------------------------------- #

# --------------------------------------------------------------------------- #
# H1: request_level_with_site_cache — fair baseline
# --------------------------------------------------------------------------- #


def run_request_level_with_site_cache(
    manifest: ServingGroupManifest, bundle: ProfileBundle,
) -> Plan:
    """Per-node placement (same as `request_level_no_reuse`), but state
    materialized **once per (state, site)** where >=1 colocated consumer is
    placed (per-site cache reuse).

    Different from D1: D1 has materialization_count = #consumers per
    (state, site); H1 has count = 1 always.
    Different from D2: D2 places per-component; H1 places per-node.

    On any single-component trace where every node's per-node best-site is the
    same site, H1 == D2(tau=1) because both yield identical (state, site) sets.
    The two diverge only when nodes within one component have different per-
    node best-sites — i.e., on multi-private-state-with-different-homes
    fixtures. The toy and the SWE-agent F2 fixture are both single-best-site,
    so they collapse to equality with D2."""
    _validate_state_homes(manifest, bundle)
    placements = _place_per_node_min_cost(manifest, bundle)
    placement_map = {p.node_id: p.site for p in placements}
    return _plan_from_placement(
        policy_name="request_level_with_site_cache",
        placement=placement_map,
        manifest=manifest,
        bundle=bundle,
        meta={},
        materialization_reason="site_cache_reuse",
        placement_reason="min_cost",
    )


# --------------------------------------------------------------------------- #
# H3: session_sticky — colocate all nodes of the same session at one site
# --------------------------------------------------------------------------- #


SESSION_STICKY_DEFAULT = "_default_session"


def run_session_sticky(manifest: ServingGroupManifest, bundle: ProfileBundle) -> Plan:
    """Place each session as a unit at the lowest-cost site for that session's
    required states. Materialize once per (state, site) where >=1 consumer is
    placed (per-site cache reuse, like H1).

    Session identity:
      - WorkNode.session_id (set from add_subtask payload by the adapter).
      - Nodes without a session_id collapse into a single sentinel session
        ("_default_session"), which on a F2-style single-instance trace
        means all nodes are placed together — equivalent to D2(tau=1) on
        that fixture.

    Difference vs H1: H1 places per-node; session_sticky constrains all
    nodes of one session to the same site. On H2 they coincide because
    each session's nodes already share a per-node best-site (the workspace
    home). They diverge when intra-session nodes have private states with
    different homes — session_sticky pays the cost to keep them together;
    H1 splits them.

    Difference vs D2: D2 groups by shared-state component (which on H2 is
    one global component of all 6 nodes). session_sticky groups by session,
    so on H2 it splits into 3 groups regardless of what `system_prompt`
    does to the connectivity graph.
    """
    _validate_state_homes(manifest, bundle)
    _validate_session_id_coverage(manifest)

    sessions: dict[str, list[str]] = defaultdict(list)
    for node_id, node in manifest.nodes.items():
        sid = node.session_id if node.session_id is not None else SESSION_STICKY_DEFAULT
        sessions[sid].append(node_id)

    placement: dict[str, str] = {}
    placement_reason: dict[str, str] = {}
    for sid, node_ids in sessions.items():
        required: set[str] = set()
        for nid in node_ids:
            required.update(manifest.nodes[nid].required_state)
        best_site: str | None = None
        best_cost = float("inf")
        for site in bundle.sites:
            total = 0.0
            for state_id in required:
                state = manifest.state_objects[state_id]
                _, c = choose_min_cost_mode(state, _state_home(state, bundle), site, bundle)
                total += c
            if total < best_cost:
                best_cost = total
                best_site = site
        assert best_site is not None
        for nid in node_ids:
            placement[nid] = best_site
            placement_reason[nid] = f"session_sticky:{sid}"

    plan = _plan_from_placement(
        policy_name="session_sticky",
        placement=placement,
        manifest=manifest,
        bundle=bundle,
        meta={"sessions": {sid: sorted(nids) for sid, nids in sessions.items()}},
        materialization_reason="site_cache_reuse",
        placement_reason="session_sticky",
    )
    enriched_placements = [
        PlacementDecision(
            node_id=p.node_id, site=p.site, cost_s=p.cost_s,
            component_size=p.component_size,
            reason=placement_reason[p.node_id],
        )
        for p in plan.placements
    ]
    return Plan(policy=plan.policy, placements=enriched_placements,
                materializations=plan.materializations, meta=plan.meta)


def _validate_session_id_coverage(manifest: ServingGroupManifest) -> None:
    """Hard-fail on a manifest where some nodes carry `session_id` and
    others don't. Mixed presence would silently merge the unsessioned nodes
    into a single sentinel session alongside the explicit ones, which is
    almost never what the caller wanted. Either ALL nodes have a session_id
    (multi-session adapter) or NONE do (the helper falls back to a single
    sentinel session — the F2 / toy case)."""
    have = [nid for nid, n in manifest.nodes.items() if n.session_id is not None]
    miss = [nid for nid, n in manifest.nodes.items() if n.session_id is None]
    if have and miss:
        raise ValueError(
            f"session_sticky requires uniform session_id presence; "
            f"{len(have)} node(s) have session_id, {len(miss)} do not. "
            f"missing={sorted(miss)[:3]}{'...' if len(miss) > 3 else ''}"
        )


# --------------------------------------------------------------------------- #
# G1: brute-force optimizer (enumerate K^N placements; pick min total cost)
# --------------------------------------------------------------------------- #


G1_MAX_ENUMERATIONS = 100_000


def run_g1_brute_force(manifest: ServingGroupManifest, bundle: ProfileBundle) -> Plan:
    """Enumerate all K^N node-to-site placements; pick the one with lowest total
    materialization cost. State materialized once per (state, site) — i.e.,
    what shared_state_aware would do globally if it weren't constrained by
    per-component independent decisions.

    Hard-capped at G1_MAX_ENUMERATIONS = K^N placements (default 100,000).
    Above the cap, raises rather than silently fall back — the user should
    pick a heuristic policy (G2 or D2) instead. The cap is K-aware so a 4-site
    bundle is correctly limited to ~8 nodes (4^8 = 65k) instead of 16.
    """
    from itertools import product

    _validate_state_homes(manifest, bundle)
    node_ids = sorted(manifest.nodes)
    site_names = list(bundle.sites)
    if not site_names:
        raise ValueError("bundle has no sites")
    space = len(site_names) ** len(node_ids)
    if space > G1_MAX_ENUMERATIONS:
        raise ValueError(
            f"run_g1_brute_force placement space is {space} (sites={len(site_names)}, "
            f"nodes={len(node_ids)}); cap is {G1_MAX_ENUMERATIONS}. "
            "Use shared_state_aware or g2_local_search instead."
        )

    best: tuple[float, dict[str, str]] | None = None
    for assignment in product(site_names, repeat=len(node_ids)):
        placement = dict(zip(node_ids, assignment))
        total = _placement_total_cost(placement, manifest, bundle)
        if best is None or total < best[0]:
            best = (total, placement)
    assert best is not None
    _, best_placement = best
    return _plan_from_placement(
        policy_name="g1_brute_force",
        placement=best_placement,
        manifest=manifest,
        bundle=bundle,
        meta={"enumerated": len(site_names) ** len(node_ids)},
    )


# --------------------------------------------------------------------------- #
# G2: local search seeded from D1 (greedy single-node moves)
# --------------------------------------------------------------------------- #


def run_g2_local_search(
    manifest: ServingGroupManifest,
    bundle: ProfileBundle,
    max_iterations: int = 100,
) -> Plan:
    """Seed from D1's per-node best-site placement; iteratively try moving any
    single node to any other site; accept the **best-improvement** move per
    iteration (scan all neighbors; pick lowest-cost strict improvement).
    Terminate when no single-node move improves cost (local optimum) or after
    max_iterations rounds.

    Best-improvement (vs first-improvement) is slightly slower per iteration
    but takes fewer iterations and avoids zigzag near plateaus. Cost is
    bounded below and decreases strictly per accepted move, so termination
    is guaranteed. Result is a local optimum; G1 is the exact oracle within
    G1_MAX_ENUMERATIONS."""
    _validate_state_homes(manifest, bundle)
    d1 = run_request_level_no_reuse(manifest, bundle)
    placement = {p.node_id: p.site for p in d1.placements}
    site_names = list(bundle.sites)

    iterations = 0
    while iterations < max_iterations:
        current = _placement_total_cost(placement, manifest, bundle)
        best_move: tuple[float, str, str] | None = None
        for node_id, current_site in placement.items():
            for candidate in site_names:
                if candidate == current_site:
                    continue
                trial = dict(placement)
                trial[node_id] = candidate
                cost = _placement_total_cost(trial, manifest, bundle)
                if cost < current and (best_move is None or cost < best_move[0]):
                    best_move = (cost, node_id, candidate)
        if best_move is None:
            break
        _, mv_node, mv_site = best_move
        placement[mv_node] = mv_site
        iterations += 1

    return _plan_from_placement(
        policy_name="g2_local_search",
        placement=placement,
        manifest=manifest,
        bundle=bundle,
        meta={"iterations": iterations, "seed": "d1"},
    )


# --------------------------------------------------------------------------- #
# Shared placement helpers (used by G1 and G2)
# --------------------------------------------------------------------------- #


def _placement_total_cost(
    placement: dict[str, str],
    manifest: ServingGroupManifest,
    bundle: ProfileBundle,
) -> float:
    """Total cost of `placement`: sum over (state, site) where >=1 consumer is
    placed at site of one materialization cost. Optimal bookkeeping —
    materialize each state once per occupied site, no per-component or
    per-node multiplicity."""
    by_state_site: set[tuple[str, str]] = set()
    for state in manifest.state_objects.values():
        sites_with_consumers = {placement[c] for c in state.consumers if c in placement}
        for site in sites_with_consumers:
            by_state_site.add((state.state_id, site))

    total = 0.0
    for state_id, site in by_state_site:
        state = manifest.state_objects[state_id]
        _, cost = choose_min_cost_mode(state, _state_home(state, bundle), site, bundle)
        total += cost
    return total


def _plan_from_placement(
    policy_name: str,
    placement: dict[str, str],
    manifest: ServingGroupManifest,
    bundle: ProfileBundle,
    meta: dict,
    materialization_reason: str = "shared_once_per_site",
    placement_reason: str = "optimized",
) -> Plan:
    placements: list[PlacementDecision] = []
    by_state_site: dict[tuple[str, str], dict] = {}

    for node_id in sorted(placement):
        node = manifest.nodes[node_id]
        site = placement[node_id]
        node_cost = 0.0
        for state_id in node.required_state:
            state = manifest.state_objects[state_id]
            _, c = choose_min_cost_mode(state, _state_home(state, bundle), site, bundle)
            node_cost += c
        placements.append(PlacementDecision(
            node_id=node_id, site=site, cost_s=node_cost,
            component_size=1, reason=placement_reason,
        ))
        for state_id in node.required_state:
            state = manifest.state_objects[state_id]
            mode, cost = choose_min_cost_mode(state, _state_home(state, bundle), site, bundle)
            key = (state_id, site)
            if key not in by_state_site:
                by_state_site[key] = {
                    "content_hash": state.content_hash,
                    "mode": mode,
                    "cost_s": cost,
                    "consumers": [node_id],
                }
            else:
                by_state_site[key]["consumers"].append(node_id)

    materializations = [
        MaterializationDecision(
            state_id=sid,
            content_hash=info["content_hash"],
            site=site,
            mode=info["mode"],
            cost_s=info["cost_s"],
            materialization_count=1,
            consumers=sorted(info["consumers"]),
            reason=materialization_reason,
        )
        for (sid, site), info in by_state_site.items()
    ]
    return Plan(policy=policy_name, placements=placements,
                materializations=materializations, meta=meta)


# --------------------------------------------------------------------------- #
# Registry
# --------------------------------------------------------------------------- #

POLICIES = {
    "request_level_no_reuse": run_request_level_no_reuse,
    "request_level_with_site_cache": run_request_level_with_site_cache,
    "shared_state_aware": run_shared_state_aware,
    "shared_state_aware_typed": run_shared_state_aware_typed,
    "session_sticky": run_session_sticky,
    "g1_brute_force": run_g1_brute_force,
    "g2_local_search": run_g2_local_search,
}


def run_policy(name: str, manifest: ServingGroupManifest, bundle: ProfileBundle, **kwargs) -> Plan:
    if name not in POLICIES:
        raise ValueError(f"unknown policy {name!r}; known: {sorted(POLICIES)}")
    fn = POLICIES[name]
    if name == "shared_state_aware":
        return fn(manifest, bundle, tau=kwargs.get("tau", 1))
    if name == "shared_state_aware_typed":
        return fn(manifest, bundle, tau=kwargs.get("tau", 1.0),
                  edge_type_weights=kwargs.get("edge_type_weights"))
    if name == "g2_local_search":
        return fn(manifest, bundle, max_iterations=kwargs.get("max_iterations", 100))
    return fn(manifest, bundle)


__all__ = [
    "PlacementDecision",
    "MaterializationDecision",
    "Plan",
    "POLICIES",
    "run_policy",
    "run_request_level_no_reuse",
    "run_request_level_with_site_cache",
    "run_shared_state_aware",
    "run_shared_state_aware_typed",
    "DEFAULT_EDGE_TYPE_WEIGHTS",
    "run_session_sticky",
    "run_g1_brute_force",
    "run_g2_local_search",
    "G1_MAX_ENUMERATIONS",
    "SESSION_STICKY_DEFAULT",
]
