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
# Registry
# --------------------------------------------------------------------------- #

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
            component_size=1, reason="optimized",
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
            reason="shared_once_per_site",
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
    "shared_state_aware": run_shared_state_aware,
    "g1_brute_force": run_g1_brute_force,
    "g2_local_search": run_g2_local_search,
}


def run_policy(name: str, manifest: ServingGroupManifest, bundle: ProfileBundle, **kwargs) -> Plan:
    if name not in POLICIES:
        raise ValueError(f"unknown policy {name!r}; known: {sorted(POLICIES)}")
    fn = POLICIES[name]
    if name == "shared_state_aware":
        return fn(manifest, bundle, tau=kwargs.get("tau", 1))
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
    "run_shared_state_aware",
    "run_g1_brute_force",
    "run_g2_local_search",
    "G1_MAX_NODES",
]
