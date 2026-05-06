"""Reconstitution policies (Workstream K5).

Six policies that produce per-(workflow, state) reconstitution plans
for a MobilityEpisode. K4's fluid simulator consumes these plans;
K7's falsification gauntlet evaluates them under three capacity
regimes (T1 infinite, T2 prefill-only, T3 multi-resource).

Each policy: `policy_fn(episode, manifests, bundle, warmness) ->
dict[workflow_id, list[ReconstitutionAction]]`. The output plan is
static (no online interaction with the simulator); `mixed_min_pressure`
is a *greedy* upfront approximation of what an online oracle would
choose.

Per A3 audit: reconstitution is at the (state, site) granularity (L1
abstraction), NOT (component, site) (L2). K3's resource model already
enforces this; K5 just picks modes and destinations.
"""
from __future__ import annotations

import math
from dataclasses import dataclass

from .costs import (
    ARTIFACT_COPY,
    CONTEXT_REPLAY,
    KV_TRANSFER,
    TEXT_TRANSFER,
    allowed_modes_for_state,
)
from .episode import MobilityEpisode
from .fluid_sim import ReconstitutionAction
from .manifest import ServingGroupManifest, StateObject
from .profiles import ProfileBundle
from .resources import (
    WARM_REUSE,
    WORKSPACE_HYDRATE,
    reconstitution_cost,
)
from .warmness import WarmnessMap


# ---------------------------------------------------------------------------
# Per-policy planners
# ---------------------------------------------------------------------------


def min_cost_independent(
    episode: MobilityEpisode,
    manifests: dict[str, ServingGroupManifest],
    bundle: ProfileBundle,
    warmness: WarmnessMap,
) -> dict[str, list[ReconstitutionAction]]:
    """Per-(workflow, state), pick the (mode, dst_site) pair that
    minimizes wallclock_s. Greedy with no global view — the failure
    mode collaborator 2 calls out: every workflow picks the fastest
    site, stampeding it under capacity.
    """
    plan: dict[str, list[ReconstitutionAction]] = {}
    for wf in episode.workflows:
        manifest = manifests[wf.workflow_id]
        actions: list[ReconstitutionAction] = []
        src = wf.src_site or episode.source_sites[0]
        for sid, state in sorted(manifest.state_objects.items()):
            best_action = _pick_min_cost_action(
                state, src, episode.destination_sites, bundle, warmness, wf.workflow_id,
            )
            if best_action is not None:
                actions.append(best_action)
        plan[wf.workflow_id] = actions
    return plan


def replay_all(
    episode: MobilityEpisode,
    manifests: dict[str, ServingGroupManifest],
    bundle: ProfileBundle,
    warmness: WarmnessMap,
) -> dict[str, list[ReconstitutionAction]]:
    """Force CONTEXT_REPLAY for prompt-context; ARTIFACT_COPY for
    workspace; TEXT_TRANSFER for memory. Stampedes prefill under
    finite prefill capacity (gauntlet T2's expected failure mode for
    fixed-mode policies)."""
    return _force_mode_per_layer_plan(
        episode, manifests, bundle, warmness,
        prompt_mode=CONTEXT_REPLAY,
        workspace_mode=ARTIFACT_COPY,
        memory_mode=TEXT_TRANSFER,
    )


def kv_all(
    episode: MobilityEpisode,
    manifests: dict[str, ServingGroupManifest],
    bundle: ProfileBundle,
    warmness: WarmnessMap,
) -> dict[str, list[ReconstitutionAction]]:
    """Force KV_TRANSFER for prompt-context; ARTIFACT_COPY for workspace;
    TEXT_TRANSFER for memory. Stampedes network under finite link bps."""
    return _force_mode_per_layer_plan(
        episode, manifests, bundle, warmness,
        prompt_mode=KV_TRANSFER,
        workspace_mode=ARTIFACT_COPY,
        memory_mode=TEXT_TRANSFER,
    )


def cache_reuse(
    episode: MobilityEpisode,
    manifests: dict[str, ServingGroupManifest],
    bundle: ProfileBundle,
    warmness: WarmnessMap,
) -> dict[str, list[ReconstitutionAction]]:
    """Warm hit at any destination -> WARM_REUSE. Cold -> cheaper of
    replay/KV per state (calls min_cost_independent's per-action picker
    for the cold path). Routes each workflow to the destination with
    the most warm hits, breaking ties on min wallclock."""
    plan: dict[str, list[ReconstitutionAction]] = {}
    for wf in episode.workflows:
        manifest = manifests[wf.workflow_id]
        src = wf.src_site or episode.source_sites[0]
        # Pick dst that maximizes warm hits for this workflow's states.
        dst_score: list[tuple[int, float, str]] = []
        for dst in episode.destination_sites:
            warm_hits = sum(1 for sid in manifest.state_objects
                            if warmness.is_warm(sid, dst))
            total_cost = sum(
                _min_cold_cost(manifest.state_objects[sid], src, dst, bundle, warmness)
                for sid in manifest.state_objects
            )
            dst_score.append((-warm_hits, total_cost, dst))
        dst_score.sort()
        best_dst = dst_score[0][2]
        actions: list[ReconstitutionAction] = []
        for sid, state in sorted(manifest.state_objects.items()):
            if warmness.is_warm(sid, best_dst):
                actions.append(ReconstitutionAction(
                    workflow_id=wf.workflow_id, state_id=sid, mode=WARM_REUSE,
                    src_site=src, dst_site=best_dst, reason="warm_hit",
                ))
            else:
                action = _pick_min_cost_action(
                    state, src, (best_dst,), bundle, warmness, wf.workflow_id,
                )
                if action is not None:
                    actions.append(action)
        plan[wf.workflow_id] = actions
    return plan


def workspace_sticky(
    episode: MobilityEpisode,
    manifests: dict[str, ServingGroupManifest],
    bundle: ProfileBundle,
    warmness: WarmnessMap,
) -> dict[str, list[ReconstitutionAction]]:
    """Forbid moving workspace state cross-site. Route workflow to a
    destination where its workspace is already warm OR same as src
    (so workspace_hydrate is local). Replay everything else."""
    plan: dict[str, list[ReconstitutionAction]] = {}
    for wf in episode.workflows:
        manifest = manifests[wf.workflow_id]
        src = wf.src_site or episode.source_sites[0]
        # Find dst sites where the workspace is already locally available.
        workspace_states = [s for s in manifest.state_objects.values() if s.layer == "workspace"]
        eligible_dsts = []
        for dst in episode.destination_sites:
            if all(
                warmness.is_warm(ws.state_id, dst) or dst == src
                for ws in workspace_states
            ):
                eligible_dsts.append(dst)
        if not eligible_dsts:
            # No sticky destination: fall back to src (workspace stays put).
            eligible_dsts = [src] if src in episode.destination_sites else [episode.destination_sites[0]]
        # Among eligible dsts, pick the one with cheapest non-workspace replay.
        best_dst = eligible_dsts[0]
        best_cost = float("inf")
        for dst in eligible_dsts:
            total = sum(
                _min_cold_cost(s, src, dst, bundle, warmness)
                for s in manifest.state_objects.values() if s.layer != "workspace"
            )
            if total < best_cost:
                best_cost = total
                best_dst = dst
        actions: list[ReconstitutionAction] = []
        for sid, state in sorted(manifest.state_objects.items()):
            if warmness.is_warm(sid, best_dst):
                actions.append(ReconstitutionAction(
                    workflow_id=wf.workflow_id, state_id=sid, mode=WARM_REUSE,
                    src_site=src, dst_site=best_dst, reason="warm_hit",
                ))
            elif state.layer == "workspace":
                # Workspace is sticky: hydrate locally at dst (no wire transfer).
                actions.append(ReconstitutionAction(
                    workflow_id=wf.workflow_id, state_id=sid, mode=WORKSPACE_HYDRATE,
                    src_site=best_dst, dst_site=best_dst, reason="workspace_sticky",
                ))
            else:
                action = _pick_min_cost_action(
                    state, src, (best_dst,), bundle, warmness, wf.workflow_id,
                )
                if action is not None:
                    actions.append(action)
        plan[wf.workflow_id] = actions
    return plan


def mixed_min_pressure(
    episode: MobilityEpisode,
    manifests: dict[str, ServingGroupManifest],
    bundle: ProfileBundle,
    warmness: WarmnessMap,
) -> dict[str, list[ReconstitutionAction]]:
    """Greedy load-aware heuristic (NOT an offline oracle).

    For each workflow in deterministic order, scans candidate
    (prompt_mode, dst_site) pairs and picks the one that minimizes the
    NEW maximum predicted resource utilization across (network,
    prefill, workspace_hydrate). Tracks per-resource cumulative
    "demand" units as we plan; each new workflow's contribution is
    estimated from K3's resource_cost (assuming no warm hits — warm
    hits short-circuit before the load-aware logic).

    This is a real bin-packing greedy heuristic, replacing the prior
    round-robin that the K-architectural critic flagged as load-
    unaware. Still NOT offline-optimal (that would be the deferred
    K9 ILP); a strict ordering or future workload could leave it worse
    than offline. But it is provably better than round-robin under
    skewed demand.

    For workspace, picks `WORKSPACE_HYDRATE` (local) when dst's
    workspace_hydrate_bps × num_dst-loaded < cross-site bytes-per-link
    × num_link-loaded; otherwise `ARTIFACT_COPY`. This is the
    bin-packing "where does the byte go" decision the round-robin
    version skipped.
    """
    plan: dict[str, list[ReconstitutionAction]] = {}
    workflows_sorted = sorted(episode.workflows, key=lambda w: w.workflow_id)
    # Cumulative load tracking, per-(resource, site_or_link).
    prefill_load: dict[str, float] = {dst: 0.0 for dst in episode.destination_sites}
    workspace_load: dict[str, float] = {dst: 0.0 for dst in episode.destination_sites}
    network_load: dict[tuple[str, str], float] = {}

    def link_key(a: str, b: str) -> tuple[str, str]:
        return tuple(sorted([a, b]))

    def predicted_max_pressure(prefill_add: dict[str, float],
                               workspace_add: dict[str, float],
                               network_add: dict[tuple[str, str], float]) -> float:
        """Highest single-resource load if we apply the proposed adds.
        Resources are normalized by their estimated capacity at default-
        scale (no per-axis weights — the optimizer just tries to keep
        them balanced)."""
        b_max = 0.0
        for site, add in prefill_add.items():
            new_load = prefill_load.get(site, 0.0) + add
            cap = bundle.site(site).prefill_tok_s
            if cap > 0 and cap < math.inf:
                b_max = max(b_max, new_load / cap)
        for site, add in workspace_add.items():
            new_load = workspace_load.get(site, 0.0) + add
            cap = bundle.site(site).workspace_hydrate_bps
            if cap > 0 and cap < math.inf:
                b_max = max(b_max, 8.0 * new_load / cap)
        for link, add in network_add.items():
            new_load = network_load.get(link, 0.0) + add
            try:
                cap = bundle.link(*link).effective_bps
            except (ValueError, KeyError):
                cap = math.inf
            if cap > 0 and cap < math.inf:
                b_max = max(b_max, 8.0 * new_load / cap)
        return b_max

    for wf in workflows_sorted:
        manifest = manifests[wf.workflow_id]
        src = wf.src_site or episode.source_sites[0]

        # Search over (prompt_mode, dst) pairs.
        candidates: list[tuple[float, str, str, list[ReconstitutionAction]]] = []
        for dst in episode.destination_sites:
            for prompt_mode in (CONTEXT_REPLAY, KV_TRANSFER):
                # Estimate this assignment's load contribution.
                prefill_add: dict[str, float] = {}
                workspace_add: dict[str, float] = {}
                network_add: dict[tuple[str, str], float] = {}
                actions: list[ReconstitutionAction] = []
                feasible = True
                for sid, state in sorted(manifest.state_objects.items()):
                    if warmness.is_warm(sid, dst):
                        actions.append(ReconstitutionAction(
                            workflow_id=wf.workflow_id, state_id=sid, mode=WARM_REUSE,
                            src_site=src, dst_site=dst, reason="warm_hit",
                        ))
                        continue
                    if state.layer == "prompt_context":
                        mode = prompt_mode if not (prompt_mode == KV_TRANSFER and src == dst) else CONTEXT_REPLAY
                        if mode == CONTEXT_REPLAY:
                            prefill_add[dst] = prefill_add.get(dst, 0.0) + state.tokens
                        else:  # KV_TRANSFER
                            if src != dst:
                                key = link_key(src, dst)
                                network_add[key] = network_add.get(key, 0.0) + state.tokens * bundle.model.kv_bytes_per_token
                        actions.append(ReconstitutionAction(
                            workflow_id=wf.workflow_id, state_id=sid, mode=mode,
                            src_site=src, dst_site=dst, reason=f"mixed_load_aware:{mode}",
                        ))
                    elif state.layer == "workspace":
                        # Pick artifact_copy vs workspace_hydrate based on which
                        # would push the bottleneck higher.
                        bytes_ = state.bytes or 0
                        # Tentatively try ARTIFACT_COPY (cross-site).
                        ac_pressure = predicted_max_pressure(
                            prefill_add, workspace_add,
                            {**network_add, link_key(src, dst): network_add.get(link_key(src, dst), 0.0)
                             + (bytes_ if src != dst else 0)},
                        )
                        # Try WORKSPACE_HYDRATE (local at dst).
                        wh_pressure = predicted_max_pressure(
                            prefill_add,
                            {**workspace_add, dst: workspace_add.get(dst, 0.0) + bytes_},
                            network_add,
                        )
                        if wh_pressure < ac_pressure or src == dst:
                            workspace_add[dst] = workspace_add.get(dst, 0.0) + bytes_
                            actions.append(ReconstitutionAction(
                                workflow_id=wf.workflow_id, state_id=sid, mode=WORKSPACE_HYDRATE,
                                src_site=dst, dst_site=dst, reason="load_aware:hydrate",
                            ))
                        else:
                            if src != dst:
                                key = link_key(src, dst)
                                network_add[key] = network_add.get(key, 0.0) + bytes_
                            actions.append(ReconstitutionAction(
                                workflow_id=wf.workflow_id, state_id=sid, mode=ARTIFACT_COPY,
                                src_site=src, dst_site=dst, reason="load_aware:copy",
                            ))
                    elif state.layer == "memory":
                        if state.bytes is not None and src != dst:
                            key = link_key(src, dst)
                            network_add[key] = network_add.get(key, 0.0) + state.bytes
                        actions.append(ReconstitutionAction(
                            workflow_id=wf.workflow_id, state_id=sid, mode=TEXT_TRANSFER,
                            src_site=src, dst_site=dst, reason="mixed_memory",
                        ))
                if not feasible:
                    continue
                pressure = predicted_max_pressure(prefill_add, workspace_add, network_add)
                candidates.append((pressure, dst, prompt_mode, actions))

        if not candidates:
            plan[wf.workflow_id] = []
            continue
        # Pick the candidate with the smallest predicted max pressure.
        candidates.sort(key=lambda c: (c[0], c[1], c[2]))
        _, chosen_dst, chosen_mode, chosen_actions = candidates[0]
        plan[wf.workflow_id] = chosen_actions
        # Commit the chosen candidate's loads to cumulative tracking.
        for action in chosen_actions:
            if action.mode == WARM_REUSE:
                continue
            state = manifest.state_objects[action.state_id]
            if state.layer == "prompt_context":
                if action.mode == CONTEXT_REPLAY:
                    prefill_load[action.dst_site] = prefill_load.get(action.dst_site, 0.0) + state.tokens
                elif action.mode == KV_TRANSFER and action.src_site != action.dst_site:
                    key = link_key(action.src_site, action.dst_site)
                    network_load[key] = network_load.get(key, 0.0) + state.tokens * bundle.model.kv_bytes_per_token
            elif state.layer == "workspace":
                bytes_ = state.bytes or 0
                if action.mode == WORKSPACE_HYDRATE:
                    workspace_load[action.dst_site] = workspace_load.get(action.dst_site, 0.0) + bytes_
                elif action.mode == ARTIFACT_COPY and action.src_site != action.dst_site:
                    key = link_key(action.src_site, action.dst_site)
                    network_load[key] = network_load.get(key, 0.0) + bytes_
            elif state.layer == "memory":
                if state.bytes is not None and action.src_site != action.dst_site:
                    key = link_key(action.src_site, action.dst_site)
                    network_load[key] = network_load.get(key, 0.0) + state.bytes
    return plan


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _pick_min_cost_action(
    state: StateObject,
    src: str,
    candidate_dsts: tuple[str, ...],
    bundle: ProfileBundle,
    warmness: WarmnessMap,
    workflow_id: str,
) -> ReconstitutionAction | None:
    """Pick (mode, dst) minimizing wallclock_s for one state. Returns
    None if no allowed mode exists for this state's layer."""
    allowed = allowed_modes_for_state(state)
    if not allowed:
        return None
    best: tuple[float, str, str] | None = None
    for dst in candidate_dsts:
        for mode in allowed:
            try:
                cost = reconstitution_cost(state, mode, src, dst, bundle, warmness)
            except ValueError:
                # mode infeasible at (src, dst, state); skip
                continue
            key = (cost.wallclock_s, mode, dst)
            if best is None or key < (best[0], best[1], best[2]):
                best = key
    if best is None:
        return None
    cost_s, mode, dst = best
    return ReconstitutionAction(
        workflow_id=workflow_id, state_id=state.state_id, mode=mode,
        src_site=src, dst_site=dst, reason="min_cost_independent",
    )


def _min_cold_cost(state: StateObject, src: str, dst: str,
                   bundle: ProfileBundle, warmness: WarmnessMap) -> float:
    """Lower bound on wall-clock if state is cold at dst. Used by
    cache_reuse + workspace_sticky for destination ranking."""
    allowed = allowed_modes_for_state(state)
    best = float("inf")
    for mode in allowed:
        try:
            cost = reconstitution_cost(state, mode, src, dst, bundle, warmness)
        except ValueError:
            continue
        if cost.wallclock_s < best:
            best = cost.wallclock_s
    return best


def _force_mode_per_layer_plan(
    episode: MobilityEpisode,
    manifests: dict[str, ServingGroupManifest],
    bundle: ProfileBundle,
    warmness: WarmnessMap,
    *,
    prompt_mode: str,
    workspace_mode: str,
    memory_mode: str,
) -> dict[str, list[ReconstitutionAction]]:
    """Shared helper for replay_all / kv_all: force a specific mode per
    state layer across all workflows. Workflows go to the first
    destination site (no load balancing — that's the point)."""
    plan: dict[str, list[ReconstitutionAction]] = {}
    dst = episode.destination_sites[0]
    for wf in episode.workflows:
        manifest = manifests[wf.workflow_id]
        src = wf.src_site or episode.source_sites[0]
        actions: list[ReconstitutionAction] = []
        for sid, state in sorted(manifest.state_objects.items()):
            if warmness.is_warm(sid, dst):
                mode = WARM_REUSE
            elif state.layer == "prompt_context":
                # KV_TRANSFER same-site falls back to CONTEXT_REPLAY (per costs.py).
                mode = prompt_mode if not (prompt_mode == KV_TRANSFER and src == dst) else CONTEXT_REPLAY
            elif state.layer == "workspace":
                mode = workspace_mode
            elif state.layer == "memory":
                mode = memory_mode
            else:
                # Skip unrecognized layers.
                continue
            actions.append(ReconstitutionAction(
                workflow_id=wf.workflow_id, state_id=sid, mode=mode,
                src_site=src, dst_site=dst, reason=f"forced:{mode}",
            ))
        plan[wf.workflow_id] = actions
    return plan


# ---------------------------------------------------------------------------
# Registry
# ---------------------------------------------------------------------------

def random_mode(
    episode: MobilityEpisode,
    manifests: dict[str, ServingGroupManifest],
    bundle: ProfileBundle,
    warmness: WarmnessMap,
    *,
    seed: int = 0,
) -> dict[str, list[ReconstitutionAction]]:
    """Sanity-check baseline: assigns a random feasible mode per state.

    A successful gauntlet must show that `mixed_min_pressure` strictly
    beats `random_mode` — otherwise the diversification heuristic is no
    better than chance and the K abstraction is not earning its keep.
    Per K-architectural-critic finding: this is the missing baseline."""
    import random as _random
    rng = _random.Random(seed)
    plan: dict[str, list[ReconstitutionAction]] = {}
    for wf in sorted(episode.workflows, key=lambda w: w.workflow_id):
        manifest = manifests[wf.workflow_id]
        src = wf.src_site or episode.source_sites[0]
        dst = rng.choice(list(episode.destination_sites))
        actions: list[ReconstitutionAction] = []
        for sid, state in sorted(manifest.state_objects.items()):
            if warmness.is_warm(sid, dst):
                actions.append(ReconstitutionAction(
                    workflow_id=wf.workflow_id, state_id=sid, mode=WARM_REUSE,
                    src_site=src, dst_site=dst, reason="warm_hit",
                ))
                continue
            allowed = allowed_modes_for_state(state)
            if not allowed:
                continue
            mode = rng.choice(list(allowed))
            # Avoid infeasible same-site KV_TRANSFER (matches _force_mode_per_layer_plan).
            if mode == KV_TRANSFER and src == dst:
                mode = CONTEXT_REPLAY
            actions.append(ReconstitutionAction(
                workflow_id=wf.workflow_id, state_id=sid, mode=mode,
                src_site=src, dst_site=dst, reason=f"random:{mode}",
            ))
        plan[wf.workflow_id] = actions
    return plan


RECONSTITUTION_POLICIES = {
    "min_cost_independent": min_cost_independent,
    "replay_all": replay_all,
    "kv_all": kv_all,
    "cache_reuse": cache_reuse,
    "workspace_sticky": workspace_sticky,
    "mixed_min_pressure": mixed_min_pressure,
    "random_mode": random_mode,
}


def run_reconstitution_policy(
    name: str,
    episode: MobilityEpisode,
    manifests: dict[str, ServingGroupManifest],
    bundle: ProfileBundle,
    warmness: WarmnessMap,
) -> dict[str, list[ReconstitutionAction]]:
    if name not in RECONSTITUTION_POLICIES:
        raise ValueError(f"unknown reconstitution policy {name!r}; "
                         f"known: {sorted(RECONSTITUTION_POLICIES)}")
    return RECONSTITUTION_POLICIES[name](episode, manifests, bundle, warmness)


__all__ = [
    "RECONSTITUTION_POLICIES",
    "run_reconstitution_policy",
    "min_cost_independent",
    "replay_all",
    "kv_all",
    "cache_reuse",
    "workspace_sticky",
    "mixed_min_pressure",
]
