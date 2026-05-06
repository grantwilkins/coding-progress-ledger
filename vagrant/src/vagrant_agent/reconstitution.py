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
    """Greedy fluid-aware oracle (upfront approximation of an online
    oracle). For prompt-context states, ROUND-ROBINS workflows across
    {CONTEXT_REPLAY, KV_TRANSFER} so neither resource is uniformly
    saturated. For workspace, distributes destinations across the
    available destination sites to avoid landing-pressure storms.

    Not offline-optimal (that would be K9 deferred); empirically should
    beat any single fixed-mode policy under multi-resource saturation
    by intentionally diversifying."""
    plan: dict[str, list[ReconstitutionAction]] = {}
    workflows_sorted = sorted(episode.workflows, key=lambda w: w.workflow_id)
    n_dst = len(episode.destination_sites)
    for i, wf in enumerate(workflows_sorted):
        manifest = manifests[wf.workflow_id]
        src = wf.src_site or episode.source_sites[0]
        # Round-robin destinations to spread landing pressure.
        dst = episode.destination_sites[i % n_dst]
        # Round-robin between CONTEXT_REPLAY and KV_TRANSFER for prompt states.
        prompt_mode = CONTEXT_REPLAY if i % 2 == 0 else KV_TRANSFER
        actions: list[ReconstitutionAction] = []
        for sid, state in sorted(manifest.state_objects.items()):
            if warmness.is_warm(sid, dst):
                actions.append(ReconstitutionAction(
                    workflow_id=wf.workflow_id, state_id=sid, mode=WARM_REUSE,
                    src_site=src, dst_site=dst, reason="warm_hit",
                ))
                continue
            if state.layer == "prompt_context":
                # Use the round-robin assignment if feasible; fall back
                # to context_replay (always feasible).
                mode = prompt_mode
                if mode == KV_TRANSFER and src == dst:
                    mode = CONTEXT_REPLAY
                actions.append(ReconstitutionAction(
                    workflow_id=wf.workflow_id, state_id=sid, mode=mode,
                    src_site=src, dst_site=dst,
                    reason=f"mixed_round_robin:{prompt_mode}",
                ))
            elif state.layer == "workspace":
                actions.append(ReconstitutionAction(
                    workflow_id=wf.workflow_id, state_id=sid, mode=ARTIFACT_COPY,
                    src_site=src, dst_site=dst, reason="mixed_workspace",
                ))
            elif state.layer == "memory":
                actions.append(ReconstitutionAction(
                    workflow_id=wf.workflow_id, state_id=sid, mode=TEXT_TRANSFER,
                    src_site=src, dst_site=dst, reason="mixed_memory",
                ))
        plan[wf.workflow_id] = actions
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

RECONSTITUTION_POLICIES = {
    "min_cost_independent": min_cost_independent,
    "replay_all": replay_all,
    "kv_all": kv_all,
    "cache_reuse": cache_reuse,
    "workspace_sticky": workspace_sticky,
    "mixed_min_pressure": mixed_min_pressure,
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
