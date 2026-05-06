"""Fluid batch simulator for mobility episodes (Workstream K4).

`simulate_fluid(episode, manifests, plan, bundle, warmness, budget)` runs
N reconstitution plans concurrently under per_resource fluid capacity:

  * Each in_flight action consumes resources from a 4_axis budget
    (network bps per link, prefill tok/s per site, workspace_hydrate
    bytes/s per site, kv_memory bytes per site).
  * At each event horizon, every resource's capacity is split
    EQUALLY among in_flight actions that need it: each demander gets
    `capacity / num_demanders`. **This is equal share, not true max_min
    fair share** — an action bottlenecked on prefill does NOT release its
    network share to a network_bottlenecked peer. The bias is
    *conservative* for T3 (it overstates contention, making mixed
    planning's win look smaller than it would under true max_min). Per
    A4 audit, this matches the "additive cost" assumption of the
    static cost model; a future M_workstream could refine.
  * Time advances to the next event (an action finishes, or a new
    action becomes eligible to start because its predecessor finished).
  * KV memory is a CAPACITY (not bandwidth): both at episode start
    (initial warmness from `episode.state_warmness`) and after each
    action completes, if a site's resident KV bytes exceed
    `kv_memory_bytes_per_site`, LRU eviction runs on that site's
    warmness entries until under cap.
  * No queues, no admission control, no scheduler. The fluid carve_out
    in CLAUDE.md is the only relaxation of the original "no capacity"
    rule.

K4 produces an `ActionTrace` per action (started_s, finished_s, dominant
bottleneck) and a final `WarmnessMap`. K7's gauntlet tests assert the
three falsification properties (T1/T2/T3) using these traces.

Determinism: action ordering within a single time step is deterministic
in (workflow_id, state_id, mode) lex order. The simulator does not use
random tie_breaks anywhere. Frozen_dataclass inputs are not mutated.

Per CLAUDE.md hard rule: K4 is the ONLY module that mutates a warmness
map. Returns a new map; never mutates in place.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Iterable

from .episode import MobilityEpisode
from .manifest import ServingGroupManifest, StateObject
from .profiles import ProfileBundle
from .resources import (
    ResourceBudget,
    ResourceCost,
    reconstitution_cost,
)
from .warmness import WarmnessMap

# Action ordering convention: actions for a workflow run sequentially
# in the plan order. K4 starts the head of each workflow's queue when
# the previous action in that workflow finishes.

# Resource axis names (used in bottleneck attribution).
NETWORK = "network"
PREFILL = "prefill"
WORKSPACE = "workspace"
KV_MEMORY = "kv_memory"
NONE = "none"

ALL_RESOURCES = (NETWORK, PREFILL, WORKSPACE, KV_MEMORY)


@dataclass(frozen=True)
class ReconstitutionAction:
    """A single (workflow, state, mode, src→dst) reconstitution step."""
    workflow_id: str
    state_id: str
    mode: str
    src_site: str | None
    dst_site: str
    reason: str = ""


@dataclass(frozen=True)
class ActionTrace:
    """K4 output: per_action wall_clock + dominant bottleneck.

    `bottleneck` is the resource axis that was binding at the moment
    this action finished — i.e., the resource for which this action
    received the smallest share_per_unit_demand.

    `wallclock_lower_bound_s` is what `costs.materialize_cost` would
    have returned for this action in isolation; used by K7 T1 to verify
    that under infinite capacity, simulated time matches the lower bound.
    """
    workflow_id: str
    state_id: str
    mode: str
    started_s: float
    finished_s: float
    bottleneck: str
    wallclock_lower_bound_s: float


@dataclass(frozen=True)
class SimulationResult:
    """Final K4 output."""
    actions: tuple[ActionTrace, ...]
    final_warmness: WarmnessMap
    makespan_s: float
    per_workflow_finish_s: dict[str, float]

    def p50_resume_s(self) -> float:
        """Time at which 50% of workflows have completed reconstitution."""
        if not self.per_workflow_finish_s:
            return 0.0
        finishes = sorted(self.per_workflow_finish_s.values())
        idx = max(0, math.ceil(0.5 * len(finishes)) - 1)
        return finishes[idx]

    def p90_resume_s(self) -> float:
        if not self.per_workflow_finish_s:
            return 0.0
        finishes = sorted(self.per_workflow_finish_s.values())
        # Conservative: take the (ceil(0.9 * N) - 1)-th element.
        idx = max(0, math.ceil(0.9 * len(finishes)) - 1)
        return finishes[idx]

    def p95_resume_s(self) -> float:
        if not self.per_workflow_finish_s:
            return 0.0
        finishes = sorted(self.per_workflow_finish_s.values())
        # Conservative: take the (ceil(0.95 * N) - 1)-th element.
        idx = max(0, math.ceil(0.95 * len(finishes)) - 1)
        return finishes[idx]


# ---------------------------------------------------------------------------
# Simulator
# ---------------------------------------------------------------------------


def simulate_fluid(
    episode: MobilityEpisode,
    manifests: dict[str, ServingGroupManifest],
    plan: dict[str, list[ReconstitutionAction]],
    bundle: ProfileBundle,
    warmness: WarmnessMap,
    budget: ResourceBudget,
    *,
    max_steps: int = 100_000,
) -> SimulationResult:
    """Run the fluid simulation. See module docstring for semantics.

    `plan[workflow_id]` is the ordered list of actions for that
    workflow. Each action runs sequentially within its workflow; across
    workflows, actions run concurrently under fluid capacity.

    `manifests[workflow_id]` is the ServingGroupManifest for that
    workflow (used to look up StateObject by id when computing
    per_action ResourceCost).
    """
    # Defensive checks.
    for wf in episode.workflows:
        if wf.workflow_id not in plan:
            raise ValueError(f"plan missing workflow {wf.workflow_id!r}")
        if wf.workflow_id not in manifests:
            raise ValueError(f"manifests missing workflow {wf.workflow_id!r}")

    # Per_workflow action queues (head is up next).
    queues: dict[str, list[ReconstitutionAction]] = {
        wid: list(actions) for wid, actions in plan.items()
    }
    finish_time: dict[str, float] = {}

    # Active in_flight actions: list of dicts with mutable state.
    # Each dict carries: action, manifest, remaining (per_resource), lower_bound.
    active: list[dict] = []

    # Mutable warmness (we'll thread through with_added/with_evicted).
    # Enforce KV cap on the INITIAL warmness too: episode.state_warmness
    # may already exceed dst capacity; if so, LRU evict at episode start
    # (per_site) before any actions run. Fixes a correctness gap flagged
    # in the K2/K3/K4 implementation review (initial warmness wasn't
    # cap_checked).
    wm = warmness
    for site in bundle.sites:
        kv_cap = budget.kv_memory_bytes_per_site.get(site, math.inf)
        if kv_cap < math.inf:
            wm = _enforce_kv_capacity(wm, site, kv_cap, manifests, bundle)

    # Output traces.
    traces: list[ActionTrace] = []

    t_now = float(episode.trigger_t_s)
    steps = 0

    def start_eligible() -> None:
        """Promote head_of_queue actions to active for any workflow whose
        previous action just finished (or that hasn't started yet)."""
        nonlocal wm
        # Collect workflows currently active.
        active_workflows = {a["action"].workflow_id for a in active}
        for wid, q in queues.items():
            if wid in active_workflows:
                continue
            if wid in finish_time:
                continue
            if not q:
                # Workflow with no actions left -> mark finished now.
                finish_time[wid] = t_now
                continue
            # Start the next action.
            action = q[0]
            manifest = manifests[wid]
            if action.state_id not in manifest.state_objects:
                raise ValueError(
                    f"action references unknown state {action.state_id!r} "
                    f"in workflow {wid!r}"
                )
            state = manifest.state_objects[action.state_id]
            cost = reconstitution_cost(
                state, action.mode, action.src_site or wid_src(episode, wid),
                action.dst_site, bundle, wm,
            )
            active_keys = {
                (a["action"].state_id, a["action"].dst_site)
                for a in active
            }
            if (action.state_id, action.dst_site) in active_keys:
                # Another workflow is already materializing this state at
                # this destination. Wait for that action to finish; warmness
                # will make this action zero_cost on the next event horizon.
                continue
            # Build remaining_work dict (positive iff there's work on that axis).
            remaining = {
                NETWORK: float(cost.network_bytes) * 8.0,    # convert to bits to match bps
                PREFILL: float(cost.prefill_tokens),
                WORKSPACE: float(cost.workspace_bytes),
                KV_MEMORY: 0.0,  # KV memory is a capacity check, not a bandwidth
            }
            active.append({
                "action": action,
                "started": t_now,
                "remaining": remaining,
                "lower_bound": cost.wallclock_s,
                "kv_resident": cost.kv_resident_bytes,
                "manifest": manifest,
            })

    def link_key(a: str, b: str) -> tuple[str, str]:
        return tuple(sorted([a, b]))

    def axis_demand_and_capacity(active: list[dict]) -> dict:
        """Per_axis: list of action indices needing it, capacities split.

        Returns:
            {axis: {capacity: float, demanders: [(idx, key)...]}}
        """
        out: dict[str, dict] = {}
        # Network: per_link.
        for axis in (NETWORK,):
            link_groups: dict[tuple, list[int]] = {}
            for i, a in enumerate(active):
                if a["remaining"][axis] <= 0.0:
                    continue
                act = a["action"]
                if act.src_site and act.src_site != act.dst_site:
                    key = link_key(act.src_site, act.dst_site)
                else:
                    continue  # same_site actions don't load network
                link_groups.setdefault(key, []).append(i)
            out[axis] = link_groups
        # Prefill, workspace: per_site at dst.
        for axis in (PREFILL, WORKSPACE):
            site_groups: dict[str, list[int]] = {}
            for i, a in enumerate(active):
                if a["remaining"][axis] <= 0.0:
                    continue
                site_groups.setdefault(a["action"].dst_site, []).append(i)
            out[axis] = site_groups
        return out

    def per_action_rate(action_idx: int, demand_groups: dict) -> tuple[float, str]:
        """Compute the slowest progress rate this action gets across all
        axes it needs. Returns (rate, bottleneck_axis_name).

        The action progresses on ALL its needed axes simultaneously. The
        "rate" returned is a uniform progress rate that respects every
        axis: rate_i = capacity_axis / num_demanders_axis (max_min fair
        share at this instant)."""
        a = active[action_idx]
        action = a["action"]
        needed_axes: list[tuple[str, float, float]] = []  # (axis, capacity_share, remaining)

        if a["remaining"][NETWORK] > 0.0 and action.src_site != action.dst_site:
            link = link_key(action.src_site, action.dst_site)
            cap = budget.network_bps_per_link.get(link, math.inf)
            n_users = len(demand_groups[NETWORK].get(link, []))
            share = (cap / n_users) if n_users else math.inf
            needed_axes.append((NETWORK, share, a["remaining"][NETWORK]))
        if a["remaining"][PREFILL] > 0.0:
            cap = budget.prefill_tok_s_per_site.get(action.dst_site, math.inf)
            n_users = len(demand_groups[PREFILL].get(action.dst_site, []))
            share = (cap / n_users) if n_users else math.inf
            needed_axes.append((PREFILL, share, a["remaining"][PREFILL]))
        if a["remaining"][WORKSPACE] > 0.0:
            cap = budget.workspace_hydrate_bps_per_site.get(action.dst_site, math.inf)
            n_users = len(demand_groups[WORKSPACE].get(action.dst_site, []))
            share = (cap / n_users) if n_users else math.inf
            needed_axes.append((WORKSPACE, share, a["remaining"][WORKSPACE]))

        if not needed_axes:
            # All demands are zero — action has no work to do (e.g.,
            # warm cache hit, or zero_byte state). Finishes immediately
            # at the current event horizon (dt = 0).
            return 0.0, NONE

        # Each axis takes (remaining / share) seconds to drain. The
        # action's wall_time_to_finish on this axis is that. The slowest
        # axis is the bottleneck.
        times = [(rem / share if share > 0 else math.inf, axis) for axis, share, rem in needed_axes]
        # Bottleneck is the axis with the LARGEST time_to_finish (slowest).
        time_to_finish, bottleneck = max(times, key=lambda x: x[0])
        return time_to_finish, bottleneck

    # Main event loop.
    while True:
        steps += 1
        if steps > max_steps:
            raise RuntimeError(f"K4 simulator exceeded max_steps={max_steps}; "
                               "likely a bug or non_terminating plan")

        # Promote eligible actions to active.
        start_eligible()

        if not active:
            # All workflows finished.
            break

        # Find the next event time across all active actions.
        demand_groups = axis_demand_and_capacity(active)
        next_event_dt = math.inf
        action_finish_times: list[tuple[float, int, str]] = []  # (dt, idx, bottleneck)
        for i in range(len(active)):
            dt, bn = per_action_rate(i, demand_groups)
            action_finish_times.append((dt, i, bn))
            if dt < next_event_dt:
                next_event_dt = dt

        if next_event_dt == math.inf:
            # Should not happen — would mean all active actions have
            # zero remaining and zero needed_axes simultaneously.
            raise RuntimeError("K4 simulator: zero progress on all actions; bug")

        # Determine which actions finish at next_event_dt.
        # Use small epsilon for float robustness.
        EPS = 1e-12
        finishing_indices: list[int] = []
        for dt, i, bn in action_finish_times:
            if dt <= next_event_dt + EPS:
                finishing_indices.append(i)

        # Advance time on all active actions.
        for i, a in enumerate(active):
            # Reduce remaining work by share * dt for each axis.
            # Recompute share from demand_groups (cached).
            action = a["action"]
            for axis in (NETWORK, PREFILL, WORKSPACE):
                if a["remaining"][axis] <= 0.0:
                    continue
                if axis == NETWORK:
                    if action.src_site == action.dst_site:
                        continue
                    link = link_key(action.src_site, action.dst_site)
                    cap = budget.network_bps_per_link.get(link, math.inf)
                    n_users = len(demand_groups[NETWORK].get(link, []))
                    share = (cap / n_users) if n_users else math.inf
                elif axis == PREFILL:
                    cap = budget.prefill_tok_s_per_site.get(action.dst_site, math.inf)
                    n_users = len(demand_groups[PREFILL].get(action.dst_site, []))
                    share = (cap / n_users) if n_users else math.inf
                elif axis == WORKSPACE:
                    cap = budget.workspace_hydrate_bps_per_site.get(action.dst_site, math.inf)
                    n_users = len(demand_groups[WORKSPACE].get(action.dst_site, []))
                    share = (cap / n_users) if n_users else math.inf
                else:
                    continue
                a["remaining"][axis] = max(0.0, a["remaining"][axis] - share * next_event_dt)

        t_now += next_event_dt
        if next_event_dt > 0.0:
            wm = wm.with_aged(next_event_dt)

        # Finalize finishing actions (deterministic order: by workflow_id,
        # state_id, mode).
        finishing = sorted(
            ((i, action_finish_times[i][2]) for i in finishing_indices),
            key=lambda pair: (active[pair[0]]["action"].workflow_id,
                              active[pair[0]]["action"].state_id,
                              active[pair[0]]["action"].mode),
        )
        for i, bn in finishing:
            a = active[i]
            action = a["action"]
            traces.append(ActionTrace(
                workflow_id=action.workflow_id,
                state_id=action.state_id,
                mode=action.mode,
                started_s=a["started"],
                finished_s=t_now,
                bottleneck=bn,
                wallclock_lower_bound_s=a["lower_bound"],
            ))
            # Update warmness: action successfully reconstituted state at dst.
            wm = wm.with_added(action.state_id, action.dst_site,
                               mode=action.mode, age_s=0.0)
            # KV_memory pressure: if dst now exceeds capacity, LRU_evict.
            kv_cap = budget.kv_memory_bytes_per_site.get(action.dst_site, math.inf)
            if kv_cap < math.inf:
                wm = _enforce_kv_capacity(wm, action.dst_site, kv_cap, manifests, bundle)
            # Pop the action from its workflow's queue.
            wid = action.workflow_id
            if queues[wid] and queues[wid][0] == action:
                queues[wid].pop(0)

        # Remove finished actions from active.
        active = [a for j, a in enumerate(active) if j not in {i for i, _ in finishing}]

    makespan = t_now - episode.trigger_t_s
    return SimulationResult(
        actions=tuple(traces),
        final_warmness=wm,
        makespan_s=makespan,
        per_workflow_finish_s=dict(finish_time),
    )


def wid_src(episode: MobilityEpisode, workflow_id: str) -> str:
    """Look up the workflow's src_site, falling back to the first source
    site of the episode for cold starts."""
    for wf in episode.workflows:
        if wf.workflow_id == workflow_id:
            return wf.src_site if wf.src_site is not None else episode.source_sites[0]
    raise ValueError(f"unknown workflow {workflow_id!r}")


# ---------------------------------------------------------------------------
# KV_memory capacity enforcement (LRU eviction)
# ---------------------------------------------------------------------------


def _enforce_kv_capacity(
    wm: WarmnessMap,
    site: str,
    kv_cap: float,
    manifests: dict[str, ServingGroupManifest],
    bundle: ProfileBundle,
) -> WarmnessMap:
    """If site's resident KV bytes exceed kv_cap, LRU_evict entries
    until under cap. Aggregates resident bytes by looking up each warm
    state's tokens × kv_bytes_per_token across all manifests."""
    state_to_kv_bytes: dict[str, int] = {}
    for manifest in manifests.values():
        for sid, state in manifest.state_objects.items():
            if state.layer in ("prompt_context", "model_execution"):
                state_to_kv_bytes[sid] = state.tokens * bundle.model.kv_bytes_per_token

    def resident_at(wm: WarmnessMap) -> int:
        return sum(state_to_kv_bytes.get(sid, 0) for sid in wm.states_warm_at(site))

    while resident_at(wm) > kv_cap:
        wm, evicted = wm.lru_evict(site)
        if evicted is None:
            break  # nothing to evict; capacity is below the smallest entry
    return wm


__all__ = [
    "ReconstitutionAction",
    "ActionTrace",
    "SimulationResult",
    "simulate_fluid",
    "NETWORK", "PREFILL", "WORKSPACE", "KV_MEMORY", "NONE",
    "ALL_RESOURCES",
]
