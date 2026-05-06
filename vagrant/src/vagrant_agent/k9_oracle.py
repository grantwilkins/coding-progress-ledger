"""K9 small-N offline oracle.

The oracle is deliberately exponential and simulator-backed.  It is not
meant to become a production policy; it answers whether a K8/K7 cell has
headroom above the strong per-site reuse baseline.
"""
from __future__ import annotations

import csv
import itertools
from dataclasses import dataclass
from pathlib import Path

from .costs import ARTIFACT_COPY, CONTEXT_REPLAY, KV_TRANSFER, TEXT_TRANSFER
from .episode import MobilityEpisode
from .fluid_sim import ReconstitutionAction, SimulationResult, simulate_fluid
from .manifest import ServingGroupManifest
from .profiles import ProfileBundle
from .reconstitution import cache_reuse, mixed_min_pressure
from .resources import ResourceBudget, WORKSPACE_HYDRATE
from .warmness import WarmnessMap


@dataclass(frozen=True)
class OracleResult:
    n_workflows: int
    min_candidates_per_workflow: int
    max_candidates_per_workflow: int
    enumerated_plans: int
    oracle_p50_resume_s: float
    oracle_p90_resume_s: float
    oracle_makespan_s: float
    strong_reuse_p50_resume_s: float
    mixed_p50_resume_s: float
    best_plan: dict[str, str]

    @property
    def oracle_vs_strong_gap_frac(self) -> float:
        if self.strong_reuse_p50_resume_s <= 0:
            return 0.0
        return (
            self.strong_reuse_p50_resume_s - self.oracle_p50_resume_s
        ) / self.strong_reuse_p50_resume_s

    @property
    def oracle_vs_mixed_gap_frac(self) -> float:
        if self.mixed_p50_resume_s <= 0:
            return 0.0
        return (
            self.mixed_p50_resume_s - self.oracle_p50_resume_s
        ) / self.mixed_p50_resume_s


@dataclass(frozen=True)
class OracleScenarioResult:
    scenario: str
    state_scale: str
    prefill_capacity: str
    link_gbps: int
    result: OracleResult


def enumerate_oracle_plans(
    episode: MobilityEpisode,
    manifests: dict[str, ServingGroupManifest],
    bundle: ProfileBundle,
    warmness: WarmnessMap,
    budget: ResourceBudget,
    *,
    max_workflows: int = 6,
) -> tuple[
    dict[str, list[ReconstitutionAction]],
    dict[str, str],
    SimulationResult,
    int,
    list[int],
]:
    """Shared oracle enumeration core.

    Returns `(best_plan, best_choice_label, best_result, enumerated, candidate_counts)`.
    Used by both `run_small_n_oracle` and `oracle_diff.compute_oracle_diff`
    so the two never drift on candidate space, objective, or termination.
    """
    workflows = tuple(sorted(episode.workflows, key=lambda w: w.workflow_id))
    if len(workflows) > max_workflows:
        raise ValueError(
            f"K9 exact oracle is exponential; got {len(workflows)} workflows, "
            f"max_workflows={max_workflows}"
        )
    candidate_by_wf = {
        wf.workflow_id: _workflow_candidates(episode, manifests[wf.workflow_id], wf.workflow_id)
        for wf in workflows
    }
    enumerated = 0
    best_result: SimulationResult | None = None
    best_plan: dict[str, list[ReconstitutionAction]] | None = None
    best_choice: dict[str, str] = {}
    for combo in itertools.product(*(candidate_by_wf[wf.workflow_id] for wf in workflows)):
        plan = {wf.workflow_id: list(actions) for wf, (_label, actions) in zip(workflows, combo)}
        result = simulate_fluid(episode, manifests, plan, bundle, warmness, budget)
        enumerated += 1
        if best_result is None or _objective(result) < _objective(best_result):
            best_result = result
            best_plan = plan
            best_choice = {
                wf.workflow_id: label for wf, (label, _actions) in zip(workflows, combo)
            }
    if best_result is None or best_plan is None:
        raise RuntimeError("oracle enumerated no plans")
    return best_plan, best_choice, best_result, enumerated, [
        len(candidate_by_wf[wf.workflow_id]) for wf in workflows
    ]


def run_small_n_oracle(
    episode: MobilityEpisode,
    manifests: dict[str, ServingGroupManifest],
    bundle: ProfileBundle,
    budget: ResourceBudget,
    warmness: WarmnessMap | None = None,
    *,
    max_workflows: int = 6,
) -> OracleResult:
    """Exhaustively enumerate workflow-level mode/destination choices.

    Each workflow candidate fixes one destination, one prompt-context mode
    (`context_replay` or `kv_transfer`), and one workspace mode
    (`artifact_copy` or `workspace_hydrate`).  All state-level actions are
    still evaluated by K4, including warm reuse and sequential workflow
    dependencies.
    """
    if warmness is None:
        warmness = WarmnessMap.from_episode_seed(episode.state_warmness)
    _, best_choice, best_result, enumerated, candidate_counts = enumerate_oracle_plans(
        episode, manifests, bundle, warmness, budget, max_workflows=max_workflows,
    )

    strong_plan = cache_reuse(episode, manifests, bundle, warmness, budget)
    strong_result = simulate_fluid(episode, manifests, strong_plan, bundle, warmness, budget)
    mixed_plan = mixed_min_pressure(episode, manifests, bundle, warmness, budget)
    mixed_result = simulate_fluid(episode, manifests, mixed_plan, bundle, warmness, budget)
    return OracleResult(
        n_workflows=len(episode.workflows),
        min_candidates_per_workflow=min(candidate_counts),
        max_candidates_per_workflow=max(candidate_counts),
        enumerated_plans=enumerated,
        oracle_p50_resume_s=best_result.p50_resume_s(),
        oracle_p90_resume_s=best_result.p90_resume_s(),
        oracle_makespan_s=best_result.makespan_s,
        strong_reuse_p50_resume_s=strong_result.p50_resume_s(),
        mixed_p50_resume_s=mixed_result.p50_resume_s(),
        best_plan=best_choice,
    )


def write_oracle_artifacts(result: OracleResult, out_dir: str | Path) -> None:
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    with (out / "oracle_gap_table.csv").open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=[
            "n_workflows", "min_candidates_per_workflow", "max_candidates_per_workflow", "enumerated_plans",
            "oracle_p50_resume_s", "strong_reuse_p50_resume_s", "mixed_p50_resume_s",
            "oracle_vs_strong_gap_frac", "oracle_vs_mixed_gap_frac",
        ])
        writer.writeheader()
        writer.writerow({
            "n_workflows": result.n_workflows,
            "min_candidates_per_workflow": result.min_candidates_per_workflow,
            "max_candidates_per_workflow": result.max_candidates_per_workflow,
            "enumerated_plans": result.enumerated_plans,
            "oracle_p50_resume_s": f"{result.oracle_p50_resume_s:.9g}",
            "strong_reuse_p50_resume_s": f"{result.strong_reuse_p50_resume_s:.9g}",
            "mixed_p50_resume_s": f"{result.mixed_p50_resume_s:.9g}",
            "oracle_vs_strong_gap_frac": f"{result.oracle_vs_strong_gap_frac:.9g}",
            "oracle_vs_mixed_gap_frac": f"{result.oracle_vs_mixed_gap_frac:.9g}",
        })
    with (out / "oracle_best_plan.csv").open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["workflow_id", "choice"])
        writer.writeheader()
        for workflow_id, choice in sorted(result.best_plan.items()):
            writer.writerow({"workflow_id": workflow_id, "choice": choice})


def write_oracle_sweep_artifacts(
    rows: list[OracleScenarioResult],
    out_dir: str | Path,
) -> None:
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    with (out / "oracle_gap_table.csv").open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=[
            "scenario", "state_scale", "prefill_capacity", "link_gbps",
            "n_workflows", "min_candidates_per_workflow", "max_candidates_per_workflow",
            "enumerated_plans", "objective", "candidate_space",
            "oracle_p50_resume_s", "strong_reuse_p50_resume_s", "mixed_p50_resume_s",
            "oracle_p90_resume_s", "oracle_makespan_s",
            "oracle_vs_strong_gap_frac", "oracle_vs_mixed_gap_frac",
        ])
        writer.writeheader()
        for row in rows:
            result = row.result
            writer.writerow({
                "scenario": row.scenario,
                "state_scale": row.state_scale,
                "prefill_capacity": row.prefill_capacity,
                "link_gbps": row.link_gbps,
                "n_workflows": result.n_workflows,
                "min_candidates_per_workflow": result.min_candidates_per_workflow,
                "max_candidates_per_workflow": result.max_candidates_per_workflow,
                "enumerated_plans": result.enumerated_plans,
                "objective": "minimize(p50,p90,makespan)",
                "candidate_space": "workflow_level_dst_prompt_mode_workspace_mode_v1",
                "oracle_p50_resume_s": f"{result.oracle_p50_resume_s:.9g}",
                "strong_reuse_p50_resume_s": f"{result.strong_reuse_p50_resume_s:.9g}",
                "mixed_p50_resume_s": f"{result.mixed_p50_resume_s:.9g}",
                "oracle_p90_resume_s": f"{result.oracle_p90_resume_s:.9g}",
                "oracle_makespan_s": f"{result.oracle_makespan_s:.9g}",
                "oracle_vs_strong_gap_frac": f"{result.oracle_vs_strong_gap_frac:.9g}",
                "oracle_vs_mixed_gap_frac": f"{result.oracle_vs_mixed_gap_frac:.9g}",
            })
    with (out / "oracle_best_plan.csv").open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["scenario", "workflow_id", "choice"])
        writer.writeheader()
        for row in rows:
            for workflow_id, choice in sorted(row.result.best_plan.items()):
                writer.writerow({
                    "scenario": row.scenario,
                    "workflow_id": workflow_id,
                    "choice": choice,
                })


def _workflow_candidates(
    episode: MobilityEpisode,
    manifest: ServingGroupManifest,
    workflow_id: str,
) -> list[tuple[str, tuple[ReconstitutionAction, ...]]]:
    wf = next(wf for wf in episode.workflows if wf.workflow_id == workflow_id)
    src = wf.src_site or episode.source_sites[0]
    candidates: list[tuple[str, tuple[ReconstitutionAction, ...]]] = []
    for dst in episode.destination_sites:
        for prompt_mode in (CONTEXT_REPLAY, KV_TRANSFER):
            if prompt_mode == KV_TRANSFER and src == dst:
                continue
            for workspace_mode in (ARTIFACT_COPY, WORKSPACE_HYDRATE):
                actions: list[ReconstitutionAction] = []
                for sid, state in sorted(manifest.state_objects.items()):
                    if state.layer in ("prompt_context", "model_execution"):
                        mode = prompt_mode
                    elif state.layer == "workspace":
                        mode = workspace_mode
                        action_src = dst if mode == WORKSPACE_HYDRATE else src
                        actions.append(ReconstitutionAction(
                            workflow_id=workflow_id, state_id=sid, mode=mode,
                            src_site=action_src, dst_site=dst,
                            reason=f"oracle:{dst}:{prompt_mode}:{workspace_mode}",
                        ))
                        continue
                    elif state.layer == "memory":
                        mode = TEXT_TRANSFER
                    else:
                        raise ValueError(
                            f"K9 candidate-space v1 does not handle layer {state.layer!r} "
                            f"for state {sid!r}"
                        )
                    actions.append(ReconstitutionAction(
                        workflow_id=workflow_id, state_id=sid, mode=mode,
                        src_site=src, dst_site=dst,
                        reason=f"oracle:{dst}:{prompt_mode}:{workspace_mode}",
                    ))
                label = f"dst={dst};prompt={prompt_mode};workspace={workspace_mode}"
                candidates.append((label, tuple(actions)))
    return candidates


def _objective(result: SimulationResult) -> tuple[float, float, float]:
    return (result.p50_resume_s(), result.p90_resume_s(), result.makespan_s)
