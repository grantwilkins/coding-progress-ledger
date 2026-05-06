"""O2 — Oracle_vs_policy plan_diff reporter.

For each K9 diagnostic cell, this module runs the small_N exact oracle
plus the strong_reuse / mixed_min_pressure / random_mode policies
through K4, then emits a per_workflow plan_diff and a per_policy
resource_bottleneck breakdown so the source of any oracle_vs_heuristic
gap is *explained*, not just measured.

The output answers three questions for each cell:

  * Where does the oracle's plan differ from `mixed_min_pressure`?
    (destination choice, prompt_context mode, workspace mode)
  * Which resource is the heuristic spending time on that the oracle
    avoids?
  * How much of the oracle's win comes from candidate_space choices
    that the heuristic does not even consider?

Per CLAUDE.md (`Do not score "winning policy" on a single episode`),
this is *diagnostic* — the reports are inputs to deciding whether to
tune `mixed_min_pressure` or whether the heuristic's candidate space
is fundamentally too narrow.
"""
from __future__ import annotations

import csv
import json
from dataclasses import dataclass, field
from pathlib import Path

from .costs import ARTIFACT_COPY, CONTEXT_REPLAY, KV_TRANSFER, TEXT_TRANSFER
from .episode import MobilityEpisode
from .fluid_sim import (
    KV_MEMORY,
    NETWORK,
    PREFILL,
    WORKSPACE,
    ReconstitutionAction,
    SimulationResult,
    simulate_fluid,
)
from .k9_oracle import enumerate_oracle_plans
from .manifest import ServingGroupManifest
from .profiles import ProfileBundle
from .reconstitution import (
    cache_reuse,
    mixed_min_pressure,
    random_mode,
)
from .resources import ResourceBudget, WORKSPACE_HYDRATE, WARM_REUSE
from .warmness import WarmnessMap


@dataclass(frozen=True)
class WorkflowChoice:
    """The (dst, prompt_mode, workspace_mode) triplet a policy picked
    for one workflow.

    Inferred from the policy's emitted actions: prompt_mode is the mode
    chosen for any prompt_context state, workspace_mode is the mode
    chosen for any workspace state. `mixed` if a policy uses different
    modes within the same layer for one workflow (uncommon but possible
    with mixed_min_pressure)."""
    workflow_id: str
    dst_site: str
    prompt_mode: str
    workspace_mode: str

    def differs_from(self, other: "WorkflowChoice") -> dict[str, tuple[str, str]]:
        diffs: dict[str, tuple[str, str]] = {}
        if self.dst_site != other.dst_site:
            diffs["dst_site"] = (self.dst_site, other.dst_site)
        if self.prompt_mode != other.prompt_mode:
            diffs["prompt_mode"] = (self.prompt_mode, other.prompt_mode)
        if self.workspace_mode != other.workspace_mode:
            diffs["workspace_mode"] = (self.workspace_mode, other.workspace_mode)
        return diffs


@dataclass(frozen=True)
class PolicyDiagnostic:
    policy_name: str
    p50_resume_s: float
    p90_resume_s: float
    makespan_s: float
    bottleneck_seconds: dict[str, float]  # axis name -> total elapsed s
    per_workflow_choice: dict[str, WorkflowChoice]

    @property
    def bottleneck_fractions(self) -> dict[str, float]:
        total = sum(self.bottleneck_seconds.values())
        if total <= 0:
            return {axis: 0.0 for axis in self.bottleneck_seconds}
        return {axis: s / total for axis, s in self.bottleneck_seconds.items()}

    @property
    def attributed_fraction_of_makespan(self) -> float:
        """Fraction of the simulation makespan covered by attributed
        bottleneck time. Below ~0.6 means most of the simulation was
        sequential_wait or warm_reuse (zero bottleneck) and the
        per_axis fractions describe a small slice — read with caution.
        """
        if self.makespan_s <= 0:
            return 1.0
        return min(1.0, sum(self.bottleneck_seconds.values()) / self.makespan_s)


@dataclass(frozen=True)
class OracleDiffReport:
    scenario: str
    cell: dict[str, object]
    oracle: PolicyDiagnostic
    mixed: PolicyDiagnostic
    random: PolicyDiagnostic
    strong_reuse: PolicyDiagnostic
    enumerated_oracle_plans: int

    @property
    def oracle_vs_mixed_gap_frac(self) -> float:
        return _gap_frac(self.mixed.p50_resume_s, self.oracle.p50_resume_s)

    @property
    def oracle_vs_random_gap_frac(self) -> float:
        return _gap_frac(self.random.p50_resume_s, self.oracle.p50_resume_s)

    @property
    def strong_vs_random_gap_frac(self) -> float:
        return _gap_frac(self.random.p50_resume_s, self.strong_reuse.p50_resume_s)

    def per_workflow_diffs(self) -> list[dict[str, object]]:
        rows: list[dict[str, object]] = []
        for wf_id in sorted(self.oracle.per_workflow_choice):
            o = self.oracle.per_workflow_choice[wf_id]
            m = self.mixed.per_workflow_choice.get(wf_id)
            row: dict[str, object] = {
                "scenario": self.scenario,
                "workflow_id": wf_id,
                "oracle_dst": o.dst_site,
                "oracle_prompt_mode": o.prompt_mode,
                "oracle_workspace_mode": o.workspace_mode,
                "mixed_dst": m.dst_site if m else "",
                "mixed_prompt_mode": m.prompt_mode if m else "",
                "mixed_workspace_mode": m.workspace_mode if m else "",
                "diff_dst": bool(m and m.dst_site != o.dst_site),
                "diff_prompt_mode": bool(m and m.prompt_mode != o.prompt_mode),
                "diff_workspace_mode": bool(m and m.workspace_mode != o.workspace_mode),
            }
            rows.append(row)
        return rows


def compute_oracle_diff(
    scenario: str,
    cell: dict[str, object],
    episode: MobilityEpisode,
    manifests: dict[str, ServingGroupManifest],
    bundle: ProfileBundle,
    budget: ResourceBudget,
    warmness: WarmnessMap | None = None,
    *,
    random_seed: int = 0,
) -> OracleDiffReport:
    """Run oracle + strong + mixed + random for one cell, build the diff.

    The oracle is the same exhaustive enumeration over (dst, prompt_mode,
    workspace_mode) tuples that K9 uses; we run it inline here so we can
    capture each policy's emitted *actions* (and hence per_workflow
    choices), not just the headline p50.
    """
    if warmness is None:
        warmness = WarmnessMap.from_episode_seed(episode.state_warmness)

    # --- run policies + collect per_policy diagnostics --------------------
    strong_plan = cache_reuse(episode, manifests, bundle, warmness, budget)
    mixed_plan = mixed_min_pressure(episode, manifests, bundle, warmness, budget)
    random_plan = random_mode(
        episode, manifests, bundle, warmness, budget, seed=random_seed,
    )
    strong_diag = _diagnose_policy(
        "strong_reuse", strong_plan, episode, manifests, bundle, warmness, budget,
    )
    mixed_diag = _diagnose_policy(
        "mixed_min_pressure", mixed_plan, episode, manifests, bundle, warmness, budget,
    )
    random_diag = _diagnose_policy(
        "random_mode", random_plan, episode, manifests, bundle, warmness, budget,
    )

    # --- exhaustive oracle -------------------------------------------------
    # Delegate to k9_oracle.enumerate_oracle_plans so candidate_space and
    # objective stay in lockstep with `run_small_n_oracle`.
    best_plan, _best_choice, _best_result, enumerated, _counts = enumerate_oracle_plans(
        episode, manifests, bundle, warmness, budget,
    )
    oracle_diag = _diagnose_policy(
        "oracle", best_plan, episode, manifests, bundle, warmness, budget,
    )

    return OracleDiffReport(
        scenario=scenario,
        cell=cell,
        oracle=oracle_diag,
        mixed=mixed_diag,
        random=random_diag,
        strong_reuse=strong_diag,
        enumerated_oracle_plans=enumerated,
    )


# ---------------------------------------------------------------------------
# Artifact emission
# ---------------------------------------------------------------------------


def write_oracle_diff_artifacts(
    reports: list[OracleDiffReport],
    out_dir: str | Path,
) -> None:
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    _write_summary_csv(reports, out / "oracle_diff_summary.csv")
    _write_per_workflow_csv(reports, out / "oracle_diff_per_workflow.csv")
    _write_bottleneck_csv(reports, out / "oracle_diff_bottlenecks.csv")
    _write_markdown(reports, out / "oracle_diff_report.md")
    (out / "oracle_diff_summary.json").write_text(
        json.dumps([_summary_dict(r) for r in reports], indent=2) + "\n"
    )
    (out / "README.md").write_text(
        "# O2 oracle_vs_policy diagnostic artifacts\n\n"
        "`oracle_diff_summary.csv` — per_cell gaps (oracle vs mixed, "
        "oracle vs random, strong vs random) and bottleneck fractions.\n"
        "`oracle_diff_per_workflow.csv` — per-(scenario, workflow) "
        "(dst, prompt_mode, workspace_mode) for oracle and mixed plus "
        "diff flags.\n"
        "`oracle_diff_bottlenecks.csv` — time_weighted bottleneck "
        "fractions per policy.\n"
        "`oracle_diff_report.md` — narrative summary.\n"
    )


def _summary_dict(report: OracleDiffReport) -> dict[str, object]:
    return {
        "scenario": report.scenario,
        "cell": report.cell,
        "oracle_p50_s": report.oracle.p50_resume_s,
        "mixed_p50_s": report.mixed.p50_resume_s,
        "random_p50_s": report.random.p50_resume_s,
        "strong_reuse_p50_s": report.strong_reuse.p50_resume_s,
        "oracle_vs_mixed_gap_frac": report.oracle_vs_mixed_gap_frac,
        "oracle_vs_random_gap_frac": report.oracle_vs_random_gap_frac,
        "strong_vs_random_gap_frac": report.strong_vs_random_gap_frac,
        "oracle_bottleneck_fractions": report.oracle.bottleneck_fractions,
        "mixed_bottleneck_fractions": report.mixed.bottleneck_fractions,
        "enumerated_oracle_plans": report.enumerated_oracle_plans,
    }


def _write_summary_csv(reports: list[OracleDiffReport], path: Path) -> None:
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=[
            "scenario", "n_workflows", "state_scale", "prefill_capacity",
            "link_gbps", "oracle_p50_s", "mixed_p50_s", "random_p50_s",
            "strong_reuse_p50_s", "oracle_vs_mixed_gap_frac",
            "oracle_vs_random_gap_frac", "strong_vs_random_gap_frac",
            "enumerated_oracle_plans",
        ])
        writer.writeheader()
        for r in reports:
            writer.writerow({
                "scenario": r.scenario,
                "n_workflows": r.cell.get("n_workflows"),
                "state_scale": r.cell.get("state_scale"),
                "prefill_capacity": r.cell.get("prefill_capacity"),
                "link_gbps": r.cell.get("link_gbps"),
                "oracle_p50_s": f"{r.oracle.p50_resume_s:.9g}",
                "mixed_p50_s": f"{r.mixed.p50_resume_s:.9g}",
                "random_p50_s": f"{r.random.p50_resume_s:.9g}",
                "strong_reuse_p50_s": f"{r.strong_reuse.p50_resume_s:.9g}",
                "oracle_vs_mixed_gap_frac": f"{r.oracle_vs_mixed_gap_frac:.9g}",
                "oracle_vs_random_gap_frac": f"{r.oracle_vs_random_gap_frac:.9g}",
                "strong_vs_random_gap_frac": f"{r.strong_vs_random_gap_frac:.9g}",
                "enumerated_oracle_plans": r.enumerated_oracle_plans,
            })


def _write_per_workflow_csv(reports: list[OracleDiffReport], path: Path) -> None:
    fieldnames = [
        "scenario", "workflow_id",
        "oracle_dst", "oracle_prompt_mode", "oracle_workspace_mode",
        "mixed_dst", "mixed_prompt_mode", "mixed_workspace_mode",
        "diff_dst", "diff_prompt_mode", "diff_workspace_mode",
    ]
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for r in reports:
            for row in r.per_workflow_diffs():
                writer.writerow(row)


def _write_bottleneck_csv(reports: list[OracleDiffReport], path: Path) -> None:
    axes = (NETWORK, PREFILL, WORKSPACE, KV_MEMORY)
    fieldnames = [
        "scenario", "policy", "p50_resume_s",
        *(f"frac_{axis}" for axis in axes),
    ]
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for r in reports:
            for diag in (r.oracle, r.mixed, r.random, r.strong_reuse):
                fractions = diag.bottleneck_fractions
                row = {
                    "scenario": r.scenario,
                    "policy": diag.policy_name,
                    "p50_resume_s": f"{diag.p50_resume_s:.9g}",
                }
                for axis in axes:
                    row[f"frac_{axis}"] = f"{fractions.get(axis, 0.0):.6f}"
                writer.writerow(row)


def _write_markdown(reports: list[OracleDiffReport], path: Path) -> None:
    lines: list[str] = ["# O2 — oracle_vs_policy plan diff", ""]
    for r in reports:
        lines.append(f"## {r.scenario}")
        cell = r.cell
        lines.append(
            f"- cell: n={cell.get('n_workflows')}, scale={cell.get('state_scale')}, "
            f"prefill={cell.get('prefill_capacity')}, link={cell.get('link_gbps')} Gbps"
        )
        lines.append(
            f"- gaps: oracle vs mixed = {r.oracle_vs_mixed_gap_frac*100:.1f}%; "
            f"oracle vs random = {r.oracle_vs_random_gap_frac*100:.1f}%; "
            f"strong vs random = {r.strong_vs_random_gap_frac*100:.1f}%"
        )
        lines.append(
            f"- p50 (s): oracle={r.oracle.p50_resume_s:.3g}, "
            f"mixed={r.mixed.p50_resume_s:.3g}, "
            f"strong={r.strong_reuse.p50_resume_s:.3g}, "
            f"random={r.random.p50_resume_s:.3g}"
        )
        lines.append("")
        lines.append(
            "Per_policy bottleneck fractions (time_weighted; "
            "`attr` = fraction of makespan with an attributed bottleneck — "
            "low values mean the breakdown describes a small slice):"
        )
        for diag in (r.oracle, r.mixed, r.random, r.strong_reuse):
            fr = diag.bottleneck_fractions
            lines.append(
                f"- `{diag.policy_name}`: "
                f"network={fr.get(NETWORK, 0)*100:.0f}%, "
                f"prefill={fr.get(PREFILL, 0)*100:.0f}%, "
                f"workspace={fr.get(WORKSPACE, 0)*100:.0f}%, "
                f"kv_memory={fr.get(KV_MEMORY, 0)*100:.0f}% "
                f"(attr={diag.attributed_fraction_of_makespan*100:.0f}%)"
            )
        lines.append("")
        diffs_per_workflow = r.per_workflow_diffs()
        n_diff_dst = sum(1 for row in diffs_per_workflow if row["diff_dst"])
        n_diff_prompt = sum(1 for row in diffs_per_workflow if row["diff_prompt_mode"])
        n_diff_ws = sum(1 for row in diffs_per_workflow if row["diff_workspace_mode"])
        n_total = len(diffs_per_workflow)
        lines.append(
            f"Per_workflow oracle vs mixed differences: "
            f"dst={n_diff_dst}/{n_total}, prompt_mode={n_diff_prompt}/{n_total}, "
            f"workspace_mode={n_diff_ws}/{n_total}"
        )
        lines.append("")
    path.write_text("\n".join(lines) + "\n")


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _diagnose_policy(
    name: str,
    plan: dict[str, list[ReconstitutionAction]],
    episode: MobilityEpisode,
    manifests: dict[str, ServingGroupManifest],
    bundle: ProfileBundle,
    warmness: WarmnessMap,
    budget: ResourceBudget,
) -> PolicyDiagnostic:
    result = simulate_fluid(episode, manifests, plan, bundle, warmness, budget)
    bottleneck_seconds = _bottleneck_seconds(result)
    per_wf_choice = _per_workflow_choices(plan, manifests)
    return PolicyDiagnostic(
        policy_name=name,
        p50_resume_s=result.p50_resume_s(),
        p90_resume_s=result.p90_resume_s(),
        makespan_s=result.makespan_s,
        bottleneck_seconds=bottleneck_seconds,
        per_workflow_choice=per_wf_choice,
    )


def _bottleneck_seconds(result: SimulationResult) -> dict[str, float]:
    out: dict[str, float] = {NETWORK: 0.0, PREFILL: 0.0, WORKSPACE: 0.0, KV_MEMORY: 0.0}
    for action in result.actions:
        if action.bottleneck not in out:
            continue
        elapsed = max(action.finished_s - action.started_s, 0.0)
        out[action.bottleneck] += elapsed
    return out


_PROMPT_MODES = (CONTEXT_REPLAY, KV_TRANSFER, WARM_REUSE, TEXT_TRANSFER)
_WORKSPACE_MODES = (ARTIFACT_COPY, WORKSPACE_HYDRATE, WARM_REUSE)


def _per_workflow_choices(
    plan: dict[str, list[ReconstitutionAction]],
    manifests: dict[str, ServingGroupManifest],
) -> dict[str, WorkflowChoice]:
    """Infer a (dst, prompt_mode, workspace_mode) triplet from emitted actions.

    Per_layer mode disagreements collapse to "mixed". The dst is the
    most common dst across the workflow's actions (usually all the same
    for K9_style oracle plans, since the candidate space picks one dst
    per workflow)."""
    out: dict[str, WorkflowChoice] = {}
    for wf_id, actions in plan.items():
        dsts = [a.dst_site for a in actions]
        dst = _majority(dsts)
        prompt_modes: list[str] = []
        workspace_modes: list[str] = []
        manifest = manifests.get(wf_id)
        if manifest is not None:
            for action in actions:
                state = manifest.state_objects.get(action.state_id)
                if state is None:
                    continue
                if state.layer in ("prompt_context", "model_execution"):
                    prompt_modes.append(action.mode)
                elif state.layer == "workspace":
                    workspace_modes.append(action.mode)
        out[wf_id] = WorkflowChoice(
            workflow_id=wf_id,
            dst_site=dst or "",
            prompt_mode=_unify_modes(prompt_modes, _PROMPT_MODES),
            workspace_mode=_unify_modes(workspace_modes, _WORKSPACE_MODES),
        )
    return out


def _majority(values: list[str]) -> str:
    if not values:
        return ""
    counts = {v: values.count(v) for v in set(values)}
    return max(counts.items(), key=lambda kv: (kv[1], kv[0]))[0]


def _unify_modes(modes: list[str], allowed: tuple[str, ...]) -> str:
    """If all modes (filtered to `allowed`) agree, return that mode;
    if they disagree, return 'mixed'; if empty, return 'none'."""
    filtered = [m for m in modes if m in allowed]
    if not filtered:
        return "none"
    unique = set(filtered)
    if len(unique) == 1:
        return next(iter(unique))
    # Drop WARM_REUSE if it appears alongside a real mode — the diff
    # consumer cares about the *cold* path the policy chose.
    cold = unique - {WARM_REUSE}
    if len(cold) == 1:
        return next(iter(cold))
    return "mixed"


def _gap_frac(slow_s: float, fast_s: float) -> float:
    if slow_s <= 0:
        return 0.0
    return (slow_s - fast_s) / slow_s
