"""Gate 2 workload-anchor evidence tables."""
from __future__ import annotations

import csv
from dataclasses import asdict, dataclass
from pathlib import Path

from .fluid_sim import NETWORK, PREFILL, WORKSPACE, SimulationResult, simulate_fluid
from .k8_regime import RegimeCell, default_bundle, make_k8_budget
from .profiles import ProfileBundle
from .reconstitution import cache_reuse, mixed_min_pressure
from .warmness import WarmnessMap
from .workloads import ANCHORS, WorkloadAnchor


PROMPT_LAYERS = {
    "tool_output_context",
    "prompt_summaries",
    "shared_task_context",
    "private_subagent_transcript",
    "merge_review_buffer",
}
WORKSPACE_LAYERS = {"base_repo_checkout", "uncommitted_diff", "subagent_workspace"}
ARTIFACT_LAYERS = {"cleaned_intermediates", "generated_plots", "test_logs"}
CACHE_LAYERS = {"dependency_cache", "build_artifacts"}
RETRIEVED_LAYERS = {"retrieved_documents"}
TRANSCRIPT_LAYERS = {"private_subagent_transcript", "shared_task_context", "merge_review_buffer"}
TOOL_OUTPUT_LAYERS = {"tool_output_context"}
GLOBAL_LAYERS = {"base_repo_checkout", "base_data_bundle", "vector_index_shards", "dependency_cache"}
W3_MULTIPLICITY = {"private_subagent_transcript": 4, "subagent_workspace": 4}


@dataclass(frozen=True)
class AnchorLayerRow:
    anchor_name: str
    anchor_description: str
    n_workflows: int
    model_name: str
    prompt_context_tokens: int
    kv_equivalent_bytes: int
    workspace_bytes: int
    artifact_bytes: int
    dependency_build_cache_bytes: int
    retrieved_document_bytes: int
    tool_output_bytes: int
    transcript_subagent_bytes: int
    globally_available_bytes: int
    mobile_bytes: int
    measured_bytes: int
    trace_derived_bytes: int
    estimated_bytes: int
    synthetic_bytes: int
    provenance: str


@dataclass(frozen=True)
class AnchorRegimeRow:
    anchor_name: str
    n_workflows: int
    cell_id: str
    exact_regime: str
    exact_dominant_bottleneck: str
    strong_reuse_p50_s: float
    mixed_min_pressure_p50_s: float
    mixed_vs_strong_gap_frac: float
    measured_fields: str
    synthetic_fields: str


def run_workload_anchor_gate(
    bundle: ProfileBundle,
    *,
    n_workflows: int = 8,
    budget_cell: RegimeCell | None = None,
) -> tuple[list[AnchorLayerRow], list[AnchorRegimeRow]]:
    cell = budget_cell or RegimeCell(n_workflows, "medium", "tight", 1, seed=9200)
    budget = make_k8_budget(cell)
    layer_rows: list[AnchorLayerRow] = []
    regime_rows: list[AnchorRegimeRow] = []
    for anchor in ANCHORS.values():
        layer_rows.append(anchor_layer_row(anchor, bundle, n_workflows))
        regime_rows.append(anchor_regime_row(anchor, bundle, budget, cell, n_workflows))
    return layer_rows, regime_rows


def anchor_layer_row(anchor: WorkloadAnchor, bundle: ProfileBundle, n_workflows: int) -> AnchorLayerRow:
    totals = _layer_totals(anchor, n_workflows)
    prompt_bytes = _sum_layers(totals, PROMPT_LAYERS)
    prompt_tokens = _bytes_to_tokens(prompt_bytes)
    synthetic = sum(totals.values())
    mobility = {layer.name: layer.mobility_class for layer in anchor.state_layers}
    return AnchorLayerRow(
        anchor_name=anchor.name,
        anchor_description=anchor.description,
        n_workflows=n_workflows,
        model_name=bundle.model.name,
        prompt_context_tokens=prompt_tokens,
        kv_equivalent_bytes=prompt_tokens * bundle.model.kv_bytes_per_token,
        workspace_bytes=_sum_layers(totals, WORKSPACE_LAYERS),
        artifact_bytes=_sum_layers(totals, ARTIFACT_LAYERS),
        dependency_build_cache_bytes=_sum_layers(totals, CACHE_LAYERS),
        retrieved_document_bytes=_sum_layers(totals, RETRIEVED_LAYERS),
        tool_output_bytes=_sum_layers(totals, TOOL_OUTPUT_LAYERS),
        transcript_subagent_bytes=_sum_layers(totals, TRANSCRIPT_LAYERS),
        globally_available_bytes=_sum_layers(totals, GLOBAL_LAYERS),
        mobile_bytes=sum(
            byte_count for name, byte_count in totals.items()
            if mobility[name] == "must_move"
        ),
        measured_bytes=0,
        trace_derived_bytes=0,
        estimated_bytes=0,
        synthetic_bytes=synthetic,
        provenance="hypothesis_fixture_synthetic_bytes",
    )


def anchor_regime_row(
    anchor: WorkloadAnchor,
    bundle: ProfileBundle,
    budget,
    cell: RegimeCell,
    n_workflows: int,
) -> AnchorRegimeRow:
    episode, manifests = anchor.build_episode(n_workflows, seed=cell.seed)
    warmness = WarmnessMap.from_episode_seed(episode.state_warmness)
    strong = simulate_fluid(
        episode, manifests,
        cache_reuse(episode, manifests, bundle, warmness, budget),
        bundle, warmness, budget,
    )
    mixed = simulate_fluid(
        episode, manifests,
        mixed_min_pressure(episode, manifests, bundle, warmness, budget),
        bundle, warmness, budget,
    )
    bottleneck = _dominant_bottleneck(strong)
    regime = _regime_from_result(strong, mixed, bottleneck)
    strong_p50 = strong.p50_resume_s()
    mixed_p50 = mixed.p50_resume_s()
    return AnchorRegimeRow(
        anchor_name=anchor.name,
        n_workflows=n_workflows,
        cell_id=cell.cell_id,
        exact_regime=regime,
        exact_dominant_bottleneck=bottleneck,
        strong_reuse_p50_s=strong_p50,
        mixed_min_pressure_p50_s=mixed_p50,
        mixed_vs_strong_gap_frac=((strong_p50 - mixed_p50) / strong_p50 if strong_p50 > 0 else 0.0),
        measured_fields="",
        synthetic_fields="all_state_layer_bytes",
    )


def write_workload_anchor_artifacts(
    layer_rows: list[AnchorLayerRow],
    regime_rows: list[AnchorRegimeRow],
    out_dir: str | Path,
) -> None:
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    _write_rows(layer_rows, out / "state_layer_table.csv", AnchorLayerRow)
    _write_rows(regime_rows, out / "exact_anchor_regime_table.csv", AnchorRegimeRow)


def main(repo_root: str | Path) -> None:
    repo = Path(repo_root)
    bundle = default_bundle(repo)
    layer_rows, regime_rows = run_workload_anchor_gate(bundle)
    write_workload_anchor_artifacts(layer_rows, regime_rows, repo / "runs" / "workload_anchors")


def _write_rows(rows, path: Path, row_type) -> None:
    fields = list(row_type.__dataclass_fields__.keys())
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            writer.writerow(asdict(row))


def _sum_layers(totals: dict[str, int], names: set[str]) -> int:
    return sum(byte_count for name, byte_count in totals.items() if name in names)


def _layer_totals(anchor: WorkloadAnchor, n_workflows: int) -> dict[str, int]:
    totals: dict[str, int] = {}
    for layer in anchor.state_layers:
        multiplicity = 1
        if anchor.name == "w3_multi_agent_fanout":
            multiplicity = W3_MULTIPLICITY.get(layer.name, 1)
        totals[layer.name] = layer.bytes_per_workflow * multiplicity * n_workflows
    return totals


def _bytes_to_tokens(byte_count: int) -> int:
    return max(0, byte_count // 4)


def _dominant_bottleneck(result: SimulationResult) -> str:
    weights: dict[str, float] = {}
    for action in result.actions:
        if action.bottleneck == "none":
            continue
        elapsed = max(action.finished_s - action.started_s, 0.0)
        weights[action.bottleneck] = weights.get(action.bottleneck, 0.0) + elapsed
    if not weights:
        return "none"
    return max(weights.items(), key=lambda kv: (kv[1], kv[0]))[0]


def _regime_from_result(strong: SimulationResult, mixed: SimulationResult, bottleneck: str) -> str:
    if not _has_contention(strong):
        return "reuse regime"
    if _is_multi_resource(strong) and _gap_frac(strong, mixed) >= 0.10:
        return "multi-resource regime"
    if bottleneck == PREFILL:
        return "prefill-pressure regime"
    if bottleneck == NETWORK:
        return "network-pressure regime"
    if bottleneck == WORKSPACE:
        return "workspace-pressure regime"
    return "reuse regime"


def _gap_frac(strong: SimulationResult, mixed: SimulationResult) -> float:
    strong_p50 = strong.p50_resume_s()
    if strong_p50 <= 0:
        return 0.0
    return (strong_p50 - mixed.p50_resume_s()) / strong_p50


def _has_contention(result: SimulationResult, tol: float = 0.05) -> bool:
    for action in result.actions:
        elapsed = action.finished_s - action.started_s
        if action.wallclock_lower_bound_s > 0 and elapsed > action.wallclock_lower_bound_s * (1.0 + tol):
            return True
    return False


def _is_multi_resource(result: SimulationResult, share_threshold: float = 0.25) -> bool:
    weights: dict[str, float] = {}
    for action in result.actions:
        if action.bottleneck == "none":
            continue
        elapsed = max(action.finished_s - action.started_s, 0.0)
        weights[action.bottleneck] = weights.get(action.bottleneck, 0.0) + elapsed
    total = sum(weights.values())
    if total <= 0:
        return False
    return sum(1 for weight in weights.values() if weight / total >= share_threshold) >= 2


__all__ = [
    "AnchorLayerRow",
    "AnchorRegimeRow",
    "main",
    "run_workload_anchor_gate",
    "write_workload_anchor_artifacts",
]
