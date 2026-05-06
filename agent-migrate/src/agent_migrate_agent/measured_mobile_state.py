"""Measured mobile-state analysis from coding-data-collection snapshots."""
from __future__ import annotations

import csv
import math
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Iterable

from .episode import MobilityEpisode, Workflow
from .fluid_sim import SimulationResult, simulate_fluid
from .k8_regime import STATE_SCALES_BYTES
from .manifest import ServingGroupManifest, StateObject, WorkNode
from .profiles import load_bundle
from .reconstitution import run_reconstitution_policy
from .resources import ResourceBudget
from .warmness import WarmnessMap


POLICIES = ("cache_reuse", "random_mode", "mixed_min_pressure")
INT_FIELDS = {
    "clean_repo_bytes",
    "initial_workspace_bytes",
    "final_workspace_bytes",
    "unchanged_initial_bytes",
    "modified_file_bytes",
    "new_file_bytes",
    "deleted_file_bytes",
    "final_diff_bytes",
    "touched_file_bytes",
    "read_file_bytes",
    "tool_output_bytes",
    "test_log_bytes",
    "build_artifact_bytes",
    "dependency_cache_bytes",
    "retrieved_document_bytes",
    "workspace_total_bytes",
    "hidden_or_protected_bytes",
    "skipped_symlink_count",
    "setup_command_count",
    "lockfile_count",
    "leakage_hit_count",
}
BOOL_FIELDS = {"agent_workspace_retained", "leakage_passed", "row_usable_for_claims"}
LAYER_FIELDS = {
    "clean_repo": "clean_repo_bytes",
    "initial_workspace": "initial_workspace_bytes",
    "full_workspace_snapshot": "final_workspace_bytes",
    "unchanged_initial_workspace": "unchanged_initial_bytes",
    "modified_files_patch": "modified_file_bytes",
    "new_or_touched_files": "new_file_bytes",
    "uncommitted_diff": "final_diff_bytes",
    "files_touched": "touched_file_bytes",
    "files_read": "read_file_bytes",
    "tool_outputs": "tool_output_bytes",
    "test_logs": "test_log_bytes",
    "build_artifacts": "build_artifact_bytes",
    "dependency_cache": "dependency_cache_bytes",
    "retrieved_documents": "retrieved_document_bytes",
}
MOBILITY_CLASS = {
    "clean_repo": "globally_available",
    "initial_workspace": "globally_available_if_reconstructable",
    "full_workspace_snapshot": "representation_cost_not_mobile_headline",
    "unchanged_initial_workspace": "globally_available_if_reconstructable",
    "modified_files_patch": "must_move",
    "new_or_touched_files": "must_move_upper_bound",
    "uncommitted_diff": "must_move",
    "files_touched": "must_move",
    "files_read": "cheaply_rehydratable",
    "tool_outputs": "must_move",
    "test_logs": "can_be_discarded",
    "build_artifacts": "can_be_recomputed",
    "dependency_cache": "cheaply_rehydratable",
    "retrieved_documents": "must_move",
}


@dataclass(frozen=True)
class SnapshotRow:
    run_id: str
    run_dir: str
    run_status: str
    final_success: str
    eligible_for_l_gate: str
    agent_workspace_retained: bool
    run_validation_passed: str = ""
    leakage_passed: bool = False
    leakage_hit_count: int = 0
    row_usable_for_claims: bool = False
    clean_repo_bytes: int = 0
    clean_repo_bytes_provenance: str = "missing"
    initial_workspace_bytes: int = 0
    initial_workspace_bytes_provenance: str = "missing"
    final_workspace_bytes: int = 0
    final_workspace_bytes_provenance: str = "missing"
    unchanged_initial_bytes: int = 0
    unchanged_initial_bytes_provenance: str = "missing"
    modified_file_bytes: int = 0
    modified_file_bytes_provenance: str = "missing"
    new_file_bytes: int = 0
    new_file_bytes_provenance: str = "missing"
    deleted_file_bytes: int = 0
    deleted_file_bytes_provenance: str = "missing"
    final_diff_bytes: int = 0
    final_diff_bytes_provenance: str = "missing"
    touched_file_bytes: int = 0
    touched_file_bytes_provenance: str = "missing"
    read_file_bytes: int = 0
    read_file_bytes_provenance: str = "missing"
    tool_output_bytes: int = 0
    tool_output_bytes_provenance: str = "missing"
    test_log_bytes: int = 0
    test_log_bytes_provenance: str = "missing"
    build_artifact_bytes: int = 0
    build_artifact_bytes_provenance: str = "missing"
    dependency_cache_bytes: int = 0
    dependency_cache_bytes_provenance: str = "missing"
    retrieved_document_bytes: int = 0
    retrieved_document_bytes_provenance: str = "missing"
    workspace_total_bytes: int = 0
    workspace_total_bytes_provenance: str = "missing"
    hidden_or_protected_bytes: int = 0
    hidden_or_protected_bytes_provenance: str = "missing"
    skipped_symlink_count: int = 0
    setup_command_count: int = 0
    lockfile_count: int = 0
    final_diff_semantics: str = "patch_file_bytes_not_touched_file_payload"

    @property
    def safe_for_claims(self) -> bool:
        return self.agent_workspace_retained and self.row_usable_for_claims and self.leakage_passed

    @property
    def measured_dirty_bytes(self) -> int:
        # Patch bytes and touched-file bytes overlap. Use the larger directional
        # estimate, then add transcript-observed tool output separately.
        dirty_file_bytes = max(self.modified_file_bytes, self.new_file_bytes, self.final_diff_bytes)
        return dirty_file_bytes + self.tool_output_bytes

    @property
    def snapshot_representation_bytes(self) -> int:
        return max(0, self.final_workspace_bytes - self.hidden_or_protected_bytes)


@dataclass(frozen=True)
class LayerDistributionRow:
    layer: str
    source_field: str
    mobility_class: str
    runs_with_measured_or_trace_bytes: int
    runs_missing: int
    p50_bytes: int
    p75_bytes: int
    p90_bytes: int
    p95_bytes: int
    max_bytes: int
    provenance_summary: str


@dataclass(frozen=True)
class ThresholdRow:
    metric: str
    threshold_bytes: int
    n_runs: int
    n_over_threshold: int
    fraction_over_threshold: float
    usable_for_claims_only: bool


@dataclass(frozen=True)
class RestartPressureRow:
    episode_id: str
    n_workflows: int
    policy: str
    p50_resume_proxy_s: float
    p90_resume_proxy_s: float
    p99_resume_proxy_s: float
    makespan_s: float
    dominant_bottleneck: str
    best_policy: str
    mixed_beats_strong: bool
    mixed_beats_random: bool


@dataclass(frozen=True)
class PackageRow:
    run_id: str
    cut_point_kind: str
    package_type: str
    structurally_valid: bool
    validity_reason: str
    bytes_moved: int
    lazy_bytes_required_later: int
    model_resume_s: float
    environment_resume_s: float
    task_resume_s: float
    provenance: str


def read_snapshot_index(path: str | Path) -> list[SnapshotRow]:
    with Path(path).open(newline="") as f:
        return [_snapshot_from_dict(row) for row in csv.DictReader(f)]


def write_measured_artifacts(snapshots: list[SnapshotRow], out_dir: str | Path, repo_root: str | Path) -> None:
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    _write_csv(out / "layer_distribution.csv", layer_distribution(snapshots))
    _write_csv(out / "mobile_state_thresholds.csv", threshold_rows(snapshots))
    pressure, comparisons = measured_restart_pressure(snapshots, repo_root=repo_root)
    _write_csv(out / "exact_restart_pressure.csv", pressure)
    _write_csv(out / "policy_comparison_on_measured_state.csv", comparisons)
    _write_csv(out / "measured_restart_package_table.csv", measured_restart_packages(snapshots))


def layer_distribution(snapshots: list[SnapshotRow]) -> list[LayerDistributionRow]:
    rows: list[LayerDistributionRow] = []
    for layer, field in LAYER_FIELDS.items():
        values = [getattr(s, field) for s in snapshots if s.safe_for_claims]
        provenance_field = f"{field}_provenance"
        present = [
            s for s in snapshots
            if s.safe_for_claims
            and getattr(s, provenance_field, "missing") not in {"missing", "not_applicable"}
        ]
        provenance_values = sorted({getattr(s, provenance_field, "missing") for s in snapshots if s.safe_for_claims})
        rows.append(LayerDistributionRow(
            layer=layer,
            source_field=field,
            mobility_class=MOBILITY_CLASS[layer],
            runs_with_measured_or_trace_bytes=len(present),
            runs_missing=sum(1 for s in snapshots if s.safe_for_claims) - len(present),
            p50_bytes=_quantile(values, 0.50),
            p75_bytes=_quantile(values, 0.75),
            p90_bytes=_quantile(values, 0.90),
            p95_bytes=_quantile(values, 0.95),
            max_bytes=max(values, default=0),
            provenance_summary=";".join(provenance_values),
        ))
    return rows


def threshold_rows(snapshots: list[SnapshotRow]) -> list[ThresholdRow]:
    usable = [s for s in snapshots if s.safe_for_claims]
    specs: list[tuple[str, int, list[int]]] = [
        ("dirty_payload_gt_1mb", 1_000_000, [s.measured_dirty_bytes for s in usable]),
        ("dirty_payload_gt_10mb", 10_000_000, [s.measured_dirty_bytes for s in usable]),
        ("dirty_payload_gt_100mb", 100_000_000, [s.measured_dirty_bytes for s in usable]),
        (
            "dependency_build_cache_gt_100mb",
            100_000_000,
            [s.dependency_cache_bytes + s.build_artifact_bytes for s in usable],
        ),
        (
            "dependency_build_cache_gt_1gb",
            1_000_000_000,
            [s.dependency_cache_bytes + s.build_artifact_bytes for s in usable],
        ),
        (
            "tool_test_output_gt_10mb",
            10_000_000,
            [s.tool_output_bytes + s.test_log_bytes for s in usable],
        ),
        (
            "tool_test_output_gt_100mb",
            100_000_000,
            [s.tool_output_bytes + s.test_log_bytes for s in usable],
        ),
    ]
    for name, threshold in STATE_SCALES_BYTES.items():
        if name == "tiny":
            continue
        specs.append((
            f"total_mobile_state_gt_{name}",
            threshold,
            [s.measured_dirty_bytes for s in usable],
        ))
    rows: list[ThresholdRow] = []
    for metric, threshold, values in specs:
        n = len(values)
        over = sum(1 for value in values if value > threshold)
        rows.append(ThresholdRow(metric, threshold, n, over, over / n if n else 0.0, True))
    return rows


def measured_restart_pressure(
    snapshots: list[SnapshotRow],
    *,
    repo_root: str | Path,
    max_workflows: int = 20,
) -> tuple[list[RestartPressureRow], list[RestartPressureRow]]:
    selected = _select_pressure_snapshots(snapshots, max_workflows=max_workflows)
    if not selected:
        return [], []
    episode, manifests = _episode_from_snapshots(selected)
    repo = Path(repo_root)
    bundle = load_bundle(
        repo / "configs" / "model_profiles.yaml",
        repo / "configs" / "sites_3site.yaml",
        "compact_kv",
    )
    budget = ResourceBudget.from_bundle(bundle)
    warmness = WarmnessMap.from_episode_seed(episode.state_warmness)
    results: dict[str, SimulationResult] = {}
    for policy in POLICIES:
        plan = run_reconstitution_policy(policy, episode, manifests, bundle, warmness, budget)
        results[policy] = simulate_fluid(episode, manifests, plan, bundle, warmness, budget)
    best = min(results, key=lambda p: results[p].p50_resume_s())
    mixed_beats_strong = results["mixed_min_pressure"].p50_resume_s() < results["cache_reuse"].p50_resume_s()
    mixed_beats_random = results["mixed_min_pressure"].p50_resume_s() < results["random_mode"].p50_resume_s()
    rows = [
        RestartPressureRow(
            episode_id=episode.episode_id,
            n_workflows=len(selected),
            policy=policy,
            p50_resume_proxy_s=result.p50_resume_s(),
            p90_resume_proxy_s=result.p90_resume_s(),
            p99_resume_proxy_s=_quantile_float(result.per_workflow_finish_s.values(), 0.99),
            makespan_s=result.makespan_s,
            dominant_bottleneck=_dominant_bottleneck(result),
            best_policy=best,
            mixed_beats_strong=mixed_beats_strong,
            mixed_beats_random=mixed_beats_random,
        )
        for policy, result in sorted(results.items())
    ]
    return rows, rows


def measured_restart_packages(
    snapshots: list[SnapshotRow],
    *,
    max_rows: int = 5,
) -> list[PackageRow]:
    selected = [
        s for s in snapshots
        if s.safe_for_claims and (s.measured_dirty_bytes > 0 or s.snapshot_representation_bytes > 0)
    ]
    selected.sort(key=lambda s: (-s.snapshot_representation_bytes, -s.measured_dirty_bytes, s.run_id))
    selected = selected[:max_rows]
    rows: list[PackageRow] = []
    for snapshot in selected:
        transcript_bytes = snapshot.tool_output_bytes
        dirty_env = snapshot.measured_dirty_bytes
        snapshot_env = snapshot.snapshot_representation_bytes
        lazy_env = dirty_env + snapshot.test_log_bytes
        for package_type, valid, reason, moved, lazy in (
            (
                "prompt_transcript_only",
                dirty_env == 0,
                "lacks_measured_dirty_environment_state" if dirty_env else "no_dirty_environment_state",
                transcript_bytes,
                lazy_env,
            ),
            (
                "base_repo_plus_diff",
                snapshot.final_diff_bytes > 0,
                "diff_patch_available" if snapshot.final_diff_bytes > 0 else "no_nonempty_diff_for_dirty_workspace",
                transcript_bytes + snapshot.final_diff_bytes,
                max(0, lazy_env - snapshot.final_diff_bytes),
            ),
            (
                "full_workspace_snapshot",
                snapshot.agent_workspace_retained,
                "workspace_retained" if snapshot.agent_workspace_retained else "missing_workspace_snapshot",
                transcript_bytes + snapshot_env,
                0,
            ),
        ):
            model_s = moved / 5e9
            env_s = (moved + lazy) / 1e9 if valid else math.inf
            rows.append(PackageRow(
                run_id=snapshot.run_id,
                cut_point_kind="post_run_measured_snapshot",
                package_type=package_type,
                structurally_valid=valid,
                validity_reason=reason,
                bytes_moved=moved,
                lazy_bytes_required_later=lazy,
                model_resume_s=model_s,
                environment_resume_s=env_s,
                task_resume_s=max(model_s, env_s),
                provenance="measured_snapshot_cost_dirty_payload_headline",
            ))
    return rows


def _snapshot_from_dict(row: dict[str, str]) -> SnapshotRow:
    kwargs = {}
    for field, spec in SnapshotRow.__dataclass_fields__.items():
        raw = row.get(field, spec.default)
        if field in INT_FIELDS:
            kwargs[field] = int(raw or 0)
        elif field in BOOL_FIELDS:
            kwargs[field] = str(raw).lower() == "true"
        else:
            kwargs[field] = raw
    return SnapshotRow(**kwargs)


def _select_pressure_snapshots(snapshots: list[SnapshotRow], *, max_workflows: int) -> list[SnapshotRow]:
    retained = [s for s in snapshots if s.safe_for_claims]
    retained.sort(key=lambda s: (-s.measured_dirty_bytes, s.run_id))
    return retained[:max_workflows]


def _episode_from_snapshots(snapshots: list[SnapshotRow]) -> tuple[MobilityEpisode, dict[str, ServingGroupManifest]]:
    workflows: list[Workflow] = []
    manifests: dict[str, ServingGroupManifest] = {}
    for idx, snapshot in enumerate(snapshots):
        wid = f"measured_{idx:04d}"
        workflows.append(Workflow(workflow_id=wid, manifest_path=f"<inline:measured:{wid}>", src_site="phoenix"))
        manifests[wid] = _manifest_from_snapshot(wid, snapshot)
    return MobilityEpisode(
        episode_id=f"measured_mobile_state_n{len(snapshots)}",
        source_sites=("phoenix",),
        destination_sites=("seattle", "austin"),
        workflows=tuple(workflows),
        state_warmness={},
        notes="measured coding-data-collection post-run snapshots; no harness execution",
    ), manifests


def _manifest_from_snapshot(wid: str, snapshot: SnapshotRow) -> ServingGroupManifest:
    tokens = max(1, _tokens_from_run_manifest(snapshot.run_dir))
    states = {
        f"prompt_{wid}": StateObject(
            state_id=f"prompt_{wid}",
            content_hash=f"hash_prompt_{wid}",
            layer="prompt_context",
            lifetime="private",
            tokens=tokens,
            bytes=None,
            home_site="phoenix",
        )
    }
    workspace_bytes = snapshot.measured_dirty_bytes
    if workspace_bytes > 0:
        states[f"workspace_{wid}"] = StateObject(
            state_id=f"workspace_{wid}",
            content_hash=f"hash_workspace_{wid}",
            layer="workspace",
            lifetime="private",
            tokens=0,
            bytes=workspace_bytes,
            home_site="phoenix",
        )
    memory_bytes = snapshot.test_log_bytes
    if memory_bytes > 0:
        states[f"tool_memory_{wid}"] = StateObject(
            state_id=f"tool_memory_{wid}",
            content_hash=f"hash_tool_memory_{wid}",
            layer="memory",
            lifetime="private",
            tokens=0,
            bytes=memory_bytes,
            home_site="phoenix",
        )
    node = WorkNode(
        node_id=f"resume_{wid}",
        node_type="llm_call",
        parent_node_id=None,
        workflow_id=wid,
        label=None,
        status="complete",
        required_state=list(states),
        produced_state=[],
        session_id=wid,
    )
    return ServingGroupManifest(
        workflow_id=wid,
        root_task=snapshot.run_id,
        nodes={node.node_id: node},
        state_objects=states,
        edges=[],
    )


def _tokens_from_run_manifest(run_dir: str) -> int:
    path = Path(run_dir) / "run_manifest.json"
    if not path.is_file():
        return 2_000
    import json
    raw = json.loads(path.read_text(encoding="utf-8"))
    metrics = raw.get("metrics") or {}
    calls = int(metrics.get("total_model_calls") or metrics.get("max_model_calls") or 1)
    total = int(metrics.get("total_tokens_in") or metrics.get("tokens_in") or 2_000)
    return max(1, total // max(1, calls))


def _dominant_bottleneck(result: SimulationResult) -> str:
    weights: dict[str, float] = {}
    for action in result.actions:
        if action.bottleneck == "none":
            continue
        weights[action.bottleneck] = weights.get(action.bottleneck, 0.0) + max(
            action.finished_s - action.started_s, 0.0
        )
    if not weights:
        return "none"
    return max(weights.items(), key=lambda kv: (kv[1], kv[0]))[0]


def _quantile(values: Iterable[int], q: float) -> int:
    vals = sorted(values)
    if not vals:
        return 0
    idx = max(0, min(len(vals) - 1, math.ceil(q * len(vals)) - 1))
    return vals[idx]


def _quantile_float(values: Iterable[float], q: float) -> float:
    vals = sorted(values)
    if not vals:
        return 0.0
    idx = max(0, min(len(vals) - 1, math.ceil(q * len(vals)) - 1))
    return vals[idx]


def _write_csv(path: Path, rows: list) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].__dataclass_fields__.keys()))
        writer.writeheader()
        for row in rows:
            writer.writerow(asdict(row))


__all__ = [
    "SnapshotRow",
    "layer_distribution",
    "measured_restart_packages",
    "measured_restart_pressure",
    "read_snapshot_index",
    "threshold_rows",
    "write_measured_artifacts",
]
