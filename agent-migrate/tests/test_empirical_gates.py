from __future__ import annotations

import csv
from pathlib import Path

from agent_migrate_agent.claim_cells import run_claim_cells, write_claim_cell_table
from agent_migrate_agent.cut_points import find_cut_points, load_trace_jsonl
from agent_migrate_agent.k8_regime import RegimeCell, default_bundle
from agent_migrate_agent.k8_validation import ValidationTarget
from agent_migrate_agent.restart_gate import run_minimal_restart_gate, write_minimal_restart_table
from agent_migrate_agent.resume_packages import WorkspaceFileEntry
from agent_migrate_agent.workload_gate import (
    run_workload_anchor_gate,
    write_workload_anchor_artifacts,
)


REPO = Path(__file__).resolve().parents[1]
H5A_TRACE = REPO / "examples" / "traces" / "h5a_multi_trajectory_swe.jsonl"


def test_gate1_claim_cell_table_contains_exact_policy_comparisons(tmp_path: Path):
    bundle = default_bundle(REPO)
    target = ValidationTarget(
        "tiny_prefill_smoke",
        RegimeCell(10, "tiny", "tight", 100, seed=9301),
        "smoke cell",
    )
    rows = run_claim_cells(bundle, targets=(target,))

    assert len(rows) == 1
    row = rows[0]
    assert row.exact_best_policy
    assert row.aggregate_best_policy
    assert row.resume_metric_kind == "k4_reconstitution_proxy_not_c5_task_resume"
    assert row.exact_p99_k4_resume_proxy_s >= row.exact_p90_k4_resume_proxy_s >= row.exact_p50_k4_resume_proxy_s
    assert isinstance(row.mixed_beats_strong_reuse, bool)
    assert isinstance(row.mixed_beats_random_diversification, bool)

    out = tmp_path / "exact_claim_cell_table.csv"
    write_claim_cell_table(rows, out)
    parsed = list(csv.DictReader(out.open()))
    assert parsed[0]["target"] == "tiny_prefill_smoke"
    assert "mixed_beats_random_diversification" in parsed[0]
    assert "aggregate_exact_best_policy_bottleneck_agrees" in parsed[0]


def test_gate2_workload_anchor_artifacts_expose_provenance_and_exact_regime(tmp_path: Path):
    bundle = default_bundle(REPO)
    layer_rows, regime_rows = run_workload_anchor_gate(
        bundle,
        n_workflows=2,
        budget_cell=RegimeCell(2, "medium", "tight", 1, seed=9302),
    )

    assert {row.anchor_name for row in layer_rows} == {
        "w1_large_repo_coding",
        "w2_data_rag_heavy",
        "w3_multi_agent_fanout",
    }
    assert all(row.synthetic_bytes > 0 for row in layer_rows)
    assert all(row.provenance == "hypothesis_fixture_synthetic_bytes" for row in layer_rows)
    assert all(row.exact_regime.endswith("regime") for row in regime_rows)

    write_workload_anchor_artifacts(layer_rows, regime_rows, tmp_path)
    assert (tmp_path / "state_layer_table.csv").exists()
    assert (tmp_path / "exact_anchor_regime_table.csv").exists()


def test_gate3_minimal_restart_table_uses_only_three_package_shapes(tmp_path: Path):
    events = load_trace_jsonl(H5A_TRACE)
    cuts = find_cut_points(events, trace_id="h5a")
    rows = run_minimal_restart_gate(
        events,
        cuts,
        harness_config={"cwd": "/workspace", "open_file": "", "env": {}},
        workspace_files=(WorkspaceFileEntry("workspace.tar", 1_000_000_000, "h_ws"),),
        diff_blob="diff --git a/x b/x\n--- a/x\n+++ b/x\n@@ -0,0 +1 @@\n+x\n",
        max_cuts=3,
    )

    assert len(rows) == 9
    assert {row.package_type for row in rows} == {
        "prompt_transcript_only",
        "base_repo_plus_diff",
        "full_workspace_snapshot",
    }
    assert all(row.task_resume_s >= row.model_resume_s for row in rows)
    assert all(row.task_resume_s >= row.environment_resume_s for row in rows)

    out = tmp_path / "minimal_package_table.csv"
    write_minimal_restart_table(rows, out)
    parsed = list(csv.DictReader(out.open()))
    assert parsed[0]["package_type"] in {
        "prompt_transcript_only",
        "base_repo_plus_diff",
        "full_workspace_snapshot",
    }
