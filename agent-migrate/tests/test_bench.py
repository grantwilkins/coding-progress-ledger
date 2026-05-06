import csv
import json
import subprocess
from pathlib import Path

from agent_migrate_agent.bench import AUDIT_COLUMNS, run_bench
from agent_migrate_agent.cli import bench_main

REPO = Path(__file__).resolve().parent.parent
TRACE = REPO / "examples" / "traces" / "toy_subagent_trace.jsonl"
MODELS = REPO / "configs" / "model_profiles.yaml"
SITES = REPO / "configs" / "sites_2site.yaml"


def test_run_bench_end_to_end(tmp_path: Path):
    summary = run_bench(
        trace_path=TRACE,
        out_dir=tmp_path,
        policies=["request_level_no_reuse", "shared_state_aware"],
        model_path=MODELS,
        sites_path=SITES,
        model_name="compact_kv",
        tau=1,
    )
    assert (tmp_path / "manifest.json").exists()
    assert (tmp_path / "results.csv").exists()
    assert (tmp_path / "summary.json").exists()
    assert (tmp_path / "state_materialization_breakdown.csv").exists()
    assert (tmp_path / "request_level_no_reuse" / "placement_plan.json").exists()
    assert (tmp_path / "request_level_no_reuse" / "materialization_plan.json").exists()
    assert (tmp_path / "shared_state_aware" / "placement_plan.json").exists()

    a = summary["policies"]["request_level_no_reuse"]["cost_weighted_duplication_factor"]
    c = summary["policies"]["shared_state_aware"]["cost_weighted_duplication_factor"]
    assert c == 1.0
    assert a > c


def test_audit_csv_schema(tmp_path: Path):
    run_bench(TRACE, tmp_path,
              ["request_level_no_reuse", "shared_state_aware"],
              MODELS, SITES, "compact_kv", tau=1)
    rows = list(csv.DictReader((tmp_path / "state_materialization_breakdown.csv").open()))
    assert rows
    assert set(rows[0].keys()) == set(AUDIT_COLUMNS)
    for row in rows:
        assert int(row["materialization_count"]) >= 1
        assert int(row["ideal_materialization_count"]) == 1
        assert float(row["cost_s"]) >= 0
        assert float(row["total_cost_s"]) >= float(row["cost_s"])


def test_audit_csv_total_cost_reconciles_with_summary(tmp_path: Path):
    import pytest
    summary = run_bench(TRACE, tmp_path,
                        ["request_level_no_reuse", "shared_state_aware"],
                        MODELS, SITES, "compact_kv", tau=1)
    rows = list(csv.DictReader((tmp_path / "state_materialization_breakdown.csv").open()))
    by_policy: dict[str, float] = {}
    for row in rows:
        by_policy[row["policy"]] = by_policy.get(row["policy"], 0.0) + float(row["total_cost_s"])
    for policy_name, info in summary["policies"].items():
        assert by_policy[policy_name] == pytest.approx(info["total_cost_s"], abs=1e-9)


def test_headline_metric_reproducible_from_csv(tmp_path: Path):
    """The audit principle: cost_weighted_duplication_factor must be reproducible
    from the breakdown CSV alone, without re_running policies."""
    import pytest
    summary = run_bench(TRACE, tmp_path,
                        ["request_level_no_reuse", "shared_state_aware"],
                        MODELS, SITES, "compact_kv", tau=1)
    rows = list(csv.DictReader((tmp_path / "state_materialization_breakdown.csv").open()))
    by_policy: dict[str, tuple[float, float]] = {}
    for row in rows:
        cost = float(row["cost_s"])
        count = int(row["materialization_count"])
        paid, ideal = by_policy.get(row["policy"], (0.0, 0.0))
        by_policy[row["policy"]] = (paid + cost * count, ideal + cost)
    for policy_name, info in summary["policies"].items():
        paid, ideal = by_policy[policy_name]
        recomputed = paid / ideal if ideal > 0 else 1.0
        assert recomputed == pytest.approx(info["cost_weighted_duplication_factor"], abs=1e-9)


def test_results_csv_has_one_row_per_policy(tmp_path: Path):
    run_bench(TRACE, tmp_path,
              ["request_level_no_reuse", "shared_state_aware"],
              MODELS, SITES, "compact_kv", tau=1)
    rows = list(csv.DictReader((tmp_path / "results.csv").open()))
    assert {row["policy"] for row in rows} == {"request_level_no_reuse", "shared_state_aware"}


def test_bench_cli_python(tmp_path: Path, capsys):
    rc = bench_main([
        "--trace", str(TRACE),
        "--out", str(tmp_path),
    ])
    assert rc == 0
    captured = capsys.readouterr()
    assert "request_level_no_reuse" in captured.out
    assert "shared_state_aware" in captured.out

    plot = tmp_path / "plots" / "duplication_factor.png"
    assert plot.exists()
    header = plot.read_bytes()[:8]
    assert header == b"\x89PNG\r\n\x1a\n", f"not a PNG: {header!r}"


def test_bench_cli_unknown_policy_hard_fails(tmp_path: Path):
    import pytest
    with pytest.raises(SystemExit):
        bench_main([
            "--trace", str(TRACE),
            "--out", str(tmp_path),
            "--policies", "nonexistent_policy",
            "--no_plot",
        ])
