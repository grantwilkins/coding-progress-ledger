"""Claim: the shadow matrix reacts once only when a repair helps."""

import csv
from pathlib import Path

from repair_shadow_campaign import prepare, run_campaign


def test_shadow_matrix_has_one_stable_decision_per_cut(tmp_path):
    out, raw = tmp_path / "out", tmp_path / "raw"

    rows = run_campaign(out, raw, "test")

    assert len(rows) == 9
    assert all(row["repair_requests"] == 1 and row["pre_cut_requests"] == 0
               for row in rows)
    assert {row["outcome"] for row in rows if row["condition"] != "both"} \
        == {"proposal"}
    assert {row["outcome"] for row in rows if row["condition"] == "both"} \
        == {"revised_maximum"}
    assert len(list(csv.DictReader((out / "summary.csv").open()))) == 9
    assert len(list(raw.glob("*.jsonl"))) == 9


def test_prepare_requests_two_ramr_a100s(tmp_path):
    sbatch = prepare(tmp_path / "job", Path("/scratch/raw"))
    text = sbatch.read_text()

    assert "--partition=ramr" in text
    assert "--gres=gpu:2" in text
    assert "GPU_SKU:A100_SXM4&GPU_MEM:80GB" in text
    assert "nvidia-smi" in text
