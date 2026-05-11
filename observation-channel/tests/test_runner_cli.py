from pathlib import Path

from observation_channel.cli import main
from observation_channel.io import read_turns, write_jsonl, write_turns
from observation_channel.models import Turn
from observation_channel.runner import annotate_file


def test_annotate_file_writes_rows_and_summary(tmp_path: Path) -> None:
    turn_file = tmp_path / "trace.jsonl"
    out_dir = tmp_path / "out"
    write_turns(
        turn_file,
        [
            Turn(step=1, kind="action", tool="ls", command="ls", instance_id="trace"),
            Turn(step=2, kind="observation", response="ok", instance_id="trace"),
        ],
    )

    summary = annotate_file(turn_file, out_dir)

    assert summary.final_total == 1
    assert (out_dir / "trace.csv").exists()
    assert (out_dir / "trace.summary.csv").exists()


def test_preprocess_cli_from_raw_jsonl(tmp_path: Path) -> None:
    raw = tmp_path / "raw.jsonl"
    out = tmp_path / "turns"
    write_jsonl(
        raw,
        [
            {
                "instance_id": "case",
                "trajectory": [
                    {"role": "assistant", "content": "create app.py"},
                    {"role": "user", "observation": "ok"},
                ],
            }
        ],
    )

    rc = main(["preprocess", "--source", "swe-agent", "--input-jsonl", str(raw), "--out-dir", str(out)])

    assert rc == 0
    turns = read_turns(out / "swe-agent" / "case.jsonl")
    assert [turn.kind for turn in turns] == ["action", "observation"]


def test_empirical_bayes_eval_cli_writes_report_and_bundle(tmp_path: Path) -> None:
    traces = tmp_path / "traces.csv"
    turns = tmp_path / "turns.csv"
    report_dir = tmp_path / "report"
    bundle = tmp_path / "lookup.json"
    traces.write_text(
        "trace_key,source,final_total,total_turns,first_stuck_step,censored_right_tail,parse_error\n"
        "a,s,5,3,,False,\n"
        "b,s,5,3,,False,\n",
        encoding="utf-8",
    )
    turns.write_text(
        "trace_key,source,instance_id,step,total,done,current_category,current_unit_age,kind,tool\n"
        "a,s,a,1,4,0,PRODUCT,1,action,bash\n"
        "b,s,b,1,4,0,PRODUCT,1,action,bash\n",
        encoding="utf-8",
    )

    rc = main(
        [
            "empirical-bayes-eval",
            "--turns-csv",
            str(turns),
            "--traces-csv",
            str(traces),
            "--report-dir",
            str(report_dir),
            "--bundle-path",
            str(bundle),
            "--min-support",
            "1",
        ]
    )

    assert rc == 0
    assert bundle.exists()
    assert (report_dir / "REPORT.md").exists()
    assert (report_dir / "heldout_predictions.csv").exists()
    assert "B=400" in (report_dir / "REPORT.md").read_text(encoding="utf-8")
