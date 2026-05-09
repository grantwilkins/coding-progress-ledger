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
