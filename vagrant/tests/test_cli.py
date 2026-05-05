import json
import subprocess
from pathlib import Path

from vagrant_agent.cli import manifest_main, trace_main

REPO = Path(__file__).resolve().parent.parent
TRACE = REPO / "examples" / "traces" / "toy_subagent_trace.jsonl"


def test_manifest_build_via_python(tmp_path: Path, capsys):
    out = tmp_path / "toy.json"
    rc = manifest_main(["build", "--trace", str(TRACE), "--out", str(out)])
    assert rc == 0
    assert out.exists()
    data = json.loads(out.read_text())
    assert data["workflow_id"] == "toy_workflow_v1"
    assert set(data["nodes"]) == {"S1", "S2", "S3", "S4"}
    captured = capsys.readouterr()
    assert "4 nodes" in captured.out


def test_manifest_build_via_subprocess(tmp_path: Path):
    out = tmp_path / "toy.json"
    result = subprocess.run(
        ["uv", "run", "vagrant-manifest", "build", "--trace", str(TRACE), "--out", str(out)],
        cwd=REPO, capture_output=True, text=True,
    )
    assert result.returncode == 0, result.stderr
    assert out.exists()


def test_trace_summarize(capsys):
    rc = trace_main(["summarize", str(TRACE)])
    assert rc == 0
    captured = capsys.readouterr()
    assert "subtasks: 4" in captured.out
    assert "events: 33" in captured.out
    assert "state_read" in captured.out


def test_manifest_main_no_args_exits():
    import pytest
    with pytest.raises(SystemExit):
        manifest_main([])


def test_trace_main_no_args_exits():
    import pytest
    with pytest.raises(SystemExit):
        trace_main([])
