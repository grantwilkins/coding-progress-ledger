import json

import service_headroom_driver as driver


def test_atomic_status_and_complete_detection(tmp_path):
    path = tmp_path / "status.json"
    driver.write_json(path, {"status": "complete", "value": 1})

    assert json.loads(path.read_text()) == {"status": "complete", "value": 1}
    assert driver.complete(path)
    assert not driver.complete(tmp_path / "missing.json")


def test_driver_retries_only_the_current_cell(tmp_path, monkeypatch):
    plan = {"schema": "test"}
    plan_path = tmp_path / "plan.json"
    runs = tmp_path / "runs"
    calls = []

    def invoke(*args):
        calls.append(args)
        cell_id = args[args.index("--cell-id") + 1]
        path = runs / cell_id / "result.json"
        path.parent.mkdir(parents=True, exist_ok=True)
        if len(calls) == 1:
            path.write_text('{"status":"invalid"}\n')
            raise driver.subprocess.CalledProcessError(1, args)
        path.write_text('{"status":"complete"}\n')

    monkeypatch.setattr(driver, "invoke", invoke)
    monkeypatch.setattr(driver.time, "sleep", lambda _seconds: None)
    driver.run_cells(plan, plan_path, ["cell-a", "cell-b"], runs, None,
                     tmp_path / "status.json", "stage", 0, 3)

    assert [call[call.index("--cell-id") + 1] for call in calls] \
        == ["cell-a", "cell-a", "cell-b"]
    status = json.loads((tmp_path / "status.json").read_text())
    assert status["completed_cells"] == 2
