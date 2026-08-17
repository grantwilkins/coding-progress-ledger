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


def test_confirmation_mode_runs_frozen_order_then_reduces(tmp_path, monkeypatch):
    plan = {"schema": "confirmation", "hardware": "a100",
            "run_order": ["held-a", "held-b"]}
    core = {"schema": "core"}
    plan_path, core_path = tmp_path / "plan.json", tmp_path / "core.json"
    scout_path, normalization = tmp_path / "scout.json", tmp_path / "norm.json"
    scout_path.write_text('{"selection_ready":true}\n')
    calls = []

    monkeypatch.setattr(
        driver.campaign, "read_plan",
        lambda path: plan if path == plan_path else core,
    )
    monkeypatch.setattr(
        driver.campaign, "validate_confirmation_source",
        lambda observed, source, scout: calls.append(
            ("validate", observed, source, scout)),
    )
    monkeypatch.setattr(
        driver, "run_cells",
        lambda *args: calls.append(("run", args[2], args[3], args[4])),
    )

    def invoke(*args):
        calls.append(("reduce", args))
        out = args[args.index("--out") + 1]
        out.write_text('{"planner_usable":true,"supported_bound":0.7}\n')

    monkeypatch.setattr(driver, "invoke", invoke)
    driver.execute_confirmation(
        tmp_path, "a100", plan_path, core_path, scout_path,
        normalization, 0, 3,
    )

    assert calls[1] == ("run", plan["run_order"], tmp_path / "confirmation",
                        normalization)
    assert calls[2][0] == "reduce"
    status = json.loads((tmp_path / "confirmation-status.json").read_text())
    assert status["planner_usable"] and status["supported_bound"] == .7
    assert status["completed_cells"] == 2
