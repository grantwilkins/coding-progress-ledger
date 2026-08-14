import json
import math

import live_timing_campaign as timing


def cluster(tmp_path):
    path = tmp_path / "cluster.json"
    path.write_text(json.dumps({
        "schema": "queue-haul-azure-cluster-v1",
        "source": {"id": "source", "region": "westus3", "host": "10.0.0.1",
                   "ssh_user": "u", "repo_root": "/r", "run_root": "/d"},
        "destinations": [{
            "id": name, "region": region, "host": host, "ssh_user": "u",
            "repo_root": "/r", "run_root": "/d",
        } for name, region, host in (
            ("east", "australiaeast", "10.0.0.2"),
            ("germany", "southcentralus", "10.0.0.3"))],
    }))
    return path


def test_pilot_plan_is_live_holdout_across_both_real_regions(tmp_path):
    manifest = tmp_path / "manifest.json"
    manifest.write_text(json.dumps({"sessions": [
        {"id": f"s{index}", "rank": index} for index in range(8)]}))
    plan = timing.make_plan(
        manifest, cluster(tmp_path), tmp_path / "plan.json", "pilot")

    timing.validate_plan(plan)
    assert len(plan["scenarios"]) == 8
    assert {row["region"] for row in plan["scenarios"]} == {
        "australiaeast", "southcentralus"}
    assert {row["context_tokens"] for row in plan["scenarios"]} == {
        8192, 31488}
    assert all(row["design"] == "timing_live"
               and row["split"] == "holdout" for row in plan["scenarios"])


def test_targeted_plan_separates_scus_kv_calibration_from_unseen_holdout(
        tmp_path):
    manifest = tmp_path / "manifest.json"
    manifest.write_text(json.dumps({"sessions": [
        {"id": "s", "rank": 0}]}))

    plan = timing.make_plan(
        manifest, cluster(tmp_path), tmp_path / "plan.json", "targeted")
    timing.validate_plan(plan)
    calibration = [row for row in plan["scenarios"]
                   if row["split"] == "calibration"]
    holdout = [row for row in plan["scenarios"] if row["split"] == "holdout"]

    assert len(calibration) == 4 and len(holdout) == 5
    assert {(row["region"], row["method"]) for row in calibration} == {
        ("southcentralus", "kv_transfer")}
    assert {row["context_tokens"] for row in calibration} == {8192, 31488}
    assert {row["context_tokens"] for row in holdout} == {16384, 24576}
    assert {(row["region"], row["method"]) for row in holdout} == {
        (region, method) for region in ("australiaeast", "southcentralus")
        for method in timing.METHODS}


def test_service_load_uses_only_the_aligned_window(tmp_path):
    path = tmp_path / "load.jsonl"
    path.write_text("\n".join(json.dumps(row) for row in (
        {"start_ns": 5_000_000_000, "prompt_tokens": 100,
         "output_tokens": 10},
        {"start_ns": 9_000_000_000, "prompt_tokens": 200,
         "output_tokens": 20},
    )))

    assert timing.service_load(path, 10_000_000_000, 4, 100, 10) == 1


def test_live_fit_uses_condition_holdouts_and_passes_exact_data(monkeypatch,
                                                               tmp_path):
    paths = [f"{region}:{method}" for region in (
        "australiaeast", "southcentralus") for method in timing.METHODS]
    rows = []
    for condition in range(40):
        for path in paths:
            row = {
                "scenario_id": str(condition), "condition_index": condition,
                "split": "prior_live", "destination": path.split(":")[0],
                "method": path.split(":")[1], "path": path,
                "context_tokens": timing.CONTEXTS[condition % 4],
                "width": 4, "same_path_width": 2,
                "destination_width": 4, "order_fraction": .5,
                "destination_load": (condition % 3) * .25,
                "time_to_first_token_s": 1, "decode_tail_s": 1,
                "api_upload_s": None, "remote_response_start_s": None,
                "response_header_to_token_s": None,
                "response_stream_s": None, "client_residual_s": None,
                "kv_transfer_window_s": None,
            }
            row["observed_s"] = math.exp(sum(timing.features(row, paths)))
            rows.append(row)
    monkeypatch.setattr(timing, "collect", lambda _root: rows)

    model = timing.fit(tmp_path, tmp_path / "fit")

    assert model["gate"]["passed"]
    assert model["source"] == "measured_live_transfers"
    assert len(model["cross_validation"]) == 5


def test_targeted_refit_uses_only_calibration_split(monkeypatch, tmp_path):
    paths = [f"{region}:{method}" for region in (
        "australiaeast", "southcentralus") for method in timing.METHODS]
    names = timing.feature_names(paths)
    model_path = tmp_path / "model.json"
    model_path.write_text(json.dumps({
        "paths": paths, "feature_names": names,
        "coefficients": [0] * len(names)}))

    def row(path, context, split):
        value = {"path": path, "context_tokens": context, "width": 1,
                 "same_path_width": 1, "destination_width": 1,
                 "destination_load": 0, "order_fraction": 0,
                 "split": split}
        correction = math.exp(.5 + .2 * math.log(context / 8192)) \
            if path == "southcentralus:kv_transfer" else 1
        return {**value, "observed_s": correction}

    rows = [row("southcentralus:kv_transfer", context, "calibration")
            for context in (8192, 8192, 31488, 31488)]
    rows += [row(path, 16384, "holdout") for path in paths]
    rows += [row("southcentralus:kv_transfer", 24576, "holdout")]
    monkeypatch.setattr(timing, "collect", lambda _root: rows)

    report = timing.refit_targeted(
        model_path, tmp_path / "run", tmp_path / "out")

    assert report["passed"]
    assert report["overall"]["median_absolute_percentage_error"] < 1e-10
