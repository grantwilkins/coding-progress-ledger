"""Reduce existing bounded telemetry without collecting data or fitting coefficients."""
import csv
import hashlib
import json
import statistics
import subprocess
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
OUT = Path(__file__).resolve().parent
OLD = ROOT / "outputs/a100-replay-live-20260909T1920"
INPUTS = [OLD / name for name in ("unloaded-analysis.json", "report.json", "policy-verification.json")]
INPUTS += [ROOT / name for name in ("pool_replay_resident.py", "pool_replay_measure.py", "pool_replay_report.py", "pool_shed_execution.py", "pool_shed_campaign.py", "pool_shed_calibration.py", "pool_shed_planner.py", "pool_shed_replay_audit.py", "loaded_service_model.py", "README.md")]
rows = json.loads(INPUTS[0].read_text())["rows"]
report = json.loads(INPUTS[1].read_text())
policy = json.loads(INPUTS[2].read_text())
valid = [r for r in rows if r["seed"] == 7102 and r["phase_valid"]]
assert len(valid) == report["heldout"]["valid_second_repeat_phases"]
assert statistics.median(abs(r["model_minus_observed_s"]) for r in valid) == report["heldout"]["median_absolute_baseline_error_s"]
assert max(r["optimistic_error_s"] for r in valid) == report["heldout"]["max_optimistic_baseline_error_s"]
assert all(not result["resident_latency_validated"] for cell in policy["cells"] for result in cell["results"].values())
assert all(json.loads(r["prompt_counts"]) == [r["context"] + (0 if r["phase"] == "initial" else r["append"])] * r["requests"] for r in rows)

def write_csv(name, data):
    with (OUT / name).open("w") as handle:
        writer = csv.DictWriter(handle, list(data[0]))
        writer.writeheader()
        writer.writerows(data)

scaling = []
for seed in (7101, 7102):
    for context in (2048, 8192, 30000):
        for append in (32, 2048):
            for phase in ("initial", "catch_up", "cold_updated"):
                pair = [next(r for r in rows if (r["seed"], r["context"], r["append"], r["phase"], r["width"]) == (seed, context, append, phase, width)) for width in (1, 8)]
                usable = all(r["phase_valid"] for r in pair)
                ratio = pair[1]["elapsed_s"] / pair[0]["elapsed_s"] if usable else None
                scaling.append(dict(seed=seed, context=context, append=append, phase=phase, valid_pair=usable,
                    width1_elapsed_s=pair[0]["elapsed_s"], width8_elapsed_s=pair[1]["elapsed_s"],
                    width8_over_width1_elapsed=ratio, width8_over_eight_singletons=ratio / 8 if ratio else None))
write_csv("model-width-scaling.csv", scaling)

metrics = ("request_queue_time_seconds_sum", "request_prefill_time_seconds_sum", "request_decode_time_seconds_sum", "request_inference_time_seconds_sum", "request_prefill_kv_computed_tokens_sum")
decomposition = []
for r in rows:
    value = {k: r[k] for k in ("episode", "seed", "context", "append", "width", "phase", "phase_valid", "requests", "http_complete", "state_valid", "exact_requests", "cache_state", "elapsed_s", "model_full_rebuild_local_s", "model_minus_observed_s")}
    value.update({k: r[k] for k in metrics})
    value.update({"mean_completed_request_" + k.removeprefix("request_").removesuffix("_sum"): r[k] / r["http_complete"] if r[k] is not None and r["http_complete"] == r["width"] else None for k in metrics})
    decomposition.append(value)
write_csv("model-phase-decomposition.csv", decomposition)

errors = []
for phase in ("initial", "catch_up", "cold_updated"):
    for width in (1, 8):
        group = [r for r in valid if r["phase"] == phase and r["width"] == width]
        errors.append(dict(phase=phase, width=width, valid_conditions=len(group),
            median_absolute_error_s=statistics.median(abs(r["model_minus_observed_s"]) for r in group),
            max_optimistic_error_s=max(r["optimistic_error_s"] for r in group),
            max_pessimistic_error_s=max(max(0, r["model_minus_observed_s"]) for r in group),
            false_30s_completion_predictions=sum(r["model_predicts_30s_but_observed_exceeds"] for r in group)))
write_csv("model-heldout-errors.csv", errors)
result = dict(
    commit=subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip(),
    dirty_state_before_audit=subprocess.check_output(["git", "status", "--short"], cwd=ROOT, text=True),
    command="python3 outputs/a100-replay-completion-20260910T0116/model-audit.py",
    input_sha256={str(p.relative_to(ROOT)): hashlib.sha256(p.read_bytes()).hexdigest() for p in INPUTS},
    validation_checks={"old_report_heldout_summary_recomputed": True, "all_20_policy_latency_flags_false": True, "all_sent_unloaded_rendered_lengths_match_full_context_targets": True},
    valid_second_repeat_phases=len(valid), heldout_by_phase_and_width=errors,
    interpretation={
        "width": "Elapsed ratios are measured batching/packing comparisons. Eight singleton elapsed intervals are not measured divisible GPU work. Both append groups retain independent initial trials; invalid pairs are null.",
        "decomposition": "Engine histogram differences sum completed-request durations and may overlap in wall time; per-request arithmetic means are not additive GPU time, per-request trace attribution or percent-of-episode shares. All unloaded requests share one coordinator clock. Null means unavailable. No source, transfer or recovery time is supplied by unloaded traces.",
        "full_context": "All issued unloaded requests render exactly C for initial and C+append for both warm/cold updated phases; generation limit remains 512. This verifies full-message construction, not evolving source behavior.",
        "main_catchup_gap": "Prior main driver submits a fixed synthetic 32-token append, marks source_active=False, and resumes incoming completion from initial_history rather than captured evolving source history. It measures destination interference only; it cannot establish source quiescence, actual catch-up growth or state-preserving ownership switch.",
        "inventory_gap": "sample_fleet allocates gpus*8 session instances across 24 representative trajectory families. Prior hardware instantiates all 24 families on one GPU and distributes the offered rate by cohort weights. It is not the modeled eight physical sessions/GPU; cache footprint and per-session causal concurrency differ.",
        "correction": "No simulator coefficient correction is supported for fleet cache reuse/queueing by these destination-only measurements. Keep raw warm evidence and identify full-rebuild catch-up as contradicted locally. The smallest next adapter correction is actual history/state continuity through source boundary pause, and a matched physical resident inventory; do not fit an assumed cache percentage.",
        "readiness": "The existing negative campaign-ready gate and all twenty negative latency-validation flags are consistent with these data. Low median baseline error must not hide large warm-phase overprediction. Short-window tests and completed-only timing cannot establish tail compliance."},
    checks="Assertions in this reproduction script; no model or instrumentation code changed.")
(OUT / "model-audit.json").write_text(json.dumps(result, indent=2, allow_nan=False) + "\n")
print(json.dumps({"valid_second_repeat_phases": len(valid), "heldout_by_phase_and_width": errors}, indent=2))
