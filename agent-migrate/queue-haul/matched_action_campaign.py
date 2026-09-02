"""Compare action choices on one completed A100 East/Germany campaign."""

from __future__ import annotations

import argparse
import csv
import importlib.util
import json
import sys
from collections import Counter
from dataclasses import asdict, replace
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

import network_campaign
import plot_style
from migration_profiler import file_hash, object_hash
from planner import plan, source_power
from profiles import ModelProfile, PhasePower, RateCurve


ROOT = Path(__file__).parent
SCHEMA = "queue-haul-matched-action-campaign-v2"
SCENARIO_ID = "4ce7626a1f20a5c3"
PARENT = ROOT / "outputs/east-germany-frontier-20260808/pilot/plan.json"
RESULTS = PARENT.with_name("results.csv")
DECISION = PARENT.parent / f"scenarios/{SCENARIO_ID}/attempt-0001/decision.json"
KV_SOURCE = ROOT.parent / "kv-transfer-early-experiment/migration_ratio.py"
CALIBRATIONS = {
    "openai/gpt-oss-20b": ROOT / "profiles/matched_action_h100_prefill/gpt-oss-20b.json",
    "Qwen/Qwen3.8-27B": ROOT / "profiles/matched_action_h100_prefill/qwen3.8-27b.json",
    "google/gemma-4-26B-A4B-it": ROOT / "profiles/matched_action_h100_prefill/gemma-4-26b.json",
}
POWER_CALIBRATIONS = {
    "openai/gpt-oss-20b": ROOT / "profiles/matched_action_h100_power/gpt-oss-20b.json",
    "Qwen/Qwen3.8-27B": ROOT / "profiles/matched_action_h100_power/qwen3.8-27b.json",
    "google/gemma-4-26B-A4B-it": ROOT / "profiles/matched_action_h100_power/gemma-4-26b.json",
}
ARMS = (
    ("gpt_oss_20b_a100", "A100", "openai/gpt-oss-20b", "gpt-oss-20b"),
    ("gpt_oss_20b_h100", "H100", "openai/gpt-oss-20b", "gpt-oss-20b"),
    ("qwen3_8_27b_h100", "H100", "Qwen/Qwen3.8-27B", "Qwen3.8 27B"),
    ("gemma_4_26b_h100", "H100", "google/gemma-4-26B-A4B-it", "Gemma 4 26B-A4B"),
)
ACTIONS = ("east_replay", "east_kv_transfer", "germany_replay", "germany_kv_transfer")


def _load_kv_source():
    spec = importlib.util.spec_from_file_location("queue_haul_kv_model", KV_SOURCE)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot load {KV_SOURCE}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def _manifest_path(parent: dict) -> Path:
    path = Path(parent["manifest"]["path"])
    return path if path.is_absolute() else ROOT.parent / path


def load_inputs():
    parent = json.loads(PARENT.read_text())
    network_campaign.validate_plan(parent)
    selected = [row for row in parent["scenarios"] if row["scenario_id"] == SCENARIO_ID]
    if len(selected) != 1:
        raise ValueError("matched scenario must occur exactly once")
    scenario = selected[0]
    signature = (scenario["design"], scenario["pack"], len(scenario["sessions"]),
                 {row["initial_tokens"] for row in scenario["sessions"]},
                 scenario["source_load"], scenario["requested_shed_fraction"],
                 scenario["deadline_s"], scenario["background"], scenario["bandwidth"])
    expected = ("frontier", "8x16k", 8, {16_384}, .8, .8, 30,
                {"east": [0, 0], "germany": [0, 0]}, "natural")
    if signature != expected:
        raise ValueError("selected A100 campaign cell changed")
    manifest_path = _manifest_path(parent)
    if file_hash(manifest_path) != parent["manifest"]["sha256"]:
        raise ValueError("campaign manifest changed")
    manifest = json.loads(manifest_path.read_text())
    with RESULTS.open(newline="") as handle:
        observed = [row for row in csv.DictReader(handle)
                    if row["scenario_id"] == SCENARIO_ID]
    if len(observed) != 1 or observed[0]["status"] != "complete" \
            or observed[0]["deadline_met"] != "True" \
            or observed[0]["target_met"] != "True":
        raise ValueError("selected A100 campaign was not completed successfully")
    decision = json.loads(DECISION.read_text())
    observed_actions = Counter(
        f"{row['destination_instance']}_{row['method']}" for row in decision["moves"])
    if observed_actions != {"germany_kv_transfer": 8}:
        raise ValueError("completed A100 action mix changed")
    summaries = {model: json.loads(path.read_text())
                 for model, path in CALIBRATIONS.items()}
    for model, summary in summaries.items():
        if summary.get("schema") != "queue-haul-h100-serving-calibration-v1" \
                or summary.get("model") != model \
                or summary.get("kv_capacity_tokens", 0) < 8 * 16_384 \
                or len(summary.get("server_info_sha256", "")) != 64 \
                or not any(row["context_tokens"] == 16_384 for row in summary["curve"]):
            raise ValueError(f"invalid H100 prefill calibration for {model}")
    powers = {model: json.loads(path.read_text())
              for model, path in POWER_CALIBRATIONS.items()}
    for model, power in powers.items():
        phase = PhasePower.parse(power["phase_power"])
        if power.get("schema") != "queue-haul-campaign-power-fit-v1" \
                or power.get("model") != model or power.get("hardware") != "H100" \
                or power.get("validation", {}).get("gate_passed") is not True \
                or len(power.get("bootstrap_curve_counts", ())) \
                != len(phase.measured_power_bootstrap) \
                or sum(power["bootstrap_curve_counts"]) != 200:
            raise ValueError(f"invalid H100 power calibration for {model}")
    return parent, scenario, manifest, observed[0], observed_actions, summaries, powers


def build_profiles(summaries: dict, powers: dict):
    bases = {
        "A100": ModelProfile.load(ROOT / "profiles/gpt_oss_20b_a100_tp1_azure_300w.json"),
        "H100": ModelProfile.load(ROOT / "profiles/gpt_oss_20b_h100_tp1.json"),
    }
    kv_source = _load_kv_source()
    kv_models = {label: kv_source.model(label) for *_, label in ARMS}
    profiles = {}
    for arm, hardware, model, kv_label in ARMS:
        base, summary = bases[hardware], summaries[model]
        case = base.case()
        measured = RateCurve.parse({"1": [
            [row["context_tokens"], row["prefill_tps_median"]]
            for row in summary["curve"]
        ]}) if hardware == "H100" else case.prefill
        contexts = [row["context_tokens"] for row in summary["curve"]]
        kv_model = kv_models[kv_label]
        transfer = replace(
            case.kv_transfer, block_tokens=1, block_bytes=int(kv_model.kv_bytes(1)),
            setup_s=0, initial_completion_s=0, catch_up_fixed_s=0,
            bytes_by_context=tuple((tokens, int(kv_model.kv_bytes(tokens)))
                                   for tokens in contexts),
        )
        power = powers[model]
        case = replace(
            case,
            F=(power["F_prefill_tps"] if hardware == "H100" else case.F),
            G=(power["G_decode_tps"] if hardware == "H100" else case.G),
            prefill=measured, replay=measured, replay_completion_s=0,
            kv_transfer=transfer,
            phase_power=(PhasePower.parse(power["phase_power"])
                         if hardware == "H100" else case.phase_power),
        )
        profiles[arm] = replace(
            base, profile_id=f"matched-action-{arm}", model=model,
            kv_capacity_tokens=(base.kv_capacity_tokens if hardware == "A100"
                                else summary["kv_capacity_tokens"]),
            max_power_load=(base.max_power_load if hardware == "A100"
                            else power["max_power_load"]),
            cases={**base.cases, "central": case}, kv_geometry=None,
        )
    return profiles


def _input_identity(parent: dict, manifest_path: Path) -> tuple[str, dict]:
    paths = {
        "campaign_plan": PARENT, "campaign_results": RESULTS,
        "campaign_decision": DECISION, "campaign_manifest": manifest_path,
        "a100_profile": ROOT / "profiles/gpt_oss_20b_a100_tp1_azure_300w.json",
        "h100_profile": ROOT / "profiles/gpt_oss_20b_h100_tp1.json",
        "kv_model": KV_SOURCE, "campaign_code": Path(__file__),
        **{f"prefill_{model}": path for model, path in CALIBRATIONS.items()},
        **{f"power_{model}": path for model, path in POWER_CALIBRATIONS.items()},
    }
    files = {name: {"path": str(path), "sha256": file_hash(path)}
             for name, path in paths.items()}
    identity = {
        "schema": SCHEMA, "scenario_id": SCENARIO_ID, "files": files,
        "planner": "lp_work_first", "kv_endpoint_residuals_s": 0,
        "parent_manifest_sha256": parent["manifest"]["sha256"],
    }
    return object_hash(identity), identity


def _evaluate(arm: str, hardware: str, model: str, profile: ModelProfile,
              scenario: dict, manifest: dict, input_sha256: str) -> dict:
    problem, architecture, routes, _requested, _demand = network_campaign._scenario_problem(
        scenario, manifest, profile)
    result = plan(problem, profile, routes, "lp_work_first",
                  seed=scenario["planner_seed"], destination=architecture)
    minimum = source_power(problem, profile,
                           (session.session_id for session in problem.sessions))
    removable = result.initial_source_power_w - minimum
    actions = Counter(f"{move.destination_instance}_{move.method}"
                      for move in result.moves)
    row = {
        "schema": SCHEMA, "input_sha256": input_sha256, "arm_id": arm,
        "hardware": hardware, "model": model, "feasible": bool(result.feasible),
        "target_met": bool(result.planned_source_power_w <= problem.power_limit_w + 1e-9),
        "session_count": len(problem.sessions), "move_count": len(result.moves),
        "kv_capacity_tokens": profile.kv_capacity_tokens,
        "action_counts": {action: actions[action] for action in ACTIONS},
        "method_counts": {method: sum(actions[action] for action in ACTIONS
                                      if action.endswith(method))
                          for method in ("replay", "kv_transfer")},
        "attained_fraction": float((result.initial_source_power_w
                                    - result.planned_source_power_w) / removable),
        "initial_source_power_w": float(result.initial_source_power_w),
        "planned_source_power_w": float(result.planned_source_power_w),
        "predicted_migration_makespan_s": (
            None if result.predicted_migration_makespan_s is None
            else float(result.predicted_migration_makespan_s)),
        "binding_resources": list(result.binding_resources),
        "moves": [asdict(move) for move in result.moves],
    }
    return row


def _arm(path: Path, args, input_sha256: str) -> dict:
    if path.exists():
        row = json.loads(path.read_text())
        if row.get("schema") != SCHEMA or row.get("input_sha256") != input_sha256 \
                or row.get("arm_id") != args[0]:
            raise ValueError(f"stale checkpoint {path}")
        return row
    row = _evaluate(*args, input_sha256)
    network_campaign.write_checkpoint(path, row)
    return row


def _write_csv(path: Path, arms: list[dict]) -> None:
    fields = ("arm_id", "hardware", "model", "move_count", "not_moved",
              *ACTIONS, "replay", "kv_transfer", "attained_fraction",
              "predicted_migration_makespan_s", "feasible", "target_met")
    temporary = path.with_suffix(".csv.tmp")
    with temporary.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, lineterminator="\n")
        writer.writeheader()
        for arm in arms:
            writer.writerow({
                **{key: arm[key] for key in ("arm_id", "hardware", "model", "move_count",
                                             "attained_fraction",
                                             "predicted_migration_makespan_s", "feasible",
                                             "target_met")},
                "not_moved": arm["session_count"] - arm["move_count"],
                **arm["action_counts"], **arm["method_counts"],
            })
    temporary.replace(path)


def _bootstrap_profiles(profiles: dict, powers: dict, scenario: dict,
                        manifest: dict, input_sha256: str) -> dict:
    result = {}
    for arm, hardware, model, _label in ARMS:
        if hardware != "H100":
            continue
        profile, fit = profiles[arm], powers[model]
        phase = profile.case().phase_power
        outcomes = Counter()
        for curve, count in zip(phase.measured_power_bootstrap,
                                fit["bootstrap_curve_counts"]):
            sampled = replace(phase, measured_power_curve=curve)
            varied = replace(profile, cases={**profile.cases, "central": replace(
                profile.case(), phase_power=sampled)})
            row = _evaluate(arm, hardware, model, varied, scenario, manifest,
                            input_sha256)
            outcomes[(row["feasible"], row["target_met"], *(
                row["action_counts"][action] for action in ACTIONS))] += count
        result[arm] = {
            "samples": sum(outcomes.values()), "unique_power_curves": len(
                phase.measured_power_bootstrap),
            "outcomes": [{"samples": count, "probability": count / sum(outcomes.values()),
                          "feasible": counts[0], "target_met": counts[1],
                          "action_counts": dict(zip(ACTIONS, counts[2:])),
                          "method_counts": {
                              method: sum(value for action, value in zip(ACTIONS, counts[2:])
                                          if action.endswith(method))
                              for method in ("replay", "kv_transfer")}}
                         for counts, count in sorted(outcomes.items())],
        }
    return result


def _write_bootstrap(path: Path, bootstrap: dict) -> None:
    fields = ("arm_id", "samples", "probability", "feasible", "target_met",
              *ACTIONS, "replay", "kv_transfer")
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, lineterminator="\n")
        writer.writeheader()
        for arm, summary in bootstrap.items():
            for outcome in summary["outcomes"]:
                writer.writerow({"arm_id": arm, "samples": outcome["samples"],
                                 "probability": outcome["probability"],
                                 "feasible": outcome["feasible"],
                                 "target_met": outcome["target_met"],
                                 **outcome["action_counts"], **outcome["method_counts"]})


def _plot(path: Path, arms: list[dict]) -> None:
    plot_style.apply()
    fig, axis = plt.subplots(figsize=(10, 4.5))
    bottom = [0] * len(arms)
    for action in (*ACTIONS, "not_moved"):
        values = [arm["action_counts"].get(action, 0) if action != "not_moved"
                  else arm["session_count"] - arm["move_count"] for arm in arms]
        axis.bar(range(len(arms)), values, bottom=bottom,
                 color=plot_style.ACTION_COLORS[action],
                 hatch=plot_style.ACTION_HATCHES[action],
                 label=plot_style.ACTION_NAMES[action])
        bottom = [old + value for old, value in zip(bottom, values)]
    axis.set_xticks(range(len(arms)), [
        f"{plot_style.MODEL_NAMES[arm['model']]}\n{arm['hardware']}"
        + ("\nTarget unmet" if not arm["target_met"] else "") for arm in arms
    ])
    axis.tick_params(axis="x", labelsize=12)
    axis.set(ylabel="Sessions", ylim=(0, 9.8), yticks=range(0, 9, 2),
             title="Planner action mix: same 8 × 16K East/Germany campaign")
    axis.legend(frameon=False, ncol=3, fontsize=8, loc="upper center")
    fig.tight_layout()
    for suffix in ("png", "pdf"):
        fig.savefig(path.with_suffix(f".{suffix}"))
    plt.close(fig)


def run(out: Path) -> dict:
    parent, scenario, manifest, observed, observed_actions, summaries, powers = load_inputs()
    profiles = build_profiles(summaries, powers)
    input_sha256, identity = _input_identity(parent, _manifest_path(parent))
    out.mkdir(parents=True, exist_ok=True)
    arms = [_arm(out / "arms" / f"{arm}.json",
                 (arm, hardware, model, profiles[arm], scenario, manifest), input_sha256)
            for arm, hardware, model, _label in ARMS]
    by_id = {arm["arm_id"]: arm for arm in arms}
    a100, gpt_h100 = by_id["gpt_oss_20b_a100"], by_id["gpt_oss_20b_h100"]
    h100 = [arm for arm in arms if arm["hardware"] == "H100"]
    bootstrap = _bootstrap_profiles(profiles, powers, scenario, manifest, input_sha256)
    gates = {
        "a100_model_reproduces_completed_action_mix":
            a100["feasible"] and a100["target_met"]
            and a100["action_counts"] == {
                action: observed_actions[action] for action in ACTIONS},
        "hardware_changes_method_mix": a100["method_counts"] != gpt_h100["method_counts"],
        "models_change_method_mix": len({tuple(arm["method_counts"].values())
                                          for arm in h100}) > 1,
        "all_h100_model_action_mixes_distinct": len({
            tuple(arm["action_counts"].values()) for arm in h100}) == len(h100),
        "all_model_power_repeat_gates_pass": all(
            power["validation"]["gate_passed"] for power in powers.values()),
        "all_h100_power_bootstraps_complete": all(
            row["samples"] == 200 for row in bootstrap.values()),
        "model_specific_power_changes_feasibility":
            len({arm["feasible"] for arm in h100}) > 1,
    }
    if not all(gates.values()):
        raise RuntimeError(f"matched action gates failed: {gates}")
    gpt_h100_sha256 = object_hash(gpt_h100)
    summary = {
        "schema": SCHEMA, "status": "complete", "input_sha256": input_sha256,
        "campaign": {
            "scenario_id": SCENARIO_ID, "regions": ["east", "germany"],
            "sessions": 8, "context_tokens_per_session": 16_384,
            "requested_shed_fraction": .8, "deadline_s": 30,
            "bandwidth_mbps": scenario["bandwidth_mbps"], "background": scenario["background"],
        },
        "completed_a100_run": {
            "status": observed["status"], "deadline_met": True, "target_met": True,
            "migration_s": float(observed["migration_s"]),
            "requested_shed_w": float(observed["requested_shed_w"]),
            "realized_shed_w": float(observed["realized_shed_w"]),
            "request_failures": int(observed["request_failures"]),
            "action_counts": {action: observed_actions[action] for action in ACTIONS},
        },
        "arms": arms, "power_bootstrap": bootstrap, "gates": gates,
        "comparisons": {
            "hardware": {"arms": [a100["arm_id"], gpt_h100["arm_id"]],
                         "gpt_h100_result_sha256": gpt_h100_sha256},
            "models": {"arms": [arm["arm_id"] for arm in h100],
                       "gpt_h100_result_sha256": gpt_h100_sha256},
        },
        "inputs": identity,
        "scope": (
            "Central action proof plus a 200-draw calibration bootstrap, not a workload "
            "population estimate. Every H100 arm uses its own measured coding-path power, "
            "prefill, and decode rates; BF16 KV bytes are analytic and endpoint residuals "
            "are zero. Only the archived A100 run is a physical migration; predicted "
            "makespans are scheduling diagnostics."
        ),
    }
    if summary["comparisons"]["hardware"]["gpt_h100_result_sha256"] \
            != summary["comparisons"]["models"]["gpt_h100_result_sha256"]:
        raise RuntimeError("GPT-OSS/H100 result was not reused")
    _write_csv(out / "action_mix.csv", arms)
    _write_bootstrap(out / "bootstrap_action_mix.csv", bootstrap)
    _plot(out / "action_mix", arms)
    network_campaign.write_checkpoint(out / "summary.json", summary)
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path,
                        default=ROOT / "outputs/matched-action-campaign-power-20260821")
    summary = run(parser.parse_args().out)
    print(json.dumps({"status": summary["status"], "gates": summary["gates"]}, indent=2))


if __name__ == "__main__":
    main()
