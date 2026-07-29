"""Cache and plot canonical fixed-contract residual headroom."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
from dataclasses import replace
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns

from destination_bench import (
    CLASSES,
    DEFAULT_MANIFEST,
    DEFAULT_MODEL,
    KV_BLOCK_TOKENS,
    Pressure,
    WORKLOADS,
    architecture,
    log_bytes_per_token,
    pack_source,
    sample_sessions,
    scenario,
    trace_shapes,
)
from power_model import ExpectedPower
from profiles import ModelProfile
from requirement_frontier import _actions


ROOT = Path(__file__).parent
DEFAULT_OUT = ROOT / "outputs/fixed-contract-residuals"
SESSIONS = 100_000
TARGETS = tuple(i / 40 for i in range(1, 41))
PRESSURE = Pressure(service=.8, bandwidth_gbps=5, migration_s=115)
FLEX = .10
DEBT = .10
SEED = 0
DATA_VERSION = 4


def result_row(requested, achieved, resource, used, capacity, **fields):
    if capacity <= 0:
        raise ValueError("resource capacity must be positive")
    slack = (capacity - used) / capacity
    return {
        "requested_shed_w": requested,
        "achieved_shed_w": achieved,
        "unmet_shed_w": max(0, requested - achieved),
        "resource": resource,
        "used": used,
        "capacity": capacity,
        "normalized_slack": slack,
        "planned_slack": slack,
        **fields,
    }


def validate_slacks(target_met, slacks):
    violated = {resource: slack for resource, slack in slacks.items()
                if slack < -1e-7}
    if target_met and violated:
        raise AssertionError(
            f"Target marked met despite resource violations: {violated}"
        )


def minimum_slack(rows, resources):
    values = {
        (float(row["target_fraction"]), row["resource"]):
            float(row["normalized_slack"])
        for row in rows if row["resource"] in resources
    }
    fractions = sorted({fraction for fraction, _ in values})
    return tuple(
        min(resources, key=lambda resource: values[fraction, resource])
        for fraction in fractions
    )


def _mixed_sessions(manifest, profile):
    counts = [SESSIONS // len(CLASSES)] * len(CLASSES)
    counts[0] += SESSIONS - sum(counts)
    sessions = []
    for job_class, count in zip(CLASSES, counts):
        sampled = sample_sessions(
            trace_shapes(manifest, job_class), count, SEED,
            log_bytes_per_token(WORKLOADS[job_class]),
        )
        sessions.extend(
            replace(session, session_id=f"{job_class}:{session.session_id}")
            for session in sampled
        )
    return pack_source(tuple(sessions), profile)


def _candidates(actions, capacities, normal, horizon, profile):
    rate = profile.case().kv_transfer.destination_bytes_per_s
    candidates = []
    for action in actions:
        ongoing = float(normal @ action.service_work)
        transition = float(normal @ action.transition_work)
        use = {
            "WAN transfer bytes": action.route_bytes,
            "Replay reconstruction GPU time": transition,
            "KV-ingest GPU time":
                action.route_bytes / rate if action.method == "kv_transfer" else 0,
            "Ongoing integrated serving load": ongoing,
            "Queued serving work": horizon * ongoing + transition,
            "KV-cache blocks": action.kv_blocks,
        }
        cost = action.duration_s / horizon + sum(
            use[resource] / capacity for resource, capacity in capacities.items()
        )
        candidates.append((action, use, action.source_power_gain_w / max(cost, 1e-12)))
    return sorted(candidates, key=lambda item: (-item[2], item[0].session_id, item[0].method))


def _select(candidates, target, capacities, horizon, power):
    used = dict.fromkeys(capacities, 0.0)
    source_used, sessions, selected = {}, set(), []
    initial = power.power(True)
    for action, use, _ in candidates:
        if initial - power.power(True) >= target - 1e-8:
            break
        source = source_used.get(action.source_instance, 0)
        if action.session_id in sessions or source + action.duration_s > horizon + 1e-8 \
                or any(used[name] + value > capacities[name] + 1e-8
                       for name, value in use.items()):
            continue
        selected.append(action)
        sessions.add(action.session_id)
        source_used[action.source_instance] = source + action.duration_s
        for name, value in use.items():
            used[name] += value
        power.remove(action.session_id)
    return tuple(selected), used, source_used, initial - power.power(True)


def generate(model_path, manifest_path):
    profile = ModelProfile.load(model_path)
    manifest = json.loads(manifest_path.read_text())
    packed, replicas = _mixed_sessions(manifest, profile)
    scenario_ = scenario(profile, packed, replicas, PRESSURE)
    architecture_ = architecture(profile, scenario_.sessions, replicas, PRESSURE)
    pool = replace(
        architecture_.pools[0],
        event_flex_fraction=FLEX,
        service_debt_fraction=DEBT,
    )
    architecture_ = replace(architecture_, pools=(pool,))
    q = architecture_.types[0]
    horizon = scenario_.deadline_s - profile.power_window_s
    bandwidth = PRESSURE.bandwidth_gbps * 125_000_000
    normal = np.asarray(q.normals[0])
    baseline = sum(
        (np.asarray(replica.baseline_work) for replica in pool.replicas),
        start=np.zeros(2),
    )
    baseline_service = float(normal @ baseline)
    stable = replicas * q.bounds["stable"][0]
    event = min(
        replicas * (
            q.bounds["normal"][0] + FLEX * q.bounds["stable"][0]
        ),
        stable,
    )
    baseline_blocks = sum(
        replica.baseline_kv_tokens // KV_BLOCK_TOKENS for replica in pool.replicas
    )
    capacities = {
        "WAN transfer bytes": bandwidth * horizon,
        "Replay reconstruction GPU time":
            replicas * horizon * profile.max_destination_replays,
        "KV-ingest GPU time": replicas * horizon * profile.max_destination_kv_streams,
        "Ongoing integrated serving load": event - baseline_service,
        "Queued serving work": horizon * (
            stable - baseline_service + DEBT * stable
        ),
        "KV-cache blocks": replicas * (q.kv_capacity_tokens // q.kv_block_tokens)
        - baseline_blocks,
    }
    candidates = _candidates(
        _actions(scenario_, profile, q, bandwidth, 0, horizon, "central"),
        capacities, normal, horizon, profile,
    )
    _, _, _, maximum = _select(
        candidates, float("inf"), capacities, horizon,
        ExpectedPower(scenario_, profile),
    )
    rows = []
    for fraction in TARGETS:
        requested = fraction * maximum
        selected, used, source_used, achieved = _select(
            candidates, requested, capacities, horizon,
            ExpectedPower(scenario_, profile),
        )
        makespan = max(
            [0, used["WAN transfer bytes"] / bandwidth]
            + [action.duration_s for action in selected]
            + list(source_used.values())
        )
        uses = {
            "Source migration-stream time": max(source_used.values(), default=0),
            **used,
            "Planned makespan lower bound": makespan,
        }
        display_capacities = {
            "Source migration-stream time": horizon,
            **capacities,
            "Planned makespan lower bound": horizon,
        }
        slacks = {
            resource: (display_capacities[resource] - use)
            / display_capacities[resource]
            for resource, use in uses.items()
        }
        target_met = achieved >= requested - 1e-7
        validate_slacks(target_met, slacks)
        binding = "|".join(
            resource for resource, slack in slacks.items() if slack <= 1e-6
        )
        common = {
            "target_fraction": fraction,
            "maximum_modeled_shed_w": maximum,
            "sessions": SESSIONS,
            "source_replicas": replicas,
            "deadline_s": scenario_.deadline_s,
            "replay_moves": sum(action.method == "replay" for action in selected),
            "kv_moves": sum(action.method == "kv_transfer" for action in selected),
            "target_met": target_met,
            "solver_status":
                "greedy_target_met" if target_met else "greedy_best_effort",
            "binding_resources": binding,
            "slack_kind": "planned",
            "execution_validated": False,
            "evidence_status": "sensitivity",
        }
        rows.extend(
            result_row(
                requested, achieved, resource, uses[resource], capacity, **common,
            )
            for resource, capacity in display_capacities.items()
        )
    return rows


def _fingerprint(model_path, manifest_path):
    paths = (
        model_path, manifest_path, *WORKLOADS.values(), ROOT / "destination.py",
        ROOT / "destination_bench.py", ROOT / "planner.py",
        ROOT / "pool_planner.py", ROOT / "requirement_frontier.py",
    )
    digest = hashlib.sha256()
    for path in paths:
        digest.update(path.read_bytes())
    return {
        "data_version": DATA_VERSION,
        "input_sha256": digest.hexdigest(),
        "sessions": SESSIONS,
        "workloads": list(CLASSES),
        "seed": SEED,
        "deadline_s": 120,
        "bandwidth_gbps": PRESSURE.bandwidth_gbps,
        "baseline_service_fraction": PRESSURE.service,
        "event_flex_fraction": FLEX,
        "service_debt_fraction": DEBT,
        "target_fractions": list(TARGETS),
        "maximum_shed_definition": "contract-constrained greedy planned shed",
    }


def load_or_generate(out, model_path, manifest_path, refresh=False):
    data, metadata = out / "fixed_contract_residuals.csv", out / "metadata.json"
    expected = _fingerprint(model_path, manifest_path)
    if data.exists() and metadata.exists() and not refresh:
        if json.loads(metadata.read_text()) != expected:
            raise ValueError("cached inputs changed; rerun with --refresh")
        with data.open(newline="") as stream:
            return list(csv.DictReader(stream))
    rows = generate(model_path, manifest_path)
    out.mkdir(parents=True, exist_ok=True)
    with data.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, tuple(rows[0]), lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)
    metadata.write_text(json.dumps(expected, indent=2) + "\n")
    return rows


def plot(rows, out):
    for row in rows:
        for field in (
            "requested_shed_w", "normalized_slack", "target_fraction",
            "unmet_shed_w",
        ):
            row[field] = float(row[field])
    primary = (
        ("Source migration-stream time", "Source time", "#8C1515", "-"),
        ("Ongoing integrated serving load", "GPU load", "#175E54", ":"),
        ("Queued serving work", "GPU time", "#B83A4B", ":"),
    )
    loose = (
        "WAN transfer bytes", "Replay reconstruction GPU time",
        "KV-ingest GPU time", "KV-cache blocks",
    )
    labels = {resource: label for resource, label, _, _ in primary}
    colors = {resource: color for resource, _, color, _ in primary}
    fractions = sorted({row["target_fraction"] for row in rows})
    x = np.asarray(fractions) * 100
    sns.set_theme(style="whitegrid")
    fig, (axis, strip) = plt.subplots(
        2, 1, figsize=(7, 4.5), sharex=True, layout="constrained",
        gridspec_kw={"height_ratios": (12, .7), "hspace": .08},
    )
    representatives = {
        row["target_fraction"]: row for row in rows
        if row["resource"] == primary[0][0]
    }
    failed = [fraction * 100 for fraction in fractions
              if representatives[fraction]["unmet_shed_w"] > 1e-7]
    all_resources = tuple(resource for resource, *_ in primary) + loose
    values = {}
    for resource in all_resources:
        selected = sorted(
            (row for row in rows if row["resource"] == resource),
            key=lambda row: row["target_fraction"],
        )
        values[resource] = np.asarray(
            [row["normalized_slack"] for row in selected]
        )
    loose_values = np.asarray([values[resource] for resource in loose])
    axis.fill_between(
        x, loose_values.min(0), loose_values.max(0),
        color="#D5D5D5", alpha=.8, linewidth=0,
        label="WAN / GPU / memory",
    )
    for resource, label, color, linestyle in primary:
        axis.plot(
            x, values[resource], color=color, linestyle=linestyle, linewidth=2.5,
            marker="o", markersize=3, markevery=4, zorder=3, label=label,
        )
    axis.axhline(0, color="black", linewidth=1.5)
    if failed:
        axis.scatter(
            failed, np.full(len(failed), -.035), marker="x", color="black",
            s=45, linewidth=1.5, label="Unmet", clip_on=False,
        )
    axis.legend(title="Resource", frameon=False, loc="center left",
                bbox_to_anchor=(1.01, .5))
    axis.set(
        ylabel="Normalized slack",
        ylim=(-.06, 1.05),
        xlim=(0, 102.5),
    )
    bindings = minimum_slack(rows, all_resources)
    bounds = np.r_[
        x[0] - (x[1] - x[0]) / 2,
        (x[:-1] + x[1:]) / 2,
        x[-1] + (x[-1] - x[-2]) / 2,
    ]
    for i, resource in enumerate(bindings):
        strip.axvspan(
            bounds[i], bounds[i + 1], color=colors.get(resource, "#777777"),
            linewidth=0,
        )
    start = 0
    for end in range(1, len(bindings) + 1):
        if end == len(bindings) or bindings[end] != bindings[start]:
            if bounds[end] - bounds[start] >= 12:
                strip.text(
                    (bounds[start] + bounds[end]) / 2, .5,
                    labels.get(bindings[start], "Other"), color="white",
                    ha="center", va="center", fontsize=8,
                )
            start = end
    strip.set(
        xlabel="Requested shed (%)", yticks=[], ylim=(0, 1),
        xticks=(0, 25, 50, 75, 100),
    )
    strip.set_ylabel("Closest\nto binding", rotation=0, ha="right", va="center")
    strip.grid(False)
    sns.despine(ax=axis)
    sns.despine(ax=strip, left=True, bottom=True)
    for extension in ("png", "pdf"):
        fig.savefig(
            out / f"fixed_contract_residuals.{extension}",
            dpi=180, facecolor="white", bbox_inches="tight",
        )
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--model", type=Path, default=DEFAULT_MODEL)
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument("--refresh", action="store_true")
    args = parser.parse_args()
    rows = load_or_generate(
        args.out, args.model, args.manifest, args.refresh,
    )
    plot(rows, args.out)
    print(f"rows={len(rows)} output={args.out}")


if __name__ == "__main__":
    main()
