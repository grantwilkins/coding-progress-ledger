"""Plot pooled time to power-target attainment across bound states."""

from __future__ import annotations

import argparse
import csv
import json
from dataclasses import replace
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

import plot_style
import workload_adaptation_campaign as campaign
from planner import _expected_scenario, _resident_tokens, plan
from simulate import predict


POLICIES = {
    "queue_haul": "lp_highs", "greedy": "greedy",
    "isolated_fastest": "isolated_fastest", "kv_only": "kv_only",
    "replay_only": "replay_only",
}
FIGSIZE = (3.85, 2.5)
LABEL_SIZE = 10
LEGEND_SIZE = 7.5
plot_style.apply()


def attainment_time(commits, shed, target, power_window_s):
    moved = []
    for time_s, session_id in sorted(commits):
        moved.append(session_id)
        if shed(moved) >= target - 1e-8:
            return time_s + power_window_s
    return None


def execution_commits(primary, tail=(), deadline_s=30):
    if any(row.committed_s is not None and row.committed_s <= deadline_s
           for row in tail):
        raise RuntimeError("post-deadline tail committed before the deadline")
    return [(row.committed_s, row.session_id) for row in (*primary, *tail)
            if row.committed_s is not None]


def landed_architecture(architecture, problem, moves, horizon_s):
    sessions = {session.session_id: session for session in problem.sessions}
    pools = []
    for pool in architecture.pools:
        q = architecture.type_by_id[pool.type_id]
        replicas = []
        for replica in pool.replicas:
            work, tokens = np.asarray(replica.baseline_work), replica.baseline_kv_tokens
            for move in moves:
                if move.destination_instance != replica.replica_id:
                    continue
                session = sessions[move.session_id]
                resident = _resident_tokens(session, horizon_s)
                work = work + q.work(
                    session.expected_f, session.expected_g, resident,
                    q.migration is not None,
                )
                tokens += -(-resident // q.kv_block_tokens) * q.kv_block_tokens
            replicas.append(replace(
                replica, baseline_work=tuple(work), baseline_kv_tokens=tokens,
            ))
        pools.append(replace(pool, replicas=tuple(replicas)))
    return replace(architecture, pools=tuple(pools),
                   residency_horizon_s=horizon_s)


def policy_moves(problem, profile, routes, architecture, solver, seed, horizon_s):
    result = plan(problem, profile, routes, solver, seed=seed,
                  destination=architecture, admission_mode="normal")
    admitted = {move.session_id for move in result.moves}
    remaining = tuple(session for session in problem.sessions
                      if session.session_id not in admitted)
    if not remaining:
        return result.moves, (), None, None
    tail_problem = replace(
        problem, sessions=remaining, controller_delay_s=30,
        deadline_s=horizon_s, end_s=horizon_s,
    )
    tail_problem = replace(tail_problem, power_limit_w=campaign.source_power(
        tail_problem, profile, (session.session_id for session in remaining),
    ))
    tail_architecture = landed_architecture(
        architecture, problem, result.moves, horizon_s,
    )
    tail = plan(
        tail_problem, profile, routes, "isolated_fastest", seed=seed,
        destination=tail_architecture, admission_mode="normal",
    ).moves
    return result.moves, tail, tail_problem, tail_architecture


def attainment_rows(samples=1000, seed=campaign.DEFAULT_SEED, sessions=28,
                    target_fraction=2 / 3, horizon_s=90):
    if samples < 1 or sessions < 1 or not 0 < target_fraction <= 1 \
            or horizon_s < 30:
        raise ValueError("invalid policy-attainment controls")
    profile = campaign.ModelProfile.load(campaign.PROFILE)
    templates, _ = campaign.load_templates(campaign.MANIFEST, profile)
    timing_rows = campaign.read_csv(campaign.TIMING)
    parent = json.loads(campaign.TIMING_PARENT.read_text())
    campaign.central_timing_fits()
    rng, rows = np.random.default_rng(seed), []
    for replicate in range(samples):
        draw = campaign.sample_draw(
            profile, templates, timing_rows, parent, rng, replicate, seed,
            sessions,
        )
        sampled_profile, pack, fits, power_index, timing_hash, _ = draw
        for case_id, label, constraints in campaign.factorial_cases():
            problem, architecture, routes, target = campaign.build_problem(
                sampled_profile, pack, constraints, target_fraction, fits,
            )
            initial = campaign.source_power(problem, sampled_profile)
            for policy, solver in POLICIES.items():
                moves, tail, tail_problem, tail_architecture = policy_moves(
                    problem, sampled_profile, routes, architecture, solver,
                    replicate, horizon_s,
                )
                execution = predict(
                    _expected_scenario(
                        replace(problem, end_s=horizon_s), moves,
                    ), sampled_profile, moves, destination=architecture,
                )
                tail_sessions = ()
                if tail:
                    tail_sessions = predict(
                        _expected_scenario(tail_problem, tail),
                        sampled_profile, tail, destination=tail_architecture,
                    ).sessions
                commits = execution_commits(
                    execution.sessions, tail_sessions, problem.deadline_s,
                )
                time_s = attainment_time(
                    commits,
                    lambda moved: initial - campaign.source_power(
                        problem, sampled_profile, moved),
                    target, sampled_profile.power_window_s,
                )
                rows.append({
                    "replicate": replicate, "case_id": case_id,
                    "bound_constraint": label, "policy": policy,
                    "power_bootstrap_index": power_index,
                    "timing_fit_sha256": timing_hash, "target_w": target,
                    "requested_fraction": target_fraction, "deadline_s": 30,
                    "horizon_s": horizon_s,
                    "attainment_time_s": "" if time_s is None else time_s,
                    "target_met_by_30s": time_s is not None and time_s <= 30,
                })
    return rows


def attainment_curve(rows, policy):
    selected = [row for row in rows if row["policy"] == policy]
    cases = {(int(row["replicate"]), row["case_id"]) for row in selected}
    if not selected or len(selected) != len(cases):
        raise RuntimeError("attainment CDF requires one row per paired case")
    events = sorted(float(row["attainment_time_s"]) for row in selected
                    if row["attainment_time_s"] not in (None, ""))
    return np.r_[0, events], np.r_[0, np.arange(1, len(events) + 1) / len(cases)]


def write_plot(rows, path):
    deadline = 30
    horizon = max(float(row["horizon_s"]) for row in rows)
    fraction = float(rows[0]["requested_fraction"])
    fig, axis = plt.subplots(figsize=FIGSIZE)
    for policy in POLICIES:
        x, y = attainment_curve(rows, policy)
        axis.step(np.r_[x, horizon], np.r_[y, y[-1]], where="post",
                  **plot_style.policy_style(
                      policy, names=plot_style.COMPACT_POLICY_NAMES,
                  ))
    axis.axvline(deadline, color="black", linestyle="--", linewidth=1.5)
    axis.text(
        deadline, .4, "30 s deadline", transform=axis.get_xaxis_transform(),
        ha="center", va="center", rotation=90, fontstyle="italic",
        fontsize=LABEL_SIZE,
        bbox={"facecolor": "white", "edgecolor": "none", "pad": 1},
    )
    axis.set(xlim=(0, min(horizon, 60)), ylim=(0, 1.02),
             xlabel="Time to Power Target (s)",
             ylabel="Cumulative Distribution")
    axis.tick_params(labelsize=LABEL_SIZE)
    axis.xaxis.label.set_size(LABEL_SIZE)
    axis.yaxis.label.set_size(LABEL_SIZE)
    axis.grid(alpha=.25)
    handles, labels = axis.get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", bbox_to_anchor=(.5, .01),
        ncol=3, frameon=False, fontsize=LEGEND_SIZE, handlelength=1.8,
        handletextpad=.5, columnspacing=.5)
    fig.subplots_adjust(left=.19, right=.98, bottom=.36, top=.98)
    path.parent.mkdir(parents=True, exist_ok=True)
    for suffix in ("png", "pdf"):
        fig.savefig(path.with_suffix(f".{suffix}"), dpi=plot_style.SAVE_DPI)
    plt.close(fig)


def write_csv(rows, path):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as output:
        writer = csv.DictWriter(output, rows[0], lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--samples", type=int, default=1000)
    parser.add_argument("--seed", type=int, default=campaign.DEFAULT_SEED)
    parser.add_argument("--sessions", type=int, default=28)
    parser.add_argument("--target", type=float, default=2 / 3)
    parser.add_argument("--horizon-s", type=float, default=90)
    parser.add_argument("--out", type=Path, default=campaign.OUT / "policy_attainment")
    args = parser.parse_args()
    rows = attainment_rows(
        args.samples, args.seed, args.sessions, args.target, args.horizon_s,
    )
    write_csv(rows, args.out.with_suffix(".csv"))
    write_plot(rows, args.out)


if __name__ == "__main__":
    main()
