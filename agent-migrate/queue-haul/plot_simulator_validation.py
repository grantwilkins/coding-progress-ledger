"""Plot a hand-checkable shared-link and source-power validation case."""

from __future__ import annotations

import argparse
import csv
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from profiles import (
    ACTION_POWER,
    SOURCE_SECTIONS,
    ActionPower,
    KVTransfer,
    ModelProfile,
    PowerCurve,
    ProfileCase,
    RateCurve,
    Source,
)
from simulate import (
    ExecutionScenario,
    NetworkLink,
    PlannedMove,
    PowerNode,
    ServingInstance,
    SimSession,
    execute,
)


def _profile() -> ModelProfile:
    rate = RateCurve.parse({"1": [[1, 100], [1000, 100]]})
    case = ProfileCase(
        "central",
        100,
        100,
        PowerCurve.parse([[0, 10], [0.5, 30], [1, 40]]),
        rate,
        rate,
        rate,
        0,
        KVTransfer(10, 100, 0, 100, 0, 0, 100),
        1,
        -8,
        1,
        2,
        {action: ActionPower.parse({"1": [0, 0], "2": [0, 0]})
         for action in ACTION_POWER},
    )
    source = Source("calculated", "hand-checkable validation", (0, 1000), 0)
    return ModelProfile(
        "simulator-validation",
        "validated",
        "synthetic",
        "synthetic",
        "bf16",
        1,
        1,
        "gpu",
        1,
        1,
        1000,
        2,
        2,
        {name: source for name in SOURCE_SECTIONS},
        {"central": case},
    )


def validation_result():
    sessions = tuple(SimSession(str(i), "source", 10, 25, 0, 1) for i in range(2))
    scenario = ExecutionScenario(
        5,
        5,
        10,
        "awake",
        0,
        (PowerNode("source-node", 1, True), PowerNode("destination-node", 1, False)),
        (
            ServingInstance("source", ("source-node",)),
            ServingInstance("destination", ("destination-node",)),
        ),
        sessions,
        (NetworkLink("link", 100),),
    )
    moves = tuple(
        PlannedMove(str(i), "destination", "kv_transfer", i, ("link",))
        for i in range(2)
    )
    return execute(scenario, _profile(), moves)


def _rows(result) -> list[dict]:
    if len(result.network) != 2 or len(result.sessions) != 2:
        raise AssertionError("simulator validation requires exactly two transfers")
    expected = {
        "session_0_transfer_start_s": 0,
        "session_0_transfer_complete_s": 2,
        "session_0_transfer_bytes": 100,
        "session_0_commit_s": 3,
        "session_1_transfer_start_s": 0,
        "session_1_transfer_complete_s": 2,
        "session_1_transfer_bytes": 100,
        "session_1_commit_s": 3,
        "initial_source_power_w": 30,
        "source_power_drop_s": 3,
        "source_power_after_both_commits_w": 10,
    }
    actual = {}
    for row in result.network:
        prefix = f"session_{row.session_id}_transfer"
        actual[f"{prefix}_start_s"] = row.start_s
        actual[f"{prefix}_complete_s"] = row.end_s
        actual[f"{prefix}_bytes"] = row.transferred_bytes
    for row in result.sessions:
        actual[f"session_{row.session_id}_commit_s"] = row.committed_s
    initial_power = result.power[0][1]
    drop = next(point for point in result.power if point[1] != initial_power)
    actual.update(
        initial_source_power_w=initial_power,
        source_power_drop_s=drop[0],
        source_power_after_both_commits_w=drop[1],
    )
    if any(abs(actual[name] - value) > 1e-9 for name, value in expected.items()):
        raise AssertionError(
            f"simulator validation failed: expected={expected}, actual={actual}"
        )
    return [
        {"quantity": name, "expected": value, "simulated": actual[name]}
        for name, value in expected.items()
    ]


def write(out: Path) -> None:
    # TODO(validation-profile): add a measured A100 regression after clean profiling.
    result = validation_result()
    rows = _rows(result)
    out.mkdir(parents=True, exist_ok=True)
    with (out / "simulator_validation.csv").open("w", newline="") as file:
        writer = csv.DictWriter(file, tuple(rows[0]), lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)

    background = "#FAFAF7"
    copy_color, switch_color, power_color = "#2F80ED", "#F2C94C", "#176B52"
    fig = plt.figure(figsize=(11, 6), facecolor=background)
    grid = fig.add_gridspec(
        2,
        2,
        left=0.08,
        right=0.96,
        top=0.82,
        bottom=0.12,
        width_ratios=(2.3, 1),
        hspace=0.38,
        wspace=0.28,
    )
    axes = (fig.add_subplot(grid[0, 0]), fig.add_subplot(grid[1, 0]))
    sessions = {row.session_id: row for row in result.sessions}
    for y, row in enumerate(result.network):
        session = sessions[row.session_id]
        axes[0].barh(
            y,
            row.end_s - row.start_s,
            left=row.start_s,
            height=0.55,
            color=copy_color,
        )
        axes[0].barh(
            y,
            session.committed_s - session.switch_s,
            left=session.switch_s,
            height=0.55,
            color=switch_color,
        )
        axes[0].scatter(
            session.committed_s,
            y,
            marker=">",
            s=90,
            color=power_color,
            zorder=3,
        )
        if y == 1:
            axes[0].text(1, y, "Copy cache", ha="center", va="center", color="white")
            axes[0].text(2.5, y, "Switch route", ha="center", va="center")
            axes[0].text(
                3.15, y, "Ready at destination", va="center", color=power_color
            )
    axes[0].axvline(2, color=copy_color, linestyle=":", linewidth=1.5)
    axes[0].set(
        yticks=range(2),
        yticklabels=["Session A", "Session B"],
        ylabel="",
        title="1. Copy together, then switch",
        xlim=(0, 5),
    )

    time = [point[0] for point in result.power]
    power = [point[1] for point in result.power]
    axes[1].step(
        [0, 3, 5],
        [30, 10, 10],
        where="post",
        color="#CBD2D9",
        linewidth=7,
        label="Calculated",
    )
    axes[1].step(
        time,
        power,
        where="post",
        color=power_color,
        linewidth=3,
        label="Simulator",
    )
    axes[1].fill_between([3, 5], 10, 30, color="#D8F3DC", alpha=0.7)
    axes[1].axvline(3, color=power_color, linestyle=":", linewidth=1.5)
    axes[1].annotate(
        "Both routes switched\n20 W lower",
        xy=(3, 10),
        xytext=(3.35, 19),
        arrowprops={"arrowstyle": "->", "color": power_color},
        color=power_color,
        fontweight="bold",
    )
    axes[1].set(
        xlabel="Time (seconds)",
        ylabel="Source power in simulation (W)",
        title="2. Source power falls only after both moves finish",
        xlim=(0, 5),
        ylim=(8, 32),
    )
    axes[1].legend(frameon=False, ncol=2, loc="upper right")
    for ax in axes:
        ax.set_facecolor(background)
        ax.grid(axis="x", alpha=0.2)
        ax.spines[["top", "right"]].set_visible(False)

    checks = fig.add_subplot(grid[:, 1])
    checks.set_facecolor("#F1F5F3")
    checks.set_xticks([])
    checks.set_yticks([])
    for side in checks.spines.values():
        side.set_visible(False)
    checks.text(
        0.08, 0.92, "EXACT CHECK", fontsize=11, color="#667085", fontweight="bold"
    )
    checks.text(
        0.08,
        0.82,
        "Copy both sessions",
        fontsize=13,
        fontweight="bold",
        color="#1F2937",
    )
    checks.text(0.08, 0.72, "2 × 100 B ÷ 100 B/s = 2 s", fontsize=11, color="#475467")
    checks.text(
        0.08,
        0.60,
        "Switch both routes",
        fontsize=13,
        fontweight="bold",
        color="#1F2937",
    )
    checks.text(0.08, 0.50, "2 s + 1 s = 3 s", fontsize=11, color="#475467")
    checks.text(
        0.08,
        0.38,
        "Lower source power",
        fontsize=13,
        fontweight="bold",
        color="#1F2937",
    )
    checks.text(0.08, 0.28, "30 W → 10 W at 3 s", fontsize=11, color="#475467")
    checks.text(
        0.08,
        0.13,
        "✓  Simulator matches",
        fontsize=12,
        color=power_color,
        fontweight="bold",
    )
    checks.text(
        0.16,
        0.08,
        "every checked value",
        fontsize=12,
        color=power_color,
        fontweight="bold",
    )

    fig.suptitle(
        "Two sessions move together, then source power falls",
        x=0.08,
        y=0.96,
        ha="left",
        fontsize=20,
        fontweight="bold",
        color="#172B4D",
    )
    fig.text(
        0.08,
        0.89,
        "Exact simulator check using calculated values — not hardware measurements",
        fontsize=11,
        color="#667085",
    )
    fig.savefig(
        out / "simulator_validation.png",
        dpi=180,
        facecolor=background,
        transparent=False,
    )
    fig.savefig(
        out / "simulator_validation.pdf", facecolor=background, transparent=False
    )
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", type=Path, default=Path(__file__).parent / "outputs")
    write(parser.parse_args().out)


if __name__ == "__main__":
    main()
