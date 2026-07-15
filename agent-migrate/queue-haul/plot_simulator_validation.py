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
        KVTransfer(10, 100, 0, 0, 0),
        1,
        2,
        1,
        2,
        {action: (0, 0) for action in ACTION_POWER},
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
        2,
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

    fig, axes = plt.subplots(2, 1, figsize=(7, 6), sharex=True)
    for y, row in enumerate(result.network):
        axes[0].barh(y, row.end_s - row.start_s, left=row.start_s, height=0.5)
        axes[0].scatter(row.end_s, y, color="black", zorder=3)
    axes[0].axvline(2, color="black", linestyle="--", label="expected completion")
    axes[0].set(
        yticks=range(2),
        yticklabels=["session 0", "session 1"],
        ylabel="session",
        title="Two 100 B transfers share one 100 B/s link",
    )
    axes[0].legend()

    time = [point[0] for point in result.power]
    power = [point[1] for point in result.power]
    axes[1].step(time, power, where="post", linewidth=2, label="simulated")
    axes[1].step(
        [0, 3, 5], [30, 10, 10], where="post", linestyle="--", label="hand calculation"
    )
    axes[1].axvline(3, color="black", linestyle=":", label="commit")
    axes[1].set(xlabel="time (s)", ylabel="modeled source power (W)")
    axes[1].legend()
    for ax in axes:
        ax.grid(alpha=0.25)
    fig.suptitle("Simulator validation: shared link and commit-gated power")
    fig.tight_layout()
    fig.savefig(out / "simulator_validation.png", dpi=180)
    fig.savefig(out / "simulator_validation.pdf")
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", type=Path, default=Path(__file__).parent / "outputs")
    write(parser.parse_args().out)


if __name__ == "__main__":
    main()
