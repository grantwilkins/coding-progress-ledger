"""Plot genuine A100 timing and power parity from reduced measurements."""

import argparse
import csv
from collections import Counter
from itertools import product
from pathlib import Path

import plot_style
from plot_h100_power_parity import write as write_power
from plot_hardware_power_parity import METHODS
from plot_live_timing_parity import write_queue


ROOT = Path(__file__).parent
TIMING = ROOT / "outputs/timing-power-validation-20260814/timing-predictions.csv"
POWER = ROOT / "outputs/power-parity-phase-aware-20260813/power_parity.csv"
TIMING_COUNTS = {
    ("training", "replay"): 36, ("training", "kv_transfer"): 36,
    ("holdout_context", "replay"): 18,
    ("holdout_context", "kv_transfer"): 18,
}
plot_style.apply()


def _read(path: Path) -> list[dict]:
    with path.open(newline="") as stream:
        return list(csv.DictReader(stream))


def load_timing(path: Path = TIMING) -> list[dict]:
    rows = _read(path)
    if Counter((row["split"], row["method"]) for row in rows) != TIMING_COUNTS \
            or len({row["scenario_id"] for row in rows}) != len(rows):
        raise ValueError("A100 timing parity requires the complete 108-path cohort")
    return [{"hardware": "a100", "scenario_id": row["scenario_id"],
             "action": row["method"], "cohort": row["split"],
             "predicted_s": float(row["predicted_s"]),
             "measured_s": float(row["observed_s"])} for row in rows]


def load_power(path: Path = POWER) -> list[dict]:
    rows = _read(path)
    keys = {(row["method"], int(row["repeat"])) for row in rows}
    if len(rows) != 350 or keys != set(product(METHODS, range(50))) \
            or {row["model"] for row in rows} != {
                "phase_aware_saturating_scratch_fit_v1"}:
        raise ValueError("A100 power parity requires the complete phase-aware cohort")
    return [{"hardware": "a100", "scenario_id": row["scenario_id"],
             "method": row["method"], "repeat": int(row["repeat"]),
             "family": "idle" if float(row["post_load"]) == 0 else "sessions",
             "cohort": "descriptive_in_sample",
             "predicted_power_w": float(row["predicted_post_power_w"]),
             "measured_power_w": float(row["post_window_power_w"]),
             "residual_w": float(row["post_window_power_w"])
             - float(row["predicted_post_power_w"])} for row in rows]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--timing", type=Path, default=TIMING)
    parser.add_argument("--power", type=Path, default=POWER)
    parser.add_argument("--timing-out", type=Path,
                        default=ROOT / "outputs/a100_live_queue_makespan_parity")
    parser.add_argument("--power-out", type=Path,
                        default=ROOT / "outputs/a100_power_model_parity")
    args = parser.parse_args()
    write_queue(load_timing(args.timing), args.timing_out, (.7, 25))
    write_power(load_power(args.power), args.power_out)


if __name__ == "__main__":
    main()
