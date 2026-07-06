from __future__ import annotations

import argparse
import csv
import json
import re
from datetime import datetime, timezone
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

OUT_STEM = Path(__file__).resolve().parent / "outputs" / "stage1_gpt_oss_20b_a100_tp1"


def load_json(path: Path) -> dict:
    with path.open() as fh:
        data = json.load(fh)
    if not isinstance(data, dict):
        raise ValueError(f"expected JSON object in {path}")
    return data


def parse_ts(text: str) -> float | None:
    try:
        return float(text)
    except ValueError:
        pass
    for fmt in ("%Y/%m/%d %H:%M:%S.%f", "%Y/%m/%d %H:%M:%S", "%Y-%m-%d %H:%M:%S.%f", "%Y-%m-%d %H:%M:%S"):
        try:
            return datetime.strptime(text.strip(), fmt).replace(tzinfo=timezone.utc).timestamp()
        except ValueError:
            pass
    return None


def read_power(path: Path) -> tuple[np.ndarray, np.ndarray]:
    with path.open(newline="") as fh:
        reader = csv.reader(fh)
        header = [h.strip().lower() for h in next(reader, [])]
        ts_col = next((i for i, h in enumerate(header) if "time" in h), None)
        p_col = next((i for i, h in enumerate(header) if "power" in h and "draw" in h), None)
        if ts_col is None or p_col is None:
            raise ValueError(f"missing timestamp/power columns in {path}")
        grouped: dict[float, float] = {}
        for row in reader:
            if len(row) <= max(ts_col, p_col):
                continue
            ts = parse_ts(row[ts_col])
            if ts is None:
                continue
            power = float(re.sub(r"[^\d.+-]", "", row[p_col]))
            grouped[ts] = grouped.get(ts, 0.0) + power
    if not grouped:
        raise ValueError(f"no power samples in {path}")
    ts = np.array(sorted(grouped), float)
    return ts, np.array([grouped[t] for t in ts], float)


def request_arrays(requests: dict) -> dict[str, np.ndarray | list]:
    fields = ("input_lens", "output_lens", "ttfts", "itls", "request_timestamps")
    missing = [k for k in fields if k not in requests]
    if missing:
        raise ValueError(f"missing request fields: {', '.join(missing)}")
    return {
        "input_lens": np.asarray(requests["input_lens"], float),
        "output_lens": np.asarray(requests["output_lens"], float),
        "ttfts": np.asarray(requests["ttfts"], float),
        "request_timestamps": np.asarray(requests["request_timestamps"], float),
        "itls": requests["itls"],
    }


def finite_quantile(values, q: float) -> float:
    arr = np.asarray(values, float)
    arr = arr[np.isfinite(arr)]
    return float(np.quantile(arr, q)) if arr.size else float("nan")


def level_row(bundle: Path, manifest: dict, req: dict, power_t: np.ndarray, power_w: np.ndarray, level: dict) -> dict:
    start, end = float(level["t_start_epoch"]), float(level["t_end_epoch"])
    duration = end - start
    if duration <= 0:
        raise ValueError(f"nonpositive level duration in {bundle}")
    mask = (req["request_timestamps"] >= start) & (req["request_timestamps"] <= end)
    p_mask = (power_t >= start) & (power_t <= end)
    itls = [x for i, xs in enumerate(req["itls"]) if mask[i] for x in xs]
    params = level["params"]
    return {
        "bundle": bundle.name,
        "probe_type": manifest["probe"]["type"],
        "level": int(level["level"]),
        "label": level["label"],
        "concurrency": int(level["concurrency"]),
        "input_len": int(params["input_len"]),
        "output_len": int(params["output_len"]),
        "prefix_len": int(params.get("prefix_len", 0)),
        "duration_s": duration,
        "n_requests": int(mask.sum()),
        "input_tps": float(req["input_lens"][mask].sum() / duration),
        "output_tps": float(req["output_lens"][mask].sum() / duration),
        "ttft_p50_ms": 1000 * finite_quantile(req["ttfts"][mask], 0.50),
        "ttft_p95_ms": 1000 * finite_quantile(req["ttfts"][mask], 0.95),
        "tpot_p50_ms": 1000 * finite_quantile(itls, 0.50),
        "tpot_p95_ms": 1000 * finite_quantile(itls, 0.95),
        "power_mean_w": float(np.nanmean(power_w[p_mask])) if p_mask.any() else float("nan"),
    }


def discover_bundles(run_dir: Path) -> list[Path]:
    if (run_dir / "manifest.json").exists():
        return [run_dir]
    bundles = sorted(p.parent for p in run_dir.rglob("manifest.json"))
    if not bundles:
        raise ValueError(f"no probe bundles under {run_dir}")
    return bundles


def read_rows(run_dir: Path) -> list[dict]:
    rows = []
    for bundle in discover_bundles(run_dir):
        manifest = load_json(bundle / "manifest.json")
        req = request_arrays(load_json(bundle / "requests.json"))
        power_t, power_w = read_power(bundle / "power.csv")
        for level in manifest["probe"].get("levels", []):
            rows.append(level_row(bundle, manifest, req, power_t, power_w, level))
    if not rows:
        raise ValueError(f"no levels under {run_dir}")
    return rows


def write_csv(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)


def _scale_rows(data: list[dict], metric: str, throughput_key: str) -> list[dict]:
    data = sorted(data, key=lambda r: (r["input_len"], r["concurrency"]))
    if not data:
        raise ValueError(f"missing {metric} rows")
    base = float(data[0][throughput_key])
    if base <= 0:
        raise ValueError(f"{metric} baseline throughput must be positive")
    return [
        {
            "metric": metric,
            "input_len": r["input_len"],
            "concurrency": r["concurrency"],
            "throughput_tps": r[throughput_key],
            "scale_vs_short": float(r[throughput_key]) / base,
            "power_mean_w": r["power_mean_w"],
        }
        for r in data
    ]


def service_scale_rows(rows: list[dict]) -> list[dict]:
    prefill = [r for r in rows if r["probe_type"] == "prefill_staircase"]
    decode_best = []
    for T in sorted({r["input_len"] for r in rows if r["probe_type"] == "decode_staircase"}):
        levels = [r for r in rows if r["probe_type"] == "decode_staircase" and r["input_len"] == T]
        decode_best.append(max(levels, key=lambda r: r["output_tps"]))
    return _scale_rows(prefill, "rho", "input_tps") + _scale_rows(decode_best, "G", "output_tps")


def write_service_scale(rows: list[dict], out_stem: Path) -> None:
    data = service_scale_rows(rows)
    write_csv(out_stem.with_name(out_stem.name + "_service_scale.csv"), data)

def write_prefill(rows: list[dict], out_stem: Path) -> None:
    data = sorted([r for r in rows if r["probe_type"] == "prefill_staircase"], key=lambda r: r["input_len"])
    if not data:
        raise ValueError("missing prefill_staircase rows")
    write_csv(out_stem.with_name(out_stem.name + "_prefill_rho.csv"), data)
    fig, ax = plt.subplots(figsize=(6.2, 4.0))
    ax.plot([r["input_len"] for r in data], [r["input_tps"] for r in data], marker="o")
    ax.set_xscale("log")
    ax.set_xlabel("context tokens T")
    ax.set_ylabel("measured prefill throughput rho(T) [tok/s]")
    ax.set_title("Prefill rho(T)")
    fig.tight_layout()
    fig.savefig(f"{out_stem}_prefill_rho.png", dpi=160)
    fig.savefig(f"{out_stem}_prefill_rho.pdf")
    plt.close(fig)


def write_decode(rows: list[dict], out_stem: Path) -> None:
    data = sorted([r for r in rows if r["probe_type"] == "decode_staircase"], key=lambda r: (r["input_len"], r["concurrency"]))
    if not data:
        raise ValueError("missing decode_staircase rows")
    write_csv(out_stem.with_name(out_stem.name + "_decode_context.csv"), data)
    fig, ax = plt.subplots(figsize=(6.6, 4.2))
    for T in sorted({r["input_len"] for r in data}):
        sel = [r for r in data if r["input_len"] == T]
        ax.plot([r["concurrency"] for r in sel], [r["output_tps"] for r in sel], marker="o", label=f"T={T}")
    ax.set_xscale("log", base=2)
    ax.set_xlabel("decode concurrency")
    ax.set_ylabel("decode throughput G(T) [tok/s]")
    ax.set_title("Decode stability by context")
    ax.legend(fontsize=7)
    fig.tight_layout()
    fig.savefig(f"{out_stem}_decode_context.png", dpi=160)
    fig.savefig(f"{out_stem}_decode_context.pdf")
    plt.close(fig)


def write_mixed(rows: list[dict], out_stem: Path) -> None:
    data = [r for r in rows if r["probe_type"] == "mixed_grid"]
    if not data:
        raise ValueError("missing mixed_grid rows")
    write_csv(out_stem.with_name(out_stem.name + "_mixed_surface.csv"), data)
    fig, ax = plt.subplots(figsize=(6.0, 4.6))
    sc = ax.scatter([r["input_tps"] for r in data], [r["output_tps"] for r in data], c=[r["power_mean_w"] for r in data], s=45)
    ax.set_xlabel("prefill throughput [tok/s]")
    ax.set_ylabel("decode throughput [tok/s]")
    ax.set_title("Mixed prefill/decode surface")
    fig.colorbar(sc, ax=ax, label="mean node power [W]")
    fig.tight_layout()
    fig.savefig(f"{out_stem}_mixed_surface.png", dpi=160)
    fig.savefig(f"{out_stem}_mixed_surface.pdf")
    plt.close(fig)


def parse_args(argv: list[str] | None = None):
    p = argparse.ArgumentParser(description="Reduce Queue-Haul Stage 1 service-surface bundles")
    p.add_argument("--run-dir", type=Path, required=True)
    p.add_argument("--out-stem", type=Path, default=OUT_STEM)
    return p.parse_args(argv)


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)
    rows = read_rows(args.run_dir)
    write_prefill(rows, args.out_stem)
    write_decode(rows, args.out_stem)
    write_mixed(rows, args.out_stem)
    write_service_scale(rows, args.out_stem)
    print(args.out_stem)


if __name__ == "__main__":
    main()
