"""Six parameter sweeps over the staged evacuation pipeline.

Usage:
    cd evacuation && uv run python sweeps.py

Writes `outputs/sweeps.json` (per-run core output vectors + derived diagnostics).
Plotting is separate (`plot_sweeps.py`) and reads only the JSON.
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np

from instance import build_instance, MODELS
from pipeline import run_pipeline

BUCKET_LO, BUCKET_HI = 256_000.0, 1_000_000.0


@dataclass
class RunResult:
    sweep_param: str
    sweep_value: float
    seed: int
    Z_star: float
    evac_fraction: float
    phi_star: float
    H_star: float
    model_evac_frac: dict
    model_replay: dict
    model_state: dict
    model_supply_demand: dict
    total_replay: int
    total_state: int
    n_binding: int
    dest_net_pressure: list
    max_ing_pressure: float


def extract_run_result(inst, pipe, sweep_name, val, seed) -> RunResult:
    s1, s2, s3 = pipe.s1, pipe.s2, pipe.s3
    N = float(inst.n.sum())
    model_evac_frac, model_replay, model_state, model_supply_demand = {}, {}, {}, {}
    for m, name in enumerate(inst.M_names):
        mask = inst.model_idx == m
        n_m = float(inst.n[mask].sum())
        z_m = float(s3.z[mask].sum())
        model_evac_frac[name] = (n_m - z_m) / n_m
        model_replay[name] = round(float(s3.x_R[mask].sum()))
        model_state[name] = round(float(s3.x_S[mask].sum()))
        delta_m = float((inst.n[mask] * inst.T[mask] / inst.rho[mask]).sum())
        sigma_m = float(inst.D * inst.W[:, m].sum())
        model_supply_demand[name] = sigma_m / delta_m
    n_binding = sum(1 for p in s2.pressures.values() if abs(p - s2.phi_star) < 1e-6)
    dest_net = [s2.pressures[f"net|{l}"] for l in inst.L_names]
    ing = [p for k, p in s2.pressures.items() if k.startswith("ing|")]
    return RunResult(
        sweep_param=sweep_name, sweep_value=float(val), seed=int(seed),
        Z_star=float(s1.Z_star), evac_fraction=(N - s1.Z_star) / N,
        phi_star=float(s2.phi_star), H_star=float(s3.H_star),
        model_evac_frac=model_evac_frac, model_replay=model_replay,
        model_state=model_state, model_supply_demand=model_supply_demand,
        total_replay=round(float(s3.x_R.sum())), total_state=round(float(s3.x_S.sum())),
        n_binding=n_binding, dest_net_pressure=dest_net,
        max_ing_pressure=max(ing) if ing else 0.0,
    )


def run_sweep(sweep_name, param_values, base_kwargs, modifier_fn) -> list[RunResult]:
    results = []
    for val in param_values:
        kwargs = modifier_fn(dict(base_kwargs), val)
        inst = build_instance(**kwargs)
        pipe = run_pipeline(inst)
        results.append(extract_run_result(inst, pipe, sweep_name, val,
                                          kwargs.get("seed", base_kwargs["seed"])))
        print(f"  {sweep_name}={val}: Z*={results[-1].Z_star:.1f} "
              f"phi*={results[-1].phi_star:.4f} H*={results[-1].H_star:.2f}")
    return results


def _demand_proportional_W(base_kwargs):
    """uniform + proportional W matrices (constant total budget) on the base instance."""
    inst = build_instance(**base_kwargs)
    L, M = inst.W.shape
    total_W = float(inst.W.sum())
    delta = np.array([(inst.T[inst.model_idx == m] / inst.rho[inst.model_idx == m]).sum()
                      for m in range(M)])
    share = delta / delta.sum()
    uniform = np.full((L, M), total_W / (L * M))
    proportional = np.tile(total_W * share / L, (L, 1))
    return uniform, proportional


def run_all_sweeps() -> dict[str, list[RunResult]]:
    base = {"D": 300.0, "total_jobs": 10_000, "seed": 42}
    uniform_W, proportional_W = _demand_proportional_W(base)

    def w_at(alpha):
        return np.round(((1 - alpha) * uniform_W + alpha * proportional_W) * 2) / 2

    print("seed sweep")
    seed = run_sweep("seed", range(20), base, lambda k, v: {**k, "seed": v})
    print("rho_scale sweep")
    rho = run_sweep("rho_scale", [0.25, 0.5, 0.75, 1.0, 1.5, 2.0, 3.0], base,
                    lambda k, v: {**k, "rho_scale": v})
    print("W_rebalance sweep")
    wreb = run_sweep("W_rebalance", [0.0, 0.2, 0.4, 0.6, 0.8, 1.0], base,
                     lambda k, v: {**k, "W": w_at(v)})
    print("sigma_scale sweep")
    sigma = run_sweep("sigma_scale", [0.8, 1.0, 1.2, 1.4, 1.6, 1.8, 2.0], base,
                      lambda k, v: {**k, "sigma_scale": v})
    print("lambda_scale sweep")
    lam = run_sweep("lambda_scale", [0.25, 0.5, 1.0, 2.0, 4.0, 8.0], base,
                    lambda k, v: {**k, "lambda_scale": v})
    print("total_jobs sweep")
    job_vals = [10_000, *range(50_000, 500_001, 50_000)]
    jobs = run_sweep("total_jobs", job_vals, base,
                     lambda k, v: {**k, "total_jobs": int(v), "n_bins": 160})
    return {"seed": seed, "rho_scale": rho, "W_rebalance": wreb,
            "sigma_scale": sigma, "lambda_scale": lam, "total_jobs": jobs}


def _interp_crossing(xs, ys, target=0.0):
    """First x where ys crosses target (linear interp). None if no crossing."""
    xs, ys = np.asarray(xs, float), np.asarray(ys, float) - target
    for i in range(len(ys) - 1):
        if ys[i] == 0.0:
            return float(xs[i])
        if ys[i] * ys[i + 1] < 0:
            t = ys[i] / (ys[i] - ys[i + 1])
            return float(xs[i] + t * (xs[i + 1] - xs[i]))
    return float(xs[-1]) if ys[-1] == 0.0 else None


def _std_cv(vals):
    vals = np.asarray(vals, float)
    std = float(vals.std())
    mean = float(vals.mean())
    return std, (std / mean if mean else 0.0)


def compute_diagnostics(sweeps) -> dict:
    diag = {}

    s = sweeps["seed"]
    z_std, z_cv = _std_cv([r.Z_star for r in s])
    p_std, p_cv = _std_cv([r.phi_star for r in s])
    h_std, h_cv = _std_cv([r.H_star for r in s])
    names = list(s[0].model_evac_frac)
    model_evac_std = {n: float(np.std([r.model_evac_frac[n] for r in s])) for n in names}
    diag["seed"] = {"Z_star": {"std": z_std, "cv": z_cv},
                    "phi_star": {"std": p_std, "cv": p_cv},
                    "H_star": {"std": h_std, "cv": h_cv},
                    "model_evac_frac_std": model_evac_std}

    for key in ("rho_scale", "lambda_scale"):
        rs = sweeps[key]
        xs = [r.sweep_value for r in rs]
        diag[key] = {
            "action_flip_point": _interp_crossing(
                xs, [r.total_replay - r.total_state for r in rs]),
            "bottleneck_transition": _interp_crossing(
                xs, [max(r.dest_net_pressure) - r.phi_star for r in rs]),
            "feasibility_threshold": _interp_crossing(xs, [r.Z_star for r in rs]),
        }

    w = sweeps["W_rebalance"]
    diag["W_rebalance"] = {
        "gap": float(w[0].Z_star - w[-1].Z_star),
        "phi_star_curve": [r.phi_star for r in w],
    }

    sig = sweeps["sigma_scale"]
    bucket_frac = []
    for r in sig:
        inst = build_instance(D=300.0, total_jobs=10_000, seed=42, sigma_scale=r.sweep_value)
        bucket_frac.append(float(np.mean((inst.T >= BUCKET_LO) & (inst.T <= BUCKET_HI))))
    diag["sigma_scale"] = {
        "H_star_curve": [r.H_star for r in sig],
        "model_z_curve": {n: [(1 - r.model_evac_frac[n]) * (10_000 * MODELS[i].job_fraction)
                              for r in sig] for i, n in enumerate(list(sig[0].model_evac_frac))},
        "bucket_256k_1M_fraction": bucket_frac,
    }
    return diag


def main() -> None:
    out = Path(__file__).resolve().parent / "outputs"
    out.mkdir(exist_ok=True)
    sweeps = run_all_sweeps()
    payload = {"runs": {k: [asdict(r) for r in v] for k, v in sweeps.items()},
               "diagnostics": compute_diagnostics(sweeps)}
    with (out / "sweeps.json").open("w") as fh:
        json.dump(payload, fh, indent=2)
    print(f"wrote {out / 'sweeps.json'}")


if __name__ == "__main__":
    main()
