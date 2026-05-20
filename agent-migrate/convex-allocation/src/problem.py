from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from catalog import ModelParams
from workload import generate_workload


@dataclass(frozen=True, init=False)
class ProblemData:
    model: ModelParams
    regime: str
    T: np.ndarray
    d: np.ndarray
    deadline_s: np.ndarray
    lambda_Bps: np.ndarray
    rho_prefill: np.ndarray
    C_net: np.ndarray
    C_prefill: np.ndarray
    ell_net: np.ndarray
    ell_prefill: np.ndarray
    h_ctx: np.ndarray
    h_kv: np.ndarray
    retained_prefill_target_s: float
    w: float = 1.0

    def __init__(
        self,
        model: ModelParams,
        regime: str,
        T: np.ndarray,
        d: np.ndarray,
        deadline_s: np.ndarray | None = None,
        lambda_Bps: np.ndarray | None = None,
        rho_prefill: np.ndarray | None = None,
        C_net: np.ndarray | None = None,
        C_prefill: np.ndarray | None = None,
        ell_net: np.ndarray | None = None,
        ell_prefill: np.ndarray | None = None,
        h_ctx: np.ndarray | None = None,
        h_kv: np.ndarray | None = None,
        retained_prefill_target_s: float | None = None,
        w: float = 1.0,
    ) -> None:
        if deadline_s is None or retained_prefill_target_s is None:
            raise ValueError("deadline_s and retained_prefill_target_s are required")
        values = {
            "model": model,
            "regime": regime,
            "T": T,
            "d": d,
            "deadline_s": deadline_s,
            "lambda_Bps": lambda_Bps,
            "rho_prefill": rho_prefill,
            "C_net": C_net,
            "C_prefill": C_prefill,
            "ell_net": ell_net,
            "ell_prefill": ell_prefill,
            "h_ctx": h_ctx,
            "h_kv": h_kv,
            "retained_prefill_target_s": retained_prefill_target_s,
            "w": w,
        }
        missing = [key for key, value in values.items() if value is None]
        if missing:
            raise ValueError(f"missing ProblemData fields: {missing}")
        for key, value in values.items():
            object.__setattr__(self, key, value)

    @property
    def G(self) -> int:
        return int(self.T.size)

    @property
    def K(self) -> int:
        return int(self.lambda_Bps.size)

    @property
    def tau(self) -> np.ndarray:
        return self.T / self.model.prefill_tok_s


WORKLOAD_T = np.array([512, 2048, 8192, 32768, 100000, 200000], dtype=float)
WORKLOAD_D = np.array([200, 150, 100, 50, 20, 10], dtype=float)
WORKLOAD_DEADLINE_S = np.array([2, 10, 30, 60, 120, 300], dtype=float)


def make_problem(
    model: ModelParams,
    regime: str,
    retained_prefill_fraction: float = 0.4,
    deadline_scale: float = 1.0,
    w: float = 1.0,
    window_s: float = 1800.0,
    gpu_count: np.ndarray | None = None,
    workload_source: str = "generated",
    workload_seed: int | None = None,
    workload_jobs: int = 10_000,
    workload_classes: int = 48,
    workload_profile: str = "agentic_retained_sessions",
) -> ProblemData:
    if workload_source == "fixed":
        T = WORKLOAD_T.copy()
        d = WORKLOAD_D.copy()
        deadline_s = WORKLOAD_DEADLINE_S.copy() * deadline_scale
        h_ctx = np.zeros((T.size, 3))
        h_kv = np.zeros((T.size, 3))
    elif workload_source == "generated":
        workload = generate_workload(3, workload_seed, workload_jobs, workload_classes, workload_profile)
        T = workload.T
        d = workload.d
        deadline_s = workload.deadline_s * deadline_scale
        h_ctx = workload.h_ctx
        h_kv = workload.h_kv
    else:
        raise ValueError(f"unknown workload source: {workload_source}")
    default_gpu_count = gpu_count is None
    if default_gpu_count:
        gpu_count = np.full(3, 72.0 if workload_source == "generated" else 8.0)
    else:
        gpu_count = np.asarray(gpu_count, dtype=float)

    if regime == "bandwidth-spread":
        lambda_gbps = np.array([1.0, 10.0, 100.0])
        net_frac = np.array([0.5, 0.5, 0.5])
        prefill_frac = np.array([0.5, 0.5, 0.5])
    elif regime == "prefill-spread":
        lambda_gbps = np.array([25.0, 25.0, 25.0])
        net_frac = np.array([0.5, 0.5, 0.5])
        prefill_frac = np.array([0.2, 0.5, 0.8])
    elif regime == "background-load-spread":
        lambda_gbps = np.array([25.0, 25.0, 25.0])
        net_frac = np.array([0.2, 0.5, 0.8])
        prefill_frac = np.array([0.8, 0.5, 0.2])
        if workload_source == "fixed":
            h_ctx[T >= 8192, 2] = 0.5
            h_kv[T >= 8192, 2] = 0.5
    elif regime == "transition-coupled":
        if model.name != "GLM-5":
            raise ValueError("transition-coupled is calibrated for GLM-5")
        if default_gpu_count:
            gpu_count = np.array([1.0, 1.0, 1.0]) if workload_source == "generated" else np.array([2.0, 2.0, 2.0])
        lambda_gbps = np.array([4.0, 6.0, 9.0])
        net_frac = np.array([0.35, 0.35, 0.35])
        prefill_frac = np.array([0.45, 0.45, 0.45])
        if workload_source == "fixed":
            h_ctx[1] = np.array([0.75, 0.25, 0.0])
            h_ctx[4] = np.array([0.05, 0.80, 0.35])
            h_kv[2] = np.array([0.05, 0.25, 0.85])
            h_kv[5] = np.array([0.90, 0.55, 0.15])
    else:
        raise ValueError(f"unknown regime: {regime}")

    lambda_Bps = lambda_gbps * 1e9 / 8.0
    rho_prefill = model.prefill_tok_s * gpu_count
    C_net = lambda_Bps * window_s
    C_prefill = rho_prefill * window_s
    ell_net = net_frac * C_net
    ell_prefill = prefill_frac * C_prefill
    retained_prefill_target_s = retained_prefill_fraction * float(np.dot(T / model.prefill_tok_s, d))
    return ProblemData(
        model=model,
        regime=regime,
        T=T,
        d=d,
        deadline_s=deadline_s,
        lambda_Bps=lambda_Bps,
        rho_prefill=rho_prefill,
        C_net=C_net,
        C_prefill=C_prefill,
        ell_net=ell_net,
        ell_prefill=ell_prefill,
        h_ctx=h_ctx.copy(),
        h_kv=h_kv.copy(),
        retained_prefill_target_s=retained_prefill_target_s,
        w=w,
    )


def saturation_diagnostics(problem: ProblemData) -> tuple[float, float]:
    replay = float(np.dot(problem.d, problem.T))
    replay_capacity = float(np.sum(problem.C_prefill - problem.ell_prefill))
    state = float(np.dot(problem.d, problem.model.eta_bytes_per_tok * problem.T))
    state_capacity = float(np.sum(problem.C_net - problem.ell_net))
    return replay / replay_capacity, state / state_capacity
