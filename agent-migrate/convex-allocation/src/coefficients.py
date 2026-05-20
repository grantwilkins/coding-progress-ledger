from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from problem import ProblemData

REPLAY = 0
STATE = 1
ACTIONS = ("replay", "state")


@dataclass(frozen=True)
class Coefficients:
    b_net: np.ndarray
    b_prefill: np.ndarray
    R0: np.ndarray
    q: np.ndarray
    option_dest: np.ndarray
    option_action: np.ndarray

    @property
    def M(self) -> int:
        return int(self.option_dest.size)

    @property
    def b_net_flat(self) -> np.ndarray:
        return self.b_net.reshape(self.b_net.shape[0], self.M)

    @property
    def b_prefill_flat(self) -> np.ndarray:
        return self.b_prefill.reshape(self.b_prefill.shape[0], self.M)

    @property
    def q_flat(self) -> np.ndarray:
        return self.q.reshape(self.q.shape[0], self.M)


def compute_coefficients(problem: ProblemData) -> Coefficients:
    G, K = problem.G, problem.K
    beta = problem.model.beta_bytes_per_tok
    eta = problem.model.eta_bytes_per_tok
    T = problem.T[:, None]
    one_minus_ctx = 1.0 - problem.h_ctx
    one_minus_kv = 1.0 - problem.h_kv

    b_net = np.zeros((G, K, len(ACTIONS)))
    b_prefill = np.zeros_like(b_net)
    R0 = np.zeros_like(b_net)

    b_net[:, :, REPLAY] = beta * T * one_minus_ctx
    b_prefill[:, :, REPLAY] = T * one_minus_ctx
    b_net[:, :, STATE] = eta * T * one_minus_kv

    R0[:, :, REPLAY] = b_net[:, :, REPLAY] / problem.lambda_Bps + b_prefill[:, :, REPLAY] / problem.rho_prefill
    R0[:, :, STATE] = b_net[:, :, STATE] / problem.lambda_Bps
    q = R0

    option_dest = np.repeat(np.arange(K), len(ACTIONS))
    option_action = np.tile(np.arange(len(ACTIONS)), K)
    return Coefficients(b_net, b_prefill, R0, q, option_dest, option_action)


def move_view(y: np.ndarray, problem: ProblemData) -> np.ndarray:
    return np.asarray(y)[:, : problem.K * len(ACTIONS)].reshape(problem.G, problem.K, len(ACTIONS))
