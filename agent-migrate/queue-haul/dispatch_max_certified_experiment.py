"""Side experiment: maximize certified shed on the dispatch-validation fixtures."""

from dataclasses import replace

from dispatch import Event, bind_dp, solve
from impact import Movement, compute
from instance import _mean_T, class_workload, generate
from power import PoolPower

EVENT, MOVE, N_NODES = Event(dest_nodes=48, W=16), Movement(), 4
CASES = ("ordinary_chat", "long_chat_code", "reasoning_chat", "agentic_tool_loop")


def run(cls):
    wl = class_workload(cls, state_mix=(1.0, 0.0, 0.0), cache_hit=(1.0, 1.0, 1.0, 1.0))
    pool = replace(PoolPower(), mean_context_tokens=_mean_T(wl))
    pop = generate(pool, wl, n_nodes=N_NODES)
    imp = compute(pop, pool)
    return cls, pop, imp, solve(pop, pool, imp, 2 * bind_dp(imp).sum(), EVENT, MOVE)


if __name__ == "__main__":
    print("class               regime jobs max_cert_kW expected_kW moved cost_s")
    for cls, pop, imp, plan in map(run, CASES):
        print(
            f"{cls:19s} {imp.regime:6s} {len(pop):4d} "
            f"{plan.shed_guaranteed/1e3:11.1f} {plan.shed_expected/1e3:11.1f} "
            f"{plan.y.sum():5.1f} {plan.cost:6.1f}"
        )
