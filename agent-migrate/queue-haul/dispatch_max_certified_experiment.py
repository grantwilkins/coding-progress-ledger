"""Side experiment: maximize certified shed on the dispatch-validation fixtures."""

from dataclasses import replace

from dispatch import Event, bind_dp, greedy, solve
from impact import Movement, compute
from instance import _mean_T, class_workload, generate
from power import PoolPower

EVENT, MOVE, N_NODES = Event(dest_nodes=48), Movement(), 4
CASES = ("ordinary_chat", "long_chat_code", "reasoning_chat", "agentic_tool_loop")


def run(cls):
    wl = class_workload(cls, state_mix=(1.0, 0.0, 0.0), cache_hit=(1.0, 1.0, 1.0, 1.0))
    pool = replace(PoolPower(), mean_context_tokens=_mean_T(wl))
    pop = generate(pool, wl, n_nodes=N_NODES)
    imp = compute(pop, pool)
    # Deliberately exempt from the full_node_expected target basis: this probe exists to force
    # solve()'s max-shed re-solve branch, and 2x the certified sum is infeasible BY CONSTRUCTION
    # (shed <= sum(dp_certified) at y <= 1), which a node-expected basis cannot guarantee.
    target = 2 * bind_dp(imp).sum()
    return cls, pop, imp, solve(pop, pool, imp, target, EVENT, MOVE), greedy(pop, pool, imp, target, EVENT, MOVE)


if __name__ == "__main__":
    print("class               regime jobs lp_cert_kW greedy_cert_kW gap_kW lp_cost_s greedy_cost_s")
    for cls, pop, imp, lp, gr in map(run, CASES):
        print(
            f"{cls:19s} {imp.regime:6s} {len(pop):4d} "
            f"{lp.shed_guaranteed/1e3:10.1f} {gr.shed_guaranteed/1e3:14.1f} "
            f"{(lp.shed_guaranteed - gr.shed_guaranteed)/1e3:6.1f} {lp.cost:9.1f} {gr.cost:13.1f}"
        )
