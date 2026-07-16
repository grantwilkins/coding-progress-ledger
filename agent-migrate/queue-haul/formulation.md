# Queue-Haul formulation

Each session has a source instance, context, sampled requests, durable log, and
expected prefill and decode rates. A model profile supplies the measured power
curve, service rates, replay rates, KV block layout, action times, concurrency
limits, resident KV-token capacity, and uncertainty cases. Active sessions are
placed only when both compute load and resident KV fit; cold sessions consume
neither until reactivation.

The planner chooses whole sessions and one of three actions: replay, KV
transfer, or replay on request. Random, load-only, node-aware, and node-drain
selection are separate policies. Destination placement is a balanced pass.
Destination placement enforces the same compute and resident-KV limits as the
source placement.
Only local source power is constrained; destination power is reported.

The simulator follows background preparation, source quiescing, catch-up,
route switch, commit, and optional sleep or shutdown. Concurrent transfers
share every named link. Source power changes only at commit. A plan is feasible
only when central-case execution meets both the move deadline and the trailing
power window.

A run is accepted when:

1. the trailing-window modeled source power is at or below the limit;
2. every planned move commits by the deadline; and
3. every request observed by the deadline starts by the deadline.

The third condition checks routing readiness, not end-to-end request latency.
Request events affect context and timing but not dynamic power yet; output
columns therefore say `modeled_*_power`.

Unsupported context, load, concurrency, topology, or profile cases hard-fail.
Open measurement work is listed in `assumptions.md`.
