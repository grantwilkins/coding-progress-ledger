# Queue-Haul TODO

## Claim to close

Queue-Haul chooses replay or KV transfer for each session, orders the chosen
migrations, and predicts how much source accelerator power can be shed by a
deadline. The primary implementation and two-A100 experiment execute migrations
sequentially. Parallel migration is a separate sensitivity.

## Already shown

- [x] The planner chooses replay/KV actions and an order.
- [x] The simulator executes those actions with migration, route, request,
  queue, commit, first-token timing, and power state.
- [x] Pool planning accounts for service, transition work, route bytes, KV
  stock, debt, recovery, and binding resources.
- [x] Two-A100 replay and KV transfer both preserve continuation and exact
  state/byte accounting.
- [x] All 24 serial migration scenarios completed by their deadline.
- [x] All 90 migrations in the bounded campaign completed by their deadline,
  and all 105 campaign gates passed.
- [x] Incremental KV transfer passed exact block, wire, destination-state, and
  continuation gates.
- [x] Measured replay and KV timelines show source inference, bulk migration,
  request-boundary drain, catch-up, route switch, and first destination token.
- [x] Measured loaded runs cover 16K/10-Gbps and 24K/5-Gbps points. These show
  migration behavior under the recorded foreground overlap, not a destination
  capacity boundary.
- [x] Canonical workload shapes and assumed simulator axes carry explicit
  provenance.
- [x] Run `uv run pytest`.

## Required simulator result

- [ ] Freeze one sequential execution contract and one canonical assumed pool
  configuration.
- [ ] Run Queue-Haul, replay-only, KV-only, and greedy on the same scenarios.
- [ ] Report requested and achieved shed, selected action mix, last commit,
  first destination token, route bytes/queue, destination work, deadline
  status, and exposed sessions.
- [ ] Show one representative simulator schedule and a compact scale result at
  10K, 100K, and 1M sessions. Additional Cartesian sweeps are optional.
- [ ] Keep every unmeasured route, pool, and service input labeled
  `assumed/sensitivity`.
- [ ] Run `uv run pytest`.
- [ ] Commit: `Finalize sequential simulator evaluation`.

## Required two-A100 closure

- [ ] Run the prepared `policy_hardware_campaign.py` experiment.
- [ ] Use its planner-generated Queue-Haul replay/KV choice and order with the
  existing eager sequential executor.
- [ ] Compare Queue-Haul with replay-only, KV-only, greedy, and random on matched
  eight-session episodes.
- [ ] Report completion fraction, controller-to-first-token time,
  controller-to-commit time, migration TTFT, and matched next-request TTFT
  inflation.
- [ ] Add one representative mixed-action Gantt chart using the existing event
  records.
- [ ] Retain failed or incomplete episodes in the completion denominator.
- [ ] Run `uv run pytest`.
- [ ] Commit: `Validate planner-driven sequential migration`.

## Final pass

- [ ] Regenerate the simulator and two-A100 tables and figures from a clean
  checkout.
- [ ] Verify input checksums and evidence labels.
- [ ] Ensure the paper claims sequential two-A100 execution and simulated
  large-scale coordination, not measured production admission or facility
  power.
- [ ] Run `uv run pytest`.
- [ ] Commit: `Finalize Queue-Haul evaluation`.

## Optional extensions

These are useful follow-on results, not blockers for the claim above:

- parallel planner-driven migration;
- a bracketed destination normal/stable service boundary;
- 8+8 A100, H100 TP=1, or A100 TP=2 generality;
- disaggregated prefill/decode pools;
- live leases and commit-time contract revalidation;
- exhaustive deadline, bandwidth, pool-count, skew, and diversity grids; and
- whole-node shutdown or facility/grid-power claims.
