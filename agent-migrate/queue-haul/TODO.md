# Queue-Haul TODO

Complete tasks in order. After every change, run `uv run pytest`, stage all
task files, and create the listed commit.

## 1. Paper and contract

- [ ] Confirm `formulation_nsdi.md`, `PROPOSED_DESTINATION_ARCH.md`, and
  `DESTINATION_ROADMAP.md` match the pool-level contract.
- [ ] Remove stale commands and claims from `README.md`.
- [ ] Test: `uv run pytest`.
- [ ] Commit: `Align Queue-Haul paper and evaluation contracts`.

## 2. Assumptions and provenance

- [ ] Add one canonical evaluation configuration for workloads, scales,
  deadlines, routes, flex, debt, pool counts, and source skew.
- [ ] Put `# TODO: ASSUMED` immediately above every unmeasured code default.
- [ ] Record provenance, units, validity range, and replacement evidence in
  every configuration value and result row.
- [ ] Hard-fail if an assumed value is emitted as measured or accepted.
- [ ] Test: hand-worked provenance and invalid-input cases, then
  `uv run pytest`.
- [ ] Commit: `Add explicit evaluation assumptions and provenance`.

## 3. Pool service flex

- [ ] Add ongoing event admission, stable capacity, and service-debt budgets.
- [ ] Define flex as 0/5/10/20% of stable pool capacity above normal.
- [ ] Define debt as excess replica-seconds over the migration window.
- [ ] Report required recovery; positive debt with no spare capacity is
  infeasible.
- [ ] Keep memory as block-rounded stock and network as pool-route bytes/queue.
- [ ] Test: exact units, boundaries, conservation, zero-spare recovery, and no
  double-counted migration work.
- [ ] Run `uv run pytest`.
- [ ] Commit: `Add pool service flex and transition debt`.

## 4. Simulator and result schema

- [ ] Schedule pool reconstruction, service, route, and byte queues.
- [ ] Consume advertised destination baselines; do not generate unrelated
  destination traffic.
- [ ] Emit candidate choice, action mix, every capacity/use, binding set, debt,
  recovery, route queue, makespan, power, and exposed state.
- [ ] Keep Poisson and equal-mean burst arrivals only for contract validation.
- [ ] Test: planner-to-simulator accounting, pool aggregation, and deadline
  boundaries.
- [ ] Run `uv run pytest`.
- [ ] Commit: `Complete pool-level drain simulation`.

## 5. Assumed-data evaluation pipeline

- [ ] Add checksum-pinned ShareGPT conversation shapes from the supplied
  `ShareGPT_V3_unfiltered_cleaned_split.json` artifact.
- [ ] Generate coding, interactive-coding, agentic, and conversation scenarios
  at 10K, 100K, and 1M sessions.
- [ ] Run 30/60/120/300-second deadlines, 1/5/10-Gbps routes,
  0/5/10/20% flex/debt, and 1/2/4/8 pools.
- [ ] Generate one tidy result table for each Q1–Q9 question.
- [ ] Generate every planned plot from reduced tables.
- [ ] Ensure every plotted row carries measured/fitted/assumed/simulated
  provenance.
- [ ] Run `uv run pytest`.
- [ ] Commit: `Add complete NSDI evaluation pipeline`.

## 6. Source power evidence

- [ ] Split complete runs into fit, calibration, and untouched final groups.
- [ ] Measure controlled group removals across source load.
- [ ] Require measured shed to meet or exceed every credited final prediction.
- [ ] Measure last-switch-to-off time and the final five-second 0-W accelerator
  window.
- [ ] Replace matching `TODO: ASSUMED` values.
- [ ] Regenerate Q1.
- [ ] Run `uv run pytest`.
- [ ] Commit: `Validate conservative source power shed`.

## 7. Single-session evidence

- [ ] Reuse valid two-A100 replay/KV evidence.
- [ ] Add only missing 4K–8K and 5-Gbps points.
- [ ] Validate bytes, blocks, cache state, no post-commit WAN fetch,
  continuation, pause, route switch, and first token.
- [ ] Regenerate Q2.
- [ ] Run `uv run pytest`.
- [ ] Commit: `Complete replay and KV handoff evidence`.

## 8. Destination service evidence

- [ ] Build prefill-heavy, balanced, and decode-heavy mixes from the four trace
  families.
- [ ] Use safe forced tokens, unique appends or cache reset, exact cache checks,
  and complete-stream checks.
- [ ] Bracket normal and stable passing/failing points.
- [ ] Repeat only boundary points at least three times.
- [ ] Measure queued work and recovery for 0/5/10/20% bursts.
- [ ] Keep the profile `sensitivity` if any required boundary is unbracketed.
- [ ] Replace matching `TODO: ASSUMED` values only after acceptance.
- [ ] Regenerate Q3–Q4.
- [ ] Run `uv run pytest`.
- [ ] Commit: `Measure destination service flex`.

## 9. End-to-end execution

- [ ] Run the full event on two A100s.
- [ ] Run the same contract on 8+8 A100 TP=1 replicas.
- [ ] Measure accelerator power, pool/route queues, actions, pauses, switches,
  first tokens, shutdown, makespan, debt, and recovery.
- [ ] Reject any point that misses the deadline, final power window, or pool
  contract.
- [ ] Regenerate Q5.
- [ ] Run `uv run pytest`.
- [ ] Commit: `Validate end-to-end drain execution`.

## 10. Planner, scale, and diversity

- [ ] Compare exact integer, LP, rounded/packed, greedy, and focused baselines.
- [ ] Run ten seeds for all workloads and source scales.
- [ ] Run measured-normal, balanced, moderate-skew, and high-skew source
  placement.
- [ ] Separate fixed-total and fixed-per-pool experiments.
- [ ] Separate resource diversity and compatibility diversity.
- [ ] Run integrated pools and assumed prefill/decode-disaggregated pools.
- [ ] Regenerate Q6–Q9.
- [ ] Run `uv run pytest`.
- [ ] Commit: `Complete scale and diversity evaluation`.

## 11. Hardware generality

- [ ] Measure H100 TP=1.
- [ ] Measure A100 TP=2.
- [ ] Replace matching assumed source power, service, KV, replay, and ingest
  inputs.
- [ ] Regenerate hardware sensitivity plots.
- [ ] Run `uv run pytest`.
- [ ] Commit: `Add measured hardware generality`.

## 12. Final pass

- [ ] Regenerate every reduced table and figure from a clean checkout.
- [ ] Verify checksums and evidence labels.
- [ ] Remove stale README content and completed TODO details.
- [ ] Confirm all documents match implemented behavior.
- [ ] Run `uv run pytest`.
- [ ] Commit: `Finalize Queue-Haul NSDI evaluation`.
