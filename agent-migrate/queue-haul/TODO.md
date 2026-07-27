# Queue-Haul TODO

Complete tasks in order. After every change, run `uv run pytest`, stage all
task files, and create the listed commit.

## Calendar

- **Jul 27–Aug 2:** finish Tasks 1–5 software paths. Exit with one physical
  resource ledger, one realized pool-service trace, pinned workload inputs, and
  runnable assumed-data commands.
- **Aug 3–9:** run Tasks 6–7 on the two-A100 setup. Exit with held-out source
  power and complete replay/KV handoff evidence.
- **Aug 10–16:** run Task 8. Exit only after normal and stable destination
  boundaries have passing and failing points for all three service mixes.
- **Aug 17–23:** run the two-A100 part of Task 9. Exit with one accepted full
  event or a written failed gate with raw evidence.
- **Aug 24–30:** run the 8+8-A100 part of Task 9 and small exact planner cases
  from Task 10. Exit with plan-versus-execution tables.
- **Aug 31–Sep 6:** finish Task 10 at 10K, 100K, and 1M sessions. Exit with
  fixed-total and fixed-per-pool results for all four workloads.
- **Sep 7–13:** run Task 11 if hardware is available; otherwise keep those rows
  assumed. Finish Task 12 and freeze the paper figures.

## 1. Paper and contract

- [x] Confirm `formulation_nsdi.md`, `PROPOSED_DESTINATION_ARCH.md`, and
  `DESTINATION_ROADMAP.md` match the pool-level contract.
- [x] Remove stale commands and claims from `README.md`.
- [x] Test: `uv run pytest`.
- [x] Commit: `Align Queue-Haul paper and evaluation contracts`.

## 2. Assumptions and provenance

- [x] Add one canonical evaluation configuration for workloads, scales,
  deadlines, routes, flex, debt, pool counts, and source skew.
- [x] Put `# TODO: ASSUMED` immediately above every unmeasured code default.
- [x] Record provenance, units, validity range, and replacement evidence in
  every configuration value and result row.
- [x] Hard-fail if an assumed value is emitted as measured or accepted.
- [x] Test: hand-worked provenance and invalid-input cases, then
  `uv run pytest`.
- [x] Commit: `Add explicit evaluation assumptions and provenance`.

## 3. Pool service flex

- [x] Add ongoing event admission, stable capacity, and service-debt budgets.
- [x] Define flex as 0/5/10/20% of stable pool capacity above normal.
- [x] Define debt as excess replica-seconds over the migration window.
- [x] Report required recovery; positive debt with no spare capacity is
  infeasible.
- [x] Keep memory as block-rounded stock and network as pool-route bytes/queue.
- [x] Test: exact units, boundaries, conservation, zero-spare recovery, and no
  double-counted migration work.
- [x] Run `uv run pytest`.
- [x] Commit: `Add pool service flex and transition debt`.

## 4. Simulator and result schema

- [x] Schedule pool reconstruction, service, route, and byte queues.
- [x] Consume advertised destination baselines; do not generate unrelated
  destination traffic.
- [ ] Emit candidate choice, action mix, every capacity/use, binding set, debt,
  recovery, route queue, makespan, power, and exposed state.
- [ ] Keep Poisson and equal-mean burst arrivals only for contract validation.
- [ ] Test: planner-to-simulator accounting, pool aggregation, and deadline
  boundaries.
- [ ] Run `uv run pytest`.
- [ ] Commit: `Complete pool-level drain simulation`.

## 5. Assumed-data evaluation pipeline

- [x] Add checksum-pinned ShareGPT conversation shapes from the supplied
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
