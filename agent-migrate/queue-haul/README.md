# Queue-Haul

Queue-Haul plans request-boundary migration of stateful LLM sessions to reduce
source accelerator power before a deadline. It jointly chooses sessions,
replay or full-KV transfer, and compatible destination serving pools.

The primary output is an executable schedule naming each migrated session,
replay or KV transfer, destination pool, handoff timing, source shutdown, and
resource use, slack, debt, recovery, achieved shed, and unmet shed. A
requirement frontier summarizes those schedules across source-power targets.
Destination capacity is an advertised pool contract, not an inferred GPU
inventory.

## Current evidence

The repository contains:

- measured GPT-OSS-20B/A100 power curves;
- working replay and compatible KV handoff on two A100s;
- conservative replay and KV duration fits;
- exact KV block accounting;
- a one-pool requirement-frontier solver;
- LP, integer, and greedy planners;
- pool-aware planning and internal packing checks; and
- a deterministic migration, network, request, queue, and power simulator.

The source power fit still needs held-out group-removal validation. The archived
destination service campaign does not provide an accepted capacity boundary.
Its valid points are sensitivity anchors until the targeted rerun brackets
passing and failing loads.

## System boundary

A handoff prepares state in the background, quiesces at a request boundary,
performs final catch-up, switches routing, and succeeds when the destination
returns the first token. Mid-token migration, return migration, cold model
placement, unrelated destination arrivals, provider fleet policy, and facility
power are out of scope.

The public candidate is:

```text
(session, replay-or-KV, compatible destination pool)
```

The pool manager chooses a replica. A pool advertises ongoing service headroom,
stable capacity, temporary queued-work allowance, reconstruction/ingest
capacity, live-KV blocks, route capacity, compatibility, and evidence status.
The pool planner enforces ongoing event capacity and conservative
replica-second debt and reports required recovery.

## Evidence flow

```text
archived raw logs
  -> checksum-pinned reduced measurements
  -> versioned model/workload/pool inputs
  -> result tables with provenance
  -> deterministic figures
```

Every input and result must say whether it is measured, fitted, assumed, or
simulated. Assumed values are sensitivities, never admission guarantees.
The conversation workload pins ShareGPT artifact revision
`192ab2185289094fc556ec8ce5ce1e8e587154ca` and stores only token/turn shapes.

The detailed contracts are in:

- `formulation_nsdi.md`;
- `PROPOSED_DESTINATION_ARCH.md`;
- `DESTINATION_ROADMAP.md`; and
- `TODO.md`.

## Main commands

Run commands from the parent `agent-migrate` directory:

```bash
uv run pytest

uv run python queue-haul/power_drain_experiment.py \
  --workload-profile queue-haul/profiles/agentic_tool_loop.json \
  --sessions 6 --seed 3 --power-limit 500 --deadline 5 --end 5 \
  --link-bytes-per-s 125000000 --intra-dc-bytes-per-s 12500000000 \
  --solver greedy --workers 2 --out queue-haul/outputs/profile_smoke

uv run python queue-haul/plot_simulator_validation.py
uv run python queue-haul/plot_simulator_evaluation.py
uv run python queue-haul/plot_scaling_results.py
uv run python queue-haul/plot_testbed_kv_timeline.py
uv run python queue-haul/plot_testbed_kv_timeline.py --method replay
uv run python queue-haul/plot_migration_ttft_cdf.py
uv run python queue-haul/plot_fixed_contract_residuals.py
uv run python queue-haul/policy_hardware_campaign.py prepare \
  --out queue-haul/outputs/policy-hardware-plan
QH_POLICY_RUN_ROOT=/scratch/$USER/qh-policy-run \
  bash queue-haul/outputs/policy-hardware-plan/run.sh
# Or submit the resumable two-A100 Slurm job:
sbatch queue-haul/outputs/policy-hardware-plan/run.sbatch
uv run python queue-haul/paper_evaluation.py \
  --out queue-haul/outputs/paper-evaluation
```

`requirement_frontier.py` computes destination requirements without constructing
a destination inventory. `pool_planner.py` compares those requirements with
concrete pool contracts and emits physical use/capacity rows. `simulate.py`
independently schedules routes, reconstruction endpoints, requests, commits,
and power. It separately traces declared pool-service demand and debt from
realized replay and commit times. `evaluation_config.py` is the canonical
source for assumed paper operating points and their replacement evidence.
`paper_evaluation.py` writes the legacy Q1–Q9 result/plot registry while the
paper evaluation is reorganized into mechanism validation, fixed-contract
coordination, multi-pool contracts, and planner quality/scale. It rejects
tables with missing provenance.
`plot_testbed_kv_timeline.py` generates measured two-A100 concurrency-one KV
transfer and replay timelines with source inference continuing through
request-boundary drain from tidy event tables.
`plot_fixed_contract_residuals.py` caches the canonical 100K-session,
120-second fixed-contract requirement sweep and plots normalized resource
headroom against requested source-power shed. Use `--refresh` when its pinned
inputs change.
`policy_hardware_campaign.py` creates a resumable paired idle-session campaign
for eager, serial execution of Queue-Haul, greedy, and random method/order
choices plus KV-only and replay-only baselines. It does not execute the
planner's paced, quiesced schedule or measure planning latency, so it is
mechanism-path evidence rather than full-scheduler evidence. The default 50
eight-session blocks yield 400 clustered observations per policy; variants and
their control run contiguously in randomized order. Failed blocks remain in
completion denominators, while TTFT inflation uses only complete matched
controls from the same allocation, excluding blocks split by a time limit. Run
from a clean committed checkout with two A100 80GB GPUs. Multi-process logs
rotate every five blocks; transfer and byte logs are sliced per scenario.
Override `QH_APPTAINER_IMAGE` if the pinned LMCache image is not at the default
scratch path; set `QH_RESUME_FROM_GIT_SHA` when resuming after a code change.
Re-submit the job after a time limit; its stable default run root resumes
completed scenarios.

## Measurement programs

- `power_window_sensitivity.py` and `power_profile_reduce.py`: source power.
- `migration_profiler.py` and `migration_profile_fit.py`: replay/KV handoff.
- `destination_campaign.py` and `destination_runner.py`: targeted destination
  service and loaded-migration evidence.
- `service_surface_runner.py` and `service_profile_reduce.py`: isolated and
  mixed service profiles.

The verified 2026-07-23 destination bundle is retained under
`outputs/destination-v7-20260722/`. Do not treat its service rows as an accepted
capacity profile. See `FINDINGS.md` and `DATA_TO_COLLECT.md` for the forensic
audit and retained evidence.

## Development rules

- Keep implementation and documentation small.
- Hard-fail unsupported domains and invalid evidence.
- Add semantic tests for every source change.
- Run `uv run pytest` after every change.
- Commit each completed task separately with a descriptive message.
