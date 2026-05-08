# Docker Substrate Status

Date: 2026-05-05

## Scope

Workstream D is complete for the current scaffold. The substrate still needs
pilot orchestration in later workstreams, but D acceptance is met.

## Implemented

Code:

```text
src/coding_data_collection/docker_substrate.py
scripts/run_docker_substrate_smoke.py
tests/test_docker_substrate_semantics.py
```

The substrate now:

- Builds a Docker image from an extracted Terminal-Bench task directory.
- Records image id/digest, Dockerfile hash, task directory hash, CPU limit,
  memory limit, disk policy, wall-clock limit, and network policy.
- Constructs an agent phase with only:
  ```text
  run_dir/agent_workspace -> /app:rw
  run_dir/task.md         -> /task/task.md:ro
  ```
- Constructs a verifier phase with:
  ```text
  run_dir/verifier_workspace -> /app:rw
  task_dir                   -> /task:ro
  TEST_DIR=/task/tests
  ```
- Copies the agent workspace into a fresh verifier workspace before verifier
  execution.
- Disables agent network by default. Verifier network can be explicitly enabled
  with a recorded exception reason.
- Clears generated agent workspaces before run preparation, so stale outputs
  cannot make a no-op run pass.
- Rejects symlinks in the agent workspace before verifier copy.
- Finalizes agent crash/timeout runs as `agent_crash` / `agent_timeout`
  without running oracle or verifier phases.
- Gives Docker containers deterministic names during phase execution and
  removes the named container on timeout.
- Preserves oracle-produced product files under `oracle_workspace_snapshot/`
  before cleaning the verifier workspace.
- Records both human-readable shell commands and exact `command_argv` arrays
  in transcript rows.
- Returns nonzero for unexpected verifier failure; no-op negative controls must
  pass `--expect-verifier-failure`.

## Smoke Evidence

No-op run:

```text
runs/d_smoke/noop_aimo/
```

Result:

```text
run_status: completed_failure
final_success: false
termination_reason: verifier_fail
```

Semantic checks:

- The agent command ran inside Docker with `--network none`.
- The agent command asserted hidden tests were absent from both `/task/tests`
  and `/app/tests`.
- The verifier ran from a clean copied workspace and mounted hidden tests only
  in the verifier phase.
- The no-op run failed cleanly because expected product files were absent.
- The verifier network exception is recorded in `task_metadata.json`.
- `scripts/validate_run.py runs/d_smoke/noop_aimo` passed.

Oracle smoke runs:

```text
runs/d_smoke/oracle_hello_world/
runs/d_smoke/oracle_grid_pattern_transform/
runs/d_smoke/oracle_aimo_airline_departures/
```

Each oracle smoke uses the same phase boundary:

```text
agent phase    hidden tests and solution absent, network disabled
oracle phase   privileged solution.sh execution in verifier workspace
verifier phase hidden tests mounted read-only, network exception recorded
```

Each oracle smoke also preserves the oracle-generated product files in:

```text
runs/d_smoke/<run_id>/oracle_workspace_snapshot/
```

Results:

```text
oracle_hello_world              completed_success
oracle_grid_pattern_transform   completed_success
oracle_aimo_airline_departures  completed_success
```

Validation:

```bash
uv run python scripts/validate_run.py runs/d_smoke/noop_aimo
uv run python scripts/validate_run.py runs/d_smoke/oracle_hello_world
uv run python scripts/validate_run.py runs/d_smoke/oracle_grid_pattern_transform
uv run python scripts/validate_run.py runs/d_smoke/oracle_aimo_airline_departures
```

Test suite:

```bash
uv run pytest tests
# 36 passed
```

## Known Gaps

- Disk limit is recorded but not enforced by default. Docker Desktop rejected
  `--storage-opt size=...` because the local overlay backend lacks the required
  quota support.
- The smoke script is a substrate proof, not the final pilot orchestrator.
  Workstreams H-J still need task selection, retry policy, transcript capture,
  and estimator artifact production.

## Next

Move to Workstream E for Terminal-Bench smoke policy, deterministic verifier
rerun checks, and no-op/oracle coverage beyond the substrate acceptance set.
