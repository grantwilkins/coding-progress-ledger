# Terminal-Bench Smoke Status

Date: 2026-05-05

## Scope

Workstream E is complete for the chosen `hf_archive_custom` path. Harbor is
kept as a secondary path from the feasibility spike, so E does not require new
Harbor oracle runs.

## Evidence

No-op negative control:

```text
runs/d_smoke/noop_aimo/                         completed_failure
```

Oracle positive controls:

```text
runs/d_smoke/oracle_hello_world/                completed_success
runs/d_smoke/oracle_grid_pattern_transform/     completed_success
runs/d_smoke/oracle_aimo_airline_departures/    completed_success
```

All four run directories validate with `scripts/validate_run.py`.

## Deterministic Verifier Rerun

Command:

```bash
uv run python scripts/verify_verifier_determinism.py \
  --run-dir runs/d_smoke/oracle_hello_world \
  --task-dir /private/tmp/houdini_tb_d_smoke/hello-world \
  --image-tag houdini-tb-d-hello-world \
  --allow-verifier-network \
  --trials 2
```

Result:

```json
{"deterministic": true, "run_id": "oracle_hello_world", "trials": 2}
```

The rerun compared semantic verifier signatures rather than raw Docker logs.
Both trials matched the recorded verifier outcome:

```text
exit_code=0
collected=2
passed=2
failed=0
errors=0
warnings=1
failed_tests=[]
```

Detailed artifact:

```text
runs/d_smoke/oracle_hello_world/verifier_determinism_report.json
```

## Implemented

Code:

```text
src/coding_data_collection/verifier_determinism.py
scripts/verify_verifier_determinism.py
tests/test_verifier_determinism.py
```

The determinism checker:

- Chooses `oracle_workspace_snapshot/` when available, otherwise
  `agent_workspace/`.
- Copies into a fresh verifier workspace for each trial.
- Runs the same Docker verifier command shape as the substrate.
- Compares exit code, collected test count, pass/fail/error/warning counts,
  and failed test identities.
- Avoids brittle byte-for-byte comparison of package-install logs.

## Verification

```bash
uv run pytest tests/test_verifier_determinism.py
# 3 passed

uv run pytest tests
# 54 passed

uv run python scripts/validate_run.py runs/d_smoke/noop_aimo
uv run python scripts/validate_run.py runs/d_smoke/oracle_hello_world
uv run python scripts/validate_run.py runs/d_smoke/oracle_grid_pattern_transform
uv run python scripts/validate_run.py runs/d_smoke/oracle_aimo_airline_departures
```

Known limitation:

```text
The rerun command currently depends on the local extracted HF task directory
under /private/tmp. Production HF archive extraction remains a later
workstream before pilot collection.
```
