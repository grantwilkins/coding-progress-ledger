# tb_live_v2 internal task manifest

Source of truth for the 25 internal tasks targeting trajectory-shape
diversity. Each row is a task. Tasks marked `scaffold` are fully
specified in this directory; `spec` rows are designed but not yet
materialized as Docker+verifier assets.

## Coverage by shape (target = 5 each)

| target_shape           | shipped | spec | total |
|------------------------|--------:|-----:|------:|
| progress_drop          | 1       | 4    | 5     |
| validation_new_work    | 1       | 4    | 5     |
| stuck_blocked          | 1       | 4    | 5     |
| high_progress_failure  | 1       | 4    | 5     |
| low_progress_success   | 1       | 4    | 5     |
| **total**              | **5**   | **20** | **25** |

## Ship status — full task scaffolds

| task_id | difficulty | category | target_shape | expected_pass |
|---|---|---|---|---:|
| progress_drop_01_lint_then_runtime_failure | medium | file_operations | progress_drop | 0.45 |
| validation_new_work_01_test_reveals_edge_case | medium | datetime/python | validation_new_work | 0.55 |
| stuck_blocked_01_missing_dep_loop | medium | python/dependencies | stuck_blocked | 0.50 |
| high_progress_failure_01_subtasks_done_verifier_strict | hard | http/python | high_progress_failure | 0.35 |
| low_progress_success_01_oneline_fix | easy | debugging/python | low_progress_success | 0.85 |

## Spec status — designed, not yet scaffolded

The remaining 20 tasks are specified at the design level. To ship a
task, copy the closest scaffold (per the README workflow) and replace
the task-specific bits.

### progress_drop (4 more)

| task_id | difficulty | sketch |
|---|---|---|
| progress_drop_02_compile_then_test_breaks | medium | Build a tiny C library; `cc` succeeds, but linking against the test harness reveals a missing symbol the agent must investigate. |
| progress_drop_03_lint_clean_logic_wrong | medium | Write a sliding-window mean. Type-checker passes. Verifier checks edge cases (window > len(input), single-element input) where the off-by-one bites. |
| progress_drop_04_yaml_valid_schema_invalid | medium | Author a `config.yaml` matching a stated schema. YAML parses fine; the schema validator (a hidden script) flags missing fields the spec mentions. |
| progress_drop_05_first_pr_revert | hard | Implement two related functions. Both pass their first tests. A cross-function integration test fails, forcing the agent to reopen and revise. |

### validation_new_work (4 more)

| task_id | difficulty | sketch |
|---|---|---|
| validation_new_work_02_silent_io_format_drift | medium | Read a JSON-lines file. The fixture has a malformed line in the middle; agent's first attempt assumes well-formed; verifier requires graceful skip. |
| validation_new_work_03_unicode_normalization | medium | Implement a search function that matches "café" to "cafe". Naive str equality fails; verifier reveals NFC normalization is needed. |
| validation_new_work_04_tz_offset_in_log | hard | Parse log timestamps. Verifier includes lines with `+0200` and `Z`; naive impl drops the offset and miscounts. |
| validation_new_work_05_quoted_field_in_tsv | medium | Convert TSV to JSON. Verifier includes a field with literal `\t` inside quotes; naive split-on-tab corrupts it. |

### stuck_blocked (4 more)

| task_id | difficulty | sketch |
|---|---|---|
| stuck_blocked_02_ambiguous_path | medium | The task references a file by name only; two files of that name exist (one in /app/data, one in /tmp). Agent commonly loops choosing the wrong one. |
| stuck_blocked_03_perm_denied_chmod | medium | A script must run, but its permissions are 0o644. The error is misleading ("command not found"). Recovery requires `chmod +x` not `pip install`. |
| stuck_blocked_04_python_version_mismatch | medium | Code uses a 3.11 syntax feature (e.g., `Self` type). Container has 3.10 by default but 3.11 is available as `python3.11`. |
| stuck_blocked_05_make_target_typo | medium | `Makefile` has target `bulid`; task says "run `make build`". Agent loops trying to fix make. Recovery: read the Makefile. |

### high_progress_failure (4 more)

| task_id | difficulty | sketch |
|---|---|---|
| high_progress_failure_02_partial_solution_passes_smoke | hard | Implement a parser. The agent's draft passes 4/5 hidden tests but fails the most subtle edge case the verifier requires. |
| high_progress_failure_03_async_race | hard | Write a small concurrent producer/consumer. Visible runs work; verifier hammers the queue and finds a race. |
| high_progress_failure_04_idempotent_required | hard | Write an "apply migration" script. Visible run completes; verifier runs the script twice and requires no side effect on the second run. |
| high_progress_failure_05_caching_correctness | hard | Implement memoization for a recursive function. Trivial impl works; verifier mutates inputs between calls and requires defensive copy. |

### low_progress_success (4 more)

| task_id | difficulty | sketch |
|---|---|---|
| low_progress_success_02_config_flag_decisive | easy | A long-running script needs `--no-network` flag added. One-line argparse change. |
| low_progress_success_03_typo_in_string | easy | A unit test fails because a hardcoded string has a typo. Fix the typo. |
| low_progress_success_04_missing_env_export | easy | Script uses `$DATA_DIR`. Set it correctly in `.env` and the test passes. |
| low_progress_success_05_quote_glob_in_makefile | easy | `Makefile` rule fails because of unquoted glob; one quote fixes it. |

## Authoring order

1. **First:** ship one full scaffold per shape (DONE — see ship table).
2. **Second:** dynamics-shape coverage. Smoke-test the 5 shipped tasks
   against a representative agent (Arm A model with degraded budget)
   and confirm the target shape actually appears in the ledger. If not,
   iterate the task, not the agent.
3. **Third:** ship the remaining 20 spec tasks in order of expected
   diagnostic value: `high_progress_failure` (most rare) >
   `validation_new_work` > `stuck_blocked` > `progress_drop` >
   `low_progress_success`.

## Acceptance check (per task)

```text
[ ] task.yaml present and well-formed
[ ] Dockerfile installs tmux, asciinema; sets WORKDIR /app
[ ] docker-compose.yaml -> ../docker-compose.template.yaml (or override)
[ ] solution.sh runs to completion in fresh container
[ ] tests/test_outputs.py passes after solution.sh
[ ] tests/test_outputs.py FAILS on a sane wrong implementation (sanity)
[ ] shape.yaml::target_shape matches an entry in the README list
[ ] expected_pass_rate ∈ [0.10, 0.95]
```

The sanity check (verifier fails on a wrong impl) is the most
important — it is what distinguishes a real verifier from a tautology.
