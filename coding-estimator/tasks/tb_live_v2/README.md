# tasks/tb_live_v2 — internal task package

Internal Terminal-Bench-style tasks for `tb_live_v2`. These exist to
add **trajectory-shape diversity** that the upstream Terminal-Bench 2.0
pool under-represents (per `reports/TB_TASK_SPACE_REVIEW.md` § 4).

## Scope

```text
goal: 25 internal tasks total, 5 per dynamic shape:
  - progress_drop      (5)
  - validation_new_work (5)
  - stuck_blocked      (5)
  - high_progress_failure (5)
  - low_progress_success  (5)

we also want at least 5 tasks that produce late_recovery
(trajectory looks stuck, then resolves). these can overlap with the
five above — late_recovery is a label property, not a task family.
```

## Convention (mirrors upstream TB)

Every task is a directory `tasks/tb_live_v2/<task_id>/` containing:

```text
Dockerfile             # Ubuntu/Debian; install tmux, asciinema; WORKDIR /app
docker-compose.yaml    # client image; TB env vars
task.yaml              # see schema in reports/TB_TASK_SPACE_REVIEW.md § 3
solution.sh            # human oracle (one-shot)
tests/test_outputs.py  # deterministic pytest verifier
shape.yaml             # internal: declares the dynamic shape this task targets
```

`shape.yaml` is **internal-only**. The TB harness ignores it. Its
purpose is to make the dynamic-shape coverage table in
`docs/TB_LIVE_V2_SAMPLING_POLICY.md § Trajectory-shape goals`
auditable from filesystem state alone.

```yaml
# shape.yaml schema
target_shape:        progress_drop | validation_new_work | stuck_blocked |
                     high_progress_failure | low_progress_success
secondary_shapes:    []          # optional — list of shapes the task may also produce
expected_pass_rate:  0.30..0.70  # author-estimated for the chosen agent stack
notes: |
  Plain-text rationale for why this task is expected to produce the target shape.
```

## Task IDs follow the convention `<shape_short>_<n>_<one-line slug>`

```text
progress_drop_01_lint_then_runtime_failure
progress_drop_02_compile_then_test_breaks
validation_new_work_01_test_reveals_edge_case
validation_new_work_02_silent_io_format_drift
stuck_blocked_01_missing_dep_loop
stuck_blocked_02_ambiguous_path
high_progress_failure_01_subtasks_done_verifier_strict
high_progress_failure_02_partial_solution_passes_smoke
low_progress_success_01_oneline_fix
low_progress_success_02_config_flag_decisive
```

## Authoring requirements

A task must satisfy **all** of:

1. The `task.yaml::descriptions[base].description` is plain natural
   language. No hint about `tests/test_outputs.py`. No expected
   output values that aren't part of the task statement.
2. The verifier in `tests/test_outputs.py` is deterministic: it
   tests properties of the final container state (file existence,
   file contents, command exit codes), not text comparison against
   model output.
3. A human oracle in `solution.sh` solves the task in one shell
   session. Run `bash solution.sh` in a fresh container and confirm
   the verifier passes.
4. The dynamic shape declared in `shape.yaml::target_shape` actually
   manifests in the agent trajectory under the pre-registered
   `tb_live_v2` agent configuration. Verify by running once with that
   agent, inspecting the ledger, and confirming the labels in
   `coding_estimator/labels/dynamics.py` fire.
5. The task does not require GPU, network, or > 600 s wall-clock.
6. The author email is set; the difficulty is one of `easy`,
   `medium`, `hard`; `tags` includes the target shape.

## How to add a task

```bash
# 1. Pick a shape and a slug.
# 2. Copy the closest-matching example task as a template.
cp -r tasks/tb_live_v2/progress_drop_01_lint_then_runtime_failure \
      tasks/tb_live_v2/progress_drop_03_<your_slug>

# 3. Edit task.yaml, Dockerfile, tests/test_outputs.py, solution.sh, shape.yaml.

# 4. Smoke-test the verifier with the oracle:
bash tasks/tb_live_v2/<task_id>/solution.sh    # in fresh container
pytest tasks/tb_live_v2/<task_id>/tests/

# 5. Add the task ID to the candidate set in
#    docs/TB_LIVE_V2_SAMPLING_POLICY.md
#    (or to the corresponding U3 manifest once that file is added).
```

## Status

- [x] Skeleton + 5 example task scaffolds shipped (one per shape).
- [ ] 20 additional tasks to reach 25 (4 per shape × 5 shapes).
- [ ] Smoke-test all oracles in fresh containers.
- [ ] Verify each task's dynamic shape under the pre-registered agent
  configuration (see `docs/TB_LIVE_V2_SAMPLING_POLICY.md`).
