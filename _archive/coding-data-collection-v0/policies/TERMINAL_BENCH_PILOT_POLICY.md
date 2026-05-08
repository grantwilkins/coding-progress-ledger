# Terminal-Bench Pilot Policy

Pilot:

```text
12 tasks
2 arms
24 runs
```

Task selection must balance:

```text
category
difficulty
expected trajectory richness
operational risk
```

Candidate scoring fields:

```text
expected_runtime_bucket
expected_validation_visibility
expected_file_edit_complexity
expected_environment_complexity
expected_failure_modes
oracle/test leakage risk
Docker feasibility
internet requirement
large downloads/builds
```

Hard rule: Phase 0 must choose `harbor_native`, `hf_archive_custom`, or
`hybrid` before pilot collection begins.

