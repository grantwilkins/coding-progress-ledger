# Benchmark Adapters

Adapters normalize source metadata for collection planning. They do not
train models and do not expose hidden benchmark materials to agents.

## TerminalBenchHFAdapter

Source: `ia03/terminal-bench`.

Reads row fields such as:

```text
task_id
archive
task_yaml
difficulty
tags
category
base_description
max_agent_timeout_sec
max_test_timeout_sec
tar_sha256
archive_bytes
n_files
```

Redacted source sample:

```text
datasets/source_samples/terminal_bench_hf_rows.jsonl
```

Fetch/manual export path:

```bash
uv run python scripts/inspect_benchmark.py \
  --source terminal_bench_hf \
  --input-jsonl datasets/source_samples/terminal_bench_hf_rows.jsonl \
  --out manifests/terminal_bench_hf_registry.csv
```

For full local extraction during feasibility, download rows to an untracked
`datasets/raw/ia03_terminal_bench/` directory and extract archives only inside
run-local scratch space. Do not commit `archive`, `task_yaml`, tests, oracle,
or verifier internals.

## HarborTerminalBenchAdapter

Source: `terminal-bench/terminal-bench-2`.

Legacy docs also refer to `terminal-bench@2.0`; the current Harbor Hub listing
on 2026-05-05 advertises:

```bash
harbor run -d terminal-bench/terminal-bench-2
```

Current task IDs are recorded in:

```text
manifests/harbor_terminal_bench_tasks.json
```

Phase 0 decides whether Harbor is:

```text
harbor_native
hf_archive_custom
hybrid
```

Critical question:

```text
Can Harbor expose or accept per-step agent transcript events in the
observation_events.jsonl schema?
```

If yes, wrap Harbor. If no, use Harbor for oracle/verifier smoke tests and
use the archive path for instrumented collection.

## SWEBenchProAdapter

Source: `ScaleAI/SWE-bench_Pro`.

Inspect-only until Terminal-Bench gates pass. The adapter emits future
command plans containing:

```text
repo
base_commit
dockerhub_tag
problem_statement
visible_test_route
hidden_evaluation_route
patch_output_path
expected_diff_format
```

Redacted source sample:

```text
datasets/source_samples/swe_bench_pro_rows.jsonl
```

The sample excludes `patch`, `test_patch`, `fail_to_pass`, and `pass_to_pass`.
Those fields are gold/verifier material for this repo's collection boundary.
