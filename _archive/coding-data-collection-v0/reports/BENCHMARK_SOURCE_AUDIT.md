# Benchmark Source Audit

Date: 2026-05-05

## Summary

The three inspect adapters now have real, redacted samples or task IDs:

```text
TerminalBenchHFAdapter      datasets/source_samples/terminal_bench_hf_rows.jsonl
HarborTerminalBenchAdapter  manifests/harbor_terminal_bench_tasks.json
SWEBenchProAdapter          datasets/source_samples/swe_bench_pro_rows.jsonl
```

Generated inspect manifests:

```text
manifests/terminal_bench_hf_registry.csv
manifests/terminal_bench_harbor_registry.csv
manifests/swe_bench_pro_registry.csv
```

The committed samples deliberately exclude raw archives, task YAML, oracle
solutions, hidden tests, gold patches, `test_patch`, `fail_to_pass`, and
`pass_to_pass`.

## Sources Checked

Terminal-Bench HF:

- Source: https://huggingface.co/datasets/ia03/terminal-bench
- Observed public fields include `task_id`, `archive`, `task_yaml`,
  `difficulty`, `tags`, `category`, `base_description`,
  `max_agent_timeout_sec`, `max_test_timeout_sec`, `tar_sha256`,
  `archive_bytes`, and `n_files`.
- Committed sample keeps only inspect-safe metadata and excludes the archive
  payload and task YAML.

Harbor Terminal-Bench:

- Source: https://hub.harborframework.com/datasets/terminal-bench/terminal-bench-2/latest
- Current dataset command shown by Harbor Hub: `harbor run -d
  terminal-bench/terminal-bench-2`.
- Three current task IDs recorded for inspection:
  `terminal-bench/fix-git`, `terminal-bench/regex-log`,
  `terminal-bench/query-optimize`.
- Local Harbor CLI version inspected: `0.6.4`.

SWE-bench Pro:

- Source: https://huggingface.co/datasets/ScaleAI/SWE-bench_Pro
- Public fields include `repo`, `instance_id`, `base_commit`, `patch`,
  `test_patch`, `problem_statement`, `requirements`, `interface`,
  `repo_language`, `fail_to_pass`, `pass_to_pass`, `issue_specificity`,
  `issue_categories`, `before_repo_set_cmd`, `selected_test_files_to_run`, and
  `dockerhub_tag`.
- Committed sample keeps inspect-safe metadata and excludes gold/verifier
  columns.

## License And Usage Notes

Terminal-Bench HF:

- The HF dataset card lists license `mit`; it also says the dataset inherits
  original Terminal-Bench repository terms. Treat task-local third-party
  contents as potentially separately licensed until extraction audit.
- Canary marker text appears in the full HF task YAML. Do not commit task YAML
  or extracted hidden materials.

Harbor:

- Harbor itself is Apache-2.0. The current Hub task listing is metadata only.
- Running tasks may download benchmark definitions and build containers; those
  outputs stay under ignored run/scratch paths.

SWE-bench Pro:

- The dataset exposes gold `patch` and verifier `test_patch` fields. This repo
  treats those fields as non-agent, non-committed verifier material.
- Docker image route is `jefzda/sweap-images:<dockerhub_tag>` for future
  execution planning only.

## Acceptance Check

```text
At least 3 real Terminal-Bench HF rows inspectable: yes
At least 3 current Harbor task IDs inspectable: yes
At least 3 real SWE-bench Pro rows inspectable: yes
Raw archives committed: no
Hidden tests committed: no
Oracle files committed: no
Verifier internals committed: no
Gold patches committed: no
```
