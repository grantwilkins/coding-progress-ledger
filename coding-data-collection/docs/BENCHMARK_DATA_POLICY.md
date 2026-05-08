# Benchmark Data Policy

## Committed Data Boundary

Committed source samples may contain public task identifiers, descriptions,
metadata hashes, timeouts, visible selected-test routes, and future command
plans.

Committed artifacts must not contain:

```text
raw archives
hidden tests
oracle files
verifier internals
gold patches
test_patch
fail_to_pass / pass_to_pass oracle lists
```

## Manual Export Paths

Terminal-Bench HF preview rows:

```bash
uv run python scripts/inspect_benchmark.py \
  --source terminal_bench_hf \
  --input-jsonl datasets/source_samples/terminal_bench_hf_rows.jsonl \
  --out manifests/terminal_bench_hf_registry.csv
```

Harbor Terminal-Bench current task IDs:

```bash
uvx harbor run \
  -d terminal-bench/terminal-bench-2 \
  -t terminal-bench/fix-git \
  -a oracle \
  --jobs-dir runs/feasibility/harbor_oracle
```

SWE-bench Pro preview rows:

```bash
uv run python scripts/inspect_benchmark.py \
  --source swe_bench_pro \
  --input-jsonl datasets/source_samples/swe_bench_pro_rows.jsonl \
  --out manifests/swe_bench_pro_registry.csv
```

Full raw exports belong under ignored local paths such as `datasets/raw/` or a
run scratch directory. They are inputs to inspection and feasibility only, not
committed collection artifacts.

Downstream systems may train or evaluate on derived telemetry. This repo only
produces and audits trace artifacts.

It must not train general-purpose language models on:

```text
task archives
task text as pretraining data
oracle solutions
solution.sh
hidden tests
verifier internals
gold patches
test_patch contents
canary-only text
```

Rules:

- Do not publish raw task archives, hidden tests, oracle files, verifier
  internals, or gold patches in generated artifacts.
- Agent-visible logs may include official task instructions but must not
  include hidden materials.
- Redact or segregate oracle/gold materials from any model-visible prompt
  or trace.
- Record source license and usage terms in source audits.
- Treat benchmark canaries as compliance signals, not model-training data.
