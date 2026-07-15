# GPU validation before the profiling run

The LMCache 0.3.3 concurrency-2 metadata-read failure remains a blocker. Do not
submit `stage1c_benchmark.sbatch` until simultaneous sink lookups complete and
the reducer reports all 16 scenarios as complete. Concurrency-1 smoke data do
not validate concurrency-2 behavior.

Run from the `agent-migrate` repository root. Do not submit the large profiling
plan until every check below passes.

## 1. Check the code

Use a clean worktree and record the commit:

```bash
git status --short
git rev-parse HEAD
uv run pytest
```

Expected: all tests pass and there are no tracked changes. Resolve existing
changes; do not discard work you do not own.

## 2. Get a two-GPU allocation

Run the remaining commands inside one node with two 80 GB A100 GPUs, 256 GB host
memory, and the same constraints as `queue-haul/stage1c_benchmark.sbatch`.

```bash
module load gcc/14.2.0 openblas/0.3.28
PY=.venv/bin/python
TRACE=${TRACE:-/scratch/users/gfw/ptsim/tracelab/syfi_coding_trace.jsonl.gz}
ROOT=queue-haul/outputs/migration_validation_$(date +%Y%m%d_%H%M%S)
export QH_APPTAINER_IMAGE=${QH_APPTAINER_IMAGE:-/scratch/users/gfw/ptsim/vllm-openai-v0.10.1.1.sandbox}
export QH_APPTAINER_GPU_MODE=${QH_APPTAINER_GPU_MODE:-nv}
export QH_PORT_OFFSET=${QH_PORT_OFFSET:-$((SLURM_JOB_ID % 40000 + 1000))}
test -r "$TRACE"
mkdir -p "$ROOT"
```

Valid workload classes are `interactive_coding`, `coding`, and
`agentic_tool_loop`. Validate one class at a time; the commands below use
`coding`.

## 3. Check the two-model testbed

```bash
$PY queue-haul/stage1b_drain_sink.py preflight --required-gpus 2
$PY queue-haul/stage1b_drain_sink.py smoke2-live \
  --mbps 1000 \
  --run-root "$ROOT/smoke2"
$PY -c 'import json,sys; p=sys.argv[1]; r=json.load(open(p)); assert r["acceptance"]["ok"], r' \
  "$ROOT/smoke2/smoke2_manifest.json"
```

Stop if preflight, either model, KV retrieval, replay, the shared bandwidth
limit, or `acceptance.ok` fails.

## 4. Run the small migration plan

This plan has 16 scenarios. It covers both transfer methods, concurrency 1 and
2, no activity and one controlled turn, and a matched no-migration control for
every migration scenario.

```bash
$PY queue-haul/stage1c_controller.py make-manifest \
  --input "$TRACE" \
  --out "$ROOT/manifest.json" \
  --workload coding \
  --sessions 2 \
  --seed 0

$PY queue-haul/stage1c_controller.py make-plan \
  --manifest "$ROOT/manifest.json" \
  --out "$ROOT/plan.json" \
  --context-sizes 2048 \
  --concurrency 1,2 \
  --bandwidth-mbps 1000 \
  --methods replay,kv_transfer \
  --activity none,one_turn \
  --repeats 1 \
  --seed 0

$PY -c 'import json,sys; p=json.load(open(sys.argv[1])); assert len(p["scenarios"]) == 16' \
  "$ROOT/plan.json"

RUN_STATUS=0
$PY queue-haul/stage1c_controller.py run \
  --plan "$ROOT/plan.json" \
  --run-root "$ROOT/run" || RUN_STATUS=$?
$PY queue-haul/stage1c_controller.py reduce --run-root "$ROOT/run"
test "$RUN_STATUS" -eq 0
```

Always run `reduce` after `run`. The run command intentionally exits nonzero if
any scenario fails.

## 5. Check the results

Check the summary tables:

```bash
$PY -c '
import csv, sys
root = sys.argv[1]
s = list(csv.DictReader(open(f"{root}/scenarios.csv")))
m = list(csv.DictReader(open(f"{root}/migrations.csv")))
assert len(s) == 16 and all(r["status"] == "complete" for r in s), s
assert len(m) == 16 and all(r["success"] == "True" for r in m), m
assert all(float(r["logical_kv_chunks"]) == 0 for r in m if r["method"] == "replay")
assert all(float(r["processed_tokens"]) > 0 for r in m if r["method"] == "replay")
assert all(float(r["logical_kv_chunks"]) > 0 and float(r["logical_kv_bytes"]) > 0 for r in m if r["method"] == "kv_transfer")
assert all(float(r["catch_up_s"]) == 0 for r in m if r["activity"] == "none")
assert all(float(r["catch_up_s"]) > 0 for r in m if r["activity"] == "one_turn")
assert all(r.get("continuation_difference_s", "") != "" for r in s if r["kind"] == "migration")
' "$ROOT/run"
```

Check every continuation and both GPUs:

```bash
$PY -c '
import csv, glob, json, sys
root = sys.argv[1]
meta = json.load(open(f"{root}/run_metadata.json"))["config"]
for path in glob.glob(f"{root}/scenarios/*/result.json"):
    result = json.load(open(path))
    assert result["status"] == "complete", result
    scenario = json.load(open(path.replace("result.json", "scenario.json")))
    port = meta["api_proxy_port"] if scenario["kind"] == "migration" else meta["src_port"]
    assert result["continuations"] and all(r["status_code"] == 200 and r["route_port"] == port for r in result["continuations"]), result
    power = list(csv.DictReader(open(path.replace("result.json", "power.csv"))))
    assert {r["gpu"] for r in power if r["valid"] == "1"} == {"0", "1"}, path
    if scenario["kind"] == "migration":
        assert sum(result["wire_bytes"].values()) > 0, result
' "$ROOT/run"

! rg -n "Failed to reset prefix cache" "$ROOT/run/debug"
```

Open at least one replay and one KV scenario for each activity setting. Confirm:

- `migration_timeline.png`: initial copies begin before the red pause outline;
  one-turn scenarios contain catch-up work; concurrency 2 overlaps at most two
  moves.
- `resource_trace.png`: network samples are present, both GPUs have continuous
  valid power samples, and missing data are not hidden.
- `requests.jsonl`: streamed chunks, prompt/output token totals, processed replay
  tokens, KV chunks/bytes, and the logged KV layout are present.
- `cache_operations.jsonl`: the clear event reports zero entries and bytes;
  source writes and destination reads have key hashes, sizes, and timestamps.

Do not use power values or `deadline_met` as acceptance criteria in this test.

After all 16 scenarios succeed, update
`profiles/gpt_oss_20b_a100_tp1.json` from reduced tables. Replace every
`TODO(profile)` assumption with measured phase power and transition timing.
Increase destination concurrency limits only for tested levels. Keep the profile
`estimated` until held-out live scenarios meet timing and fixed-window power
error thresholds.

Before the large GPU plan, run one local profile-driven smoke:

```bash
uv run python queue-haul/power_drain_experiment.py \
  --workload-profile queue-haul/profiles/agentic_tool_loop.json \
  --sessions 6 \
  --power-limit 500 \
  --deadline 5 \
  --end 5 \
  --solver load_only \
  --out queue-haul/outputs/profile_smoke
```

Confirm that all six CSV tables are nonempty and the four plots are readable.

## 6. Submit the large plan

Proceed only if every command and inspection above passes. Create a fresh plan
and run directory for each workload class. Keep the validated commit clean, then
submit:

```bash
WORKLOAD=coding
PLAN=queue-haul/outputs/${WORKLOAD}-plan.json \
RUN_ROOT=queue-haul/outputs/${WORKLOAD}-run \
sbatch queue-haul/stage1c_benchmark.sbatch
```

Keep the validation directory and Slurm job ID with the profiling results.
