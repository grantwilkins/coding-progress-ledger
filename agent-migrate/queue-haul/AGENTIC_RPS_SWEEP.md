# Agentic RPS SLO sweep

Schema v4 measures where the fixed GPT-OSS-20B OpenHands-shaped workload
violates its 1 s P90 TTFT or 50 ms P90 TPOT SLO. H100 and A100 use the same
code, model revision, request shape, seed, optimized native vLLM 0.22.0 /
LMCache 0.5.1 stack, and statistical procedure. This is the replacement
experiment for that optimized stack, not repeated vLLM 0.19.1 evidence. Its
discarded preflight independently selects a focused formal grid on each GPU.

The frozen ascending scout candidates are 0.03125, 0.0625, 0.125, 0.25, 0.5,
1, 2, 4, 8, 10, 12, 16, 24, and 32 RPS. Each candidate uses a fresh warmed
engine. Scouting stops only after the first observed SLO violation and the next
two candidates also violate. All lower observations must be numeric SLO passes;
a later pass after a violation hard-fails the scout. If the fixed candidates do
not supply a pass, a violation, and both upper guards, make a new plan rather
than adding rates to a live run.

The discarded scout fixes the formal grid before block 0: one lower scout guard
when available, the observed pass/fail bracket divided into eight equal
intervals, and the next two violating scout anchors. The immutable preflight
record includes raw-evidence hashes, the exact selected rates, all 30 randomized
block orders, and a selection hash. Every formal cell records that hash, and
resume and reduction recompute the evidence record instead of trusting it.

Each rate has 32 open-loop Poisson requests with 3,920 prompt and exactly 1,024
output tokens. Twenty randomized complete blocks are primary; ten predeclared
blocks are available only if the primary result is unresolved. Every block
starts a fresh engine, performs a discarded 32-request warmup at 1 RPS, and
resets and drains caches between rates. Shared A100/H100 rates have identical
arrival and prompt seeds and identical relative order.

A numeric cell requires all 32 completions, 32 exact one-token streams, zero
cache hits, complete telemetry, drain, correct token counts, and at most 50 ms
send lateness. Telemetry must bracket the episode with no scrape gap over 1 s.
Instrumentation or runtime failures stop the run and retain the failed attempt.
Genuine service failures are retained as right-censored violations and are not
retried. The 1-RPS warmup is a compilation warmup, not a claimed safe rate.

At every rate, the figure shows all block-level P90 values, their median, and
an exact distribution-free median interval. The predeclared two-look rule uses
the 5th through 16th observations at 20 blocks (98.818% coverage) and the 9th
through 22nd at 30 (98.388%). A union bound gives at least 97.206% coverage for
the interval selected by the optional second look. These are pointwise
rate/metric error bars, not a simultaneous 95% band over both curves.

A boundary runs from the last clear pass to the first clear fail, allowing only
indeterminate error bars between them. It must be no wider than four refined
steps (half the scout bracket) and have another higher clear fail; any lower
clear fail or higher clear pass makes it unresolved. Aggregate medians and
intervals use offered RPS, faint raw dots use realized RPS, and the request-rate
axis is logarithmic. This describes the finite 32-request episode, not
stationary capacity.

## Prepare the frozen plans

The two plan files are committed. These commands reproduce them with the same
explicit seed and should leave a clean checkout unchanged:

```bash
uv run python agentic_rps_sweep_campaign.py prepare --error-bars \
  --seed 20260901 --hardware h100 \
  --out runs/agentic-rps-sweep-h100-v4/plan.json
uv run python agentic_rps_sweep_campaign.py prepare --error-bars \
  --seed 20260901 --hardware a100 \
  --out runs/agentic-rps-sweep-a100-v4/plan.json
git diff --exit-code -- runs/agentic-rps-sweep-h100-v4/plan.json \
  runs/agentic-rps-sweep-a100-v4/plan.json
```

## Run on H100

Use the same clean pushed commit on both sites from
`/home/azureuser/coding-progress-ledger/agent-migrate/queue-haul`.
The commands select GPU 0 on a plain node. Under a scheduler, use its existing
single-device `CUDA_VISIBLE_DEVICES` value instead.

```bash
CUDA_VISIBLE_DEVICES=0 QH_RUNTIME=native QH_LMCACHE_MODE=mp \
QH_CACHE_ROOT=/datadrive/queue-haul-cache HF_HOME=/datadrive \
uv run python agentic_rps_sweep_campaign.py preflight \
  --plan runs/agentic-rps-sweep-h100-v4/plan.json \
  --run-root /datadrive/agentic-rps-sweep-h100-v4

CUDA_VISIBLE_DEVICES=0 QH_RUNTIME=native QH_LMCACHE_MODE=mp \
QH_CACHE_ROOT=/datadrive/queue-haul-cache HF_HOME=/datadrive \
uv run python agentic_rps_sweep_campaign.py run \
  --plan runs/agentic-rps-sweep-h100-v4/plan.json \
  --run-root /datadrive/agentic-rps-sweep-h100-v4 \
  --model openai/gpt-oss-20b --blocks 20

uv run python agentic_rps_sweep_campaign.py reduce \
  --plan runs/agentic-rps-sweep-h100-v4/plan.json \
  --run-root /datadrive/agentic-rps-sweep-h100-v4 \
  --blocks 20 \
  --out /datadrive/agentic-rps-sweep-h100-v4/summary.json
```

## Run on A100

```bash
CUDA_VISIBLE_DEVICES=0 QH_RUNTIME=native QH_LMCACHE_MODE=mp \
QH_CACHE_ROOT=/datadrive/queue-haul-cache HF_HOME=/datadrive \
uv run python agentic_rps_sweep_campaign.py preflight \
  --plan runs/agentic-rps-sweep-a100-v4/plan.json \
  --run-root /datadrive/agentic-rps-sweep-a100-v4

CUDA_VISIBLE_DEVICES=0 QH_RUNTIME=native QH_LMCACHE_MODE=mp \
QH_CACHE_ROOT=/datadrive/queue-haul-cache HF_HOME=/datadrive \
uv run python agentic_rps_sweep_campaign.py run \
  --plan runs/agentic-rps-sweep-a100-v4/plan.json \
  --run-root /datadrive/agentic-rps-sweep-a100-v4 \
  --model openai/gpt-oss-20b --blocks 20

uv run python agentic_rps_sweep_campaign.py reduce \
  --plan runs/agentic-rps-sweep-a100-v4/plan.json \
  --run-root /datadrive/agentic-rps-sweep-a100-v4 \
  --blocks 20 \
  --out /datadrive/agentic-rps-sweep-a100-v4/summary.json
```

If a 20-block summary says `extend_to_30`, rerun the same `run` and `reduce`
commands with `--blocks 30`. Stop at 30 even if the result remains unresolved.

After copying both summaries to one checkout, plot them with:

```bash
uv run python plot_agentic_rps_sweep.py \
  outputs/agentic-rps-sweep-v4/a100-summary.json \
  outputs/agentic-rps-sweep-v4 \
  --h100-summary outputs/agentic-rps-sweep-v4/h100-summary.json
```

The older schema-v3 campaign remains available through `prepare` without
`--error-bars` for existing serving-calibration consumers. Its retained
single-repeat vLLM 0.19.1 H100 curve is pilot evidence only and must not be
pooled with v4.
