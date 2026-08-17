# Agentic RPS sweep

This is a deliberately small, descriptive single-A100 experiment. It supports
the claim:

> For this fixed agentic request shape on one A100, increasing offered rate
> from X to Y RPS causes TTFT and/or TPOT to violate the declared SLO.

The design follows the serving convention used by
[DistServe](https://arxiv.org/abs/2401.09670): sweep offered request rate and
evaluate P90 TTFT and TPOT against explicit latency SLOs. Finite-rate
[vLLM serving benchmarks](https://github.com/vllm-project/vllm/blob/main/docs/benchmarking/cli.md)
also use open-loop Poisson arrivals by default and leave concurrency unlimited
unless a limit is explicitly requested.

## Fixed experiment

- One A100 and one serving engine per model.
- Models: GPT-OSS-20B, Qwen3.8-27B, and Gemma-4-26B-A4B-it.
- One compact OpenHands-derived shape: 3,920 prompt tokens and exactly 1,024
  generated tokens (`ignore_eos=true`).
- Seeded open-loop Poisson arrivals at the original 0.125, 0.25, 0.5, 1, 2,
  4, and 8 RPS discovery points.
- Predeclared dense refinement points at 3, 5, 6, and 7 RPS for GPT-OSS
  and Gemma, and at 0.6, 0.7, 0.8, and 0.9 RPS for Qwen. These localize
  each observed SLO knee without wasting Qwen cells deep in overload.
- 32 requests at every rate, with no concurrency cap.
- Every rate runs even after a violation. Failures and engine exits are data,
  not campaign gates.
- Repeat the first discovery violation and the preceding rate twice more. The
  three boundary observations produce median curves and min-max whiskers.

The plan is schema v3. TPOT is the P90 over all exact post-first-token
intervals in a cell, pooled across its 32 fixed-length requests. The earlier
schema-v2 results used P90 over per-request mean TPOT and therefore cannot be
used directly; they must be re-reduced from their retained token timestamps or
rerun.

The fixed SLOs are 2.0 s TTFT / 0.1 s TPOT for GPT-OSS and 2.0 s TTFT /
0.2 s TPOT for Gemma. Qwen uses twice its 0.125-RPS P90 baseline for each
metric.

## Run

Prepare one immutable plan:

```bash
uv run python agentic_rps_sweep_campaign.py prepare \
  --out runs/agentic-rps-sweep-a100/plan.json
```

Run one model per node. These commands are independent and can run in
parallel:

```bash
HF_HOME=/datadrive uv run python agentic_rps_sweep_campaign.py run \
  --plan runs/agentic-rps-sweep-a100/plan.json \
  --run-root /datadrive/agentic-rps-sweep \
  --model 'openai/gpt-oss-20b'
```

Use the corresponding model ID for Qwen or Gemma. Completed cells are skipped,
so rerunning the same command resumes safely. After copying the three model
cell directories into one run root, reduce and plot:

```bash
uv run python agentic_rps_sweep_campaign.py reduce \
  --plan runs/agentic-rps-sweep-a100/plan.json \
  --run-root /datadrive/agentic-rps-sweep \
  --out outputs/agentic-rps-sweep/summary.json

uv run python plot_agentic_rps_sweep.py \
  outputs/agentic-rps-sweep/summary.json \
  outputs/agentic-rps-sweep
```

The plot command writes one aligned two-panel PDF and PNG. Both panels use raw
seconds and a conventional linear RPS axis; each contains one canonical model
curve, model-colored SLO lines, first-confirmed-violation markers, and min-max
whiskers at repeated boundary rates.
