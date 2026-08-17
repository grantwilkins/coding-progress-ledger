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
rerun. The v3 `rps-sweep.csv` exports this pooled metric as `p90_tpot_s` and
omits the diagnostic per-request-mean percentile.

Retained schema-v1/v2 request files can be pooled without rerunning inference:

```bash
uv run python agentic_rps_sweep_campaign.py rereduce \
  --plan runs/agentic-rps-sweep-a100/plan.json \
  --source-root /path/to/sweden/cells \
  --source-root /path/to/east/cells \
  --source-root /path/to/germany/cells \
  --source-label swedencentral \
  --source-label eastus2 \
  --source-label germanywestcentral \
  --source-origin /persistent/sweden/cells \
  --source-origin /persistent/east/cells \
  --source-origin /persistent/germany/cells \
  --run-root runs/agentic-rps-sweep-a100-pooled \
  --csv outputs/agentic-rps-sweep-a100-pooled/all-rereduced-cells.csv
```

Each derived cell records hashes of its source result and raw request file.

The fixed SLOs are 2.0 s TTFT / 0.1 s TPOT for GPT-OSS and 2.0 s TTFT /
0.2 s TPOT for Gemma. Qwen uses twice its 0.125-RPS P90 baseline for each
metric.

## Run

Prepare one immutable plan:

```bash
uv run python agentic_rps_sweep_campaign.py prepare \
  --hardware a100 \
  --out runs/agentic-rps-sweep-a100/plan.json
```

Use `--hardware h100` for the same workload on the optimized H100 runtime. It
validates the visible GPU, leaves CUDA architecture selection to that GPU, and
enables vLLM compilation and CUDA graphs; the A100 runtime remains eager.
Runtime provenance allows up to three minutes for vLLM to serialize its full
server configuration before failing the launch.

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

The plot command writes a compact, vertically stacked PDF and PNG for
GPT-OSS-20B. Both panels use raw seconds and an increasing linear RPS axis,
with the OpenHands Agentic curve and a black dotted SLO line.

## H100 GPT-OSS result

The workload-matched H100 run is retained in
`outputs/agentic-rps-sweep-h100-vllm019-20260817`. It used the same model
revision, 3,920/1,024-token request shape, Poisson seeds, rates, and 32 requests
per cell as the A100 campaign. All 352 requests completed and neither SLO was
violated through 8 RPS.

| RPS | P90 TTFT (s) | P90 TPOT (ms) |
|---:|---:|---:|
| 0.125 | 0.095 | 4.93 |
| 0.25 | 0.095 | 5.27 |
| 0.5 | 0.096 | 5.69 |
| 1 | 0.105 | 6.01 |
| 2 | 0.138 | 7.44 |
| 3 | 0.173 | 8.40 |
| 4 | 0.163 | 8.85 |
| 5 | 0.194 | 8.86 |
| 6 | 0.252 | 8.38 |
| 7 | 0.274 | 8.41 |
| 8 | 1.498 | 8.33 |

These are workload-matched, not runtime-version-matched, results. The current
vLLM 0.22 environment showed a severe GPT-OSS low/moderate-batch regression on
H100 even with fresh SM90 caches, compilation, and both Triton and Marlin MXFP4
backends. The retained result therefore uses an isolated vLLM 0.19.1 runtime
with compilation and CUDA graphs. A 16-request diagnostic improved from about
186 output tokens/s and 80 ms P90 TPOT on vLLM 0.22 to 2,050 output tokens/s
and 5.94 ms on vLLM 0.19.1, confirming that the earlier inversion was a
software-stack artifact rather than H100 performance.
