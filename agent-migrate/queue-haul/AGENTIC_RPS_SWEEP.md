# Agentic RPS SLO sweep

The default H100/A100 graph uses `quick_slo_sweep.py`. It measures the fixed
GPT-OSS-20B agentic shape against the unchanged SLOs:

- P90 TTFT must be at most 1 second.
- P90 TPOT must be at most 50 milliseconds.

One warmed engine launch runs 50 uncached requests at each predeclared rate:
0.5, 1, 2, 3, 4, 5, 6, 7, 8, 10, 12, 14, 16, 20, and 24 RPS. Rates run in a
frozen randomized order, with a full cache reset and drain between them. A
resume or engine failure may start another runtime-identical launch; each
result records its launch path. The grid covers the A100 transition around
2--4 RPS and the corrected H100 transition above 10 RPS, with several upper
violation points.

Both sites explicitly use the vLLM `TRITON_ATTN` backend and disable async
scheduling. This avoids the vLLM 0.22.0 H100 default-FA3 performance regression
while `stream_interval=1` retains the finest available stream resolution.
Startup evidence must confirm both settings. This is the quick-v4 runtime;
earlier quick results used a different backend or TPOT estimand and are not
comparable.

Each point is the measured P90 across requests. Request TPOT is `(last token -
first token) / (output tokens - 1)`, matching `vllm bench serve` even when vLLM
bundles several tokens in one streamed delta. The error bar is the envelope of
two pointwise 95% circular moving-block bootstrap intervals, using
adjacent-request blocks of 5 and 10 across 10,000 draws. The point estimate
determines pass or violation; a bar crossing the SLO reports uncertainty
without changing that classification.

These bars quantify request/queue sampling uncertainty conditional on each
rate's finite episode. They do not claim engine-restart, device, or day-to-day
reproducibility. The older v4 plans remain the long-form option for
distribution-free intervals across fresh-engine repetitions.

A numeric cell requires all 50 requests to complete with the exact 3,920-token
prompt and 1,024-token output and valid first/last-token timestamps, zero cache
hits, no dispatch more than 50 ms late, telemetry gaps no larger than three
seconds, successful drain, and unchanged model/runtime/GPU/commit identity.
Service failures remain SLO violations; surprising latency is never retried.

## Frozen plans

The committed plans are reproducible with:

```bash
uv run python quick_slo_sweep.py prepare --seed 20260902 --hardware h100 \
  --out runs/agentic-rps-sweep-h100-quick-v4/plan.json
uv run python quick_slo_sweep.py prepare --seed 20260902 --hardware a100 \
  --out runs/agentic-rps-sweep-a100-quick-v4/plan.json
git diff --exit-code -- \
  runs/agentic-rps-sweep-h100-quick-v4/plan.json \
  runs/agentic-rps-sweep-a100-quick-v4/plan.json
```

Both sites use the same clean commit and checkout path. They must use separate
run roots. The quick runtime keeps vLLM's effective configuration in its
provenance response but disables its unbounded OS-package inventory probe.

## H100

From `/home/azureuser/coding-progress-ledger/agent-migrate/queue-haul`:

```bash
CUDA_VISIBLE_DEVICES=0 QH_RUNTIME=native QH_LMCACHE_MODE=mp \
QH_CACHE_ROOT=/datadrive/queue-haul-cache HF_HOME=/datadrive \
uv run python quick_slo_sweep.py run \
  --plan runs/agentic-rps-sweep-h100-quick-v4/plan.json \
  --run-root /datadrive/agentic-rps-sweep-h100-quick-v4

uv run python plot_agentic_rps_sweep.py \
  /datadrive/agentic-rps-sweep-h100-quick-v4/summary.json \
  /datadrive/agentic-rps-sweep-h100-quick-v4/figures
```

## A100

```bash
CUDA_VISIBLE_DEVICES=0 QH_RUNTIME=native QH_LMCACHE_MODE=mp \
QH_CACHE_ROOT=/datadrive/queue-haul-cache HF_HOME=/datadrive \
uv run python quick_slo_sweep.py run \
  --plan runs/agentic-rps-sweep-a100-quick-v4/plan.json \
  --run-root /datadrive/agentic-rps-sweep-a100-quick-v4
```

After both summaries are available in one checkout, create the combined graph:

```bash
uv run python plot_agentic_rps_sweep.py \
  outputs/agentic-rps-sweep-a100-quick-v4-20260902/summary.json \
  outputs/agentic-rps-sweep-a100-h100-quick-v4-20260902 \
  --h100-summary outputs/agentic-rps-sweep-h100-quick-v4-20260902/summary.json
```

The accepted A100 and H100 quick-v4 summaries, curves, runtime identities, and
figures are in `outputs/agentic-rps-sweep-a100-quick-v4-20260902/` and
`outputs/agentic-rps-sweep-h100-quick-v4-20260902/`, respectively.
The combined figure is in
`outputs/agentic-rps-sweep-a100-h100-quick-v4-20260902/`.
The plot-only 5-second TTFT sensitivity uses `--ttft-slo-s 5` and is in
`outputs/agentic-rps-sweep-a100-h100-quick-v4-ttft5s-20260902/`; the summaries
retain their measured 1-second contract.

Rerunning the same `run` command safely reuses completed cells. Do not share a
run root between sites or combine quick-v4 results with earlier quick runs or the
fresh-engine v4 campaign.
