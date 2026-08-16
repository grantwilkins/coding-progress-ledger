# Single-A100 capacity discovery

This is the completed, non-gating single-GPU discovery run. It measures the
largest number of requests observed running simultaneously for three pinned
models at five prompt contexts. Every synchronized burst uses 32 output tokens.
The result is workload-specific descriptive evidence, not a universal serving
limit or a planner admission gate.

## Provenance and validity

- Raw root: `/datadrive/qh-single-gpu-capacity-a100-20260815`
- Raw launch commit: `b5aabce8e79bc92a551d54017082e9c037acdb71`
- Frozen campaign-object hash:
  `f215ed68ec45e4fff11c3c94b21c18c7983f6d08f835666a27fa387f99dc68d3`
- Frozen plan-file SHA-256:
  `76c38b22a24b0da4d5d62e92f4b13a56067c9e0f0c270bd19867754c626a2556`
- GPU: NVIDIA A100 80GB PCIe,
  `GPU-16f1b098-2d58-d5e4-c60e-85267354942d`
- Runtime: native vLLM 0.22.0 and LMCache 0.5.1, BF16 KV, TP1,
  `max_model_len=32768`, `max_num_seqs=256`, 90% GPU-memory allocation,
  chunked prefill, prefix caching, eager execution, and the hybrid KV manager.
- Service exit: success, exit code 0, zero systemd restarts.
- All 15 cells launched on their first attempt. All recorded bursts returned
  exact token counts, `[DONE]`, `finish_reason=length`, drained, and left the
  engine healthy. There were no OOM, launch, request, or infrastructure
  failures.
- All 15 runtime identities pin the same launch commit and GPU. BF16 is proven
  in every cell. Every Qwen cell proves a 784-token unified block and separate
  hybrid object groups.

The raw root retains request token IDs/events, full Prometheus scrapes,
queue/running/KV traces, power samples, server configuration, logs, commands,
versions, and runtime identity hashes. Streaming sometimes coalesced multiple
token IDs into one event. Those requests still count as successful service but
are excluded from exact TPOT quantiles rather than assigned invented timing.

## Result

| Model | Prompt contexts | Maximum simultaneous running requests |
|---|---:|---:|
| GPT-OSS-20B | 4,096 / 8,192 / 16,384 / 24,576 / 32,256 | 65 / 33 / 17 / 12 / 10 |
| Gemma-4-26B-A4B | 4,096 / 8,192 / 16,384 / 24,576 / 32,256 | 65 / 33 / 17 / 12 / 10 |
| Qwen3.8-27B | 3,920 / 7,840 / 15,680 / 24,304 / 32,144 | 8 / 5 / 3 / 2 / 2 |

Every final tested burst completed, so the completed-burst values are
right-censored lower bounds. They are deliberately not reported as failure
limits. The running-request values are repeated plateaus under the frozen
32-output-token burst shape.

Runtime-reported KV geometry differs sharply: GPT-OSS exposes 1,952,597 KV
tokens (56.06 GiB), Gemma 286,068 (20.83 GiB), and Qwen 285,354 (20.03 GiB).
At the first saturated burst, observed KV usage is only 11.3--12.9% for
GPT-OSS and 15.6--20.1% for Qwen, versus 45.0--89.2% for Gemma. Matching
request-count curves therefore do not mean matching binding resources. Service
flow, KV capacity, prefill, decode, and power must remain separate measured
constraints.

The queueing signature is visible without a derived metric. For Qwen at 24K,
repeating the plateau from width 4 to 8 doubles P90 TTFT from 34.51 to 69.17 s
while P90 mean TPOT stays 0.305--0.307 s/token. At 32K, GPT-OSS and Gemma show
the same pattern: TTFT is about 65 s at width 16 and about 130 s at width 32,
while TPOT remains nearly flat within each model. Additional requests wait;
the per-token decode rate does not collapse.

## Files

- `plan.json`: frozen campaign plan.
- `summary.json`: validated reduced evidence.
- `capacity.csv`: one row per model/context, including first-saturation KV,
  TTFT, TPOT, and exact-timing coverage.
- `single-gpu-capacity.pdf` and `.png`: canonical styled figure. Open markers in
  the completed-burst panel denote right-censored lower bounds.
