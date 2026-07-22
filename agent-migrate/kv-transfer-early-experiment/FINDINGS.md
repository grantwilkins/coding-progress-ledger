# KV state transfer vs. public context replay

`migration_ratio.py` now uses public H200 TTFT measurements, not peak-FLOP or
assumed-MFU replay estimates. Quantitative models must be non-Llama releases from
July 2025 onward with public model dimensions and a compatible H100/H200 benchmark.

## Benchmark-backed result

Qwen3.5-35B-A3B was measured on one H200 SXM with concurrency 1, fresh context,
no prompt caching, no speculative decoding, and full-precision KV. Its published
TTFT rises from 77 ms at 1K to 12.4 s at 256K. Kimi-K2.5 publishes one exact
single-request point: 112 ms at 1K on eight H200s. Its page does not state the
cache policy, so the Kimi point has lower confidence.

| Model and context | Runnable state | Public TTFT | Crossover |
|---|---:|---:|---:|
| Qwen3.5-35B-A3B, 1K | 0.09 GB | 0.077 s | 8.92 Gbps |
| Qwen3.5-35B-A3B, 32K | 0.74 GB | 0.600 s | 9.81 Gbps |
| Qwen3.5-35B-A3B, 96K | 2.08 GB | 2.700 s | 6.16 Gbps |
| Qwen3.5-35B-A3B, 256K | 5.43 GB | 12.400 s | 3.51 Gbps |
| Kimi-K2.5, 1K | 0.07 GB | 0.112 s | 5.14 Gbps |

GB and Gbps are decimal. Crossover is state bits divided by measured TTFT. Above
the crossover bandwidth, transferring runnable state is faster; below it, replay
is faster. State sizes describe tensor payloads and exclude allocator padding and
transport overhead.

Qwen runnable state includes ten BF16 full-attention GQA caches, thirty FP32
Gated-DeltaNet recurrent matrices, and thirty BF16 convolution states. Kimi uses
61 BF16 MLA caches containing a 512-value latent and 64-value RoPE key per token.
These state sizes are derived from public configurations and reference cache
shapes; the TTFT values themselves are measured.

## Why the old GLM curve fell below one

The previous plot divided a low-confidence B200 FLOP-model replay estimate by
transfer time. At 100K and 25 Gbps it estimated about 0.79 s of replay but 1.44 s
of transfer, producing 0.55. That was arithmetic consistency, not benchmark
validation. An earlier version was also mislabeled: the GLM-named function plotted
Qwen3-235B. The replacement plot contains only published H200 replay points and is
named `benchmark_context_ratio_bandwidths`.

## Current model catalogue

The requested frontier models remain useful architecture records but do not enter
the ratio plot:

| Model | Release | Exclusion |
|---|---|---|
| Inkling NVFP4 | July 2026 | No public no-cache TTFT |
| GLM-5.2 | July 2026 | Public H200 TTFT workload has 90% KV hits |
| DeepSeek-V4-Pro | April 2026 | No comparable public TTFT |
| Kimi K3 | July 2026 | Weights and full configuration pending |
| Qwen3.7-Max | July 2026 | Closed dimensions and no comparable public TTFT |
| Nemotron 3 Ultra | April 2026 | No comparable public TTFT |

They are not assigned synthetic TTFTs. No Llama model is used.

## Confidence

- Public TTFT transcription and transfer arithmetic: high.
- Qwen3.5 state tensor geometry: medium-high; runtime allocation overhead is not
  included.
- Kimi-K2.5 state tensor geometry: medium-high; benchmark cache policy is unknown.
- Cross-model comparison: medium-low because one H200 and eight H200 deployments
  are different systems. The hardware is printed in every legend.
- Requested 2026 models' architecture-only entries: medium to high where configs
  are public; no performance claim is made.

## Sources

- [Qwen3.5 H200 benchmark and methodology](https://cdn.millstoneai.cloud/benchmarks/qwen3-5-35b-a3b-fp8-1x-h200-sxm/qwen3-5-35b-a3b-fp8-1x-h200-sxm.pdf)
- [Qwen3.5 configuration](https://huggingface.co/Qwen/Qwen3.5-35B-A3B-FP8/blob/main/config.json)
- [Qwen3.5 reference cache implementation](https://github.com/huggingface/transformers/blob/main/src/transformers/models/qwen3_5/modeling_qwen3_5.py)
- [Kimi-K2.5 H200 benchmark](https://www.luminal.com/report/moonshotai-kimi-k2-5-8xh200)
- [Kimi-K2.5 configuration](https://huggingface.co/moonshotai/Kimi-K2.5/blob/main/config.json)
- [GLM-5.2 cached H200 benchmark](https://docs.nvidia.com/dynamo/dev/recipes/glm-5-2)
- [Inkling architecture](https://thinkingmachines.ai/news/introducing-inkling/)
- [DeepSeek-V4-Pro configuration](https://huggingface.co/deepseek-ai/DeepSeek-V4-Pro/blob/main/inference/config.json)
- [Kimi K3 announcement](https://www.kimi.com/it-it/blog/kimi-k3)
- [Qwen model limits](https://docs.qwencloud.com/developer-guides/getting-started/text-generation-models)
- [Nemotron 3 Ultra configuration](https://huggingface.co/nvidia/NVIDIA-Nemotron-3-Ultra-550B-A55B-BF16/blob/main/config.json)

## Files

- `migration_ratio.py`: public benchmark data, state geometry, console table, and plots.
- `test_migration_ratio.py`: source-data, geometry, boundary, and invariant tests.
- `migration_ratio.{png,pdf}`: crossover bandwidth at each published TTFT point.
- `benchmark_context_ratio_bandwidths.{png,pdf}`: Qwen3.5 measured-context ratios.
- `prefill-breakeven.py` and its sweep artifacts: retained legacy H100/BF16 model.
