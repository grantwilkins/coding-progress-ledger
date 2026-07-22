# KV state transfer vs. context replay

`migration_ratio.py` compares runnable-state transfer time with a transparent
prefill-compute estimate. It does not claim to predict end-to-end TTFT, which also
depends on queueing, tokenization, runtime, parallelism, cache hits, and first-token
decode.

## Reference and result

The replay estimate uses one 8x B200 system's 36 PFLOP/s dense FP8 peak at 35%
utilization: 12.6 PFLOP/s effective. The old 8x H100 BF16 arithmetic was correct
(2.77 PFLOP/s effective), but those GPUs cannot run the current NVFP4 checkpoints
natively and the larger BF16 checkpoints do not fit one node.

At 100,000 text tokens:

| Model | Migrated state | Modeled replay | Crossover |
|---|---:|---:|---:|
| Inkling NVFP4 | 4.74 GB | 0.80 s | 47.30 Gbps |
| GLM-5.2 | 4.49 GB | 0.79 s | 45.76 Gbps |
| DeepSeek-V4-Pro | 0.45 GB | 0.88 s | 4.07 Gbps |
| Nemotron 3 Ultra NVFP4 | 0.82 GB | 1.03 s | 6.39 Gbps |
| Kimi K3 | undisclosed | not modeled | not modeled |
| Qwen3.7-Max | undisclosed | not modeled | not modeled |

GB is decimal. State sizes exclude allocator/block padding and transfer-protocol
overhead. The crossover is state bits divided by modeled replay seconds.

## Number audit

- The supplied `glm5_context_ratio_bandwidths` figure was actually generated from
  `Qwen3 235B`. It now uses GLM-5.2 and is capped at its public 1M context window.
- FP4 describes checkpoint weights, not KV precision. The modeled serving formats
  are Inkling BF16 KV; GLM-5.2 FP8 MLA; DeepSeek FP8 shared KV plus FP4 indexer;
  and Nemotron FP8 KV plus FP16 recurrent state.
- Inkling has 66 layers: 11 global layers with 8 KV heads and 55 sliding-window
  layers with 16 KV heads and a 512-token retained window.
- GLM-5.2 has 78 MLA/DSA layers. Each token stores a 512-value latent and 64-value
  RoPE key per layer. IndexShare runs 21 indexers for 78 top-2048 attention layers;
  it reduces compute, not the MLA cache width.
- DeepSeek-V4-Pro has 30 c4a and 31 c128a layers, a 128-token local window,
  512-value shared cache entries, and a 128-value c4a index cache. The prior code
  charged BF16 for every component; current Blackwell recipes use FP8 KV and FP4
  index state.
- Nemotron 3 Ultra has 12 attention and 48 Mamba layers. Migration includes its
  fixed recurrent and convolution state; counting attention KV alone is incomplete.
  Its config is native to 262,144 tokens; NVIDIA permits 1M extrapolation with a
  workload-specific quality warning.
- Kimi K3's public announcement says 2.8T total parameters, KDA/MLA, 896 experts
  with 16 routed, and 1M context, but says the weights and technical report arrive
  July 27, 2026. Active parameters, layer dimensions, and cache layout are not yet
  public.
- Qwen3.7-Max is a closed API model. Qwen publishes its 1M input limit but not the
  layer/head/cache dimensions needed for this calculation.

## TTFT evidence

Published TTFT values are not interchangeable with the zero-cache replay estimate.
For example, NVIDIA reports GLM-5.2 p50 TTFT of 356 ms on an aggregated B200 target
and 1.94 s on a disaggregated target, but both use a 64K-median agent trace with 90%
KV hits and concurrency 64/128. Inkling's current NVIDIA recipe explicitly publishes
no benchmark. The other requested models do not publish comparable no-hit,
single-request, 100K TTFT measurements, so no measured TTFT was substituted into the
curves.

## Primary sources

- [Inkling announcement and architecture](https://thinkingmachines.ai/news/introducing-inkling/)
- [Inkling configuration](https://huggingface.co/thinkingmachines/Inkling/blob/main/config.json)
- [GLM-5.2 configuration](https://huggingface.co/zai-org/GLM-5.2/blob/main/config.json)
- [GLM-5.2 IndexShare explanation](https://huggingface.co/blog/zai-org/glm-52-blog)
- [DeepSeek-V4-Pro configuration](https://huggingface.co/deepseek-ai/DeepSeek-V4-Pro/blob/main/inference/config.json)
- [DeepSeek V4 cache derivation](https://vllm.ai/blog/2026-04-24-deepseek-v4)
- [Kimi K3 announcement](https://www.kimi.com/it-it/blog/kimi-k3)
- [Qwen model limits](https://docs.qwencloud.com/developer-guides/getting-started/text-generation-models)
- [Nemotron 3 Ultra configuration](https://huggingface.co/nvidia/NVIDIA-Nemotron-3-Ultra-550B-A55B-BF16/blob/main/config.json)
- [Nemotron 3 Ultra technical report](https://research.nvidia.com/labs/nemotron/files/NVIDIA-Nemotron-3-Ultra-Technical-Report.pdf)
- [NVIDIA GLM-5.2 benchmark](https://docs.nvidia.com/dynamo/dev/recipes/glm-5-2)
- [NVIDIA Inkling deployment note](https://docs.nvidia.com/dynamo/recipes/inkling)
- [DGX B200 specifications](https://www.nvidia.com/en-au/data-center/dgx-b200/)

## Files

- `migration_ratio.py`: audited model, console summary, and plots.
- `test_migration_ratio.py`: hand-derived unit, precision, boundary, and invariant tests.
- `migration_ratio.{png,pdf}`: modeled ratio across bandwidth at 100K tokens.
- `glm5_context_ratio_bandwidths.{png,pdf}`: GLM-5.2 ratio across context and bandwidth.
- `prefill-breakeven.py` and its sweep artifacts are the older H100/BF16 experiment;
  they are retained for provenance and are not the current model catalogue.
