# KV transfer vs prompt replay

Question: when moving an in-flight request to another site, is it faster to ship
the KV cache or replay the prompt/context?

## Setup

- Hardware: 8x H100 SXM, dense bf16 peak = 7.92 PFLOP/s.
- Assumed MFU: 35%, so effective prefill compute = 2.77 PFLOP/s.
- KV state is bf16. Quantized KV and allocator padding are not modeled.
- Replay time is recomputed from prefill FLOPs at each context length; it is not a
  fixed tokens/sec benchmark. Attention counts the pairs each layout actually
  computes, including DSA top-k (GLM-5 k=2048, DeepSeek CSA k=1024).
- Reference table below is at 100,000 tokens.

## Result

Above the crossover bandwidth, ship KV. Below it, replay.

| Model | KV @ 100k | Replay | Prefill | Crossover |
|---|---:|---:|---:|---:|
| DeepSeek-V4-Pro | 0.99 GB | 4.01 s | 24.9k tok/s | 1.98 Gbps |
| Kimi-K2.6 | 7.03 GB | 6.82 s | 14.7k tok/s | 8.24 Gbps |
| GLM-5 | 8.99 GB | 3.26 s | 30.7k tok/s | 22.04 Gbps |
| Qwen3-235B-A22B | 19.25 GB | 7.15 s | 14.0k tok/s | 21.55 Gbps |
| Qwen3.5-397B-A17B | 3.07 GB | 2.11 s | 47.3k tok/s | 11.62 Gbps |
| Qwen3-Next-80B-A3B | 2.46 GB | 0.57 s | 175.0k tok/s | 34.41 Gbps |

## Takeaways

- Crossovers are modest: about 2-34 Gbps at 100k tokens, not 100+ Gbps.
- GLM-5 is no longer a heavy full-MHA outlier. The corrected model treats it as
  MLA + DSA compressed KV.
- DeepSeek-V4-Pro has the lowest crossover because CSA/HCA keeps KV small enough
  that transfer usually wins.
- Qwen3-Next has the highest crossover because replay is extremely fast.
- Context length only matters where attention is still quadratic. Dense models
  keep the T^2 replay term, so long contexts favour KV transfer. GLM-5's DSA
  caps every query at 2048 entries, so above ~2k tokens its replay is linear and
  its decision no longer depends on context length at all.
- GLM-5 and Qwen3-235B now cross over within 2% of each other (22.04 vs 21.55
  Gbps), so their curves overlap in `migration_ratio.png`.
- `deepseekv4_context_ratio_bandwidths.{png,pdf}` shows the DeepSeek-V4-Pro ratio
  from 1k to 10M tokens across 500 linear-spaced 0.1-25 Gbps links. Its HCA
  layers have no top-k, so the quadratic term survives and the boundary still
  moves with context.

## Files

- `prefill-breakeven.py`: main corrected sweep and plots.
- `migration_sweep_corrected.csv`: generated sweep data.
- `migration_times.{png,pdf}`: transfer vs replay time by model.
- `crossover_bandwidth.{png,pdf}`: crossover summary.
- `migration_ratio.py`: ratio plots for the bandwidth and context sweeps.
- `deepseekv4_context_ratio_bandwidths.{png,pdf}`: DeepSeek-V4-Pro replay/transfer
  ratio vs context size across fixed bandwidth lines.
- `appendix.{tex,bib}`: writeup appendix for `migration_ratio.py` — cost model,
  per-model architecture data with sources, and assumptions. `\input`-able;
  compile standalone with a minimal preamble.
