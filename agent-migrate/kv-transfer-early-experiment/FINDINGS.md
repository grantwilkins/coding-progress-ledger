# KV transfer vs prompt replay

Question: when moving an in-flight request to another site, is it faster to ship
the KV cache or replay the prompt/context?

## Setup

- Hardware: one instance is 8x H100 SXM, dense bf16 peak = 7.92 PFLOP/s, 640 GB HBM.
- Assumed MFU: 35%, so one instance sustains 2.77 PFLOP/s of prefill.
- Models too big for one instance get the fewest that hold FP8 weights + 100 GB
  headroom, and prefill sees all of them. Instance count scales replay only; KV
  bytes are per-token and do not move.
- KV state is bf16. Quantized KV and allocator padding are not modeled.
- Replay time is recomputed from prefill FLOPs at each context length; it is not a
  fixed tokens/sec benchmark. Attention counts the pairs each layout actually
  computes, including top-k, sliding-window, and hybrid linear/full layouts.
- Reference table below is at 100,000 tokens.

## Result

Above the crossover bandwidth, ship KV. Below it, replay.

| Model | Nodes | KV @ 100k | Replay | Prefill | Crossover |
|---|---:|---:|---:|---:|---:|
| DeepSeek-V4-Pro | 3 | 0.99 GB | 1.34 s | 74.8k tok/s | 5.94 Gbps |
| Gemma-4-26B-A4B | 1 | 1.23 GB | 0.59 s | 170.9k tok/s | 16.87 Gbps |
| gpt-oss-20b | 1 | 2.46 GB | 0.62 s | 162.3k tok/s | 31.94 Gbps |
| Qwen3.8-27B | 1 | 6.55 GB | 2.66 s | 37.6k tok/s | 19.72 Gbps |
| Kimi-K2.6 | 2 | 7.03 GB | 3.41 s | 29.3k tok/s | 16.49 Gbps |
| GLM-5 | 2 | 8.99 GB | 1.63 s | 61.3k tok/s | 44.08 Gbps |

## Takeaways

- Crossovers are modest: about 6-44 Gbps at 100k tokens, not 100+ Gbps.
- GLM-5 is no longer a heavy full-MHA outlier. The corrected model treats it as
  MLA + DSA compressed KV.
- DeepSeek-V4-Pro still has the lowest crossover: CSA/HCA keeps KV small enough
  that transfer usually wins, even though 3 instances make its replay fast.
- GLM-5 has the highest. Its DSA-capped prefill is cheap and 2 instances make it
  cheaper, so the link has to be very fast before shipping 8.99 GB pays off.
- The three added models cross at 16.87--31.94 Gbps. Their local or linear
  layers keep migration state small, while their remaining global layers retain
  the quadratic replay term.
- Context length only matters where attention is still quadratic. Dense and
  hybrid-global layers keep the T^2 replay term, so long contexts favour KV
  transfer. GLM-5's DSA caps every query at 2048 entries, so above ~2k tokens
  its replay is linear and its decision no longer depends on context length.
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
