# KV-transfer vs context-replay — findings

**Question:** when migrating an in-flight LLM request between sites, ship the KV cache or ship the prompt and re-prefill?

## Reference setup (single source of truth)

- **Hardware:** 8× NVIDIA H100 SXM5, bf16 — peak **7.91 PFLOPS**
- **Engine:** vLLM with chunked prefill, **MFU = 35%** — effective **2.77 PFLOPS**
- **Reference context:** 100,000 tokens (where prefill rate is evaluated)
- **No vendor benchmark numbers used** — every prefill rate derived from architecture.

## Math

```
prefill_FLOPs/token(T) = 2·active_params              (dense forward)
                       + 4·L_softmax·n_q·head_dim·T   (attention, MAC×2)
prefill_tok_s          = effective_FLOPS / prefill_FLOPs/token(T_ref)

t_transmit = T·kv_bpt·8 / bw_bps
t_replay   = T / prefill_tok_s
crossover  = 8·kv_bpt·prefill_tok_s     (T cancels)
```

Above the crossover bandwidth: ship the KV. Below: replay.

## Models — every number derived consistently

| Model | active | KV B/tok | dense FLOP | attn FLOP | Prefill | Crossover |
|---|---:|---:|---:|---:|---:|---:|
| **GLM-5** | 40 B | 1248 KB | 0.08 TF | 0.13 TF | 13.3 k tok/s | **136 Gbps** |
| Qwen3-Next-80B-A3B | 3.9 B | 24 KB | 0.01 TF | 0.02 TF | 100.8 k tok/s | 20 Gbps |
| Qwen3-235B-A22B | 22 B | 188 KB | 0.04 TF | 0.31 TF | 7.9 k tok/s | 12 Gbps |
| Qwen3.5-397B-A17B | 17 B | 30 KB | 0.03 TF | 0.05 TF | 33.3 k tok/s | 8 Gbps |
| Kimi-K2.6 | 32 B | 69 KB | 0.06 TF | 0.30 TF | 7.6 k tok/s | 4 Gbps |
| **DeepSeek-V4-Pro** | 49 B | 69 KB | 0.10 TF | 1.60 TF | 1.6 k tok/s | **0.9 Gbps** |

(FLOPs columns are per-token, evaluated at T_ref = 100,000.)

## Headline finding

The crossover bandwidth spans **>150×** across the catalog (0.9 → 136 Gbps), driven by two architectural levers:

1. **KV-bytes-per-token** — full MHA (GLM-5, 1.25 MB) is **52× heavier** than hybrid (Qwen3-Next, 24 KB).
2. **Attention compute** — DeepSeek-V4-Pro's `head_dim=512` makes its attention term `4·L·n_q·hd·T` enormous (1.60 TF/token vs Qwen3-Next's 0.02 TF), which crashes its prefill rate and pulls its crossover to <1 Gbps.

These two effects pull in **opposite directions**:
- **GLM-5** has heavy KV but cheap attention → replay wins until 136 Gbps. Replay is the right answer for cross-region (~25 Gbps) and most inter-DC links.
- **DeepSeek-V4-Pro** has light KV (MLA) but expensive attention → KV-transfer wins above 1 Gbps. Almost any link favors shipping the KV.

## Plot reading

**`transmission_time.png`** — six small panels, one per model, fixed at 1 M token context. Blue = KV-transfer time vs bandwidth. Red dashed = replay time (constant). Green shading = ship KV; orange = re-prefill. The panel titles give the crossover bandwidth and replay time directly.

**`crossover_bandwidth.png`** — single bar chart sorted by crossover bandwidth. Each bar annotated with kv/tok, derived prefill rate, and architecture note. Vertical reference lines at 10 G / 100 G / 400 G / 1 T link tiers.

## When the hypothesis (replay wins) holds

- **GLM-5**: any link below ~100 Gbps. Full MHA punishes KV-transfer hard.
- **Qwen3-Next**, **Qwen3-235B**, **Qwen3.5-397B**: only sub-20 Gbps links (rare for inter-DC).
- **Kimi-K2.6**: only sub-5 Gbps (mobile / WAN edge).
- **DeepSeek-V4-Pro**: essentially never — its expensive attention compute makes replay the slow option even at 1 Gbps.

## Caveats explicitly priced in

- **MFU = 35%** is one number; ±10 percentage points moves crossovers by ~30%. Doesn't change ordering.
- **Single fixed prefill rate at T_ref = 100k**: at 1 M context, real prefill rate would be lower (attention is T²) so replay actually loses by more than shown — current numbers are the *generous* case for replay.
- **MLA attention compute** is approximated as the equivalent dense-MHA FLOPs at `n_q × head_dim`. The compressed-latent compute is similar; the saving is in cache size.
- **Hybrid models** count only softmax-attention layers in the attention term; the Gated-DeltaNet branches contribute a small constant cost ignored here.

## Files

- `prefill-breakeven.py` — full simulation (~200 lines)
- `migration.csv` — 6 × 4 × 200 sweep
- `transmission_time.{png,pdf}` — small multiples
- `crossover_bandwidth.{png,pdf}` — single-number summary
