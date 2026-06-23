# Turn, Token, and Context Corrections

Checked June 23, 2026. This note updates the queue-haul workload model for
modern interactive, reasoning, and agentic serving.

## Decision

Keep the model:

```text
f   = active * turn_rate * Delta
g   = active * turn_rate * Y
ell = f / rho(T) + g / G
m   = eta * T
```

but tighten the meanings:

| symbol | meaning |
|---|---|
| `turn_rate` | model calls per second while a session is active |
| `Delta` | new input tokens since the cached prefix |
| `Y` | all generated tokens: answer, hidden reasoning, and tool-call text |
| `T` | resident context already held for the session |

The important correction is cache-aware prefill. In a multi-turn agent, `Delta`
is not the full prompt. Most prior context should hit the KV cache, so the
per-turn prefill work is only the newly appended text unless the cache was
evicted.

## Source Values

| source | values we should use |
|---|---|
| [LMSYS-Chat-1M, 2023](https://arxiv.org/abs/2309.11998) | 1M real conversations, 210K users, average 2.0 turns, average 69.5 prompt tokens and 214.5 response tokens per turn. Good lower anchor for ordinary chat, not agentic work. |
| [BurstGPT, 2024](https://arxiv.org/abs/2401.17644) | Real GPT service traces. Older version reports campus traces over two months, ChatGPT conversation service around 0.019 RPS and API service around 0.21 RPS; request lengths are short-heavy and response lengths vary by model/service. Use as a reminder that aggregate service RPS is not per-session `turn_rate`. |
| [ServeGen, 2025](https://arxiv.org/abs/2505.09999) | 3.54B production requests, 12 models, 4 months. Input lengths fit Pareto+lognormal mixtures, output lengths fit exponential distributions, input/output correlation is weak, and both shift independently over time. |
| ServeGen reasoning workload | DeepSeek-R1 reasoning outputs are longer and more variable than normal outputs. Reason tokens can exceed answer tokens, and reason-answer ratios are bimodal. In one 12-hour window: 188,986 multi-turn requests out of 1,964,415 total, forming 57,205 conversations; average conversation length 3.5; inter-turn times concentrate around 100 s with a long tail. |
| [Agentic AI Workload Characteristics, 2026](https://arxiv.org/abs/2605.26297) | Agent tasks average 12-62 turns in most configurations, with long tails up to hundreds of turns. SWE-bench Pro mean context is 68.7K-80.1K tokens, with maxima 146K-166K. Qwen Terminal Bench and GAIA reach mean contexts of 63.5K-65.1K and 52.5K-54.5K. |
| Agentic AI Workload Characteristics, cache result | Prefix/cache hit ratios are 84.6-99.5% empirically. Raw input/output ratios are 53.9-559.8, but append/output ratios are only 1.5-7.3, often below 1.5 median. Decode is 91.0-98.6% of LLM time when cache state stays resident. |
| Agentic AI Workload Characteristics, output makeup | Thinking tokens are 45.8-67.6% of Gemma Thinking output and 29.0-40.7% of Qwen Thinking output. Tool-call text dominates many instant-agent outputs: 87.8-98.2% for Gemma Instant and 70.4-81.6% for Qwen Instant. |
| [Continuum, 2025/2026](https://arxiv.org/abs/2511.02230) | Agent workflows interleave model calls and tool calls. Tool pauses can cause KV eviction; retaining KV cache across turns is central to multi-turn agent performance. |

## Current Queue-Haul Values

From `instance.py`:

| quantity | current value | interpretation |
|---|---:|---|
| active / idle / cold | 0.30 / 0.25 / 0.45 | reasonable as an evacuation snapshot, not a trace-wide traffic mix |
| agentic fraction | 0.50 | reasonable sweep knob |
| reasoning fraction within agentic | 0.30 | reasonable sweep knob |
| agentic `turn_rate` | 0.15/s | one model call every 6.7 s |
| chat `turn_rate` | 0.02/s | one model call every 50 s |
| agentic `Delta` | LogN(8.0, 1.0) | median 2,981 tokens, mean 4,915 tokens |
| chat `Delta` | LogN(5.5, 1.0) | median 245 tokens, mean 403 tokens |
| agentic `Y` | geometric mean 600 | generated tokens per call |
| chat `Y` | geometric mean 800 | high for ordinary chat; plausible for long chat/code |
| reasoning `Y` | geometric mean 4,000 | stress value for hidden-reasoning output |
| center `T` | mixture mean 66,058 tokens | plausible for agentic, too high for ordinary chat |
| short sweep `T` | mixture mean 17,828 tokens | long for ordinary chat, useful for short-context stress |
| load-bound fixture `T` | mean 3,378 tokens | good ordinary-chat/load-bound fixture |
| long sweep `T` | mixture mean 151,946 tokens | aggressive long-agent / memory-pressure case |

Two doc fixes:

1. `Delta` values above are lognormal medians in prose if written as "~3k" and
   "~250"; the means are 4.9K and 403.
2. NumPy's geometric draw has mean `1/p`, so `np.random.geometric(1 / y_mean)`
   has mean `y_mean`, not `(1-p)/p`.

## Corrected Workload Classes

Use class-specific defaults. One shared `T` distribution for chat and agents is
not defensible.

| class | `turn_rate` | `Delta` | `Y` | `T` | status |
|---|---:|---:|---:|---:|---|
| ordinary chat | 0.005-0.02/s | median 70-300, mean 200-500 | mean 200-400 | mean 1K-4K | source-backed by LMSYS and useful as load-bound baseline |
| long chat / code help | 0.005-0.02/s | median 250-1K, tail to several K | mean 500-1.5K | mean 4K-20K | production-relevant, should be swept |
| reasoning chat | 0.005-0.015/s | median 100-1K | mean 1K-4K, bimodal tail | mean 4K-30K | ServeGen says inter-turn times center near 100 s; outputs are longer and more variable |
| agentic tool loop | 0.05-0.30/s | append/output ratio 1.5-7.3 | mean 400-1.5K non-reasoning, 1K-4K reasoning | mean 50K-80K, tail 150K+ | modern agent setting; cache-aware `Delta` is required |
| failed/retry agent tail | 0.05-0.30/s | error/tool-output heavy | same or higher than agentic | mean can increase up to about 1.8x | include as stress case because failed agents cost real serving resources |
| stateless API/batch | do not use per-session `turn_rate` | request input length | request output length | no durable session unless API maintains history | model as aggregate arrivals, not active-session turns |

Recommended queue-haul center:

```text
chat:
  turn_rate = 0.01/s
  Delta median = 150 tokens, mean = 250-350 tokens
  Y mean = 300 tokens
  T mean = 3K-4K tokens

agentic:
  turn_rate = 0.15/s
  Delta mean = append_to_output_ratio * Y, ratio swept 1.5-7.3
  Y mean = 600 tokens non-reasoning, 4,000 tokens reasoning stress
  T mean = 66K tokens, tail to 150K+

reasoning chat:
  turn_rate = 0.01/s
  Delta median = 250-1K tokens
  Y mean = 2K-4K tokens
  T mean = 10K-30K tokens
```

The current center should be described as an agentic long-context center, not as
a generic mixed chat workload.

## Formulation Wording

Use this in `formulation.md`:

> A turn is one model re-entry while a session is active. `Delta` is the new
> text appended since the cached prefix, not the full context. `T` is the
> resident context already held for the session. `Y` is the number of generated
> tokens, including hidden reasoning and tool-call text. Thus `f = turn_rate *
> Delta` and `g = turn_rate * Y` are token rates, while `ell = f/rho(T) + g/G`
> is the fraction of one serving node's time used by the job.

Add the cache caveat:

> If the KV cache is retained, prefill handles only `Delta`. If the cache is
> evicted or the session is replayed cold, prefill handles the full context `T`.

## Code Changes To Make Later

1. Store `rate`, `Delta`, `Y`, `f`, and `g` in `JobPopulation`; they are now
   discarded after `ell_pre` and `ell_dec` are formed.
2. Split `T` by class: ordinary chat should not draw from the same context
   distribution as agents.
3. Replace `delta_agentic=(8.0,1.0)` style fields with explicit median/mean
   helpers or document them as log-space parameters.
4. Add tests for sampled `turn_rate`, `Delta`, `Y`, and `T` marginals.
5. Add a cache-hit knob for agentic workloads. With hit rate near 1, prefill
   uses `Delta`; with eviction, prefill uses `T`.
