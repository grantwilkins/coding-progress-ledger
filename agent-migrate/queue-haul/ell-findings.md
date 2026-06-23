# ell and job power: review finding

This note answers the question we actually need for the paper: what should `ell`
mean, and how should we estimate the power effect of moving one job?

## Short answer

Keep `ell` as the "how full is the node?" number:

```
ell = prefill_tokens_per_second / prefill_rate(T)
    + decode_tokens_per_second / decode_rate
```

It is a capacity number, not a power meter. It tells us how much serving time a
job uses on a colocated node. That is the right input for placement, destination
admission, and the load-vs-memory regime test.

Use a separate power estimate for the work the job causes:

```
work_power = c_prefill(T) * prefill_tokens_per_second
           + c_decode * decode_tokens_per_second
```

The units are simple: joules per token times tokens per second gives watts.
Measured traces say decode costs more energy per token than prefill. That does
not mean decode has higher instant power. Prefill can draw high power for a
short burst; decode draws lower power for much longer.

For the job's effect on future node power, include both pieces:

```
future_power_impact = base_power_per_load * ell + work_power
```

The first term is the share of node base power that exists because the job takes
capacity. The second term is the token work. This is the clean separation:
`ell` prices the node that must exist, and token energies price the work done on
that node.

If we only use `c_prefill * f + c_decode * g`, we are estimating the job's
average work power, not the full power saved if the job helps drain a node.

## What was right

`ell = f/rho(T) + g/G` should stay.

The node is not split into separate prefill and decode pools. Both phases share
one wall-clock budget on the same node, so their busy fractions add. Long context
also makes prefill slower, so the `1/rho(T)` term is real capacity cost, not a
modeling artifact.

The guaranteed low-end power reduction should also stay tied to `ell`:

```
guaranteed_power = plateau_slope * ell
```

That is the power reduction we can claim even if the node stays on.

## What was wrong

The old two-price code treats prefill and decode as two made-up prices derived
from the single average load price:

```
old_expected = p_prefill * (f / rho(T)) + p_decode * (g / G)
```

with `p_decode = 5 * p_prefill`.

That fixed `5x` split is not measured. It also changes with context once written
as per-token energy:

```
old_decode_energy / old_prefill_energy = 5 * rho(T) / G
```

So the old model quietly makes the prefill-vs-decode ratio depend on context and
on the chosen decode rate. In the default workload, that implied ratio is often
far larger than the measured `5x` to `14x` per-token decode premium.

The direction was useful: decode-heavy jobs should usually have higher work
power. The calibration was not defensible.

## What the measurements support

Measured A100/H100 vLLM traces in `~/powertrace-sim/` support three plain claims.

1. A busy serving node reaches near-peak power early, then changes slowly with
   more load. This is why we keep a conservative `plateau_slope * ell` floor.
2. Over a whole request, decode costs more energy per token than prefill.
3. A single `c_prefill` value is an average over the measured trace mix. If the
   workload has very long contexts, prefill energy should be context-aware or
   clearly labeled as an average approximation.

The measured fits give useful analogs, not a final Qwen3-235B-A22B calibration:

| analog node | prefill J/token | decode J/token | decode / prefill |
|---|---:|---:|---:|
| gpt-oss-120b A100 TP8, MoE | 0.062 | 0.87 | 14.1 |
| llama-3-70b H100 TP8, dense | 0.148 | 1.76 | 11.9 |
| llama-3-405b H100 TP8, dense | 0.290 | 1.52 | 5.3 |

For Qwen3-235B-A22B, use these as labeled placeholders or sweep points until we
fit a real trace.

## Recommended model

Use three power numbers per job.

```
load_floor      = plateau_slope * ell
average_work    = c_prefill(T) * f + c_decode * g
future_impact   = base_power_per_load * ell + average_work
memory_impact   = memory_price * T / mean_T
```

Definitions:

- `f`: prefill tokens per second for this job.
- `g`: decode tokens per second for this job.
- `rho(T)`: prefill tokens per second at this job's context length.
- `G`: decode tokens per second for the node.
- `ell`: fraction of one node's serving time used by the job.

Pick `base_power_per_load` by calibration, not by guesswork. The simplest
calibration is:

```
base_power_per_load =
    total_node_power_per_load - average_work_power_per_load
```

where `total_node_power_per_load` is `P_busy / rho_star` and
`average_work_power_per_load` is measured on the same trace or on the synthetic
workload being used for the experiment. Clamp it at zero if an analog trace
overestimates the target node.

If no Qwen trace exists yet:

1. Use the measured decode/prefill ratio for ranking.
2. Scale or calibrate the absolute values so the whole workload matches
   `P_busy / rho_star`.
3. Label the result as a Qwen placeholder and sweep the ratio.

## Code status

The formulation text already points in the right direction, but the code still
uses the old `p_prefill` / `p_decode` split.

Needed code changes:

| file | change |
|---|---|
| `power.py` | replace `phase_ratio`, `p_pre`, and `p_dec` with measured token-energy fields and a calibrated base-load field |
| `instance.py` | store raw `f` and `g` alongside `ell_pre` and `ell_dec` |
| `impact.py` | compute `average_work` from `f` and `g`; compute `future_impact` as base-load plus work |
| tests | assert that `ell` controls capacity while token energies control average work power |

The paper-facing wording should be:

> `ell` is how much node time a job consumes. It is the right number for capacity
> and for the guaranteed power floor. The job's average work power is measured
> from prefill and decode token rates. Its future node-power impact is the sum of
> that work power and the job's share of the node base power.
