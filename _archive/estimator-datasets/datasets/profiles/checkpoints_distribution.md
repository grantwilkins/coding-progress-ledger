# Checkpoint-distribution profile (F2)

Per canonical source: how checkpoints are distributed across progress / elapsed-fraction buckets, leaf-count quantiles, and validation/blocked state rates. A heavily skewed distribution (e.g. >80% of checkpoints at progress=0) flags an evaluation regime where most baselines look good for the wrong reason.

### Coding progress buckets

| source | n | [0.0, 0.25) | [0.25, 0.5) | [0.5, 0.75) | [0.75, 1.0] |
| --- | --- | --- | --- | --- | --- |
| hermes_pilot_h5_v2 | 896 | 255 | 19 | 266 | 356 |
| swe_agent_pilot | 599 | 192 | 3 | 259 | 145 |
| tb_live | 83 | 24 | 3 | 23 | 33 |

### Elapsed-fraction buckets (tb_live only)

| source | n | [0.0, 0.25) | [0.25, 0.5) | [0.5, 0.75) | [0.75, 1.0] |
| --- | --- | --- | --- | --- | --- |
| hermes_pilot_h5_v2 | 896 | 0 | 0 | 0 | 0 |
| swe_agent_pilot | 599 | 0 | 0 | 0 | 0 |
| tb_live | 83 | 0 | 0 | 0 | 0 |

### Leaf-count quantiles

| source | p25 | p50 | p75 |
| --- | --- | --- | --- |
| hermes_pilot_h5_v2 | 1.00 | 3.00 | 5.00 |
| swe_agent_pilot | 1.00 | 2.00 | 3.00 |
| tb_live | 1.00 | 2.00 | 4.00 |

### Validation + blocked rates

| source | validation_started | validation_complete | any_blocked |
| --- | --- | --- | --- |
| hermes_pilot_h5_v2 | 0.68 | 0.57 | 0.05 |
| swe_agent_pilot | 0.15 | 0.08 | 0.01 |
| tb_live | 0.43 | 0.29 | 0.00 |

