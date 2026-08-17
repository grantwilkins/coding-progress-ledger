# Pooled A100 agentic RPS sweep with P90 TPOT

This schema-v3 evidence re-reduces the retained token timestamps from the
Sweden Central, East US 2, and Germany West Central A100 runs. It does not
rerun inference. TPOT is the P90 over every exact post-first-token interval in
a cell, pooled across its fixed-length requests. The CSVs export this metric as
`p90_tpot_s` and omit the previous per-request-mean percentile.

| Model | Last TTFT-SLO-met rate | First confirmed violation | Maximum P90 TPOT | TPOT SLO |
|---|---:|---:|---:|---:|
| GPT-OSS-20B | 5 RPS: 1.463 s | 6 RPS: 2.968 s TTFT | 0.0332 s | 0.1000 s |
| Qwen3.8-27B | 0.6 RPS: 3.916 s | 0.7 RPS: 9.843 s TTFT | 0.0888 s | 0.1638 s |
| Gemma-4-26B-A4B | 4 RPS: 1.868 s | 5 RPS: 2.638 s TTFT | 0.0595 s | 0.2000 s |

No model violates its TPOT SLO. All first confirmed violations remain TTFT
violations. The primary curve contains 45 selected cells; every cell completed
32/32 requests with zero failures. It contains 1,468,005 exact TPOT samples,
with 30,690 to 32,736 samples per cell. Four cells have exact timestamps for
30 or 31 of their 32 completed requests; TPOT excludes only those inexact
requests.

## Provenance

All 59 timestamp-bearing historical cells were re-reduced. The primary CSV
selects the 45 discovery and current-boundary cells used by the figure; the
all-cells CSV also retains 14 superseded coarse-boundary repeats.

| Source label | Raw run root | Primary rows |
|---|---|---:|
| `swedencentral` | `/datadrive/qh-agentic-rps-sweep-a100-dense-20260817-sweden/cells` | 23 |
| `eastus2` | `/datadrive/qh-agentic-rps-sweep-a100-20260816-eastus2-v3/cells` | 7 |
| `germanywestcentral` | `/datadrive/qh-agentic-rps-sweep-a100-dense-20260817-germany/cells` | 15 |

Every derived row records the source label, source schema and plan hash, raw
root, source-result SHA-256, and source-request SHA-256. The schema-v3 plan
object hash is
`362af5ca07b59f83b9ac5f610f4f225c7539b89cf7bfa98a3d8e44e250dedcf7`.

## Artifacts

- `rps-sweep.csv`: 45 figure inputs plus header.
- `all-rereduced-cells.csv`: all 59 re-reduced cells plus header.
- `summary.json`: reduced curves, SLOs, boundary repeats, and source hashes.
- `agentic-rps-sweep.pdf` and `.png`: compact stacked GPT-OSS-20B TTFT and pooled-P90 TPOT panels for the OpenHands Agentic workload.
- `plan.json`: immutable schema-v3 plan.

SHA-256:

- `plan.json`: `20d644adee121f07ce9f146fd96657278df15312a5074a4cd172604fb61ec293`
- `summary.json`: `7b87cfa93c5ce3e1360c784685a79673964cb6726afd024f869711c4d9d88494`
- `rps-sweep.csv`: `004ea5dbc9e8b32382ff1811347b8ff1a42c225426ff49aee7a6746324fd78de`
- `all-rereduced-cells.csv`: `694289b8bc62b1692bed0cb526cac6e33b9819eda8363f22e4a448d02fbdeec5`
- `agentic-rps-sweep.pdf`: `41aabbd007b6cd9a0c0035da736a4b532c59251ce492a6efd9a8a915b52ec435`
- `agentic-rps-sweep.png`: `47f2f01a858bae283b1c619ed1100d79bef288717525fcd1d12e5779eafcc698`
