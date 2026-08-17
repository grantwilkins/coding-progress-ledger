# Completed A100 agentic RPS sweep

The reduced evidence combines one model shard from each A100 node:

| Region | Model | Raw run root |
|---|---|---|
| Sweden | `openai/gpt-oss-20b` | `/datadrive/qh-agentic-rps-sweep-a100-20260816-sweden-v2` |
| East US 2 | `Qwen/Qwen3.8-27B` | `/datadrive/qh-agentic-rps-sweep-a100-20260816-eastus2-v3` |
| Germany | `google/gemma-4-26B-A4B-it` | `/datadrive/qh-agentic-rps-sweep-a100-20260816-germany-v2` |

All shards use the same plan file SHA-256
`76853a89fc7e5e7a52ceeb01ff632903a67b2c831db131cc55bb4283814b9634`
and per-result plan object hash
`4709014a6cbaa32104531be1c9e0482094a4f3ac6d155fb44d015f13473b67ed`.
The campaign source SHA-256 on all three nodes was
`2aac7aacc96861573c756c26f9d80fb9665203218a1cf8cf759d47f53f5294f0`.

The first confirmed P90 latency violation is 8 RPS for GPT-OSS, 1 RPS for
Qwen, and 8 RPS for Gemma. Every result used by the reducer completed all
32 requests with zero failures. Gemma uses the corrected fixed SLOs of 2.0 s
TTFT and 0.2 s mean TPOT. Its earlier 2-RPS repeats from the superseded
relative-SLO selection remain in the Germany raw root but are intentionally
excluded from the reduced fixed-SLO boundary.

Artifacts:

- `summary.json`: reduced curves, SLOs, boundary selections, and medians.
- `rps-sweep.csv`: per-cell measurement records.
- `agentic-rps-sweep.pdf` and `.png`: aligned TTFT/TPOT paper figure.
