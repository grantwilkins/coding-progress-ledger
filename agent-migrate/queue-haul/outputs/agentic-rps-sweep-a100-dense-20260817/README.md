# Completed dense A100 agentic RPS sweep

This evidence extends the original power-of-two discovery sweep with
predeclared local refinement points. All points use one A100, 3,920 prompt
tokens, exactly 1,024 output tokens, 32 requests, seeded open-loop Poisson
arrivals, and no concurrency cap or SLO control-flow gate.

| Model | Added discovery rates (RPS) | Last SLO-met point | First confirmed TTFT violation |
|---|---:|---:|---:|
| GPT-OSS-20B | 3, 5, 6, 7 | 5 RPS: 1.463 s | 6 RPS: 2.968 s |
| Qwen3.8-27B | 0.6, 0.7, 0.8, 0.9 | 0.6 RPS: 3.916 s | 0.7 RPS: 9.843 s |
| Gemma-4-26B-A4B | 3, 5, 6, 7 | 4 RPS: 1.868 s | 5 RPS: 2.638 s |

GPT-OSS and Gemma use fixed 2.0 s TTFT SLOs. Qwen's TTFT SLO is
6.9649 s (twice its 0.125-RPS baseline). The TPOT SLOs are 0.1 s for
GPT-OSS, 0.1894 s for Qwen, and 0.2 s for Gemma. No model violated its TPOT
SLO in this sweep.

The reduced summary contains 45 selected cells. Every cell completed 32/32
requests with zero failures. The dense extension added 24 new cells (768
requests); the remaining cells are provenance-linked measurements from the
immutable parent sweep. Repeated dense boundary points use the median for the
curve and min-max for the whiskers.

## Node provenance

| Measurement | Region | Raw run root |
|---|---|---|
| GPT-OSS parent + dense cells | Sweden Central | `/datadrive/qh-agentic-rps-sweep-a100-dense-20260817-sweden` |
| Qwen parent cells | East US 2 | `/datadrive/qh-agentic-rps-sweep-a100-20260816-eastus2-v3` |
| Qwen dense cells | Sweden Central | `/datadrive/qh-agentic-rps-sweep-a100-dense-20260817-sweden` |
| Gemma parent + dense cells | Germany West Central | `/datadrive/qh-agentic-rps-sweep-a100-dense-20260817-germany` |

East US 2 was unreachable during the extension (repeated SSH timeouts), so
the four new Qwen rates ran on the idle Sweden A100 where the same pinned
revision was already cached. Both sides of Qwen's refined 0.6-to-0.7 boundary
therefore come from the same Sweden node. The original East result records
were imported from the committed reduced summary; their raw request and
engine traces remain at the East raw root above.

## Immutable identities

- Dense plan file SHA-256: `427eee6cb7c9cd7fbac6c631c5e7e8646a21ac34e518b3b52b648078552d6b51`.
- Dense plan object hash: `194ad7d6e376e903fb7ce3db7f40df925942f8cb21b91e2d6fb890a39825512d`.
- Parent plan object hash: `4709014a6cbaa32104531be1c9e0482094a4f3ac6d155fb44d015f13473b67ed`.
- Frozen campaign source SHA-256: `1cb324d5ad6d398380788f14cd86de8ba35c65530c45d72410d1f3c8ba99e931`.
- Sweden reduced-input bundle SHA-256: `d18e7d1f8fe6ac25a69a4701dc412e448b15645c30e632d0894fb51bd1460110`.
- Germany reduced-input bundle SHA-256: `944e6c0d43dbcad0cfe81a91ddc931f5040a13fabbdfce69864ee8d554e879ee`.

Artifacts:

- `plan.json`: the immutable schema-v2 campaign plan.
- `qwen-parent-import.json`: the East-to-Sweden parent-result import record.
- `summary.json`: reduced curves, SLOs, medians, ranges, and boundary selections.
- `rps-sweep.csv`: the 45 selected per-cell result records.
- `agentic-rps-sweep.pdf` and `.png`: compact stacked GPT-OSS-20B TTFT/TPOT panels for the OpenHands Agentic workload, with black dotted SLO lines.
