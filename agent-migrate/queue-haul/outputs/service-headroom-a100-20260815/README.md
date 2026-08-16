# A100 service-headroom evidence, 2026-08-15

This directory contains the reduced and held-out evidence from the clean A100
service-headroom run rooted at
`/datadrive/qh-headroom-a100-20260815-r7`. The raw root is 3.8 GB and remains on
the A100 node; it contains exact request/token events, engine metrics, prewarm
records, commands, and logs for all 72 calibration, discovery/control, and
confirmation cells.

## Result

Collection succeeded operationally: 54/54 discovery cells and 18/18 unseen
confirmation cells completed in frozen order. Confirmation used first attempts
only, with zero service restarts, invalid cells, or cache mismatches. The start,
middle, and end held-out baselines all passed.

The proposed scalar bound did not confirm. `confirmed.json` reports
`planner_usable=false` and `supported_bound=null`:

| Held-out group | Passing blocks | Frozen check |
|---|---:|---|
| Baseline, rho=0.25 | 3/3 | pass |
| Prefill-heavy last pass, rho=0.70 | 2/3 | fail |
| Prefill-heavy first fail, rho=0.85 | 0/3 | pass |
| Balanced candidate, rho=0.70 | 2/3 | fail |
| Decode-heavy last pass, rho=0.70 | 2/3 | fail |
| Decode-heavy first fail, rho=0.85 | 1/3 | fail |

The latency curve itself is reproducible. At total offered normalized work near
0.70, held-out prefill-heavy, balanced, and decode-heavy median P90 TTFT/P90
per-request mean TPOT are 234.7/59.7 ms, 152.9/42.7 ms, and 131.7/37.2 ms.
Prefill-heavy rho=0.85 failed the declared 100-ms TPOT target in all three
blocks. Decode-heavy classifications near the candidate depended instead on
the preregistered one-fitted-request late-window stability threshold.

`rho` is not measured GPU utilization. It is offered normalized phase work:

```text
rho_f = sum(append_tokens / measured_prefill_tokens_per_s) / window_s
rho_d = sum(output_tokens / measured_decode_tokens_per_s) / window_s
rho   = rho_f + rho_d
```

The exact-stack A100 normalization is 16,758.928 prefill tok/s and 3,597.591
decode tok/s. `service-headroom-phase.csv` preserves both phase coordinates.
The composition dependence means the three sampled rays are descriptive
evidence, not a universal latency equation or a certified two-dimensional
frontier. No value from this directory may update planner admission through
`supported_bound()`.

## Figures and tables

- `service-headroom.pdf` / `.png`: discovery P90 TTFT and P90 per-request mean
  TPOT versus offered normalized phase work for the two extreme mixes, with
  held-out restart-block min--max ranges shown as open markers. TTFT is
  logarithmic so the measured 24-second overload point does not hide the
  subsecond curve. The balanced held-out treatment is reported in the tables;
  it is omitted from this panel because it has no discovery curve.
- `service-headroom-phase.pdf` / `.png`: sampled prefill/decode work rays, with
  held-out points shown as open markers and any all-repeat evidence miss marked
  with an x. It is not a fitted frontier.
- `service-headroom.csv`: discovery medians, ranges, phase coordinates, and
  repeat-level feasibility counts.
- `service-headroom-heldout.csv`: the same reduction for unseen confirmation.
- `service-headroom-phase.csv`: both stages in one table.
- `plan.json`, `normalization.json`, `scout.json`, `confirmation-plan.json`,
  `confirmed.json`, and `confirmation-status.json`: immutable campaign inputs
  and reduced evidence.

Both figure writers call `plot_style.apply()` and use the registered service
mix identities.

## Provenance

- GPU: NVIDIA A100 80GB PCIe,
  `GPU-16f1b098-2d58-d5e4-c60e-85267354942d`.
- Model: `openai/gpt-oss-20b`, revision
  `6cee5e81ee83917806bbde320786a8fb61efebee`.
- Runtime: native vLLM 0.22.0 and LMCache 0.5.1, TP1, eager,
  chunked-prefill, prefix caching, 8,192 batched tokens, 256 sequences.
- Acquisition commit: `2b7236586b14440a71f75e5135d937b0abc290a2`.
- Corrected evidence reducer commit:
  `cbb49e458792c2a20ca87dae9015abf8cec87a91`.
- Durable confirmation driver commit:
  `76b316123950404dd2643251ede56de39332736b`.
- Core plan canonical digest:
  `2435f84b2a1bde80896565f9d5b7a59bfcd6adbe12fe7a094172d01a741c03f6`.
- Confirmation plan canonical digest:
  `8575a6cbb276b5c1ae0c6b4d9e595b5b68834657f9c2981d8ff9711a133f7f5f`.
- Discovery runtime identity:
  `7ae339b33943fe457faf1b7497b5f27ccf45b77ac0a529b912b53eaa667ae808`.
- Cross-commit service runtime identity:
  `e3cb11186d494bcb719b92552dfe0530f1f995da64205cc1761ec169a7380fe6`.

Focused service-headroom semantic, plotting, and driver tests: 42 passed.
