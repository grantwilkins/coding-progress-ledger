# Strong Per-Site Reuse Baseline

`strong_site_reuse` is the paper-facing name for the K-level `cache_reuse`
policy. It is intentionally a serious baseline, not a strawman.

Contract:

```text
reuse any state already materialized at a destination;
for cold state, choose the cheapest available materialization mode;
avoid paying twice for the same state at the same site.
```

In code, `strong_site_reuse` delegates to `cache_reuse` so historical artifacts
and tests keep their old policy name while new writing can use clearer
terminology. Cold-state choice includes replay, KV transfer, artifact copy,
text transfer, or workspace hydrate when those modes are valid for the state
layer and resource budget.

This baseline is stronger than a no-reuse request-level policy. It includes
per-site warm reuse, per-state materialization choice, and per-workflow
destination preference. New documents should avoid calling weak baselines
`request_level` without the `no_reuse` qualifier.
