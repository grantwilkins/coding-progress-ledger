# Queue-Haul assumptions

The checked-in GPT-OSS-20B/A100 profile is estimated, not validated. The v7
destination run preserves state and supports exploratory migration timing for
its recorded request schedules. Its recovered archive identifies six invalid
forced-token signatures
that produce 50 empty HTTP-200 streams and invalidate 47 service runs. Sixty
of 66 complete-work runs contain cache hits extending into the nominal new
append. Five executions are forensically private-prefix-consistent and their
recorded summaries pass every policy, but absent stream-completion evidence
makes them descriptive sensitivity anchors, not admissible service points. The
sole under-hit is excluded because its prewarm likely failed. The archive
records foreground overlap in 12/18 migrations, but achieved load counts
cached prompts and remains a prewindow average rather than migration-interval
intensity.

Normal and emergency SLOs and the coding, interactive-coding, and agentic trace
sources are frozen in `destination_campaign.py`. They are experiment policies,
not validated fleet guarantees.

Service headroom is therefore an explicit sensitivity input. The v7 migration
fits apply only to its recorded concurrency-one request schedules and measured
16K/10-Gbps and 24K/5-Gbps cells; “one overlapping request” is not by itself a
workload bound. Their empirical maxima and paired foreground effects are not
95% confidence bounds.

The destination profile is valid only for its pinned model, hardware,
precision, parallel layout, engine, scheduler, and KV ABI. The current planner
uses an append-token/cold-rate normalization coordinate and sums projected
context tokens. That coordinate is not a conservative physical cached-prefill
bound. It does
not credit cross-session prefix sharing because the evidence does not establish
protected destination block identity or residency, and its unrounded token sum
is not yet a physical-block memory guarantee. Block-rounded private KV is target
v1 semantics; any future sharing credit requires exact protected block keys and
remains a separate optimization. The current vLLM 0.22.0/.75 runtime exposes
963,152 KV tokens, not the older configuration's 1,214,544.
