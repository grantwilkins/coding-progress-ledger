# Queue-Haul assumptions

The checked-in GPT-OSS-20B/A100 profile is estimated, not validated. The v7
destination run preserves state and supports exploratory low-work migration
timing. Its recovered archive identifies six invalid forced-token signatures
that produce 50 empty HTTP-200 streams and invalidate 47 service runs. All 66
complete-work runs pass every service policy, so they provide conditional
inner observations but no capacity boundary. The archive records foreground
overlap in 12/18 migrations, but achieved load counts cached prompts and
remains a prewindow average rather than migration-interval intensity.

Normal and emergency SLOs and the coding, interactive-coding, and agentic trace
sources are frozen in `destination_campaign.py`. They are experiment policies,
not validated fleet guarantees.

Service headroom is therefore an explicit sensitivity input. The v7 migration
fits apply only to concurrency one, at most one overlapping foreground request,
and the measured 16K/10-Gbps and 24K/5-Gbps cells. Their empirical maxima and
paired foreground effects are not 95% confidence bounds.
