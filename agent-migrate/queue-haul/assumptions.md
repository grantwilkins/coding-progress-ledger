# Queue-Haul assumptions

The checked-in GPT-OSS-20B/A100 profile is estimated, not validated. The v7
destination run preserves state and supports exploratory low-work migration
timing. It cannot calibrate admission because compact summaries omit request
failure causes and the realized service labels are not downward-closed. It
cannot identify migration interference because achieved load is a prewindow
average with no recorded migration overlap. Recover the archived raw records
before collecting more GPU data. `DATA_TO_COLLECT.md` is the evidence ledger.

Normal and emergency SLOs and the coding, interactive-coding, and agentic trace
sources are frozen in `destination_campaign.py`. They are experiment policies,
not validated fleet guarantees.

Service headroom is therefore an explicit sensitivity input. The v7 migration
fits apply only to concurrency one, raw anchor-normalized foreground work from
0 through 0.146, and the measured 16K/10-Gbps and 24K/5-Gbps cells. Their
empirical maxima are not 95% confidence bounds.
