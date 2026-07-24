# Queue-Haul findings

## Decision

Do not submit another GPU campaign or the current reserve bundle yet. The
compact 2026-07-23 bundle contains more usable evidence than the first
reduction recognized, but it cannot identify a service-capacity frontier or a
causal migration slowdown versus destination load. Recover the archived raw
request and engine records before collecting new data.

The run does establish:

- 4K/16K/24K anchor delivery within 91.4%–99.9% of the reused rates;
- exact replay and KV continuation for every migration;
- new-runtime migration timing at 16K/10 Gbps and 24K/5 Gbps; and
- low foreground work, from zero through 0.146 in anchor-normalized
  prefill-plus-decode work.

The checked profile therefore remains `estimated`.

## Service evidence

All 113 service summaries give identical normal, emergency, and stable labels.
Of the 47 summaries labeled unstable, 45 report a drained queue and a
queue-drift upper bound no larger than `1/180` requests/s. Under the runner's
classifier those 45 failures require an incomplete request batch. The compact
bundle omits `requests.json` and `engine.csv`, so it cannot distinguish a
legitimate overload rejection from a transient or request-generation failure.
It also cannot recover TTFT or TPOT.

The nominal radius is not realized work. The runner normalizes every request
at the cohort's mean context, uses the cohort's mean request work to choose an
arrival rate, and then sends a Poisson count through a deterministic
session order. Reconstructing each execution from its recorded request count,
session order, and per-session context rates gives these fit intervals:

| Direction | largest successful realized work | smallest failed realized work |
|---|---:|---:|
| interactive coding | 0.123546 | 0.132380 |
| coding | 0.184381 | 0.206757 |
| agentic tool loop | 0.003985 | 0.017653 |

These are descriptive intervals, not capacity bounds. Across all splits, 10 of
21 unique failed points are componentwise dominated by a successful point even
after adding realized request rate to prefill and decode work. No
downward-closed facet model in the current variables can represent those
labels. Adding more facets cannot repair the contradiction; request validity,
per-request shape/SLO eligibility, and long-run resource capacity must be
separate.

The current service reduction also repeats one majority-decided boundary once
per vote and then treats those copies as independent runs. Disputed cells
therefore receive more weight. A future reduction must consume realized
run-level work and one boundary estimate per independent run.

For existing offline analysis, retain measured `F(T)`, `G(T)`, and KV
capacity. Treat service headroom as an explicit sensitivity input and report
robust, possible, or unsupported placement. Do not fit a measured service
envelope from these summaries.

## Migration evidence

Recorded `achieved_rho` is raw normalized work divided by the rejected normal
bound, 0.1140625. The apparent range 0–1.282 is therefore only 0–0.146 in raw
normalized work. It is also a preceding 30-second throughput average, not a
measurement over the migration interval. Each treatment records only zero to
two foreground requests, `queue_at_start` refers to the sampler's first row,
and the compact bundle cannot prove that foreground work overlapped migration.

Four treatment rows have achieved throughput zero: replay and KV at both
16K/10 Gbps and 24K/5 Gbps. They are exploratory same-runtime zero-throughput
anchors, contrary to the earlier claim that the run had no zero-load
measurement. They are not empty-destination controls: the runner first
prewarms 18 background sessions totaling 264,699 prompt tokens, up to 21.8% of
reported KV capacity, and retained residency is absent from the compact
records. There is only one zero-throughput repeat per cell.

After removing context or network time, neither method has an identified load
trend. Spearman tests against achieved load give `p=0.589` for replay and
`p=0.574` for KV. A linear load term does not improve replay leave-one-out
median error and changes KV median error only from 4.2% to 3.4% with nine
observations (`p=0.462`). That does not justify a load-indexed curve.

The smallest useful exploratory timing models are method-specific:

- Keep the reused replay context curve. On the six 16K rows, central and
  upper calibration factors are 0.564585 and 0.586673. Applied to the untouched
  24K curve, they have 5.5% and 9.6% held-out median error and never
  underpredict the three held-out runs.
- Replace the KV scalar multiplier with
  `duration = sealed_bytes / route_bytes_per_s + c`. Fitting the six 16K rows
  gives `c=0.961186 s` centrally and `c=1.133822 s` conservatively. The 24K
  held-out median errors are 2.4% and 7.8%; the conservative form never
  underpredicts.

The KV result is mechanistic: GPU/runtime calibration belongs in the
post-transfer residual, not on exact route time. The current
`LoadedCoefficients` multiplies the complete migration duration, so a
calibration below one can predict less than `bytes/link_rate`. It cannot safely
encode this model yet. Replay link time and switch time likewise must remain
outside runtime scaling.

These timing fits are valid only as exploratory low-work envelopes for
concurrency one and the two measured context/bandwidth cells. Their maxima are
not 95% statistical bounds, and they do not establish foreground interference.

## Next evidence decision

First recover the archived files written by the runner:

- `service/**/{requests.json,engine.csv}` to identify request failures,
  recompute realized work, TTFT, TPOT, and backlog;
- `loaded/**/{control,replay,kv_transfer}/foreground/{requests.json,engine.csv}`
  to measure foreground work and overlap during each migration; and
- cache/KV metrics needed to establish actual resident state after prewarm.

Those records may permit a valid re-reduction without another GPU hour. If
they are unavailable, only the irreducible failed service cohorts and
contemporaneous-overlap migration cells should be rerun. The current reserve
must not be used for that purpose: it writes `reserve_tasks`, but
`destination_runner.py` ignores them and reruns the full campaign.
