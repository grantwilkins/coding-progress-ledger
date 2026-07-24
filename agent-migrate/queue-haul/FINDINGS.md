# Queue-Haul findings

## Decision

The recovered 2026-07-23 archive changes the diagnosis but still does not
justify an accepted destination profile.

- Do not launch the current reserve; its task list is ignored and it would
  repeat the full campaign.
- Do not collect more migration timing for the measured low-concurrency case.
  The raw records support conservative replay and KV timing envelopes and a
  method-affinity rule for live traffic.
- Replace the six invalid forced-token signatures before any service rerun. If
  a measured service boundary is required, rerun only the unresolved service
  frontier after that fix.

All 767 artifacts listed in `SHA256SUMS` match; the archive contains those
artifacts plus the manifest itself. The local `data/` copy is ignored by Git
and is not part of the evidence commit.

## Service evidence

The 47 summaries classified infeasible contain 50 requests with HTTP status
200, no recorded error, and zero prompt and output usage. The client requested
a forced token with `ignore_eos=true`, so these are not legitimate early model
stops. The stream ended without the required work or usage record. The other
9,131 of 9,181 requests produced exactly their planned output.

Every empty response is one of six deterministic
`(session, request_index, forced_token)` signatures. Their forced token IDs are
200110–200952; no successful request uses a forced ID at or above 200000. This
is a harness/token-eligibility defect, not evidence of overload. The
descriptive incidence is 11/470 agentic requests, 18/5,065 coding requests, and
21/3,646 interactive-coding requests.

This separates two questions that the old reduction conflated:

1. **Measurement validity:** an empty forced-token stream invalidates the
   probe; it does not establish production availability.
2. **Consumable capacity:** a run with missing work is not a capacity-boundary
   observation.

After excluding the 47 invalid runs, all 66 complete-work runs pass normal,
emergency, and stability. Their worst p90 TTFT is 0.587 s, worst p90 mean TPOT
is 0.0276 s/token, and largest queue-drift upper bound is 0.00335 requests/s.
The normal policies are 2 s TTFT and 0.1 s/token TPOT. The valid data therefore
contains no infeasible capacity point.

Using each request's actual context and planned work, the largest valid
successful normalized work rates are:

| Workload affinity | largest valid success |
|---|---:|
| agentic tool loop | 0.108677 |
| coding | 0.182805 |
| interactive coding | 0.134067 |

These are conditional inner observations, not frontier estimates or
confidence bounds. The smallest common observed success is 0.108677. Normal
and emergency cannot be distinguished because every valid run is well inside
both SLOs.

The earlier downward-closure contradictions disappear when missing-work runs
are excluded. Deleting only the zero-work rows would still leave two
nonmonotone stable failures, so contaminated executions are not salvaged as
boundary evidence. More facets or a learned request-shape model are therefore
not justified. For current analysis, keep measured `F(T)`, `G(T)`, and KV
capacity; treat service headroom as partially identified and report:

- **robust:** within a selected conditional inner bound and all compatibility
  and affinity predicates pass;
- **possible:** feasible only for some unmeasured headroom values; or
- **unsupported:** outside the measured affinity/domain or based on an invalid
  probe.

## Migration timing

Twelve of 18 migration treatments overlap a foreground request. Ten have one
request active at migration start, and two start a request during migration.
All 36 control and treatment foreground requests reuse cached prefixes; the 20
treatment requests reuse 2,608–3,168 cached tokens. Twenty-six of 27 first
engine samples record the intended 264,699 prompt tokens, 18 generated tokens,
and 18 successful prewarm requests. One control contains additional prior state
and is not cache-state-identical to its treatments. The data proves reuse, not
exact idle residency: the GPU cache gauge reports zero for evictable cached
blocks. The intended 264,699-token prewarm is 21.79% of declared capacity but
must not be modeled as measured resident occupancy. Six treatments have no
foreground overlap; four complete no foreground request at all.

The recorded `achieved_rho` remains the wrong timing covariate. It is a
preceding 30-second average divided by the rejected service bound, and it
counts cached prompt tokens as new compute. Subtracting cached prompts gives
prewindow compute rho from 0 to 0.898. Exact migration-interval foreground work
is not recoverable: aggregate request totals have no per-token timestamps,
engine counters include migration work, and 8/18 samplers end 39–222 ms before
route switch. The archive identifies overlap topology, not a load curve.

The smallest reliable duration models remain method-specific:

- Keep the reused replay compute-plus-completion context curve. On the six 16K
  rows, central and conservative calibration factors are 0.564571 and
  0.586660 after separating route bytes. Applied to the untouched 24K curve,
  they have 5.5% and 9.6% held-out median error and never underpredict the
  three held-out runs. One fitting context cannot identify a separate replay
  intercept.
- Model KV duration as `sealed_bytes / route_bytes_per_s + c`. The six 16K
  rows give `c=0.961186 s` centrally and `c=1.133822 s` conservatively. The
  24K held-out median errors are 2.4% and 7.8%; the conservative form never
  underpredicts.

A categorical overlap split improves central replay held-out median error from
5.5% to 4.2%, but its 16K idle fit contains one row. The analogous KV split
underpredicts two held-out rows. That is insufficient evidence for another
planner coefficient; the global conservative envelopes already cover both
idle and observed busy executions.

Exact route time must remain outside runtime calibration. The current
`LoadedCoefficients` multiplies the complete migration duration and can predict
less than `bytes/link_rate`, so it cannot safely encode the KV model. Replay
log transfer and switch time likewise remain physical components.

## Foreground impact and affinity

Control and treatment runs provide ten exactly matched foreground requests per
method, six of which overlap migration.

- For five already-active replay requests, paired TPOT increased by a median
  3.45 ms/token and at most 6.79 ms/token. The five KV counterparts increased
  by a median 0.42 ms/token and at most 0.67 ms/token.
- The one request arriving during replay reconstruction incurred 1.084 s
  additional TTFT. The same scheduled request arriving during KV transfer
  incurred 4.7 ms additional TTFT.

These are observations, not percentile bounds. They support a simple
eligibility rule: prefer KV transfer for latency-sensitive busy destinations;
allow replay only on an idle/drained destination or when explicit TTFT slack
covers the empirical replay penalty plus uncertainty. Foreground impact should
not be hidden inside a service-capacity facet.

## Remaining evidence

No additional migration campaign is needed for concurrency one, the measured
16K/10-Gbps and 24K/5-Gbps cells, and low foreground work. Higher concurrency,
continuous high load, or a claimed interference percentile would require new
evidence.

Service capacity remains right-censored. First select known-safe forced tokens
and make the client hard-fail a stream that lacks completion, usage, or the
requested tokens. Then reuse all 66 valid runs. If an accepted boundary is
still required, start with three independent corrected probes per workload
affinity at nominal radius 0.5 and adapt only the affinity that fails. Compute
normal, emergency, and stability from each physical run instead of rerunning
the same boundary by policy.
