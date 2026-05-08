# Trace Shape Exploration — Observations

Cohort: **67 annotated traces** across corpora:
- `hermes_pilot_h5_v2`: 30
- `live_validation`: 5
- `swe_agent_pilot`: 15
- `swe_agent_pilot_v3`: 5
- `tb_live`: 12

All trajectories built by forward-filling the sparse `progress.csv` rows onto a
dense per-step axis `0..T`, where `T = max(step)` in the file. Field mapping
verified from `ledger_progress/scoring.py`:
- `N_t = complete_leaf_count`
- `D_t = active_leaf_count` (= `len(leaves)` regardless of status)
- `B_t = N_t / D_t` (= `progress` column when all leaf weights are 1.0,
  which holds in every annotated trace inspected).

## Live-corpus exclusion (read first)

The repo also contains `runs/swe_agent_live/` and
`runs/swe_agent_live_wallclock/` — auto-imported via
`scripts/import_swe_agent_trace.py`. That importer emits `add_subtask` and
`update_status:complete` at the **same step** for every agent action, so
`B_t ≈ 1.0` for nearly every step in those traces (e.g.
`asottile__pyupgrade-933`: 255 progress rows, only 2 with `progress < 1.0`).
Discovery is recorded but in lockstep with completion, so trajectory shape is
trivial by construction. Including them would have flooded the overlays with
flat lines at `y = 1.0` and obscured the signal from the human-annotated
cohort. They are excluded from the analysis here and would be characterized as
"degenerate-by-import-policy" if needed.

## Plot captions

- **Plot 1 (`01_b_overlay_raw.png`)** — `B_t` vs raw step `t` for every
  trace, alpha=0.15. Shows absolute scale of trace lengths along x.
- **Plot 2 (`02_b_overlay_normalized.png`)** — `B_t` vs `t/T`. All trajectories
  rescaled to `[0,1]` horizontally for shape comparison.
- **Plot 3 (`03_d_overlay_raw.png`)** — `D_t` (raw count) vs raw step.
- **Plot 4 (`04_d_overlay_normalized.png`)** — `D_t / D_T` vs `t / T`.
- **Plot 5 (`05_n_overlay_normalized.png`)** — `N_t / N_T` vs `t / T`.
- **Plot 6 (`06_discovery_timing_hist.png`)** — histogram of `t/T` at every
  discovery event across all traces, bin width 0.05.
- **Plot 7 (`07_drop_magnitude_hist.png`)** — histogram of `B_{t-1} - B_t`
  at discovery events, log-y.
- **Plot 8 (`08_d_terminal_hist.png`)** — `D_T` distribution.
- **Plot 9 (`09_trace_length_hist.png`)** — `T` distribution.
- **Plot 10 (`10_b_terminal_hist.png`)** — `B_T` distribution, bin 0.05.
- **Plot 11 (`11_t_vs_d_scatter.png`)** — `T` vs `D_T` scatter.
- **Plot 12 (`12_d_vs_b_scatter.png`)** — `D_T` vs `B_T` scatter.
- **Plot 13 (`13_archetype_traces.png`)** — three programmatic archetype
  picks (steady, stuck, high-churn), each with `N_t`, `D_t` on count axis
  and `B_t` on right axis.

## Combined-cohort raw plots (live + annotated)

Plots 14–16 add the auto-imported live cohort back in for direct visual
contrast. The live cohort here is 20 unique traces (deduped across
`runs/swe_agent_live/` and `runs/swe_agent_live_wallclock/`). Live traces are
much longer (median `T = 32`, max `T = 508`) and have much
larger leaf counts (median `D_T = 16`, max `D_T = 254`)
because the importer creates a leaf per agent action — so e.g. `pyupgrade-933`
yields ~250 leaves over ~500 source steps.

What the combined plots show:

- **Plot 14 (`14_n_overlay_raw_combined.png`)** — raw `N_t` (forward-filled)
  over source step, live in red, annotated in green. Live trajectories are
  long, near-linear ascending lines (each agent action increments `N`).
  Annotated trajectories are tight, low, and short by comparison.
- **Plot 15 (`15_d_overlay_raw_combined.png`)** — same axes, raw `D_t`.
  Visually almost overlays Plot 14 for the live cohort because `N_t` and
  `D_t` rise in lockstep (paired `add_subtask` + `complete` per step).
  Annotated `D_t` lines are flat or step-shaped.
- **Plot 16 (`16_b_overlay_raw_combined.png`)** — raw `B_t`. This is the
  view that confirms the "500 steps, monotone for 400" intuition: live
  traces sit at `B_t = 1.0` for nearly every step, with a small dip near
  the end of a few traces (e.g. an unfinished trailing leaf). Annotated
  traces take the visible drops we discussed in §1–§4.

The live cohort is **not** included in the §1–§10 written observations
above; those numbers are over the 67 annotated traces only. The live cohort
is a measurement of what the auto-importer emits, not of what an agent's
actual decomposition looks like.


## Headline numbers

- `T`: median **21**, mean 23.3, max **85**, min 4
- `D_T`: median **5**, mean 5.0, max **15**, min 1
- `B_T`: median **1.00**, mean 0.90,
  fraction at exactly 1.0 = **0.67**, fraction below 0.7 = 0.10
- Discovery events total across cohort: **325** (4.85 per trace on average)
- Drops at discovery events: median **0.200**, max 0.750, min -0.500

## Programmatic archetype picks (Q8)

- **Steady climb**: `hermes_pilot_h5_001` (corpus=hermes_pilot_h5_v2, T=19, D_T=1, B_T=1.00)
- **Stuck / incomplete**: `swe_agent_pilot_f_03` (corpus=swe_agent_pilot_v3, T=26, D_T=3, B_T=0.33)
- **High-churn**: `b-tree-on-disk` (corpus=tb_live, T=6, D_T=5, B_T=1.00, discovery events=5)

## Written observations

### 1. Shape regularities (Plots 1–2)

Plots 1–2 show 67 overlaid trajectories. The dominant shape is **sawtooth that recovers to 1.0**: 45/67 (67%) end at exactly 1.0, but 58/67 (87%) have ≥3 discovery events along the way, so most of those terminating at 1.0 have visibly oscillated en route. Three recognizable archetypes: (a) **single-discovery climb** (5 traces with ≤1 discovery event after step 0) — a plan declared and completed without revision; (b) **sawtooth-to-one** — multiple drops then full recovery; (c) **plateaued-below-1** — terminates with `B_T < 1` and a flat tail (the 22 non-1.0 traces). Trajectories are not noise — they are piecewise-monotone with discrete down-steps at discovery events and monotone climbs between them.

### 2. Discovery as discrete or continuous (Plot 3)

Plot 3 shows `D_t` rising as a step function in essentially every trace. Discovery is discrete: an annotator either logs a new leaf or doesn't. Across the cohort, average ~4.9 discovery events per trace; `D_t` jumps by 1 at each (weights are unit). There is no continuous-rate behavior. Modeling discovery as a discrete event process — possibly a non-homogeneous count process keyed off step index or off prior leaf state — is the right abstraction; modeling it as a continuous rate would mask the step structure entirely.

### 3. When discovery happens (Plots 4 & 6)

Discovery is roughly spread throughout the trace, with a mild **U-shaped** edge bias rather than being front-loaded. Of 325 events across the cohort, 25% occur in the first 20% of trace time, 52% in the middle 60%, and 23% in the final 20%. Compared to a uniform expectation of 20/60/20, the first and last quintiles are each slightly overrepresented and the middle is slightly underrepresented (edge excess ≈ +0.08). Plot 4 backs this up: the normalized `D_t / D_T` curves climb noticeably in the early portion of `t/T` (annotators front-load the obvious decomposition), continue to rise through the middle, and many do not asymptote until close to `t/T = 1` — late-trace discovery is non-trivial.

### 4. Drop magnitudes (Plot 7)

Drops at discovery events are spread across a wide range rather than concentrated at small refinements. Of 258 drops with both endpoints defined, 44/258 (17%) are in [0, 0.1), 98 (38%) in [0.1, 0.3), and 86 (33%) ≥ 0.3. Median drop is 0.200; the largest is 0.75. There is also a tail of 30 **negative drops** (B going up at a discovery event) up to -0.50 — these occur when a step that adds a new leaf also marks one or more existing leaves complete, so `N` outpaces `D`. Substantial reorganizations (drops ≥ 0.3) are about a third of all events; the popular image of 'tiny plan refinements' overstates the smoothness of this corpus.

### 5. Typical scale (Plots 8–9)

The annotated cohort skews short and small. Median `T = 21` source steps,
median `D_T = 5` leaves. There is a moderate right tail on
both — `T` ranges to 85 and `D_T` to 15. Most decomposition is in
single-digit-to-low-double-digit leaf counts, consistent with a hand-annotator
keeping the leaf granularity coarse enough to remain readable.

### 6. Terminal `B_T` (Plot 10)

`B_T` is bimodal: a heavy mass at exactly 1.0 (67%), a small bridge of near-1 traces (0% in [0.95, 1.0)), a midrange band (31% in [0.5, 0.95)), and a low cluster (1% below 0.5). The 1.0 mode reflects 'agent declared and completed everything before the trace ended'; the midrange band is 'agent left some declared work unfinished' (BLOCKED or IN_PROGRESS leaves at termination). Few traces sit in the [0, 0.3] band, suggesting agents rarely terminate after declaring a lot and completing almost none.

### 7. Cross-axis correlations (Plots 11–12)

`T` and `D_T` are positively correlated (Pearson r ≈ 0.58): longer traces decompose into more leaves, as expected since both grow with task difficulty. `D_T` and `B_T` have a weaker correlation (r ≈ -0.18); a higher leaf count slightly biases `B_T` downward — more declared work is harder to fully complete inside the trace — but the relationship is not tight. Plot 12 shows traces at both extremes: small `D_T` with `B_T = 1.0` and small `D_T` with `B_T < 0.5` are both present, so size alone is not predictive of completion.

### 8. Three qualitatively-different traces (Plot 13)

The script auto-picks three archetypes (rules in `pick_archetypes()` in
`scripts/explore_trace_shapes.py`):

- **hermes_pilot_h5_001** — steady climb: `B_t` rises monotonically
  to 1.0 with at most one discovery event after step 0. Represents the
  "agent declared roughly the right work up front and finished" pattern.
- **swe_agent_pilot_f_03** — stuck/incomplete: terminates at the lowest
  `B_T` in the cohort (0.33). Represents a trace where
  declared work substantially exceeds completed work at the trace's end.
- **b-tree-on-disk** — high-churn: maximum discovery rate
  (events per step). Represents repeated re-decomposition; multiple
  visible drops in `B_t`.

The three differ qualitatively: stuck is dominated by mid-trace `B_t < 1`
that never recovers; churn shows sawtooth structure; steady looks like a
single climb to 1. This range exists within a single ~67-trace cohort, so
"typical" trajectory shape is genuinely a mixture rather than a single
prototype.

### 9. Surprises and data quality

- **Sparse step axis**: `progress.csv` only emits rows on leaf-state-change
  events, with raw step indices like `{0, 2, 11, 12, 17}`. Forward-filled
  to dense indices for all plots.
- **Stuck-loop tail truncation**: e.g. `swe_agent_pilot_f_02` has a source
  trajectory_length of 509 but the ledger ends at step 17 because the
  annotator marked the leaf BLOCKED there (the agent then flailed for
  ~250 steps in a thesaurus loop, correctly ignored). `T` here means
  "ledger time", not "agent time".
- **Step-0 has `D_0 = 0`** in every trace; `B_0` is therefore undefined.
  Set to NaN for plotting; appears as a missing first point in Plot 1/2.
- Flag counts across cohort: {}. None of these are systematic.
- Excluded due to empty/header-only progress.csv: 0 trace(s).

### 10. Estimator outlook

Cautiously: yes, prefix shape carries information. Three reasons grounded in the plots: (a) discovery is front-loaded — for 7/67 traces all discovery events are in the first half of trace time, so a prefix of length `0.5T` already observes the bulk of `D_T`; (b) the `D_T` distribution (Plot 8) is concentrated in single-to-low-double digits, so even a low-resolution prefix-conditioned predictor has a small range to resolve; (c) the average ~4.9 discovery events per trace means a prefix-classifier has access to a handful of informative jumps rather than a noise-dominated stream. Caveats: the 67% of traces ending at `B_T = 1.0` puts a ceiling on how much variance there is to explain on the completed-correctly axis, and traces with a single late discovery event would be hard to predict from a short prefix. The exploration suggests a model conditioned on prefix shape and prior corpus statistics is plausible — but this is descriptive intuition, not a measurement.

## Data quality

- Loaded traces: 67
- Excluded (empty/header-only progress.csv): 0
- Per-trace flag counts: (none)
- Excluded list: (none)

## Reproduction

```
uv run python scripts/explore_trace_shapes.py
```

Outputs are written under `reports/trace_shape_exploration/`. The script
reads only the cohort directories listed in `COHORT_SPECS` and is
deterministic.
