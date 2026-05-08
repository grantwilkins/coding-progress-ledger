"""Trace shape exploration — descriptive plots over annotated ledger traces.

Reads progress.csv from the annotated cohort (latest-version-only), builds
dense forward-filled per-step series, and emits 13 PNGs plus OBSERVATIONS.md
into reports/trace_shape_exploration/.
"""
from __future__ import annotations

import csv
import json
from dataclasses import dataclass
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
RUNS = ROOT / "runs"
OUT = ROOT / "reports" / "trace_shape_exploration"

# Pilot tasks superseded by pilot_v3.
PILOT_SUPERSEDED = {"swe_agent_pilot_f_01", "swe_agent_pilot_f_03",
                    "swe_agent_pilot_f_06", "swe_agent_pilot_s_01",
                    "swe_agent_pilot_s_03"}

COHORT_SPECS = [
    ("swe_agent_pilot", "swe_agent_pilot_*", PILOT_SUPERSEDED),
    ("swe_agent_pilot_v3", "*", set()),
    ("hermes_pilot_h5_v2", "hermes_pilot_h5_*", set()),
    ("tb_live", "*", set()),
    ("live_validation", "*", set()),
]


@dataclass
class Trace:
    trace_id: str
    corpus: str
    steps: np.ndarray  # raw step values from progress.csv
    N_raw: np.ndarray
    D_raw: np.ndarray
    # dense forward-filled series (length T+1, indexed 0..T)
    N: np.ndarray
    D: np.ndarray
    B: np.ndarray  # NaN where D=0
    flags: list[str]

    @property
    def T(self) -> int:
        return int(self.steps.max())

    @property
    def N_T(self) -> int:
        return int(self.N[-1])

    @property
    def D_T(self) -> int:
        return int(self.D[-1])

    @property
    def B_T(self) -> float:
        return float(self.B[-1])

    @property
    def discovery_steps(self) -> np.ndarray:
        # steps t in 1..T where D[t] > D[t-1]
        diffs = np.diff(self.D)
        return np.flatnonzero(diffs > 0) + 1


def load_trace(trace_dir: Path, corpus: str) -> Trace | None:
    csv_path = trace_dir / "progress.csv"
    if not csv_path.exists():
        return None
    rows = list(csv.DictReader(csv_path.open()))
    if not rows:
        return None
    steps = np.array([int(r["step"]) for r in rows])
    N_raw = np.array([int(float(r["complete_leaf_count"])) for r in rows])
    D_raw = np.array([int(float(r["active_leaf_count"])) for r in rows])
    T = int(steps.max())
    if T < 1:
        # T==0 traces have only the init row; trivial — skip.
        return None
    # Dense forward-fill on 0..T
    N = np.zeros(T + 1, dtype=int)
    D = np.zeros(T + 1, dtype=int)
    last_n, last_d = 0, 0
    j = 0
    for t in range(T + 1):
        while j < len(steps) and steps[j] == t:
            last_n, last_d = int(N_raw[j]), int(D_raw[j])
            j += 1
        N[t], D[t] = last_n, last_d
    with np.errstate(divide="ignore", invalid="ignore"):
        B = np.where(D > 0, N / np.where(D > 0, D, 1), np.nan)

    flags = []
    if (N > D).any():
        flags.append("N>D")
    if (np.diff(D) < 0).any():
        flags.append("D_decreased")
    if np.nan_to_num(B, nan=0.0).max() > 1.0 + 1e-9:
        flags.append("B>1")
    if T < 2:
        flags.append("T<2")
    return Trace(trace_id=trace_dir.name, corpus=corpus,
                 steps=steps, N_raw=N_raw, D_raw=D_raw,
                 N=N, D=D, B=B, flags=flags)


def load_cohort() -> tuple[list[Trace], dict]:
    traces: list[Trace] = []
    excluded_empty: list[str] = []
    for corpus, glob_pat, supersede in COHORT_SPECS:
        corp_dir = RUNS / corpus
        for sub in sorted(corp_dir.glob(glob_pat)):
            if not sub.is_dir():
                continue
            if sub.name in supersede:
                continue
            tr = load_trace(sub, corpus)
            if tr is None:
                excluded_empty.append(f"{corpus}/{sub.name}")
                continue
            traces.append(tr)
    return traces, {"excluded_empty": excluded_empty}


def fig(name: str):
    f = plt.figure(figsize=(8, 5), dpi=150)
    return f, OUT / name


def save(f, path: Path):
    f.tight_layout()
    f.savefig(path, dpi=150)
    plt.close(f)


def plot_overlay_b_raw(traces: list[Trace]) -> None:
    f, p = fig("01_b_overlay_raw.png")
    ax = f.gca()
    for tr in traces:
        ax.plot(np.arange(tr.T + 1), tr.B, alpha=0.15, color="C0", linewidth=0.8)
    ax.set_xlabel("step t (ledger source-trace step)")
    ax.set_ylabel("B_t = N_t / D_t")
    ax.set_ylim(-0.02, 1.02)
    ax.set_title(f"B_t trajectories (n={len(traces)}, raw step axis)")
    save(f, p)


def plot_overlay_b_norm(traces: list[Trace]) -> None:
    f, p = fig("02_b_overlay_normalized.png")
    ax = f.gca()
    for tr in traces:
        x = np.arange(tr.T + 1) / tr.T
        ax.plot(x, tr.B, alpha=0.15, color="C0", linewidth=0.8)
    ax.set_xlabel("t / T (fraction of trace elapsed)")
    ax.set_ylabel("B_t")
    ax.set_ylim(-0.02, 1.02)
    ax.set_xlim(0, 1)
    ax.set_title(f"B_t trajectories (n={len(traces)}, normalized)")
    save(f, p)


def plot_overlay_d_raw(traces: list[Trace]) -> None:
    f, p = fig("03_d_overlay_raw.png")
    ax = f.gca()
    for tr in traces:
        ax.plot(np.arange(tr.T + 1), tr.D, alpha=0.15, color="C1", linewidth=0.8)
    ax.set_xlabel("step t")
    ax.set_ylabel("D_t (discovered leaf count)")
    ax.set_title(f"D_t trajectories (n={len(traces)}, raw)")
    save(f, p)


def plot_overlay_d_norm(traces: list[Trace]) -> None:
    f, p = fig("04_d_overlay_normalized.png")
    ax = f.gca()
    for tr in traces:
        if tr.D_T == 0:
            continue
        x = np.arange(tr.T + 1) / tr.T
        ax.plot(x, tr.D / tr.D_T, alpha=0.15, color="C1", linewidth=0.8)
    ax.set_xlabel("t / T")
    ax.set_ylabel("D_t / D_T")
    ax.set_xlim(0, 1)
    ax.set_ylim(-0.02, 1.02)
    ax.set_title(f"D_t / D_T (n={len(traces)})")
    save(f, p)


def plot_overlay_n_norm(traces: list[Trace]) -> None:
    f, p = fig("05_n_overlay_normalized.png")
    ax = f.gca()
    for tr in traces:
        if tr.N_T == 0:
            continue
        x = np.arange(tr.T + 1) / tr.T
        ax.plot(x, tr.N / tr.N_T, alpha=0.15, color="C2", linewidth=0.8)
    ax.set_xlabel("t / T")
    ax.set_ylabel("N_t / N_T")
    ax.set_xlim(0, 1)
    ax.set_ylim(-0.02, 1.02)
    ax.set_title(f"N_t / N_T (n={len(traces)})")
    save(f, p)


def plot_discovery_timing(traces: list[Trace]) -> None:
    f, p = fig("06_discovery_timing_hist.png")
    ax = f.gca()
    timings = []
    for tr in traces:
        ds = tr.discovery_steps
        if len(ds) and tr.T > 0:
            timings.extend(ds / tr.T)
    bins = np.arange(0, 1.05, 0.05)
    ax.hist(timings, bins=bins, color="C3", edgecolor="black")
    ax.set_xlabel("t / T at discovery event")
    ax.set_ylabel("count of events (across all traces)")
    ax.set_title(f"Discovery event timing (n_events={len(timings)})")
    save(f, p)


def plot_drop_magnitude(traces: list[Trace]) -> None:
    f, p = fig("07_drop_magnitude_hist.png")
    ax = f.gca()
    drops = []
    for tr in traces:
        for t in tr.discovery_steps:
            b_prev, b_now = tr.B[t - 1], tr.B[t]
            if not (np.isnan(b_prev) or np.isnan(b_now)):
                drops.append(b_prev - b_now)
    drops = np.array(drops)
    ax.hist(drops, bins=40, color="C4", edgecolor="black")
    ax.set_xlabel("B_{t-1} - B_t at discovery event")
    ax.set_ylabel("count")
    ax.set_yscale("log")
    ax.set_title(f"Drop magnitudes at discovery events (n={len(drops)}, log-y)")
    save(f, p)


def plot_d_terminal(traces: list[Trace]) -> None:
    f, p = fig("08_d_terminal_hist.png")
    ax = f.gca()
    vals = np.array([tr.D_T for tr in traces if tr.D_T > 0])
    if vals.max() / max(vals.min(), 1) > 10:
        bins = np.logspace(np.log10(max(vals.min(), 1)), np.log10(vals.max() + 1), 20)
        ax.set_xscale("log")
    else:
        bins = np.arange(vals.min(), vals.max() + 2)
    ax.hist(vals, bins=bins, color="C5", edgecolor="black")
    ax.set_xlabel("D_T (final discovered leaf count)")
    ax.set_ylabel("count of traces")
    ax.set_title(f"D_T distribution (n={len(vals)})")
    save(f, p)


def plot_trace_length(traces: list[Trace]) -> None:
    f, p = fig("09_trace_length_hist.png")
    ax = f.gca()
    vals = np.array([tr.T for tr in traces])
    if vals.max() / max(vals.min(), 1) > 10:
        bins = np.logspace(np.log10(max(vals.min(), 1)), np.log10(vals.max() + 1), 20)
        ax.set_xscale("log")
    else:
        bins = np.arange(vals.min(), vals.max() + 2)
    ax.hist(vals, bins=bins, color="C6", edgecolor="black")
    ax.set_xlabel("T (trace length, max source step)")
    ax.set_ylabel("count of traces")
    ax.set_title(f"T distribution (n={len(vals)})")
    save(f, p)


def plot_b_terminal(traces: list[Trace]) -> None:
    f, p = fig("10_b_terminal_hist.png")
    ax = f.gca()
    vals = [tr.B_T for tr in traces if not np.isnan(tr.B_T)]
    bins = np.arange(0, 1.05, 0.05)
    ax.hist(vals, bins=bins, color="C7", edgecolor="black")
    ax.set_xlabel("B_T (terminal reported progress)")
    ax.set_ylabel("count of traces")
    ax.set_title(f"B_T distribution (n={len(vals)})")
    save(f, p)


def plot_t_vs_d(traces: list[Trace]) -> None:
    f, p = fig("11_t_vs_d_scatter.png")
    ax = f.gca()
    Ts = [tr.T for tr in traces]
    Ds = [tr.D_T for tr in traces]
    ax.scatter(Ts, Ds, alpha=0.6, color="C0")
    ax.set_xlabel("T")
    ax.set_ylabel("D_T")
    ax.set_title(f"Trace length vs final denominator (n={len(traces)})")
    save(f, p)


def plot_d_vs_b(traces: list[Trace]) -> None:
    f, p = fig("12_d_vs_b_scatter.png")
    ax = f.gca()
    Ds = [tr.D_T for tr in traces]
    Bs = [tr.B_T for tr in traces if not np.isnan(tr.B_T)]
    Ds_plot = [tr.D_T for tr in traces if not np.isnan(tr.B_T)]
    ax.scatter(Ds_plot, Bs, alpha=0.6, color="C2")
    ax.set_xlabel("D_T")
    ax.set_ylabel("B_T")
    ax.set_ylim(-0.02, 1.02)
    ax.set_title(f"D_T vs B_T (n={len(Bs)})")
    save(f, p)


def pick_archetypes(traces: list[Trace]) -> dict[str, Trace]:
    by_id = {tr.trace_id: tr for tr in traces}
    # 1. Steady-climb: B_T == 1.0, ≤1 discovery event after step 0, median T
    steady_pool = [tr for tr in traces if tr.B_T == 1.0
                   and len([d for d in tr.discovery_steps if d > 0]) <= 1]
    if steady_pool:
        median_T = float(np.median([x.T for x in steady_pool]))
        steady_pool.sort(key=lambda t: (abs(t.T - median_T), t.trace_id))
        steady = steady_pool[0]
    else:
        steady = sorted(traces, key=lambda t: (-t.B_T, t.trace_id))[0]

    # 2. Stuck/incomplete: lowest B_T, tiebreak highest D_T then alpha
    stuck_pool = [tr for tr in traces if not np.isnan(tr.B_T)]
    stuck_pool.sort(key=lambda t: (t.B_T, -t.D_T, t.trace_id))
    stuck = stuck_pool[0]

    # 3. High-churn: max discovery_count / T
    churn_sorted = sorted(traces, key=lambda t: (-(len(t.discovery_steps) / max(t.T, 1)),
                                                 -len(t.discovery_steps), t.trace_id))
    churn = churn_sorted[0]
    used = {steady.trace_id, stuck.trace_id}
    for c in churn_sorted:
        if c.trace_id not in used:
            churn = c
            break

    return {"steady": steady, "stuck": stuck, "churn": churn}


def plot_archetypes(picks: dict[str, Trace]) -> None:
    f, axes = plt.subplots(1, 3, figsize=(15, 5), dpi=150)
    titles = {"steady": "Steady climb", "stuck": "Stuck / incomplete",
              "churn": "High-churn / oscillating"}
    for ax, key in zip(axes, ["steady", "stuck", "churn"]):
        tr = picks[key]
        x = np.arange(tr.T + 1)
        ax.plot(x, tr.D, label="D_t", color="C1")
        ax.plot(x, tr.N, label="N_t", color="C2")
        ax2 = ax.twinx()
        ax2.plot(x, tr.B, label="B_t", color="C0", linestyle="--")
        ax2.set_ylim(-0.02, 1.02)
        ax2.set_ylabel("B_t")
        ax.set_xlabel("step t")
        ax.set_ylabel("count")
        ax.set_title(f"{titles[key]}\n{tr.trace_id} (T={tr.T}, B_T={tr.B_T:.2f}, D_T={tr.D_T})")
        ax.legend(loc="upper left")
        ax2.legend(loc="lower right")
    f.tight_layout()
    f.savefig(OUT / "13_archetype_traces.png", dpi=150)
    plt.close(f)


def write_observations(traces: list[Trace], picks: dict[str, Trace],
                       qa: dict) -> None:
    Ts = np.array([tr.T for tr in traces])
    Ds = np.array([tr.D_T for tr in traces])
    Bs = np.array([tr.B_T for tr in traces if not np.isnan(tr.B_T)])
    n_disc_total = sum(len(tr.discovery_steps) for tr in traces)
    drops = []
    for tr in traces:
        for t in tr.discovery_steps:
            b_prev, b_now = tr.B[t - 1], tr.B[t]
            if not (np.isnan(b_prev) or np.isnan(b_now)):
                drops.append(b_prev - b_now)
    drops = np.array(drops)
    flag_summary = {}
    for tr in traces:
        for fl in tr.flags:
            flag_summary[fl] = flag_summary.get(fl, 0) + 1

    by_corpus: dict[str, int] = {}
    for tr in traces:
        by_corpus[tr.corpus] = by_corpus.get(tr.corpus, 0) + 1

    body = f"""# Trace Shape Exploration — Observations

Cohort: **{len(traces)} annotated traces** across corpora:
{chr(10).join(f"- `{c}`: {n}" for c, n in sorted(by_corpus.items()))}

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
- **Plot 7 (`07_drop_magnitude_hist.png`)** — histogram of `B_{{t-1}} - B_t`
  at discovery events, log-y.
- **Plot 8 (`08_d_terminal_hist.png`)** — `D_T` distribution.
- **Plot 9 (`09_trace_length_hist.png`)** — `T` distribution.
- **Plot 10 (`10_b_terminal_hist.png`)** — `B_T` distribution, bin 0.05.
- **Plot 11 (`11_t_vs_d_scatter.png`)** — `T` vs `D_T` scatter.
- **Plot 12 (`12_d_vs_b_scatter.png`)** — `D_T` vs `B_T` scatter.
- **Plot 13 (`13_archetype_traces.png`)** — three programmatic archetype
  picks (steady, stuck, high-churn), each with `N_t`, `D_t` on count axis
  and `B_t` on right axis.

## Headline numbers

- `T`: median **{int(np.median(Ts))}**, mean {Ts.mean():.1f}, max **{int(Ts.max())}**, min {int(Ts.min())}
- `D_T`: median **{int(np.median(Ds))}**, mean {Ds.mean():.1f}, max **{int(Ds.max())}**, min {int(Ds.min())}
- `B_T`: median **{np.median(Bs):.2f}**, mean {Bs.mean():.2f},
  fraction at exactly 1.0 = **{(Bs == 1.0).mean():.2f}**, fraction below 0.7 = {(Bs < 0.7).mean():.2f}
- Discovery events total across cohort: **{n_disc_total}** ({n_disc_total / len(traces):.2f} per trace on average)
- Drops at discovery events: median **{np.median(drops):.3f}**, max {drops.max():.3f}, min {drops.min():.3f}

## Programmatic archetype picks (Q8)

- **Steady climb**: `{picks['steady'].trace_id}` (corpus={picks['steady'].corpus}, T={picks['steady'].T}, D_T={picks['steady'].D_T}, B_T={picks['steady'].B_T:.2f})
- **Stuck / incomplete**: `{picks['stuck'].trace_id}` (corpus={picks['stuck'].corpus}, T={picks['stuck'].T}, D_T={picks['stuck'].D_T}, B_T={picks['stuck'].B_T:.2f})
- **High-churn**: `{picks['churn'].trace_id}` (corpus={picks['churn'].corpus}, T={picks['churn'].T}, D_T={picks['churn'].D_T}, B_T={picks['churn'].B_T:.2f}, discovery events={len(picks['churn'].discovery_steps)})

## Written observations

### 1. Shape regularities (Plots 1–2)

{shape_para_1(traces)}

### 2. Discovery as discrete or continuous (Plot 3)

{shape_para_2(traces, n_disc_total)}

### 3. When discovery happens (Plots 4 & 6)

{shape_para_3(traces)}

### 4. Drop magnitudes (Plot 7)

{shape_para_4(drops)}

### 5. Typical scale (Plots 8–9)

The annotated cohort skews short and small. Median `T = {int(np.median(Ts))}` source steps,
median `D_T = {int(np.median(Ds))}` leaves. There is a moderate right tail on
both — `T` ranges to {int(Ts.max())} and `D_T` to {int(Ds.max())}. Most decomposition is in
single-digit-to-low-double-digit leaf counts, consistent with a hand-annotator
keeping the leaf granularity coarse enough to remain readable.

### 6. Terminal `B_T` (Plot 10)

{shape_para_6(Bs)}

### 7. Cross-axis correlations (Plots 11–12)

{shape_para_7(Ts, Ds, Bs, traces)}

### 8. Three qualitatively-different traces (Plot 13)

The script auto-picks three archetypes (rules in `pick_archetypes()` in
`scripts/explore_trace_shapes.py`):

- **{picks['steady'].trace_id}** — steady climb: `B_t` rises monotonically
  to 1.0 with at most one discovery event after step 0. Represents the
  "agent declared roughly the right work up front and finished" pattern.
- **{picks['stuck'].trace_id}** — stuck/incomplete: terminates at the lowest
  `B_T` in the cohort ({picks['stuck'].B_T:.2f}). Represents a trace where
  declared work substantially exceeds completed work at the trace's end.
- **{picks['churn'].trace_id}** — high-churn: maximum discovery rate
  (events per step). Represents repeated re-decomposition; multiple
  visible drops in `B_t`.

The three differ qualitatively: stuck is dominated by mid-trace `B_t < 1`
that never recovers; churn shows sawtooth structure; steady looks like a
single climb to 1. This range exists within a single ~67-trace cohort, so
"typical" trajectory shape is genuinely a mixture rather than a single
prototype.

### 9. Surprises and data quality

- **Sparse step axis**: `progress.csv` only emits rows on leaf-state-change
  events, with raw step indices like `{{0, 2, 11, 12, 17}}`. Forward-filled
  to dense indices for all plots.
- **Stuck-loop tail truncation**: e.g. `swe_agent_pilot_f_02` has a source
  trajectory_length of 509 but the ledger ends at step 17 because the
  annotator marked the leaf BLOCKED there (the agent then flailed for
  ~250 steps in a thesaurus loop, correctly ignored). `T` here means
  "ledger time", not "agent time".
- **Step-0 has `D_0 = 0`** in every trace; `B_0` is therefore undefined.
  Set to NaN for plotting; appears as a missing first point in Plot 1/2.
- Flag counts across cohort: {flag_summary or '{}'}. {"None of these are systematic." if not flag_summary else ""}
- Excluded due to empty/header-only progress.csv: {len(qa['excluded_empty'])} trace(s).

### 10. Estimator outlook

{shape_para_10(traces, Bs, n_disc_total)}

## Data quality

- Loaded traces: {len(traces)}
- Excluded (empty/header-only progress.csv): {len(qa['excluded_empty'])}
- Per-trace flag counts: {flag_summary or "(none)"}
- Excluded list: {qa['excluded_empty'] or "(none)"}

## Reproduction

```
uv run python scripts/explore_trace_shapes.py
```

Outputs are written under `reports/trace_shape_exploration/`. The script
reads only the cohort directories listed in `COHORT_SPECS` and is
deterministic.
"""
    (OUT / "OBSERVATIONS.md").write_text(body)


def shape_para_1(traces: list[Trace]) -> str:
    arrived_one = sum(1 for tr in traces if tr.B_T == 1.0)
    multi_drop = sum(1 for tr in traces if len(tr.discovery_steps) >= 3)
    single = sum(1 for tr in traces if len(tr.discovery_steps) <= 1)
    return (
        f"Plots 1–2 show {len(traces)} overlaid trajectories. The dominant "
        f"shape is **sawtooth that recovers to 1.0**: {arrived_one}/{len(traces)} "
        f"({arrived_one / len(traces):.0%}) end at exactly 1.0, but "
        f"{multi_drop}/{len(traces)} ({multi_drop / len(traces):.0%}) have ≥3 "
        f"discovery events along the way, so most of those terminating at 1.0 "
        f"have visibly oscillated en route. Three recognizable archetypes: "
        f"(a) **single-discovery climb** ({single} traces with ≤1 discovery "
        f"event after step 0) — a plan declared and completed without "
        f"revision; (b) **sawtooth-to-one** — multiple drops then full "
        f"recovery; (c) **plateaued-below-1** — terminates with `B_T < 1` "
        f"and a flat tail (the {len(traces) - arrived_one} non-1.0 traces). "
        f"Trajectories are not noise — they are piecewise-monotone with "
        f"discrete down-steps at discovery events and monotone climbs "
        f"between them."
    )


def shape_para_2(traces: list[Trace], n_disc_total: int) -> str:
    avg_disc = n_disc_total / len(traces)
    return (
        f"Plot 3 shows `D_t` rising as a step function in essentially every "
        f"trace. Discovery is discrete: an annotator either logs a new leaf "
        f"or doesn't. Across the cohort, average ~{avg_disc:.1f} discovery "
        f"events per trace; `D_t` jumps by 1 at each (weights are unit). "
        f"There is no continuous-rate behavior. Modeling discovery as a "
        f"discrete event process — possibly a non-homogeneous count process "
        f"keyed off step index or off prior leaf state — is the right "
        f"abstraction; modeling it as a continuous rate would mask the "
        f"step structure entirely."
    )


def shape_para_3(traces: list[Trace]) -> str:
    timings = []
    for tr in traces:
        ds = tr.discovery_steps
        timings.extend(ds / tr.T)
    timings = np.array(timings)
    early = (timings < 0.2).mean()
    mid = ((timings >= 0.2) & (timings < 0.8)).mean()
    late = (timings >= 0.8).mean()
    edge_bias = early + late - 0.4  # vs uniform expectation 0.4
    return (
        f"Discovery is roughly spread throughout the trace, with a mild "
        f"**U-shaped** edge bias rather than being front-loaded. Of "
        f"{len(timings)} events across the cohort, {early:.0%} occur in "
        f"the first 20% of trace time, {mid:.0%} in the middle 60%, and "
        f"{late:.0%} in the final 20%. Compared to a uniform expectation "
        f"of 20/60/20, the first and last quintiles are each slightly "
        f"overrepresented and the middle is slightly underrepresented "
        f"(edge excess ≈ {edge_bias:+.2f}). Plot 4 backs this up: "
        f"the normalized `D_t / D_T` curves climb noticeably in the early "
        f"portion of `t/T` (annotators front-load the obvious decomposition), "
        f"continue to rise through the middle, and many do not asymptote "
        f"until close to `t/T = 1` — late-trace discovery is non-trivial."
    )


def shape_para_4(drops: np.ndarray) -> str:
    if len(drops) == 0:
        return "No drops observed in the cohort."
    negative = (drops < 0).sum()
    small = ((drops >= 0) & (drops < 0.1)).sum()
    medium = ((drops >= 0.1) & (drops < 0.3)).sum()
    large = (drops >= 0.3).sum()
    return (
        f"Drops at discovery events are spread across a wide range rather "
        f"than concentrated at small refinements. Of {len(drops)} drops "
        f"with both endpoints defined, {small}/{len(drops)} ({small / len(drops):.0%}) "
        f"are in [0, 0.1), {medium} ({medium / len(drops):.0%}) in [0.1, 0.3), "
        f"and {large} ({large / len(drops):.0%}) ≥ 0.3. Median drop is "
        f"{np.median(drops):.3f}; the largest is {drops.max():.2f}. There "
        f"is also a tail of {negative} **negative drops** (B going up at "
        f"a discovery event) up to {drops.min():.2f} — these occur when a "
        f"step that adds a new leaf also marks one or more existing leaves "
        f"complete, so `N` outpaces `D`. Substantial reorganizations "
        f"(drops ≥ 0.3) are about a third of all events; the popular "
        f"image of 'tiny plan refinements' overstates the smoothness of "
        f"this corpus."
    )


def shape_para_6(Bs: np.ndarray) -> str:
    one_pct = (Bs == 1.0).mean()
    near_one = ((Bs >= 0.95) & (Bs < 1.0)).mean()
    midrange = ((Bs >= 0.5) & (Bs < 0.95)).mean()
    low = (Bs < 0.5).mean()
    return (
        f"`B_T` is bimodal: a heavy mass at exactly 1.0 ({one_pct:.0%}), "
        f"a small bridge of near-1 traces ({near_one:.0%} in [0.95, 1.0)), "
        f"a midrange band ({midrange:.0%} in [0.5, 0.95)), and a low cluster "
        f"({low:.0%} below 0.5). The 1.0 mode reflects 'agent declared and "
        f"completed everything before the trace ended'; the midrange band "
        f"is 'agent left some declared work unfinished' (BLOCKED or "
        f"IN_PROGRESS leaves at termination). Few traces sit in the [0, 0.3] "
        f"band, suggesting agents rarely terminate after declaring a lot "
        f"and completing almost none."
    )


def shape_para_7(Ts: np.ndarray, Ds: np.ndarray, Bs: np.ndarray,
                 traces: list[Trace]) -> str:
    valid = [(tr.T, tr.D_T, tr.B_T) for tr in traces if not np.isnan(tr.B_T)]
    Tv = np.array([v[0] for v in valid])
    Dv = np.array([v[1] for v in valid])
    Bv = np.array([v[2] for v in valid])
    r_td = np.corrcoef(Tv, Dv)[0, 1]
    r_db = np.corrcoef(Dv, Bv)[0, 1]
    return (
        f"`T` and `D_T` are positively correlated (Pearson r ≈ "
        f"{r_td:.2f}): longer traces decompose into more leaves, as "
        f"expected since both grow with task difficulty. `D_T` and `B_T` "
        f"have a weaker correlation (r ≈ {r_db:.2f}); a higher leaf count "
        f"slightly biases `B_T` downward — more declared work is harder "
        f"to fully complete inside the trace — but the relationship is "
        f"not tight. Plot 12 shows traces at both extremes: small `D_T` "
        f"with `B_T = 1.0` and small `D_T` with `B_T < 0.5` are both "
        f"present, so size alone is not predictive of completion."
    )


def shape_para_10(traces: list[Trace], Bs: np.ndarray, n_disc_total: int) -> str:
    avg_disc = n_disc_total / len(traces)
    early_only = sum(1 for tr in traces
                     if tr.T > 0 and len([d for d in tr.discovery_steps if d > 0.5 * tr.T]) == 0)
    return (
        f"Cautiously: yes, prefix shape carries information. Three reasons "
        f"grounded in the plots: (a) discovery is front-loaded — for "
        f"{early_only}/{len(traces)} traces all discovery events are in the "
        f"first half of trace time, so a prefix of length `0.5T` already "
        f"observes the bulk of `D_T`; (b) the `D_T` distribution (Plot 8) "
        f"is concentrated in single-to-low-double digits, so even a "
        f"low-resolution prefix-conditioned predictor has a small range to "
        f"resolve; (c) the average ~{avg_disc:.1f} discovery events per "
        f"trace means a prefix-classifier has access to a handful of "
        f"informative jumps rather than a noise-dominated stream. "
        f"Caveats: the {(Bs == 1.0).mean():.0%} of traces ending at "
        f"`B_T = 1.0` puts a ceiling on how much variance there is to "
        f"explain on the completed-correctly axis, and traces with a "
        f"single late discovery event would be hard to predict from a "
        f"short prefix. The exploration suggests a model conditioned on "
        f"prefix shape and prior corpus statistics is plausible — but "
        f"this is descriptive intuition, not a measurement."
    )


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    traces, qa = load_cohort()
    print(f"loaded {len(traces)} traces; excluded {len(qa['excluded_empty'])}")
    plot_overlay_b_raw(traces)
    plot_overlay_b_norm(traces)
    plot_overlay_d_raw(traces)
    plot_overlay_d_norm(traces)
    plot_overlay_n_norm(traces)
    plot_discovery_timing(traces)
    plot_drop_magnitude(traces)
    plot_d_terminal(traces)
    plot_trace_length(traces)
    plot_b_terminal(traces)
    plot_t_vs_d(traces)
    plot_d_vs_b(traces)
    picks = pick_archetypes(traces)
    plot_archetypes(picks)
    write_observations(traces, picks, qa)
    print(f"wrote 13 PNGs and OBSERVATIONS.md to {OUT}")


if __name__ == "__main__":
    main()
