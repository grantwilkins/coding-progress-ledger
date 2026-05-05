# NOT_READY_FOR_SCHEDULING — ledger_basic_v0.1

_Generated 2026-05-05T04:43:11+00:00._

## Verdict: ⚠️ INDETERMINATE

Workstream P1 has NOT cleared the no-regression gate. This document records which conditions blocked the gate and the cheapest next experiment for each.

## Conditions that blocked

- `P1.b` (indeterminate): single-class y on tb_live for `y_success_eventual` (N=12 cohort is currently 12/12 successes)
- `P1.c` (indeterminate): hermes_pilot_h5_v2 labels not built into `datasets/labels_all.parquet` — combined retrospective is not testable as the plan defines it
- `P1.d` (indeterminate): single-class y on tb_live for `y_success_eventual`
- `P1.g` (indeterminate): D5 audit artifact not provided; Workstream M is deferred — re-evaluate this condition once D5 ships with required fields ['schema_version', 'n_runs_audited', 'n_checkpoints_audited', 'findings', 'clean']

## Recommended next experiments (P + O)

Recommendations are tagged BLOCKING (must land first), DATA (unblocks indeterminate gates), or AUDIT (process artifact).

- **BLOCKING (O7)** the v0 ledger features do not carry decision-relevant signal beyond elapsed time on ['swe_agent_pilot']. Cheapest next experiment: add the deferred dynamics group (G5) and re-run O7.
- **DATA (P1.b)** tb_live cohort is 12/12 successes — collect at least 5 tb_live failures before this gate is even testable.
- **DATA (P1.c)** build hermes_pilot_h5_v2 labels into `datasets/labels_all.parquet` so the combined retrospective (~50 runs) is testable as the plan intended.
- **DATA (P1.d)** tb_live cohort is 12/12 successes — collect at least 5 tb_live failures before this gate is even testable.
- **AUDIT (P1.g)** ship the D5 behavioral leakage audit artifact (Workstream M deferred → D5 substitute is still required).

## Do not consume

- This estimator MUST NOT drive scheduling, modulation, or any other control action.
- `not_safe_for_control` flag in `model_card.json` is `true`; consumers MUST hard-block on this flag.
