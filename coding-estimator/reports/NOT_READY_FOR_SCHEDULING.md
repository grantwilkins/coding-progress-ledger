# NOT_READY_FOR_SCHEDULING — ledger_basic_v0.1

_Generated 2026-05-05T04:29:36+00:00._

## Verdict: ❌ FAIL

Workstream P1 has NOT cleared the no-regression gate. This document records which conditions blocked the gate and the cheapest next experiment for each.

## Conditions that blocked

- `P1.b` (indeterminate): single-class y on tb_live for `y_success_eventual` (N=12 cohort is currently 12/12 successes)
- `P1.c` (fail): Δ Brier (G2 − G4) = -0.009, 95% CI = [-0.050, +0.030]; CI INCLUDES zero
- `P1.d` (indeterminate): single-class y on tb_live for `y_success_eventual`
- `P1.g` (indeterminate): D5 audit artifact not provided; Workstream M is deferred — re-evaluate this condition once D5 ships

## Cheapest next experiments

- O7 timeout-bias **FAIL**: the v0 ledger features do not carry decision-relevant signal beyond elapsed time on ['swe_agent_pilot']. The cheapest next experiment is to add the deferred dynamics feature group (G5) and re-run O7.
- P1.b indeterminate: tb_live cohort is 12/12 successes — collect at least 5 tb_live failures before this gate is even testable.
- P1.d indeterminate: tb_live cohort is 12/12 successes — collect at least 5 tb_live failures before this gate is even testable.
- P1.g indeterminate: ship the D5 behavioral leakage audit artifact (Workstream M deferred → D5 substitute is still required).

## Do not consume

- This estimator MUST NOT drive scheduling, modulation, or any other control action.
- `not_safe_for_control` flag in `model_card.json` is `true`; consumers MUST hard-block on this flag.
