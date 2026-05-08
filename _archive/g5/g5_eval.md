# G5 ledger-dynamics evaluation

_Generated 2026-05-05T05:00:45+00:00._

G2 (`time_only`) vs G4 (`ledger_basic`) vs G5 (`ledger_dynamics`) vs `g4_plus_g5` per source under LORO across the recentered v0 headline targets. Lower Brier is better. Run-level bootstrap 95% CI (B=500).

**Headline framing (v1).** The primary v0 target family is **process dynamics** (`y_future_progress_drop_h5`, `y_validation_new_work_h5`). `y_success_eventual` is a **secondary / negative** target — ledger features do not yet beat elapsed time on it at the current N. Read the dynamics table first; the success table is preserved as a negative result, not a headline.

## Primary headline targets (process dynamics)

| target | source | model | n_runs | n_ckpts | pos | AUROC | Brier | Brier 95% CI | note |
|---|---|---|---:|---:|---:|---:|---:|---|---|
| y_future_progress_drop_h5 | swe_agent_pilot | time_only | 20 | 499 | 0.170 | 0.626 | 0.142 | [0.112, 0.173] |  |
| y_future_progress_drop_h5 | swe_agent_pilot | ledger_basic | 20 | 499 | 0.170 | 0.977 | 0.039 | [0.019, 0.063] |  |
| y_future_progress_drop_h5 | swe_agent_pilot | ledger_dynamics | 20 | 499 | 0.170 | 0.897 | 0.078 | [0.049, 0.107] |  |
| y_future_progress_drop_h5 | swe_agent_pilot | g4_plus_g5 | 20 | 499 | 0.170 | 0.973 | 0.042 | [0.019, 0.066] |  |
| y_future_progress_drop_h5 | tb_live | time_only | 10 | 23 | 0.130 | 0.867 | 0.133 | [0.024, 0.203] |  |
| y_future_progress_drop_h5 | tb_live | ledger_basic | 10 | 23 | 0.130 | 0.917 | 0.096 | [0.011, 0.156] |  |
| y_future_progress_drop_h5 | tb_live | ledger_dynamics | 10 | 23 | 0.130 | 0.233 | 0.124 | [0.018, 0.228] |  |
| y_future_progress_drop_h5 | tb_live | g4_plus_g5 | 10 | 23 | 0.130 | 0.933 | 0.093 | [0.011, 0.152] |  |
| y_validation_new_work_h5 | swe_agent_pilot | time_only | 20 | 499 | 0.008 | 0.037 | 0.016 | [0.000, 0.036] |  |
| y_validation_new_work_h5 | swe_agent_pilot | ledger_basic | 20 | 499 | 0.008 | 0.158 | 0.008 | [0.000, 0.024] |  |
| y_validation_new_work_h5 | swe_agent_pilot | ledger_dynamics | 20 | 499 | 0.008 | 0.149 | 0.008 | [0.000, 0.024] |  |
| y_validation_new_work_h5 | swe_agent_pilot | g4_plus_g5 | 20 | 499 | 0.008 | 0.187 | 0.008 | [0.000, 0.024] |  |
| y_validation_new_work_h5 | tb_live | time_only | 10 | 23 | 0.478 | 0.432 | 0.254 | [0.154, 0.387] |  |
| y_validation_new_work_h5 | tb_live | ledger_basic | 10 | 23 | 0.478 | 0.492 | 0.284 | [0.195, 0.383] |  |
| y_validation_new_work_h5 | tb_live | ledger_dynamics | 10 | 23 | 0.478 | 0.273 | 0.278 | [0.253, 0.310] |  |
| y_validation_new_work_h5 | tb_live | g4_plus_g5 | 10 | 23 | 0.478 | 0.485 | 0.285 | [0.196, 0.384] |  |

## Secondary headline target (terminal success — negative result)

| target | source | model | n_runs | n_ckpts | pos | AUROC | Brier | Brier 95% CI | note |
|---|---|---|---:|---:|---:|---:|---:|---|---|
| y_success_eventual | swe_agent_pilot | time_only | 20 | 599 | 0.467 | 0.281 | 0.283 | [0.263, 0.306] |  |
| y_success_eventual | swe_agent_pilot | ledger_basic | 20 | 599 | 0.467 | 0.410 | 0.291 | [0.245, 0.340] |  |
| y_success_eventual | swe_agent_pilot | ledger_dynamics | 20 | 599 | 0.467 | 0.385 | 0.272 | [0.251, 0.295] |  |
| y_success_eventual | swe_agent_pilot | g4_plus_g5 | 20 | 599 | 0.467 | 0.411 | 0.292 | [0.246, 0.340] |  |
| y_success_eventual | tb_live | time_only | 12 | 83 | 1.000 | n/a | 0.000 | [0.000, 0.000] | single-class y |
| y_success_eventual | tb_live | ledger_basic | 12 | 83 | 1.000 | n/a | 0.000 | [0.000, 0.000] | single-class y |
| y_success_eventual | tb_live | ledger_dynamics | 12 | 83 | 1.000 | n/a | 0.000 | [0.000, 0.000] | single-class y |
| y_success_eventual | tb_live | g4_plus_g5 | 12 | 83 | 1.000 | n/a | 0.000 | [0.000, 0.000] | single-class y |

## Δ Brier vs G2 by (target, source)

Positive = G2 better; negative = the named model better.

| target | source | G2 Brier | G4 - G2 | G5 - G2 | (G4+G5) - G2 |
|---|---|---:|---:|---:|---:|
| y_future_progress_drop_h5 | swe_agent_pilot | 0.142 | -0.102 | -0.064 | -0.100 |
| y_future_progress_drop_h5 | tb_live | 0.133 | -0.037 | -0.009 | -0.039 |
| y_success_eventual | swe_agent_pilot | 0.283 | +0.009 | -0.010 | +0.010 |
| y_success_eventual | tb_live | 0.000 | +0.000 | +0.000 | +0.000 |
| y_validation_new_work_h5 | swe_agent_pilot | 0.016 | -0.008 | -0.008 | -0.008 |
| y_validation_new_work_h5 | tb_live | 0.254 | +0.030 | +0.025 | +0.031 |

