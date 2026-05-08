# Source-level profile (F1)

Per canonical source: run count, label availability, success/failure split, wallclock coverage, run-length quantiles. Generated from the combined manifest (C5) + optional checkpoint frame.

| source | n_runs | n_succ | n_fail | n_unres | n_real_wc | p25_len | p50_len | p75_len | n_ckpts | n_wc_ckpts |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| hermes_pilot_h5_v2 | 30 | 0 | 0 | 30 | 0 | 19.2 | 23.0 | 30.5 | 896 | 0 |
| swe_agent_pilot | 21 | 10 | 10 | 1 | 0 | 21.0 | 28.0 | 33.0 | 599 | 0 |
| tb_live | 12 | 12 | 0 | 0 | 12 | 5.0 | 6.0 | 6.2 | 83 | 83 |
| tb_live_v2 | 102 | 81 | 21 | 0 | 102 | 4.0 | 5.0 | 7.0 | 703 | 703 |

