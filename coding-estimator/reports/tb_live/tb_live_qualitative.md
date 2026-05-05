# TB-live qualitative rollup (K3)

_Generated 2026-05-05T02:55:05+00:00._

K3 — one rollup across the TB-12 cohort. Stuck-loop precursor checkpoint = first step where `no_progress_window_5 >= 5`. Max Δp uses ledger_basic on `y_success_eventual` predictions.

## Cohort summary

- runs: 12
- runs with `no_progress_window_5 >= 5`: 0 (0%)
- runs with `repeated_observation_loop_flag` ever set: 0 (0%)
- runs with at least one validation attempt: 12 (100%)
- runs with at least one validation success: 12 (100%)

## Shape distribution

_no shape labels available for tb_live (live source — expected)_

## Per-run rollup

| run_id | n_ckpts | final_prog | first_no_prog (phase) | first_repeated_loop | first_validation | first_v_success | first_v_failure | max Δp (step) | shape tags | final_success |
|---|---:|---:|---|---:|---:|---:|---:|---|---|---:|
| b-tree-on-disk | 7 | 1.00 | - | - | 3 | 4 | - | 0.000 (1) | - | 1 |
| csv-streaming-dedup | 7 | 1.00 | - | - | 3 | 4 | - | 0.000 (1) | - | 1 |
| decouple-state-from-controller | 6 | 1.00 | - | - | 4 | 5 | - | 0.000 (1) | - | 1 |
| directory-watcher-log-rotator | 7 | 1.00 | - | - | 5 | 6 | - | 0.000 (1) | - | 1 |
| fix-broken-pyproject-build | 6 | 1.00 | - | - | 4 | 5 | - | 0.000 (1) | - | 1 |
| graph-tarjan-scc | 8 | 1.00 | - | - | 5 | 6 | - | 0.000 (1) | - | 1 |
| lru-cache-threadsafe | 7 | 1.00 | - | - | 5 | 6 | - | 0.000 (1) | - | 1 |
| markdown-to-html-cli | 6 | 1.00 | - | - | 3 | 4 | - | 0.000 (1) | - | 1 |
| recover-corrupted-sqlite | 9 | 1.00 | - | - | 4 | 5 | - | 0.000 (1) | - | 1 |
| sliding-window-rate-limiter | 5 | 1.00 | - | - | 3 | 4 | - | 0.000 (1) | - | 1 |
| tar-extract-with-traversal-guard | 10 | 1.00 | - | - | 5 | 6 | - | 0.000 (1) | - | 1 |
| xss-filter-bypass-then-fix | 5 | 1.00 | - | - | 3 | 4 | - | 0.000 (1) | - | 1 |

