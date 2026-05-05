# D5 behavioral leakage audit

- schema_version: `1.0.0`
- n_runs_audited: 62
- n_checkpoints_audited: 1578
- clean: **False**
- findings: 1

## Section results

| section | findings | detail key counts |
|---|---:|---|
| structural | 0 | keys=['forbidden_exact', 'forbidden_prefix', 'forbidden_suffix', 'hits'] |
| prefix_truncation | 0 | keys=['differing_runs', 'sampled_runs', 'skipped_runs'] |
| shuffle | 1 | keys=['sources', 'tolerance'] |
| run_constancy | 0 | keys=['audited_cells', 'offenders'] |

## Findings

- `shuffle` / `shuffled_auroc_excursion`: tb_live / y_validation_new_work_h5: mean AUROC on label-shuffled data is 0.167; |Δ from 0.5| > 0.1
