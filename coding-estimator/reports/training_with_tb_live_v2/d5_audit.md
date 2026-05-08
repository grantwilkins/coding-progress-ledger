# D5 behavioral leakage audit

- schema_version: `1.1.0`
- n_runs_audited: 102
- n_checkpoints_audited: 703
- clean: **True**
- findings: 0

## Methodology notes

- **Shuffle test** uses run-level shuffling for run-constant targets and row-level shuffling otherwise. Skipped (with `severity: info`) when n_runs < 8 or n_unmasked_rows < 30 on a (source, target) cell, because seed-variance on AUROC dominates below those floors. A finding fires when `|mean_AUROC - 0.5| > 0.1` across 3 seeds.
- **Prefix-truncation test** rebuilds the row at `mid_step` from a ledger truncated at `mid_step` and asserts byte-equality with the same row built from the full ledger.
- Findings of severity `info` are reported but do not flip `clean: false`.

## Section results

| section | findings | detail key counts |
|---|---:|---|
| structural | 0 | keys=['forbidden_exact', 'forbidden_prefix', 'forbidden_suffix', 'hits'] |
| prefix_truncation | 0 | keys=['differing_runs', 'sampled_runs', 'skipped_runs'] |
| shuffle | 0 | keys=['min_checkpoints', 'min_runs', 'sources', 'tolerance'] |
| run_constancy | 0 | keys=['audited_cells', 'offenders'] |

