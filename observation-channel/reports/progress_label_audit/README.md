# Progress Label Audit

Primary question: Does opened-unit progress reach 100% while a large fraction of the trace still remains?

This is a read-only label audit. It does not change the estimator, filter, tracker model, or classification rules.

## Artifacts

- `tail_progress_audit.csv`: all valid traces ranked by remaining fraction after opened-unit progress first reaches 100%.
- `tail_progress_audit_non_artifact.csv`: the same ranking excluding final `ARTIFACT` units.
- `progress_curve_traces.csv` and `progress_curve_traces.png`: opened-unit, closed-unit, and step progress for selected worst traces.
- `inspection_snippets.csv`: compact command/tool and observation evidence around the opened-100% point and trace tail.
- `classifier_final_unit_samples.csv`: longest final units by final category.

## Decision Rubric

If opened-unit progress is bad but closed-unit progress looks sane, switch the belief target from opened units to closed units.

If both opened and closed units are bad, keep units as features and change the prediction target to remaining steps/actions/tool calls.

If category mistakes explain the long tails, fix segmentation/classification before rerunning the tracker.

## Run Summary

- audited traces: 94104
- non-artifact traces: 42566
- selected traces with curve/snippet evidence: 91
