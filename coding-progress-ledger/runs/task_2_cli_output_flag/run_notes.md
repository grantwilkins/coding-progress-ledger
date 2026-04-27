# Run Notes

Progress started at zero while the behavior contract and repo scaffold were
unknown. It increased when the buggy baseline was committed because the task had
a reproducible starting point.

Regression tests initially expanded the active work count, so progress did not
just rise linearly. After the file-output fix looked complete, the `--output -`
sentinel was treated as newly discovered missing behavior and the implementation
work was reopened. That produced the intentional non-monotonic event: progress
dropped when completed implementation work became incomplete again.

Completions were backed by concrete evidence: files created, git baseline
commit, tests added, implementation patch applied, and pytest passing. The
ledger was useful for showing when the task widened and when confidence was
earned by evidence. The awkward part is that a simulated short run needs event
steps assigned after the fact with care, because the coding and observation
channels are separate.
