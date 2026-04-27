# Live Validation Summary

Five live validation runs used `LedgerSession` manually while making small
repo-local improvements. No observer proposal, LLM-driven update path, or
scoring semantic change was implemented.

| Run | final_coding_progress | final_overall_progress | coding_nonmonotonic | overall_nonmonotonic | evidence_audit_status | captured real discovered work |
| --- | ---: | ---: | --- | --- | --- | --- |
| `01_suite_summary_weight_source` | 1.000 | 1.000 | yes | yes | strong | yes: table validation became explicit after the report patch looked complete |
| `02_drop_category_contributions` | 1.000 | 1.000 | yes | yes | strong | yes: contribution export required mixed and single-category fixtures |
| `03_evidence_audit_by_category` | 1.000 | 1.000 | yes | yes | strong | yes: live rescoring exposed that explicit categories were being overwritten |
| `04_docs_progress_not_success` | 1.000 | 1.000 | yes | yes | weak | yes: the agent usage docs needed the same progress-not-success warning as the README |
| `05_active_incomplete_coding_leaves` | 1.000 | 1.000 | yes | yes | strong | yes: the helper task split into a reusable query and CLI wrapper |

## Findings

The ledger remained useful during natural work. It captured denominator growth
from late validation, docs scope expansion, and one real reporting bug:
`rescore_run` had been inferring categories over explicit `LedgerSession`
categories. That issue would have made per-category evidence audit results less
honest, and the ledger made the discovery visible as new active work rather
than hidden cleanup.

Evidence quality improved relative to the toy suite. Four of five live runs
audited as strong, compared with several weak toy runs in the suite summary.
The weak live run was the docs run, where the investigation completion evidence
was mostly manual narrative. Product and validation completions were generally
backed by `final_diff.patch`, focused pytest output, regenerated summaries, or
CLI output.

Ledger maintenance was most awkward around artifact leaves: saving
`task.md`, notes, patches, test output, summaries, and the ledger itself is
administrative work. It was least awkward when the task mapped directly to
ledger concepts, especially the active incomplete coding leaves helper.

Observer automation does not yet look justified for semantic ledger updates.
The manual ledger produced useful, inspectable traces without changing scoring
or introducing LLM calls. Limited automation for artifact packaging, export
checks, or command helpers does look justified; automatic observer updates
would still risk making category and completion judgments less explicit.
