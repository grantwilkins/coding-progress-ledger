# Golden semantic fixture (D0)

This fixture is the **executable definition** of feature semantics for the
coding-estimator. The prefix replay engine (D2) and every feature builder
(D3) MUST produce values matching `expected_checkpoints.json` at every
step. Drift between the fixture and the implementation is by definition a
bug in the implementation — the fixture is authority.

## Contents

| file | purpose |
|---|---|
| `ledger.jsonl` | canonical 14-step run; covers every required event type |
| `ledger_mutated_after_t_mid.jsonl` | identical to canonical for steps 0..10; diverges past `t_mid=10` |
| `expected_checkpoints.json` | hand-derived per-step expected aggregates |

## Required event-type coverage

Every event type required by the post-D plan invariants is present:

| event type | step(s) |
|---|---|
| `init` | 0 |
| `add_subtask` | 1, 3, 4 |
| `update_status` → `in_progress` | 1, 3, 4 (plus split-spawned children at 5) |
| `update_status` → `complete` | 2, 6, 7, 12, 13 |
| `update_status` → `blocked` | 11 |
| `add_evidence` | 8 |
| `split_subtask` | 5 |
| `reopen_subtask` | 9 (this is the canonical strict drop event) |
| `invalidate_subtask` | 10 |
| validation pass (validation-leaf → complete) | 7 |
| validation fail (validation-leaf → invalidate) | 10 |
| `delete_subtask` | 13 (mutated only) |

Five strict progress drops occur at or before `t_mid=10`: at steps 3, 4,
5, 9, and 10. Largest drop magnitude is 0.5 (step 3, when adding `s2`
breaks the 1/1 progress). Step 5 demonstrates a split-induced drop;
step 9 a reopen drop; step 10 an invalidation drop.

## Future-mutation invariance

`ledger.jsonl` and `ledger_mutated_after_t_mid.jsonl` are **byte-identical
through the line that ends with step 10**. Past `t_mid=10` they diverge:

- canonical: `s2` proceeds, both `s2a` and `s2b` complete, run succeeds.
- mutated:   `s2a`, `s2b` are invalidated; `s2` is deleted; run "fails."

The prefix replay engine (D2) and every feature builder (D3) must produce
**identical** feature values at every checkpoint `t <= 10` for both
fixtures. Any divergence at `t <= 10` is a future-leakage bug.

## Hand-derivation, then upstream cross-check

Expected aggregates were derived from the leaf semantics declared by the
upstream `score()` function (a leaf is an active subtask whose active
children, if any, do not exist; a coding leaf belongs to product /
validation / investigation; progress = complete leaf weight / active
leaf weight over coding categories).

After hand-derivation, each value was independently confirmed by running
upstream `replay()` + `score()` on the same fixture. The two agree
exactly. The hand derivation is authoritative; the upstream cross-check
guards against arithmetic typos.

## Reproducing the upstream cross-check

```bash
uv run python - <<'PY'
from ledger_progress.serialization import load_events_jsonl
from ledger_progress.core import replay
from ledger_progress.scoring import score
from ledger_progress.queries import CODING_CATEGORIES

events = load_events_jsonl('tests/fixtures/golden_run/ledger.jsonl')
for t in range(0, max(e.step for e in events) + 1):
    prefix = [e for e in events if e.step <= t]
    obs = score(replay(prefix), categories=CODING_CATEGORIES)
    print(f"t={t} AL={obs.active_leaf_count} CL={obs.complete_leaf_count} CP={obs.progress:.4f}")
PY
```
