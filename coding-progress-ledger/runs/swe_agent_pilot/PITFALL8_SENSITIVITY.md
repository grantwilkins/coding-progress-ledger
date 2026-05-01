# W4 — Pitfall #8 sensitivity report

Status: **W2 may proceed.** All shape tags for `f_02`/`f_03`/`f_07`/`f_10` are stable across the Pitfall #8 cleanup shift, provided W2's `high_progress_failure` threshold is set at or above 0.70.

## 1. Question

If a follow-up E1 cleanup pass applies Revision 1 (Pitfall #8 — bug-fix tasks always have implicit validation) consistently to the four pilots that v1 missed, scalar `coding_progress` shifts on each. W4 asks whether the W2 shape labels — the audit tags W2 will derive from the same ledgers — change as a result, or whether they survive the scalar move and W2 can proceed against the current values.

## 2. Scope

Four pilots. The H4 gate (`runs/swe_agent_pilot_reannotation/H4_GATE_RESULT.md` § 7) identified them as the cases v1 left without an implicit-VAL leaf:

| Pilot | Instance | Old leaves | New leaves |
|-------|----------|-----------:|-----------:|
| `f_02` | `asottile__pyupgrade-933` | 2 | 3 |
| `f_03` | `asottile__setup-cfg-fmt-132` | 2 | 3 |
| `f_07` | `openstack-charmers__zaza-36` | 3 | 4 |
| `f_10` | `walles__px-50` | 3 | 4 |

`f_05` and `f_08` already include a VAL leaf in v1, so Pitfall #8 leaves them unchanged.

## 3. Scalar shift

Computed by replaying the existing `runs/swe_agent_pilot/swe_agent_pilot_<id>/ledger.jsonl` and synthesizing one extra `ADD_SUBTASK` (category=`validation`, status=`not_started`, weight=1.0) at `last_step + 1` per pilot.

| Pilot | Old `coding_progress` | Revised `coding_progress` | Δ | Crosses 0.50? | Crosses 0.70? |
|-------|---:|---:|---:|:-:|:-:|
| `f_02` | 0.5000 | 0.3333 | −0.1667 | yes (0.50 → 0.33) | no |
| `f_03` | 0.5000 | 0.3333 | −0.1667 | yes (0.50 → 0.33) | no |
| `f_07` | 0.6667 | 0.5000 | −0.1667 | no (both ≥ 0.50) | no |
| `f_10` | 0.6667 | 0.5000 | −0.1667 | no (both ≥ 0.50) | no |

All four shifts are exactly the `1 / (n+1)` ratio H4 predicted; replay confirms the math.

## 4. Shape-label sensitivity

W2's initial tag vocabulary (TASKS.md § W2):

```
high_progress_failure
low_progress_success
stuck_loop
submit_without_validation
no_validation_frontier
validation_induced_reopen
scope_discovery_after_high_progress
hidden_work_gap
nonmonotone_recovery
```

Per-pilot evaluation. The dominant evidence comes from `annotations/swe_agent_pilot/swe_agent_pilot_<id>.notes.md` and the leaf states recorded in each ledger.

### `f_02` — pyupgrade thesaurus loop

- 250+ consecutive `find_file` calls returning identical `"No matches found"` responses, vocabulary devolving from concrete programming terms to thesaurus synonyms; harness force-quit at step 508.
- Old leaves: `S1` INVESTIGATION complete, `S2` INVESTIGATION blocked.
- Revised: + `SVAL` VALIDATION not_started.

| Tag | Old | Revised | Notes |
|-----|:-:|:-:|-----|
| `stuck_loop` | yes | yes | derived from repeated-identical-observation pattern; scalar-invariant |
| `submit_without_validation` | no | no | agent never issued `submit`; harness force-quit |
| `no_validation_frontier` | yes (no VAL leaf at all) | yes (VAL leaf exists at not_started — no validation *attempted*) | depends on tag definition; see § 5 |
| `high_progress_failure` | no (0.50) | no (0.33) | below threshold either way |
| `hidden_work_gap` | no | no | the failure is a search-strategy collapse, not work the agent did but couldn't surface |
| Others | no | no | — |

**Dominant tag:** `stuck_loop`. **Stable across the shift.**

### `f_03` — setup-cfg-fmt find_file loop

- 4-command cycle iterating from step 22 to step 112 (~23 cycles), pattern observably stuck per general § 6 (a) command-loop variant.
- Old leaves: `S1` INVESTIGATION complete, `S2` INVESTIGATION blocked.

Identical structure to f_02. **Dominant tag:** `stuck_loop`. **Stable across the shift.**

### `f_07` — zaza model defaults, validation-script loop

- Real PRODUCT edit landed at step 17 (file size +6 lines). Agent then pivoted to validation by writing a `test_config.py` repro and got stuck on five identical `edit 1:20` rewrites of that script (steps 36–45) before harness force-quit.
- Old leaves: `S1` INVESTIGATION complete, `S2` PRODUCT complete, `S3` VALIDATION blocked.

| Tag | Old | Revised | Notes |
|-----|:-:|:-:|-----|
| `stuck_loop` | yes | yes | derived from validation-script edit cycle, scalar-invariant |
| `validation_induced_reopen` | no | no | the agent didn't reopen PRODUCT; got stuck inside VAL |
| `submit_without_validation` | no | no | never submitted |
| `no_validation_frontier` | no | no | a VAL leaf already existed in v1 (at blocked); revised adds a *second* VAL leaf at not_started — VAL frontier remains present in both worlds |
| `high_progress_failure` | borderline at 0.67 | no at 0.50 | **only label whose value depends on threshold choice** — see § 5 |
| Others | no | no | — |

**Dominant tag:** `stuck_loop`. **Stable across the shift if `high_progress_failure` threshold is ≥ 0.70.**

### `f_10` — px regex edit-syntax-error loop

- ~30+ identical `edit 17:32` rewrites of `px_loginhistory.py`, every one rejected with the same syntax error; harness force-quit at step 80.
- Old leaves: `S1` INVESTIGATION complete, `S2` INVESTIGATION complete, `S3` PRODUCT blocked.

| Tag | Old | Revised | Notes |
|-----|:-:|:-:|-----|
| `stuck_loop` | yes | yes | derived from repeated-identical-edit-rejection pattern, scalar-invariant |
| `submit_without_validation` | no | no | never submitted; never even landed a syntactically valid edit |
| `no_validation_frontier` | yes (no VAL leaf) | yes (VAL leaf at not_started — no validation work attempted) | same edge as f_02 |
| `high_progress_failure` | borderline at 0.67 | no at 0.50 | same as f_07 |
| Others | no | no | — |

**Dominant tag:** `stuck_loop`. **Stable across the shift if `high_progress_failure` threshold is ≥ 0.70.**

## 5. The two ambiguous tag definitions

Two tag definitions are sensitive to W2's design choices, not to the Pitfall #8 shift. W2 must lock them in either way; this report records the risk so W2 doesn't reintroduce it.

### 5.1 `high_progress_failure` threshold

If W2 sets the threshold at:

- **≥ 0.70** — none of the four pilots carry this tag in either world. **Recommended.** Robust to the Pitfall #8 cleanup whether or not it ships.
- **≥ 0.60** — `f_07` and `f_10` carry the tag at 0.67 but *lose* it at 0.50. The label would change for two of the four pilots if cleanup runs.
- **≥ 0.50** — the cleanup tips f_07/f_10 below the threshold and pushes f_02/f_03 from on-edge to clearly off (0.50 → 0.33). Three of the four pilots' labels become Pitfall-#8-cleanup-dependent.

W2's `f_06` acceptance-criterion case (`f_06 is high_progress_failure + hidden_work_gap`) sits at coding_progress = 1.0. That criterion is satisfied at any threshold ≤ 1.0, so no constraint flows back from there. **Recommendation: anchor the threshold at 0.70.**

### 5.2 `no_validation_frontier` definition

Two coherent readings:

- **Reading A: "no VAL leaf exists in the ledger."** Under this reading, f_02/f_03/f_10 *gain* their VAL leaf under the revised scalar and lose the tag.
- **Reading B: "no validation work has been attempted (no VAL leaf has reached `in_progress` or beyond)."** Under this reading, the addition of a not_started VAL leaf is irrelevant — the tag still applies if the leaf is unattempted. f_02/f_03/f_10 keep the tag in both worlds.

The N4 PARITY_REPORT and the live-vs-retro frontier policy both use language closer to Reading B (the live channel does not invent unattempted VAL leaves; the *attempted* validation is what the policy tracks). **Recommendation: W2 adopts Reading B**, in which case `no_validation_frontier` is stable across the Pitfall #8 shift for all four pilots.

If W2 prefers Reading A, the tag becomes Pitfall-#8-dependent on f_02/f_03/f_10, and W2 must define its label rules relative to the revised values, not the current ones.

## 6. Verdict

```
Shape-tag stability across Pitfall #8 cleanup: STABLE,
provided
  - high_progress_failure threshold ≥ 0.70
  - no_validation_frontier defined as "no validation attempted"
    (Reading B, consistent with N4 frontier policy)
```

Under those two anchors, every shape tag W2 will assign to `f_02`/`f_03`/`f_07`/`f_10` is invariant to whether the Pitfall #8 cleanup pass runs first. **W2 may proceed against the current scalar values.** If W2 chooses different anchors, it must pre-commit to running the cleanup pass first and authoring shape rules against the post-cleanup scalars.

## 7. Pointers

- H4 incidental finding: `runs/swe_agent_pilot_reannotation/H4_GATE_RESULT.md` § 7
- Pitfall #8 text: `docs/SWE_AGENT_ANNOTATION_PROTOCOL_REVISIONS.md` § Revision 1
- Memory: `feedback_validation_implicit_for_bug_fix.md`
- W2 tag vocabulary and acceptance criteria: `TASKS.md` § W2
- Source ledgers: `runs/swe_agent_pilot/swe_agent_pilot_{f_02,f_03,f_07,f_10}/ledger.jsonl`
- Annotation notes: `annotations/swe_agent_pilot/swe_agent_pilot_{f_02,f_03,f_07,f_10}.notes.md`
