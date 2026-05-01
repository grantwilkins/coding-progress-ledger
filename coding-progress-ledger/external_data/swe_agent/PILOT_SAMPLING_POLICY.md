# SWE-agent pilot sampling policy (B1)

## 1. Goal

This document defines the inclusion/exclusion criteria, dedupe rule, random
seed, fallback ladder, and pilot-id scheme that govern selection of the
SWE-agent retrospective pilot sample. It exists to satisfy `TASKS.md`
§ Workstream B, task **B1**, which requires the policy to be written down
*before* any selection happens. It constrains the deterministic sampler
script (B2) and the audit (B3); it does not constrain annotation
(Workstream D), categories (D1), or any cross-model / predictive analysis
(deferred to Workstreams P and Q). All § 0 project rules apply: the
sampler must not mutate raw traces, must not infer completion from
progress, and must not use the final outcome as a feature — it only uses
`final_success` to *split* candidates into success and failure pools.

## 2. Target sample size

| Item | Value |
|------|-------|
| Total pilot traces | 20 |
| Successful traces (`final_success == True`) | 10 |
| Failed traces (`final_success == False`) | 10 |

This is the **pilot** target. Scaling to N=100 (or to a multi-model
sample) is the responsibility of Workstream M (`M1`/`M2`); B1 does not
prejudge it.

## 3. Inclusion criteria

A row from `external_data/swe_agent/manifests/swe_agent_inventory.csv`
is eligible only if **all** of the following hold. Column names are
verbatim from A3's manifest.

| # | Condition | Justification |
|---|-----------|---------------|
| I1 | `parse_status == "ok"` | Only cleanly parsed rows; A4 reports 80,036 / 80,036 already pass, but stated for forward compatibility. |
| I2 | `trajectory_available == True` | We need an actual trajectory to annotate. |
| I3 | `final_success_available == True` | We need a label to assign success vs failure (used only for stratification, never as a feature). |
| I4 | `patch_available == True` | Strong-evidence prerequisite for the validation category in retrospective annotation. |
| I5 | `eval_log_available == True` | Strong-evidence prerequisite for distinguishing weak vs strong validation evidence (Workstream K). |
| I6 | `trajectory_length >= 10` | A4 distribution: min 2, p25 21, median 35. **2,085 rows** sit below 10; they look truncated/aborted and skew the lower tail. Cutting at 10 removes only ~2.6% of the corpus and keeps any trajectory with at least a plausible plan/act/observe/validate cycle. |
| I7 | `model_name == "swe-agent-llama-70b"` | A4: 74,792 of 80,036 rows (93.4%) are this model. B1's directive is "prefer one model/scaffold first"; isolating the 70b scaffold avoids confounding model effects in the pilot's qualitative findings. The 8b and 405b variants are deferred to Workstream P. |

## 4. Exclusion criteria

The following are stated explicitly so an auditor can verify the sampler
applied them:

- Rows with `trajectory_length < 10` (**2,085 rows** per A4 — potentially malformed / aborted very early).
- Rows with `patch_available == False` **or** `eval_log_available == False` (**9,397 rows**: 9,294 with neither, 103 with patch but no eval log; A4 confirms all are failures).
- Rows with `model_name in {"swe-agent-llama-8b", "swe-agent-llama-405b"}` (**5,244 rows** combined; deferred to Workstream P cross-model analysis).
- Rows with `parse_status != "ok"` (currently zero, listed for forward compatibility).
- Duplicate rows for an already-selected `instance_id` beyond the first chosen — see § 5.

## 5. Dedupe rule (load-bearing)

A4 surfaced this as the single most important policy decision Workstream
B had to make: the manifest holds 80,036 rows but only **4,219 unique
`(instance_id, model_name)` pairs** (~19× duplication) and **3,591
unique `instance_id`s**.

**Decision:** the pilot dedupes on **`instance_id` alone**, not on
`(instance_id, model_name)`.

Rationale: criterion I7 already restricts to a single `model_name`, so
deduping on `instance_id` is equivalent to deduping on `(instance_id,
model_name)` *within the chosen model* — but it also forecloses the
hazard of selecting two near-identical trajectories of the same task.
Each pilot run should correspond to a distinct underlying SWE-bench
issue.

**Tie-break when more than one row survives the filters for a given
`instance_id`:** the sampler picks the row with the **lowest
`raw_path_or_dataset_index`** (i.e. earliest streaming-iterator index).
This is deterministic, content-independent, and depends only on the
manifest, not on the seed.

## 6. Random seed

- **Default seed: `seed = 0`.**
- The B2 sampler MUST accept `--seed <int>`.
- `seed = 0` is the canonical pilot sample; any other seed produces a
  different sample and must be labeled as such in the audit (B3).
- B2's acceptance criterion ("re-running with same seed produces
  byte-identical CSV") is restated here: same inventory CSV + same
  flags + same seed ⇒ byte-identical output.

## 7. Sampling procedure (algorithm sketch, prose)

1. Load `swe_agent_inventory.csv` as text rows; preserve the manifest
   column types via explicit casts (`int` for `trajectory_length` and
   the trailing numeric portion of `raw_path_or_dataset_index`; `bool`
   for the `*_available` columns and `final_success` via string match
   on `"True"` / `"False"`).
2. Apply filters I1–I7 from § 3 in the order listed. Call the survivors
   the **eligible pool**.
3. **Dedupe** (§ 5): for each distinct `instance_id`, keep only the row
   with the lowest `raw_path_or_dataset_index`. Call this the **deduped
   pool**.
4. Split the deduped pool by `final_success` into `eligible_success`
   and `eligible_failure`.
5. Within each list, **sort by `instance_id` lexicographically** (this
   is the determinism anchor — without it, the seed alone is not enough
   to guarantee byte-equivalence across re-runs that happen to receive
   the rows in a different order). Then sample without replacement
   using `random.Random(seed).sample(sorted_list, n)` with `n_success
   = 10` and `n_failure = 10`.
6. Tag every selected row with `selection_reason =
   "primary_balanced_10_10"`.
7. Assign `pilot_id`s per § 8.
8. Sort the final selection by `pilot_id` and write the CSV.

If any step in (4)–(5) underdelivers, descend the fallback ladder in § 9
*automatically* and re-run from the appropriate step.

## 8. Pilot IDs

Format: `swe_agent_pilot_<status_letter>_<2-digit-counter>`, where:

- `status_letter` is `s` for `final_success == True` and `f` for `False`.
- `<2-digit-counter>` is 1-based, zero-padded to width 2, scoped **per
  status group**.

Examples:

```
swe_agent_pilot_s_01 ... swe_agent_pilot_s_10
swe_agent_pilot_f_01 ... swe_agent_pilot_f_10
```

Assignment order: **after** sampling, sort the selected rows in each
status group by `instance_id` (lexicographic ascending), then enumerate.
Pilot IDs therefore depend only on which rows were selected, not on the
seed value or the sampling RNG state. Two different seeds that happen
to select the same rows produce identical `pilot_id`s.

## 9. Fallback rules (in order)

The sampler descends this ladder automatically. Each level is triggered
**iff** the previous level yields fewer than the target counts.
Each level retains the dedupe rule from § 5.

| Level | Trigger | Relaxation |
|-------|---------|------------|
| Primary | — | Filters I1–I7, target 10 success / 10 failure, `selection_reason = "primary_balanced_10_10"`. |
| Fallback 1 | `< 10` in either pool after Primary | Drop I7 (model restriction). All three models eligible. `selection_reason = "fallback_any_model"`. |
| Fallback 2 | `< 10` in either pool after Fallback 1 | Loosen I6 to `trajectory_length >= 5`. `selection_reason = "fallback_short_traj"`. |
| Fallback 3 | `< 10` in either pool after Fallback 2 | Reduce target to 5 success / 5 failure. `selection_reason = "fallback_5_5"`. |
| Fallback 4 | `< 5` in either pool after Fallback 3 | Stop. Emit whatever was reached, mark each shortfall row with `selection_reason = "fallback_imbalance"`, and **do not silently rebalance** by oversampling the abundant side. The audit (B3) MUST report which fallback level was used and the per-side counts at that level. |

A4's pool counts (after § 3 filters and § 5 dedupe) make Fallback 1+
extremely unlikely for the current manifest: the deduped strict pool
holds 384 success rows and 3,189 failure rows in the 70b slice. The
ladder exists to keep the sampler honest if a future inventory refresh
changes those numbers.

## 10. What this policy does NOT decide

- **Annotation categories.** The category set used during ledger
  annotation is fixed by D1's protocol (cross-references
  `ledger_progress/core.py:SubtaskCategory` and
  `ledger_progress/queries.py:CODING_CATEGORIES`).
- **Whether any selected trace is actually annotated.** D4 (pilot-zero,
  N=2) and E1 (full N=20) decide that.
- **Cross-model comparison.** Workstream P is responsible for any
  `model_name` axis beyond the single-model 70b slice.
- **Predictive modeling.** Workstream Q owns this; Q5 explicitly
  prohibits using `final_success` as a feature.
- **Repo balance / stratification.** § 12 lists this as an open caveat.

## 11. Determinism contract

- The inventory CSV is itself byte-deterministic per A3 acceptance ("same
  raw → same CSV byte-for-byte"; the MD5 of the canonical inventory is
  recorded in A3's commit metadata).
- Given the same inventory CSV + same flags (`--n-success 10
  --n-failure 10`) + same `--seed`, the B2 sampler MUST produce a
  byte-identical pilot sample CSV. This is enforceable by SHA-256 in CI
  if needed.
- The `pilot_id` assignment in § 8 is content-determined, not RNG-determined,
  so two seeds that select the same row set produce identical pilot ids.

## 12. Open questions / known caveats

1. **No repo stratification.** A4 shows `pydantic/pydantic` is the
   single largest repo (6,279 rows; 984 successes). The 10/10 sample may
   over-represent it. We accept this for the pilot; a future revision
   could add a per-repo cap.
2. **Patch / eval-log presence is a string-non-empty check, not a
   semantic one.** We do not require the patch to apply cleanly or the
   eval log to evidence test execution. K1 (evidence audit) will quantify
   how often the patch/eval is materially weak.
3. **Dedupe discards data.** Restricting to one row per `instance_id`
   means later variance studies (e.g. "how does the same model retry
   the same task?") cannot be run from this sample. If that question
   becomes interesting, this policy must be revised — do not silently
   relax the dedupe rule downstream.
4. **No exit-status filter.** A4 did not break out the 9 `exit_status`
   values; the strict pool may include rows that hit context-window
   exits or tool-loop exits. The annotation protocol (D1) is expected
   to surface these as evidence rather than the sampler filtering them.
5. **Single-model bias is a deliberate scope choice.** All pilot
   findings are conditional on `swe-agent-llama-70b`; transferability
   to 8b / 405b / non-Llama scaffolds is explicitly out of pilot scope.
