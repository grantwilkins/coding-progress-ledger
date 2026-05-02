# SWE-agent LedgerSet addendum

This addendum specializes the general LedgerSet protocol
(`docs/LEDGER_SET_PROTOCOL.md`) to SWE-agent pilot runs imported per
`scripts/import_swe_agent_trace.py` (C3) and annotated per
`docs/SWE_AGENT_RETROSPECTIVE_LEDGER_PROTOCOL.md`. It is the only place
where SWE-agent-specific naming and SWE-agent run-dir layout decisions
about set-level artifacts are allowed. Anything you find here that
contradicts the general protocol is a bug — **the general protocol
wins.**

In particular, this file does NOT redefine § 1-9 of the general
LedgerSet protocol. It only:

- names the SWE-agent set artifacts (§ 1),
- specifies the singleton-set convention for one pilot trace (§ 2),
- specifies the rollup-set convention for the 20-pilot dataset (§ 3),
- catalogues SWE-agent-specific pitfalls at the set layer (§ 4).

## 1. SWE-agent set artifacts

Two artifact shapes are produced by `scripts/build_pilot_ledger_sets.py`
(T4) and conform to the general protocol § 2 data model:

| Path | Members | Weights | Source |
|------|---------|---------|--------|
| `runs/swe_agent_pilot/<pilot_id>/set.jsonl` | 1 (`pilot_id`) | 1.0 | the pilot's own `ledger.jsonl` |
| `runs/swe_agent_pilot/pilot_rollup_set.jsonl` | 20 (one per pilot) | 1.0 | sibling pilot dirs' `ledger.jsonl` |

`ledger_ref` is always relative to the set file's parent directory
(`ledger.jsonl` for singletons, `<pilot_id>/ledger.jsonl` for the
rollup). This keeps the set bytes stable under repo relocation.

The set layer reads finished ledgers and aggregates per general § 1.
It does not annotate. `source_trace.json`, `normalized_trace.json`,
and `ledger.jsonl` are never edited by set-level tooling — `T4` ships
a regression test (`tests/test_pilot_ledger_sets.py`) that locks this
invariant by SHA-256 comparison across re-runs of the build script.

## 2. Singleton-set convention (one pilot)

Each pilot maps to a 1-member `LedgerSet` whose `set_id` and
`member_id` both equal the pilot's `pilot_id` (e.g.
`swe_agent_pilot_s_01`). The single member has weight `1.0` and
`status_override = None`. Per general § 4 this means

```text
singleton.score() == score(member.ledger, CODING_CATEGORIES).progress
```

i.e. the singleton set is a no-op aggregator over the pilot's
single-ledger coding-progress. The point of materializing it is API
parity: downstream consumers can treat any pilot the same way they
treat the rollup, which keeps the consumer code branch-free.

## 3. Rollup-set convention (the 20-pilot dataset)

The dataset is a 20-member `LedgerSet` with `set_id =
swe_agent_pilot_rollup`, one member per pilot, all weights `1.0`. Per
general § 4, set-level progress is the unweighted mean of the 20
per-pilot coding-progress scores. As of T4 commit, this evaluates to

```text
score_set(rollup, base_dir=runs/swe_agent_pilot) ≈ 0.8224
```

— in the success-mean (~0.97) / failure-mean (~0.68) band reported in
`runs/swe_agent_pilot/PILOT_ANNOTATION_SUMMARY.md`.

Member ordering is deterministic (sorted by `pilot_id`, so all `f_*`
appear before all `s_*`). The score is invariant under reordering, but
the byte representation of `pilot_rollup_set.jsonl` is not — pinning a
canonical sort keeps the file stable under re-runs of the build
script.

The rollup is **not** intended as a stratification key: `final_success`
is a property of each member's `source_metadata.json`, not of the set,
and the set layer does not propagate it. Splitting the 20-pilot
rollup into "success rollup" + "failure rollup" sub-sets would
privilege the upstream label exactly the way Workstream Q's locked
framing forbids. See general § 9 open question 3.

## 4. SWE-agent-specific pitfalls at the set layer

### Pitfall S1. Multi-set membership

A pilot's `ledger.jsonl` may belong to multiple sets without being
copied (general § 2). E.g. a future model-comparison set may include
the same `swe_agent_pilot_f_06` ledger as a member with weight `1.0`
alongside a different model's attempt at the same instance. **Do not
materialize a second copy of `ledger.jsonl`** when defining such a
set; just point at the existing one with a relative `ledger_ref`.

### Pitfall S2. `status_override` is not the upstream label

`LedgerSetMember.status_override` is reserved for the case where a
pilot's outcome is decided *outside* its ledger — e.g. a pilot we
later declare invalid because the upstream trace was malformed in a
way only discovered after annotation. **It is not a place to record
`final_success`**. The upstream label lives where the general
SWE-agent protocol § 3 puts it (`source_metadata.json::target`); set
membership and aggregation are oblivious to it.

### Pitfall S3. Rollup denominators if a pilot is later overridden

If a pilot is marked `status_override = INVALIDATED` or `DELETED` in
the rollup (per general § 4), it drops out of both numerator and
denominator. The rollup mean over the remaining 19 members is then
the relevant statistic; the pre-override mean over 20 members is no
longer reproducible from the file unless you keep the override out
of the canonical rollup and apply it in a derived sibling set.
Prefer the latter — keep `pilot_rollup_set.jsonl` clean and override
only in derived sets that document why.

### Pitfall S4. Don't promote rollup progress to a forecast

Per general § 8 and Workstream Q's locked framing, set-level progress
is a process-shape signal, not an outcome forecast. The 0.8224 rollup
score does not say "82% of the dataset succeeded" — it says "the mean
fraction of *discovered* coding work the agent visibly closed across
the 20 pilots is 0.82." Failures at progress 1.00 (e.g. `f_06`,
documented in `PILOT_ANNOTATION_SUMMARY.md`) and successes at <1.00
(e.g. `s_04`) are features of the channel, not bugs in the
aggregation rule.
