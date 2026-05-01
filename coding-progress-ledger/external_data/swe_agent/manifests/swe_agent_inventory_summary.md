# SWE-agent inventory summary (A4)

Computed from `external_data/swe_agent/manifests/swe_agent_inventory.csv`
(the deterministic manifest produced by A3). For schema notes on the
upstream source, see `external_data/swe_agent/SOURCE_FORMAT.md`. For the
CSV column definitions, see TASKS.md § A3.

This document does not duplicate A2/A3 content; it answers the report
and acceptance items in TASKS.md § A4 and surfaces a few extra signals
(duplicates, multi-model coverage, trajectory-length distribution) that
Workstream B's sample policy will need.

## Headline counts

| Metric                                | Value   |
|---------------------------------------|---------|
| total traces (rows)                   | 80,036  |
| traces with usable trajectory         | 80,036  |
| traces with success label             | 80,036  |
| success count (`final_success=True`)  | 13,389  |
| failure count (`final_success=False`) | 66,647  |
| missing-label count                   | 0       |
| patch availability (`patch_available=True`)         | 70,742  |
| eval-log availability (`eval_log_available=True`)   | 70,639  |
| parse_status=`ok`                     | 80,036  |
| parse_status=other                    | 0       |

Notes:

- Every row parsed cleanly; A3 reported zero `parse_error` strings.
- Trajectory and label availability are 100%. The upstream `target`
  field is always populated as a bool, and every row has a non-empty
  `trajectory` list, so there is no "missing-label" subset to set
  aside.
- Success rate: 13389 / 80036 = 16.73%, matching the dataset card's
  "16.7% resolved" claim (cross-check with A2 SOURCE_FORMAT.md).

## Patch / eval-log joint table

How many rows have each combination of `final_success` x
`patch_available` x `eval_log_available`:

| final_success | patch_available | eval_log_available | rows   |
|---------------|-----------------|--------------------|--------|
| True          | True            | True               | 13,389 |
| False         | True            | True               | 57,250 |
| False         | True            | False              | 103    |
| False         | False           | False              | 9,294  |

Observations:

- **Every successful row has both a patch and an eval log.** That makes
  the success side of a balanced sample easy to filter for "rich
  evidence".
- Failures split: ~86% have patch+eval, ~14% (9,294) have neither, and
  a tiny 103 have a patch but no eval log.
- The 9,294 failures with no patch and no eval log are the weakest
  evidence subset; B1 should probably exclude them by default unless we
  want to study agent-no-patch failures specifically.

## Top model names (top 10, ties broken by name)

The dataset only spans 3 model names, all Llama-family scaffolds of
SWE-agent.

| model_name              | rows    | success | failure |
|-------------------------|---------|---------|---------|
| swe-agent-llama-70b     | 74,792  | 12,467  | 62,325  |
| swe-agent-llama-8b      | 4,053   | 614     | 3,439   |
| swe-agent-llama-405b    | 1,191   | 308     | 883     |

Operationally: 93.4% of rows are `swe-agent-llama-70b`. A
single-model-first sample (per B1's "prefer one model/scaffold first")
is trivially satisfiable by restricting to `swe-agent-llama-70b` —
12,467 successes and 62,325 failures available there alone.

## Top repos (top 10, ties broken by name)

Total unique repos covered: **1,276**. Total unique `instance_id`s:
**3,591**. Top 10 by row count:

| repo_name                      | rows  |
|--------------------------------|-------|
| pydantic/pydantic              | 6,279 |
| iterative/dvc                  | 5,646 |
| tobymao/sqlglot                | 1,824 |
| asottile/pyupgrade             | 1,194 |
| Textualize/textual             | 656   |
| oasis-open/cti-python-stix2    | 607   |
| reata/sqllineage               | 500   |
| sqlfluff/sqlfluff              | 473   |
| reframe-hpc/reframe            | 464   |
| ResearchObject/ro-crate-py     | 406   |

Top 10 repos by **success** rows (for sanity, since success is rarer):

| repo_name                      | successes |
|--------------------------------|-----------|
| pydantic/pydantic              | 984       |
| iterative/dvc                  | 295       |
| Textualize/textual             | 173       |
| streamlink/streamlink          | 154       |
| networkx/networkx              | 131       |
| construct/construct            | 130       |
| marshmallow-code/apispec       | 124       |
| asottile/pyupgrade             | 119       |
| oasis-open/cti-python-stix2    | 112       |
| pypa/setuptools_scm            | 112       |

## Trajectory length distribution

Computed across all 80,036 rows (`trajectory_length` from the manifest):

| stat   | value |
|--------|-------|
| min    | 2     |
| p25    | 21    |
| median | 35    |
| p75    | 67    |
| max    | 817   |
| mean   | 53.87 |

Short-trace counts (potentially malformed / aborted very early):

- `trajectory_length < 5`: **107** rows
- `trajectory_length < 10`: **2,085** rows

For B1's "avoid extremely short malformed traces (< N steps)" rule, a
threshold of `N = 10` removes ~2.6% of rows and is a reasonable default;
`N = 5` is a softer cut (~0.13%) that only excludes obviously broken
traces.

## Duplicates and multi-model coverage (data-quality flag for B)

The manifest's `source_id` is constructed by A3 as
`nebius:<instance_id>:<model_name>` and is unique per row, but A3 noted
that the same `instance_id` can appear multiple times. The numbers:

| metric                                                 | value   |
|--------------------------------------------------------|---------|
| total rows                                             | 80,036  |
| unique `instance_id`                                   | 3,591   |
| unique `(instance_id, model_name)` pairs               | 4,219   |
| rows duplicating an existing `(instance_id, model_name)` | 75,817 |
| instances appearing under more than one `model_name`   | 578     |

Read this as: **the same `(instance_id, model_name)` pair is repeated
about 19x on average across the dataset** (80,036 / 4,219 ≈ 19.0). This
is consistent with multiple sampled trajectories per (instance, model)
in the upstream nebius dump. Concretely, this means that if B1 dedupes
on `(instance_id, model_name)` it has 4,219 unique combinations to draw
from; if it dedupes on `instance_id` alone it has 3,591. **B1 must
make this dedupe choice explicit** — sampling at the row level without
deduping risks selecting two near-identical trajectories of the same
instance under the same model.

578 instances are covered by more than one model, which is enough for
later cross-model comparison work (Workstream P) but not directly
relevant to the pilot.

## Pilot sampling outlook

**A balanced 10-success / 10-failure sample is trivially feasible.**
The dataset has 13,389 successes and 66,647 failures, both with full
trajectory and label coverage. Filtering to the strict-evidence subset
(patch + eval log + `trajectory_length >= 10`) leaves **12,985
successes** and **56,120 failures** to choose from — three orders of
magnitude more than the pilot needs. Restricting further to
`swe-agent-llama-70b` (B1's "prefer one model/scaffold first") still
leaves >12k successes and >60k failures. There are **no imbalances to
flag for the 10/10 target**: even the smallest model partition
(`swe-agent-llama-405b`) has 308 successes and 883 failures, which
would itself comfortably support a 10/10 sample.

The real selection constraints for B1 are therefore not "is balance
possible" but:

1. dedupe policy on `(instance_id, model_name)` (75,817 of 80,036 rows
   are duplicates of an earlier `(instance, model)` pair);
2. minimum `trajectory_length` (recommended `>= 10`, removes 2,085
   rows);
3. require `patch_available` and `eval_log_available` for richer
   evidence (drops 9,397 rows, all failures);
4. whether to lock to a single model — recommended `swe-agent-llama-70b`
   for the pilot, since 93.4% of the corpus and the bulk of both
   success and failure rows are there.

None of these threaten the 10/10 target; they only shape which slice
of the corpus the sample comes from.
