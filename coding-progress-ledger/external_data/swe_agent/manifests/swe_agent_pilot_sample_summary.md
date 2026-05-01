# SWE-agent pilot sample audit (B3)

This document is the human-readable audit produced by Workstream B's
task **B3** for the canonical 20-trace pilot sample. The sample itself
is the deterministic output of B2 (`scripts/sample_swe_agent_pilot.py`)
applied to A3's full inventory under the rules of B1's policy. For the
inclusion/exclusion criteria, dedupe rule, fallback ladder, and seed
contract see
[`PILOT_SAMPLING_POLICY.md`](../PILOT_SAMPLING_POLICY.md). The CSV
audited here is
[`swe_agent_pilot_sample.csv`](swe_agent_pilot_sample.csv); for the
corpus-wide context see
[`swe_agent_inventory_summary.md`](swe_agent_inventory_summary.md).
This audit is read-only — no CSV or upstream doc was mutated.

## 1. Headline counts

| Metric | Value |
|--------|-------|
| total selected | 20 |
| `final_success == True` (success) | 10 |
| `final_success == False` (failure) | 10 |
| distinct `selection_reason` values | `primary_balanced_10_10` (20) |
| fallback level used | none — primary level (`primary_balanced_10_10`); no descent into Fallback 1+ |
| seed | `seed = 0` (canonical pilot sample per B1 § 6) |
| source CSV (relative) | `external_data/swe_agent/manifests/swe_agent_pilot_sample.csv` |
| source CSV MD5 | `ea27475452bfc4d3fb48b4bb93a27103` |

The single `selection_reason` value confirms the sampler did not need
to relax any filter: B1's strict pool (filters I1–I7, dedupe on
`instance_id`) yielded enough rows on both sides for the 10/10 target
without descending the fallback ladder.

## 2. Model distribution

| `model_name` | rows |
|--------------|------|
| swe-agent-llama-70b | 20 |

100% `swe-agent-llama-70b`, as required by B1 criterion I7. Cross-model
comparison is deferred to Workstream P.

## 3. Repo distribution

Each of the 20 picks comes from a distinct repository: 20 unique
`repo_name` values across 20 rows. No repo dominates the sample.
Notably, `pydantic/pydantic` — the largest repo in the corpus and
explicitly flagged in B1 § 12 as a potential over-representation
hazard — contributes only 1 of 20 picks despite holding 6,279 of 80,036
inventory rows (~7.8% of the corpus). The 70b-only, dedupe-on-instance,
seed-0 sampler kept it from over-running the small pilot.

| `repo_name` | sample rows | rows in full inventory |
|---|---:|---:|
| Melevir/cognitive_complexity | 1 | 153 |
| WIPACrepo/iceprod | 1 | 59 |
| asottile/pyupgrade | 1 | 1,194 |
| asottile/setup-cfg-fmt | 1 | 164 |
| dfm/emcee | 1 | 69 |
| fairlearn/fairlearn | 1 | 156 |
| geomet/geomet | 1 | 87 |
| googleapis/python-spanner | 1 | 48 |
| hsahovic/poke-env | 1 | 65 |
| joke2k/django-environ | 1 | 165 |
| lidatong/dataclasses-json | 1 | 100 |
| mahmoud/boltons | 1 | 75 |
| mc706/changelog-cli | 1 | 32 |
| oasis-open/cti-taxii-client | 1 | 71 |
| omni-us/jsonargparse | 1 | 401 |
| openstack-charmers/zaza | 1 | 14 |
| planetlabs/planet-client-python | 1 | 348 |
| pydantic/pydantic | 1 | 6,279 |
| python-cmd2/cmd2 | 1 | 106 |
| walles/px | 1 | 6 |

The "rows in full inventory" column is informational, not a property of
the pilot row itself: `walles/px` (6 rows) and `openstack-charmers/zaza`
(14 rows) are rare repos in the corpus, while `asottile/pyupgrade`
(1,194 rows) and `pydantic/pydantic` (6,279 rows) are popular ones.
Annotators should not read these counts as "this trace is rare/typical"
— they describe the parent repo, not the picked trajectory.

## 4. Trajectory length distribution

For the 20 picks, computed from the `trajectory_length` column:

| stat | sample (n=20) | corpus (A4, n=80,036) |
|------|---:|---:|
| min    | 17  | 2 |
| p25    | 23.0 | 21 |
| median | 33.0 | 35 |
| p75    | 51.5 | 67 |
| max    | 509 | 817 |

The sample's lower-tail and median track the corpus closely (median
33 vs 35; p25 23 vs 21). The differences sit at the tails: the sample's
minimum is 17, well above the corpus floor of 2, because B1's filter I6
excludes `trajectory_length < 10` and the dedupe rule prefers the
earliest streaming-iterator index per `instance_id`. The p75 is lower
(51.5 vs 67) and the max is lower (509 vs 817), both expected from a
small sample. Overall the sample is reasonably representative of the
strict-pool body of the corpus, with the very-short and very-long
extremes deliberately absent or attenuated.

## 5. Patch / eval-log availability

| Column | True | False |
|--------|---:|---:|
| `patch_available` | 20 | 0 |
| `eval_log_available` | 20 | 0 |

Both 20/20, as required by B1 criteria I4 and I5. Any deviation here
would be a sampling bug; none was found.

## 6. Per-pick table

Sorted by `pilot_id` (failures `f_01..f_10` first, then successes
`s_01..s_10` — same order as the CSV).

| pilot_id | instance_id | final_success | trajectory_length | repo_name | raw_path_or_dataset_index |
|---|---|---|---:|---|---|
| swe_agent_pilot_f_01 | WIPACrepo__iceprod-339 | False | 17 | WIPACrepo/iceprod | nebius/SWE-agent-trajectories:train:25550 |
| swe_agent_pilot_f_02 | asottile__pyupgrade-933 | False | 509 | asottile/pyupgrade | nebius/SWE-agent-trajectories:train:48200 |
| swe_agent_pilot_f_03 | asottile__setup-cfg-fmt-132 | False | 113 | asottile/setup-cfg-fmt | nebius/SWE-agent-trajectories:train:18165 |
| swe_agent_pilot_f_04 | dfm__emcee-510 | False | 19 | dfm/emcee | nebius/SWE-agent-trajectories:train:69069 |
| swe_agent_pilot_f_05 | fairlearn__fairlearn-967 | False | 35 | fairlearn/fairlearn | nebius/SWE-agent-trajectories:train:64867 |
| swe_agent_pilot_f_06 | googleapis__python-spanner-317 | False | 33 | googleapis/python-spanner | nebius/SWE-agent-trajectories:train:52966 |
| swe_agent_pilot_f_07 | openstack-charmers__zaza-36 | False | 183 | openstack-charmers/zaza | nebius/SWE-agent-trajectories:train:37645 |
| swe_agent_pilot_f_08 | pydantic__pydantic-740 | False | 77 | pydantic/pydantic | nebius/SWE-agent-trajectories:train:56260 |
| swe_agent_pilot_f_09 | python-cmd2__cmd2-681 | False | 41 | python-cmd2/cmd2 | nebius/SWE-agent-trajectories:train:16314 |
| swe_agent_pilot_f_10 | walles__px-50 | False | 81 | walles/px | nebius/SWE-agent-trajectories:train:56446 |
| swe_agent_pilot_s_01 | Melevir__cognitive_complexity-15 | True | 43 | Melevir/cognitive_complexity | nebius/SWE-agent-trajectories:train:26 |
| swe_agent_pilot_s_02 | geomet__geomet-101 | True | 27 | geomet/geomet | nebius/SWE-agent-trajectories:train:2848 |
| swe_agent_pilot_s_03 | hsahovic__poke-env-68 | True | 37 | hsahovic/poke-env | nebius/SWE-agent-trajectories:train:66567 |
| swe_agent_pilot_s_04 | joke2k__django-environ-174 | True | 17 | joke2k/django-environ | nebius/SWE-agent-trajectories:train:76189 |
| swe_agent_pilot_s_05 | lidatong__dataclasses-json-394 | True | 33 | lidatong/dataclasses-json | nebius/SWE-agent-trajectories:train:77604 |
| swe_agent_pilot_s_06 | mahmoud__boltons-298 | True | 29 | mahmoud/boltons | nebius/SWE-agent-trajectories:train:36311 |
| swe_agent_pilot_s_07 | mc706__changelog-cli-34 | True | 23 | mc706/changelog-cli | nebius/SWE-agent-trajectories:train:16055 |
| swe_agent_pilot_s_08 | oasis-open__cti-taxii-client-11 | True | 29 | oasis-open/cti-taxii-client | nebius/SWE-agent-trajectories:train:32241 |
| swe_agent_pilot_s_09 | omni-us__jsonargparse-370 | True | 19 | omni-us/jsonargparse | nebius/SWE-agent-trajectories:train:70864 |
| swe_agent_pilot_s_10 | planetlabs__planet-client-python-389 | True | 23 | planetlabs/planet-client-python | nebius/SWE-agent-trajectories:train:69434 |

## 7. Known caveats

These are restated only briefly; the canonical list lives in B1
§ 12 (`PILOT_SAMPLING_POLICY.md`).

- **Single-model lock.** All 20 picks are `swe-agent-llama-70b`; pilot
  findings do not transfer to the 8b/405b scaffolds (deferred to
  Workstream P).
- **No repo stratification.** The strict pool was sampled without a
  per-repo cap; the 20-distinct-repos outcome here is a happy
  coincidence, not a guarantee for future re-samples.
- **Patch / eval-log are presence-only checks.** B1 requires the fields
  to be non-empty but not that the patch applies or that the eval log
  evidences test execution. K1 will quantify how often that matters.
- **Dedupe discards data.** One row per `instance_id`; same-instance
  retry/variance studies cannot be done from this sample without
  revising B1's dedupe rule.
- **No `exit_status` filter.** Context-window or tool-loop terminations
  may be present in the failure side; the annotation protocol (D1) is
  expected to surface them rather than the sampler removing them.

## 8. Annotation outlook

These 20 traces look like a reasonable starting set for D4 (pilot-zero,
N=2) and E1 (full N=20). The success side has 10 distinct
`instance_id`s drawn from 10 distinct repos with trajectory lengths
between 17 and 43 — short enough to annotate quickly and varied enough
to exercise multiple `SubtaskCategory` values. The failure side spans
17 to 509 steps (failure-only min 17, max 509), providing genuine
variety in failure modes from "gave up early" (`f_01` at 17 steps,
`f_04` at 19 steps) through mid-length stuck behaviour (`f_07` at 183,
`f_10` at 81) to a probable tool/context loop (`f_02` at 509 steps).
The two anomalies a human should know about up-front are: `f_02`
(`asottile__pyupgrade-933`, 509 steps) will be the slowest single trace
to annotate and is the most likely candidate for a context-window or
tool-loop failure mode; and `f_01` / `s_04` (both 17 steps) sit just
above B1's I6 length floor and may have fewer distinct subtasks than
the other picks. Neither is a sampling bug; they are exactly the
diversity D4/E1 are meant to surface.
