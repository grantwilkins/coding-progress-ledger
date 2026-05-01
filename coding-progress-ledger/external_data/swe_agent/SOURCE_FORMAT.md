# SWE-agent trajectory source — verified format

This document records the **first source format** chosen for the SWE-agent
retrospective pilot (Workstream A, task A2 in `TASKS.md`). The schema below
was extracted from a real row, not from documentation.

## Chosen primary source

**`nebius/SWE-agent-trajectories`** on Hugging Face.

- Page: <https://huggingface.co/datasets/nebius/SWE-agent-trajectories>
- Reason: it is the only candidate that ships, in a single row, all four
  fields the pilot needs end-to-end — full trajectory, generated patch,
  evaluation logs, and a co-located boolean success label. The fallback
  (`SWE-bench/SWE-smith-trajectories`) only has ~5k rows from a single
  model and uses a different label name (`resolved`); we keep it as a
  back-up but do not start with it.
- Scale (per the dataset card): 80,036 trajectories, 16.7% resolved
  (13,389 resolved / 80,036 total). Multi-model.
- This document is from inspection of **one row only**. No bulk download
  was performed.

## Access pattern actually used

```python
from datasets import load_dataset
ds = load_dataset("nebius/SWE-agent-trajectories", split="train", streaming=True)
row = next(iter(ds))   # row index 0 of the streaming iterator
```

Sampled row index: `train[0]` via the streaming iterator (so: the first
row served by the parquet shard list — deterministic in practice but
nominally HF-streaming-order, not random).

The exact fetch script committed alongside this doc is
`external_data/swe_agent/raw/_fetch_one.py` (kept under `raw/`, which is
gitignored — the script is reproducible from this doc).

## Real schema (from the sampled row)

Field names below are **verbatim** from the row dict's keys.

| Field             | Python type | Notes                                                                                       |
|-------------------|-------------|---------------------------------------------------------------------------------------------|
| `instance_id`     | `str`       | SWE-bench-style instance id, e.g. `"AnalogJ__lexicon-336"` (repo-slug + issue number).      |
| `model_name`      | `str`       | Agent model that produced this trajectory, e.g. `"swe-agent-llama-70b"`.                    |
| `target`          | `bool`      | **Success/failure label.** `True` = issue resolved by this trajectory, `False` = not resolved. |
| `trajectory`      | `list[dict]`| Ordered list of trajectory entries. See sub-schema below.                                   |
| `exit_status`     | `str`       | Agent termination reason, e.g. `"submitted (exit_context)"` (9 possible values per card).   |
| `generated_patch` | `str`       | Final unified-diff patch the agent submitted against the repo. Empty string possible (meaning unconfirmed for the empty case). |
| `eval_logs`       | `str`       | Stdout/stderr from running the evaluation tests against `generated_patch`.                  |

### `trajectory` item sub-schema (verified)

Each entry in `trajectory` is a dict with keys:

| Key             | Python type | Notes                                                                              |
|-----------------|-------------|------------------------------------------------------------------------------------|
| `role`          | `str`       | One of `"system"`, `"user"`, `"ai"` (observed in this row). `system` is the first entry; `user` carries environment observations / tool returns; `ai` carries model reasoning + actions. |
| `text`          | `str` \| `None` | The message text. Observed `None` for the leading `system` entry (system prompt content lives in `system_prompt` instead in that row). (meaning unconfirmed: whether `text` is ever `None` for non-system roles.) |
| `system_prompt` | `str` \| `None` | The system prompt for the agent. Populated on the leading `system` entry; `None` on subsequent entries. (meaning unconfirmed: whether it can repeat mid-trajectory.) |
| `mask`          | unconfirmed | (meaning unconfirmed) — likely a training-mask flag for the entry; not used by the pilot. |
| `cutoff_date`   | unconfirmed | (meaning unconfirmed) — likely the model knowledge cutoff date string; not used by the pilot. |

Roles observed in this row's trajectory: `{"system", "user", "ai"}`.

### Field roles for the pilot (explicit mapping)

- **Success/failure label field:** `target` (bool).
- **Patch field:** `generated_patch` (str, unified diff).
- **Eval log field:** `eval_logs` (str).
- **Trajectory field:** `trajectory` (list[dict] with `role`/`text` per entry, plus extras).

### Deviations from TASKS.md A2 pre-recommendation

The pre-recommendation said trajectory items would be `role`/`content`.
The real schema uses `role` + `text` (plus `system_prompt`, `mask`,
`cutoff_date`). All seven top-level fields named in the pre-recommendation
are present with the predicted types. No fields are missing or renamed
at the top level.

## Sample row sanity check (the row written to `raw/sample_row.json`)

- `instance_id`: `"AnalogJ__lexicon-336"`
- `model_name`: `"swe-agent-llama-70b"`
- `target`: `False`
- `exit_status`: `"submitted (exit_context)"`
- `trajectory` length: 93 entries
- `generated_patch` length: 2,190 chars (non-empty patch present)
- `eval_logs` length: 2,048 chars (non-empty eval log present)

No fields required type conversion to JSON-encode. No string fields exceeded
the 100,000-char truncation threshold for this row, so `sample_row.json`
contains the row exactly as decoded by the `datasets` library, with no
truncation applied.

## License and usage constraints

- **Dataset license:** `Creative Commons Attribution 4.0 (CC-BY-4.0)`
  (per the HF dataset card on
  <https://huggingface.co/datasets/nebius/SWE-agent-trajectories>).
- **Restrictions noted on the card we must respect:**
  1. Each trajectory is derived from a real GitHub repository's issue —
     individual repository licenses still apply to anything sourced from
     those repos (e.g. snippets quoted in patches, file contents shown
     in tool returns).
  2. Trajectories generated using Llama-family models are subject to the
     **Llama 3.1 License** if their outputs are reused. The sampled row
     here uses `model_name = "swe-agent-llama-70b"`, so this clause is
     live for at least some rows.
- **Pilot stance:** internal/research use only, no redistribution of raw
  trajectories from this repo, attribution given via this file. We are
  not retraining models from this data; we are only annotating ledgers
  retrospectively from visible evidence.

## File layout produced by A2

```
external_data/swe_agent/
├── SOURCE_FORMAT.md                # this file (committed)
└── raw/                            # gitignored
    ├── _fetch_one.py               # the one-row fetcher actually used
    ├── _schema_summary.json        # field/type summary emitted by the fetcher
    └── sample_row.json             # ONE decoded row, raw, never edited
```
