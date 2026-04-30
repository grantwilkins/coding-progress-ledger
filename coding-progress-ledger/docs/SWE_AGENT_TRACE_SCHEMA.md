# SWE-agent normalized trace schema (C1)

This document defines the **internal** schema that `coding-progress-ledger`
uses to represent a single agent trajectory. It exists to satisfy
`TASKS.md` § Workstream C, task **C1**, and is the contract the C2
normalizer (`scripts/normalize_swe_agent_trace.py`) and the C3 importer
(`scripts/import_swe_agent_trace.py`) MUST honour. The schema is decoupled
from any one upstream source: today only `nebius/SWE-agent-trajectories`
is wired up (see `external_data/swe_agent/SOURCE_FORMAT.md`), but the
schema is intentionally portable to the SWE-smith fallback.

## 1. Goals

1. Give annotators (Workstream D) a single, predictable per-step shape.
2. Preserve every raw field — nothing is silently dropped.
3. Tolerate missing fields. Upstream rows can be partial; the normalizer
   never raises for "unexpected None".
4. Stay decoupled from ledger semantics. The normalized trace is a
   *retrospective input* to ledger annotation, not a ledger event.

## 2. Top-level normalized trace

```jsonc
{
  "schema_version": 1,
  "source": "swe_agent_nebius",            // or "swe_agent_smith" later
  "instance_id": "AnalogJ__lexicon-336",
  "model_name": "swe-agent-llama-70b",
  "exit_status": "submitted (exit_context)",  // null if absent
  "final_success": false,                   // bool from upstream label, or null if missing
  "trajectory_length": 93,                  // length of `events`, including system entry
  "issue_text": "...",                      // the initial user prompt; null if absent
  "system_prompt": "...",                   // from the leading system entry; null if absent
  "events": [ <event>, ... ],
  "raw_metadata": {                         // pass-through for any upstream field NOT explicitly normalized
    "patch_length": 2190,                   // string length only — patch goes to final_diff.patch
    "eval_logs_length": 2048,               // string length only — log goes to eval_output.txt
    "extra_top_level_keys": ["..."]         // any unrecognized top-level upstream keys (names only)
  }
}
```

Notes:

- `final_success` is sourced from the upstream label field (`target` for
  nebius). It is preserved here for *stratification and audit only*; per
  § 0 of `TASKS.md`, downstream code MUST NOT use it as a feature.
- `trajectory_length` MUST equal `len(events)`. The C2 normalizer asserts
  this before writing.
- `issue_text` is an extracted convenience copy of the first non-system
  user message. The same text is also preserved verbatim inside that
  event's `raw`.

## 3. Per-step event shape

Every entry in `events` is a dict with the following keys, in this order:

```jsonc
{
  "step_index": 0,                  // 0-based, dense, matches position in events[]
  "role": "system",                 // see § 4
  "thought": null,                  // free-text reasoning (assistant only); null otherwise
  "action": null,                   // structured action description (assistant only); null otherwise
  "observation": null,              // tool/environment return text (tool/environment only); null otherwise
  "tool_name": null,                // best-effort first token of `command`; null if not derivable
  "command": null,                  // literal command issued (assistant only); null otherwise
  "files_touched": [],              // best-effort list; [] when undecidable
  "timestamp": null,                // upstream does not carry timestamps; reserved
  "raw": { ... }                    // verbatim copy of the upstream entry dict (no fields dropped)
}
```

The `raw` field is the contract that "no fields are silently dropped":
even keys the normalizer does not understand (`mask`, `cutoff_date`,
`system_prompt`, anything new) survive there. Annotators can ignore
`raw` for reading, but auditors can use it to reconstruct upstream.

## 4. Role mapping

The internal vocabulary is `system | assistant | tool | environment | unknown`.
For nebius rows, the upstream `role` field uses `system | ai | user`.
Mapping rules:

| Upstream role | Position                              | Internal role  | Rationale |
|---------------|---------------------------------------|----------------|-----------|
| `system`      | any                                   | `system`       | direct mapping; `system_prompt` carries the prompt body. |
| `ai`          | any                                   | `assistant`    | the model's reasoning + action turn. |
| `user`        | first non-system entry                | `environment`  | the issue / task description; not a tool return. |
| `user`        | any subsequent entry                  | `tool`         | bash / SWE-agent shell return following an assistant action. |
| anything else | any                                   | `unknown`      | preserved as-is in `raw.role`; never dropped. |

The schema retains the original upstream label inside `raw.role`, so
re-mapping later is lossless.

## 5. Assistant turn parsing (thought / action / command / tool_name)

SWE-agent assistant turns embed reasoning and the bash command in the
same `text` blob, formatted as

```
<discussion text...>

```
<bash command line(s)>
```
```

The C2 normalizer parses each `assistant` turn as follows:

1. Find the **first** triple-backtick fence (` ``` `) and the matching
   closing fence in `text`.
2. `thought` ← everything before the opening fence, stripped of trailing
   whitespace. Empty string becomes `null`.
3. `command` ← the literal text *inside* the fences, with surrounding
   whitespace stripped. Empty becomes `null`.
4. `action` ← `command` (these alias for SWE-agent; kept distinct in the
   schema so future sources with structured actions can fill `action`
   without overloading `command`).
5. `tool_name` ← the first whitespace-separated token of `command` if
   `command` is set, else `null`.
6. If the fence cannot be parsed (no fence, mismatched fences, or `text`
   is `None`):
   - `thought` ← the full `text` (or `null` if `text` is None)
   - `command`, `action`, `tool_name` ← `null`
   - the failure is recorded under `raw.parse_warnings`
     (e.g. `["no_fenced_block"]`).

`files_touched` is left as `[]` for SWE-agent rows: deriving it
correctly requires per-step shell state (e.g. resolving `edit` against
the currently open file). Annotators read `command` directly.

## 6. Tool-turn parsing (observation, tool_name)

Tool-role entries:

- `observation` ← `text` verbatim (or `null` if `text` is `None`).
- `tool_name` ← `command`'s first token from the **immediately preceding
  assistant turn**, if available. This makes downstream "which tool
  produced which observation" queries cheap. If the preceding turn has
  no command, `tool_name` is `null`.

Environment-role entries (the issue / task description):

- `observation` ← `text` verbatim. `tool_name` is always `null`.

System-role entries:

- All of `thought / action / observation / tool_name / command` are
  `null`. `system_prompt` lives at the top level (and inside `raw`).

## 7. Tolerance for missing / malformed fields

The normalizer MUST NOT raise on any of the following; instead it
records the issue inside `raw.parse_warnings` and continues:

- `text` is `None` for an `ai`/`user` entry.
- `role` is missing or unknown — internal role becomes `unknown`.
- `trajectory` is empty — `events` is `[]`, `trajectory_length` is `0`.
- `target` is `None` or absent — `final_success` is `null`.
- Top-level upstream key not in {`instance_id`, `model_name`, `target`,
  `trajectory`, `exit_status`, `generated_patch`, `eval_logs`} — its name
  is recorded in `raw_metadata.extra_top_level_keys`. Its content is NOT
  copied into `raw_metadata` to keep the normalized JSON small; the
  full row is preserved separately as `source_trace.json` (C3).

The only fatal condition is "input is not a dict" or "trajectory is not
a list when present" — these indicate a corrupt row, not a partial one.

## 8. Worked example — nebius row `AnalogJ__lexicon-336`

Source: `external_data/swe_agent/raw/sample_row.json` (one decoded row).
Upstream snapshot:

- `instance_id`: `"AnalogJ__lexicon-336"`
- `model_name`: `"swe-agent-llama-70b"`
- `target`: `false`
- `exit_status`: `"submitted (exit_context)"`
- `trajectory` length: 93
- `generated_patch` length: 2,190 chars
- `eval_logs` length: 2,048 chars

Normalized output (events truncated to the first three for brevity):

```jsonc
{
  "schema_version": 1,
  "source": "swe_agent_nebius",
  "instance_id": "AnalogJ__lexicon-336",
  "model_name": "swe-agent-llama-70b",
  "exit_status": "submitted (exit_context)",
  "final_success": false,
  "trajectory_length": 93,
  "issue_text": "We're currently solving the following issue ... Memset provider: TypeError: string indices must be integers ...",
  "system_prompt": "SETTING: You are an autonomous programmer ...",
  "events": [
    {
      "step_index": 0,
      "role": "system",
      "thought": null, "action": null, "observation": null,
      "tool_name": null, "command": null,
      "files_touched": [], "timestamp": null,
      "raw": { "role": "system", "text": null, "system_prompt": "SETTING: You are an autonomous programmer ...", "mask": false, "cutoff_date": "01.01.2023" }
    },
    {
      "step_index": 1,
      "role": "environment",
      "thought": null, "action": null,
      "observation": "We're currently solving the following issue ...",
      "tool_name": null, "command": null,
      "files_touched": [], "timestamp": null,
      "raw": { "role": "user", "text": "We're currently solving ...", "system_prompt": null, "mask": false, "cutoff_date": null }
    },
    {
      "step_index": 2,
      "role": "assistant",
      "thought": "To start solving this issue, we need to reproduce the error first. The issue provides a command to reproduce the error. Let's run it to see if we get the same error.",
      "action": "lexicon memset create example.com TXT --name _acme-challenge.example.com --content BLAH --ttl 300",
      "observation": null,
      "tool_name": "lexicon",
      "command": "lexicon memset create example.com TXT --name _acme-challenge.example.com --content BLAH --ttl 300",
      "files_touched": [], "timestamp": null,
      "raw": { "role": "ai", "text": "To start solving ... ```\nlexicon memset create ...\n```", "system_prompt": null, "mask": true, "cutoff_date": null }
    }
  ],
  "raw_metadata": {
    "patch_length": 2190,
    "eval_logs_length": 2048,
    "extra_top_level_keys": []
  }
}
```

## 9. Portability to SWE-smith

The fallback dataset (`SWE-bench/SWE-smith-trajectories`) uses
`resolved` instead of `target`, and carries trajectories in a slightly
different shape per its dataset card. The normalizer's
`source="swe_agent_smith"` branch is **not implemented in C2**, but the
schema is designed so that adding it is one mapping function:

- map `resolved` → `final_success`
- map upstream role labels → `{system, assistant, tool, environment}`
- preserve everything else under `raw` / `raw_metadata`

## 10. What this schema does NOT decide

- **Ledger annotation categories** — fixed by D1 and
  `ledger_progress/queries.py:CODING_CATEGORIES`. The schema carries no
  category labels.
- **Step-level success / progress** — that is what retrospective
  annotation produces. The normalized trace is the *input*, not the
  output.
- **Whether a patch is materially correct** — `final_success` is the
  upstream label only; `eval_logs` / `generated_patch` are kept as raw
  text artifacts at the importer level (C3), not normalized here.
