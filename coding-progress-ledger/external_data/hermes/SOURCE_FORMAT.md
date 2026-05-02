# Hermes — source format

Source: [`lambda/hermes-agent-reasoning-traces`](https://huggingface.co/datasets/lambda/hermes-agent-reasoning-traces).
Apache 2.0. Public, ungated. Two configs: `kimi`, `glm-5.1`. ~14.7k
total traces (kimi 7,646 / glm-5.1 7,055). Parquet at
`data/{kimi,glm-5.1}/train.parquet`.

## Top-level fields (per row, verified against real samples)

| field           | type    | example                                          |
|-----------------|---------|--------------------------------------------------|
| `id`            | string  | UUID, e.g. `1b510b01-5892-4810-8663-8f457280d904` |
| `task`          | string  | task description (human-readable prompt)         |
| `tools`         | string  | JSON array of tool definitions (function calling) |
| `category`      | string  | one of 9 (Terminal & Coding, Repository Tasks, …) |
| `subcategory`   | string  | fine-grained                                     |
| `conversations` | list    | ShareGPT format: `[{from: str, value: str}, …]`  |

**No upstream success label.** No `target` / `resolved` / `exit_status`
/ `generated_patch` / `eval_logs` field. Downstream pipelines must
treat `final_success` as `null`.

## Conversation roles

```
system  — initial system prompt (function-calling instructions, tools available)
human   — user message
gpt     — assistant message (may contain <think>…</think> and one-or-more <tool_call>…</tool_call> blocks)
tool    — tool response (<tool_response>{...}</tool_response>)
```

## Tool call / response shape

Inside `gpt` value:
```text
<think>...reasoning...</think>
[free-text response to user, optional]
<tool_call>
{"name": "...", "arguments": {...}}
</tool_call>
<tool_call>
{"name": "...", "arguments": {...}}
</tool_call>
```
Multiple `<tool_call>` blocks in one `gpt` turn are common (verified
on kimi sample: up to 4 per turn). Each tool_call has a matching
`tool` turn whose value contains:
```text
<tool_response>
{"tool_call_id": "...", "name": "...", "content": {...}}
</tool_response>
```

## License & retention

Apache 2.0. Raw parquet files belong under `external_data/hermes/raw/`
(gitignored, like SWE-agent). Sample rows for documentation are
committed under `external_data/hermes/raw/sample_row_*.json`.

## What we sampled for documentation

- `external_data/hermes/raw/sample_row_kimi.json` — `0c699abf-…`,
  category `Agent Tools`, 13 conversations.
- `external_data/hermes/raw/sample_row_glm-5.1.json` — `1b510b01-…`,
  category `Terminal & Coding`, 29 conversations.
