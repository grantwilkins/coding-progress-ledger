# Model Agent Prompt Contract

The model controller runs on the host with provider/API access. The task
sandbox runs inside Docker with network disabled. The model can interact with
the task only through structured tool actions executed by the sandbox executor.

The model sees:

```text
task.md
available tool specs
recent transcript summary
budget state
tool outputs from its own prior actions
```

The model must not see:

```text
hidden tests
oracle files
solution files
gold patches
verifier internals
post-run verifier output before the terminal observation step
terminal labels or estimator labels
```

Each turn must return exactly one JSON object:

```json
{
  "thought_summary": "I need to inspect the task and files.",
  "action": {
    "type": "read_file",
    "path": "task.md"
  }
}
```

Allowed action types:

```text
find_files(pattern, path)
grep(pattern, path, file_glob)
list_dir(path)
read_file(path, start_line, end_line)
write_file(path, content)
edit_file(path, instruction)
apply_patch(unified_diff)
shell(command)
done(summary)
```

Behavioral requirements:

- Use only the provided tools.
- Treat hidden tests and verifier internals as unavailable.
- Do not claim `done` until a reasonable validation attempt has been made
  when one is available.
- Real pilot runs may enforce `min_steps_before_done` and
  `require_validation_before_done`. If the controller rejects `done`, continue
  with useful inspection, implementation, or validation.
- A blocked `done` is allowed only when the visible transcript supports that
  the task is genuinely blocked, such as missing required files, denied tools,
  unavailable commands, or insufficient visible task information.
- Return one action per turn, not prose outside JSON.
- Keep shell commands scoped to the sandboxed workspace.
