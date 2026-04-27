# Task 1 Parser Timezone Offset

Toy repo path:

```bash
cd /Users/grantwilkins/houdini/coding-progress-ledger/runs/task_1_parser_timezone_offset/repo
```

Run tests:

```bash
/Users/grantwilkins/houdini/coding-progress-ledger/.venv/bin/python -m pytest -q
```

The initial committed version accepts `+05:30` and rejects compact offsets. The
working tree contains the fixed parser, and `../final_diff.patch` shows the
change from the initial buggy commit to the final solution.
