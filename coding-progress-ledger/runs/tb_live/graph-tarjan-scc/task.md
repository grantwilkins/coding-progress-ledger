# graph-tarjan-scc

Implement Tarjan's (or Kosaraju's) strongly connected components algorithm as a
Python package.

## What you must produce

A package importable as `tarjan_scc` exposing exactly one public function:

```python
def scc(adj: dict[int, list[int]]) -> list[list[int]]
```

`adj` is an adjacency dictionary mapping each node to its list of
out-neighbors. Keys with no out-edges may be absent; if a node appears
only as a target it must still be treated as a node. The function returns
all strongly connected components of the directed graph.

## Output contract

- Each inner list (one SCC) is sorted in ascending order.
- The outer list is sorted in ascending order by the minimum element of
  each SCC.
- A single isolated node with no edges forms an SCC of size 1.
- A self-loop `{0: [0]}` counts as a cycle and returns `[[0]]`.
- An empty adjacency dict `{}` returns `[]`.

## Examples

```python
from tarjan_scc import scc

scc({})                                      # []
scc({0: []})                                 # [[0]]
scc({0: [0]})                                # [[0]]
scc({0: [1], 1: [0]})                        # [[0, 1]]
scc({0: [1], 1: [2], 2: []})                 # [[0], [1], [2]]
scc({0: [1], 1: [0], 2: [3], 3: [2]})       # [[0, 1], [2, 3]]
scc({0: [1], 1: [2], 2: [0, 3], 3: [4], 4: [3]})  # [[0, 1, 2], [3, 4]]
scc({0: [1], 1: [0], 5: []})                 # [[0, 1], [5]]
```

## Repository layout

Use the standard `src/` layout. The verifier expects:

```
<agent_repo>/
  src/
    tarjan_scc/
      __init__.py     # exports `scc`
```

The verifier puts `<agent_repo>/src` on `PYTHONPATH` and runs `pytest`
against hidden test files. You may add a `pyproject.toml`, a `tests/`
directory, or anything else you want; only the `src/tarjan_scc/` contract
is load-bearing.

## Algorithm guidance

Tarjan's algorithm or Kosaraju's algorithm are both acceptable. The
output format (sorted inner lists, outer list sorted by min element) is
what the verifier checks — not the internal algorithm.

For Tarjan's: use iterative DFS with an explicit stack to avoid Python
recursion limits on large graphs.

For Kosaraju's: two-pass DFS — first pass on the original graph to get
finish order, second pass on the transposed graph in reverse finish order.

## What is NOT required

You do not need to support: weighted edges, parallel edges, labeled nodes,
`__main__` CLI, serialisation, or any graph format other than the
`dict[int, list[int]]` adjacency dict described above.

## How to track progress

You are running under the N_TB live ledger harness. After each
meaningful action (subtask added, started, completed, blocked, etc.),
emit one wire-format event with:

```bash
uv run python /Users/grantwilkins/houdini/coding-progress-ledger/scripts/tb_emit.py \
    /Users/grantwilkins/houdini/coding-progress-ledger/runs/tb_live/graph-tarjan-scc \
    <step_number> \
    '[{"op":"add","id":"s1","description":"...","category":"product"}]'
```

See the project's `docs/AGENT_USAGE.md` and `docs/TB_LIVE_TASK_FORMAT.md`
for the protocol. Use `product` for code-that-ships, `validation` for
tests / asserts / manual checks, `investigation` for reading / search /
trace work. Add subtasks as you discover them, not as a plan up front.
Mark complete only with concrete evidence.

## Done condition

You are done when `verifier.sh` exits 0 against your repo. The
verifier is hidden — you cannot read it. Your fastest path to done is
to write your own tests for each case in the spec above, run them, and
only declare a leaf complete when the test passes.
