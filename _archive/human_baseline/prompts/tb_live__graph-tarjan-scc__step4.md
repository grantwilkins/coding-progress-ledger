# Human baseline prompt — tb_live/graph-tarjan-scc

**Midpoint step:** 4
**Events visible:** 9 of 13 total

## Task

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


## Ledger events visible (prefix only)

```jsonl
{"event_type": "EventType.INIT", "payload": {"root_task": "graph-tarjan-scc"}, "reason": null, "step": 0, "subtask_id": null, "timestamp": "2026-05-04T07:53:40.589473Z"}
{"event_type": "EventType.ADD_SUBTASK", "payload": {"category": "investigation", "description": "read task spec", "parent_id": null, "weight": 1.0}, "reason": null, "step": 1, "subtask_id": "i1", "timestamp": "2026-05-04T07:53:40.589473Z"}
{"event_type": "EventType.UPDATE_STATUS", "payload": {"status": "in_progress"}, "reason": null, "step": 1, "subtask_id": "i1", "timestamp": "2026-05-04T07:53:40.589473Z"}
{"event_type": "EventType.UPDATE_STATUS", "payload": {"evidence": ["task spec read: implement tarjan_scc package with scc() function, src/ layout, 8 example cases, sorted output contract"], "status": "complete"}, "reason": null, "step": 2, "subtask_id": "i1", "timestamp": "2026-05-04T07:53:45.793141Z"}
{"event_type": "EventType.ADD_SUBTASK", "payload": {"category": "product", "description": "implement tarjan_scc package with scc() function", "parent_id": null, "weight": 1.0}, "reason": null, "step": 3, "subtask_id": "p1", "timestamp": "2026-05-04T07:53:56.007564Z"}
{"event_type": "EventType.UPDATE_STATUS", "payload": {"status": "in_progress"}, "reason": null, "step": 3, "subtask_id": "p1", "timestamp": "2026-05-04T07:53:56.007564Z"}
{"event_type": "EventType.ADD_SUBTASK", "payload": {"category": "validation", "description": "write tests for all 8 spec examples", "parent_id": null, "weight": 1.0}, "reason": null, "step": 3, "subtask_id": "v1", "timestamp": "2026-05-04T07:53:56.007564Z"}
{"event_type": "EventType.ADD_SUBTASK", "payload": {"category": "validation", "description": "run pytest and verify all tests pass", "parent_id": null, "weight": 1.0}, "reason": null, "step": 3, "subtask_id": "v2", "timestamp": "2026-05-04T07:53:56.007564Z"}
{"event_type": "EventType.UPDATE_STATUS", "payload": {"evidence": ["tarjan_scc/__init__.py written with recursive Tarjan SCC, all-nodes collection, sorted output contract"], "status": "complete"}, "reason": null, "step": 4, "subtask_id": "p1", "timestamp": "2026-05-04T07:54:06.324983Z"}
```

## Predict

Given only the prefix above, fill in `human_predictions.csv` with one row per target:

```csv
run_id,target,p_success
graph-tarjan-scc,y_success_eventual,<your probability in [0, 1]>
graph-tarjan-scc,y_future_progress_drop_h5,<your probability in [0, 1]>
```
