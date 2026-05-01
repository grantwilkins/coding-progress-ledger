# Event observability matrix

| Event / transition | Level | N4 note |
|---|---|---|
| INIT | mechanical | Sidecar creates one root event from the first wire timestamp. |
| ADD_SUBTASK investigation/product/artifact from emitted tool action | mechanical | N3 live runs produce these from `tool_name`/`command`. |
| ADD_SUBTASK validation from emitted validation command | mechanical | Only when the agent actually runs pytest/tox/python repro. |
| ADD_SUBTASK validation obligation without emitted validation command | annotation_only | Requires semantic judgment that validation was discovered but not attempted. |
| UPDATE_STATUS complete from tool observation | mechanical | N3 marks observed tool actions complete when a following tool observation exists. |
| UPDATE_STATUS start from command without observation | mechanical | Sidecar can emit in-progress work for issued commands with no observation. |
| UPDATE_STATUS blocked | annotation_only | Needs a semantic stuck/block judgment; not present in the N3 live adapter. |
| REOPEN_SUBTASK | annotation_only | Needs evidence that prior completion was invalidated by later work. |
| INVALIDATE_SUBTASK | annotation_only | Needs semantic replacement/deletion judgment. |
| SPLIT_SUBTASK | weakly_inferable | Explicit `ledger_ops` can produce it; raw step adapter cannot reliably infer grouping. |

## Event types seen in N3 pairs

| Instance | Retrospective event types | Live event types |
|---|---|---|
| `Melevir__cognitive_complexity-15` | `add_subtask:6, init:1, update_status:6` | `add_subtask:21, init:1, update_status:21` |
| `WIPACrepo__iceprod-339` | `add_subtask:4, init:1, update_status:3` | `add_subtask:8, init:1, update_status:8` |
