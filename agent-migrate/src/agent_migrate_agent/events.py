"""Vagrant_specific event types, carried as `event_type` strings on `ledger_progress.LedgerEvent`.

Workflow / node lifecycle (subagent spawn, status, invalidation) reuses ledger_progress
enums (ADD_SUBTASK, UPDATE_STATUS, SPLIT_SUBTASK, REOPEN_SUBTASK, INVALIDATE_SUBTASK).
This module only defines the *additional* event types agent_migrate needs.

These event types ride on LedgerEvent via the upstream pass_through hook (Workstream A2).
They append to the events list without mutating ledger.subtasks.

Payload schemas (all required unless noted):

state_declare:
    state_id: str          # primary identity (synthetic adapters supply this)
    content_hash: str      # secondary identity (real adapters supply this)
    layer: str             # one of: model_execution, prompt_context, subagent,
                           #         workspace, memory, semantic
    tokens: int            # 0 if not applicable
    bytes: int | None      # for non_token state (workspace artifacts), else null
    producer_node_id: str | None
    lifetime: str          # persistent | shared | private | ephemeral

state_read:
    state_id: str
    content_hash: str
    consumer_node_id: str
    tokens: int            # tokens consumed from this state on this read

state_write:
    state_id: str
    content_hash: str
    producer_node_id: str
    tokens: int
    bytes: int | None

state_invalidate:
    state_id: str
    reason: str

placement_decision:
    node_id: str
    site: str
    cost_s: float
    reason: str            # e.g. "min_cost", "forced_by_colocation"

materialization_plan:
    state_id: str
    site: str
    mode: str              # one of: warm_reuse, kv_transfer, context_replay,
                           #         text_transfer, artifact_copy, workspace_hydrate,
                           #         remote_workspace, summary, restart
    cost_s: float
    reason: str

migration_start:
    workflow_id: str
    src_site: str
    dst_site: str

migration_end:
    workflow_id: str
    src_site: str
    dst_site: str
    elapsed_s: float
"""

STATE_DECLARE = "state_declare"
STATE_READ = "state_read"
STATE_WRITE = "state_write"
STATE_INVALIDATE = "state_invalidate"

PLACEMENT_DECISION = "placement_decision"
MATERIALIZATION_PLAN = "materialization_plan"

MIGRATION_START = "migration_start"
MIGRATION_END = "migration_end"

ALL = (
    STATE_DECLARE,
    STATE_READ,
    STATE_WRITE,
    STATE_INVALIDATE,
    PLACEMENT_DECISION,
    MATERIALIZATION_PLAN,
    MIGRATION_START,
    MIGRATION_END,
)


STATE_LAYERS = (
    "model_execution",
    "prompt_context",
    "subagent",
    "workspace",
    "memory",
    "semantic",
)

LIFETIMES = ("persistent", "shared", "private", "ephemeral")

MATERIALIZATION_MODES = (
    "warm_reuse",
    "kv_transfer",
    "context_replay",
    "text_transfer",
    "artifact_copy",
    "workspace_hydrate",
    "remote_workspace",
    "summary",
    "restart",
)

NODE_TYPES = ("llm_call", "tool_call", "subagent", "summary", "test")
