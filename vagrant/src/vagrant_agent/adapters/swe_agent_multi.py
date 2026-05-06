"""Multi-session SWE-agent fixture (Workstream H2).

Concatenates 2-3 F2-style SWE-agent trajectories into a single vagrant trace
with:

  * ONE shared `system_prompt` state across all sessions (deduped by virtue of
    a single `state_declare`; consumed by every llm_call across every session).
  * Per-session `issue_text_<session_id>` and `tool_output_<session_id>_<NNN>`
    states. State IDs are session-prefixed so that re-using the same trajectory
    across sessions produces *distinct* state objects, even though the
    underlying content_hash is identical. This is **the intended framing** of
    H2: each session is conceptually a different agent invocation against a
    different repo, so state identity should be temporal/session, not content.
  * Per-session synthetic `workspace_<session_id>` state — declared with
    `home_site` set per session and a `bytes` field. Real SWE-agent
    trajectories do not surface workspace bytes (cf. `swe_agent.py`); H2
    augments the trace with this state so D2's grouping has something to
    fight against. The synthetic part of this fixture is the workspace bytes,
    not the trace structure.

Within-session `tool_output` content_hash dedup is preserved (a `cat empty.py`
emitted twice in one session collapses to one state object); across sessions,
state IDs are distinct, so identical content does NOT collapse. Without
per-session scoping the F2 dedup map would erase the multi-session structure
H2 is trying to expose.

This is **not** a real harness adapter and does not unlock the original
"phenomenon demonstrated" gate. It is the smallest realistic fixture that puts
D2 (`shared_state_aware`) in its natural habitat — multiple components linked
by a shared prompt, each anchored at a different site by a private workspace —
so we can measure whether H1 (`request_level_with_site_cache`) beats D2 on
something that resembles a real workload's structure.

Truncation. `max_ai_turns` caps the number of llm_call nodes per session,
because G1's brute-force enumeration is hard-capped at K^N <= 100_000
(`policies.G1_MAX_ENUMERATIONS`). For K=2 sites and 3 sessions, total nodes
must be <= 16; the canonical fixture uses 2 ai turns per session = 6 nodes
(2^6 = 64 enumerations) so G1 fits comfortably.
"""
from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

from .. import events as v_events
from ..hashing import segment_hash


def approx_tokens(text: str) -> int:
    return max(1, len(text) // 4)


@dataclass(frozen=True)
class SessionSpec:
    """One concatenated SWE-agent session.

    `session_id` must be a short, file-system-safe slug (used as a state_id
    and node_id prefix). It must be unique within a `MultiSessionConfig`.
    `workspace_home_site` is the site the session's synthetic workspace is
    anchored at (so per-node placement pulls the session toward that site).
    `workspace_bytes` is the synthetic byte size of the workspace. At the
    canonical 1 GB scale, the cross-site artifact_copy at 5 Gbps is 1.6 s,
    which dominates the prompt-context costs and produces the H1 < D2 gap.
    `max_ai_turns` truncates the trajectory to keep the total node count
    within G1's enumeration cap.
    """
    traj_path: str | Path
    session_id: str
    workspace_home_site: str
    workspace_bytes: int
    max_ai_turns: int | None = None


@dataclass(frozen=True)
class MultiSessionConfig:
    sessions: tuple[SessionSpec, ...]
    workflow_id: str = "h2_multi_session_swe"
    root_task: str = "H2: multi-session SWE-style fixture with synthetic workspaces"


def generate_events(config: MultiSessionConfig) -> list[dict]:
    if len(config.sessions) < 2:
        raise ValueError("H2 fixture requires >=2 sessions; for single-session use the F2 adapter")
    if len(config.sessions) != len({s.session_id for s in config.sessions}):
        raise ValueError("session_ids must be unique within a MultiSessionConfig")
    for spec in config.sessions:
        if spec.max_ai_turns is not None and spec.max_ai_turns < 1:
            raise ValueError(
                f"session {spec.session_id!r}: max_ai_turns must be >=1 (or None for no truncation); "
                f"got {spec.max_ai_turns}"
            )

    raw_trajs = [_load_traj(s.traj_path) for s in config.sessions]
    system_prompts = [t["sys"] for t in raw_trajs]
    if len(set(system_prompts)) != 1:
        raise ValueError(
            "H2 fixture requires identical system_prompt across sessions "
            "(otherwise the shared state collapses); supply trajectories with the same SETTING"
        )

    ev: list[dict] = []
    step = 0

    def emit(event_type: str, subtask_id: str | None, payload: dict, reason: str | None = None) -> None:
        nonlocal step
        ev.append({"step": step, "event_type": event_type, "subtask_id": subtask_id,
                   "payload": payload, "reason": reason})
        step += 1

    emit("init", None, {"root_task": config.root_task})

    sys_text = system_prompts[0]
    sys_hash = segment_hash(sys_text)
    sys_tokens = approx_tokens(sys_text)
    emit(v_events.STATE_DECLARE, None, {
        "state_id": "system_prompt",
        "content_hash": sys_hash,
        "layer": "prompt_context",
        "lifetime": "persistent",
        "tokens": sys_tokens,
        "bytes": None,
        "producer_node_id": None,
    })

    for spec, traj in zip(config.sessions, raw_trajs):
        _emit_session(emit, spec, traj, sys_tokens, sys_hash, config.workflow_id)

    return ev


def _emit_session(emit, spec: SessionSpec, traj: dict, sys_tokens: int, sys_hash: str,
                  workflow_id: str) -> None:
    sid = spec.session_id
    issue_text = traj["issue"]
    issue_state_id = f"issue_text_{sid}"
    issue_hash = segment_hash(issue_text)
    issue_tokens = approx_tokens(issue_text)
    emit(v_events.STATE_DECLARE, None, {
        "state_id": issue_state_id,
        "content_hash": issue_hash,
        "layer": "prompt_context",
        "lifetime": "shared",
        "tokens": issue_tokens,
        "bytes": None,
        "producer_node_id": None,
    })

    workspace_state_id = f"workspace_{sid}"
    emit(v_events.STATE_DECLARE, None, {
        "state_id": workspace_state_id,
        "content_hash": f"hash_workspace_{sid}_v1",
        "layer": "workspace",
        "lifetime": "private",
        "tokens": 0,
        "bytes": int(spec.workspace_bytes),
        "producer_node_id": None,
        "home_site": spec.workspace_home_site,
    })

    declared_outputs: dict[str, str] = {}
    state_meta: dict[str, dict] = {
        "system_prompt": {"hash": sys_hash, "tokens": sys_tokens},
        issue_state_id: {"hash": issue_hash, "tokens": issue_tokens},
        workspace_state_id: {"hash": f"hash_workspace_{sid}_v1", "tokens": 0},
    }
    accumulated_output_state_ids: list[str] = []

    turns = traj["turns"]
    max_turns = spec.max_ai_turns if spec.max_ai_turns is not None else len(turns)
    last_kept_turn_index = _last_kept_ai_turn_index(turns, max_turns)
    llm_call_index = 0

    for i, turn in enumerate(turns[1:]):
        absolute_index = i + 1
        if absolute_index > last_kept_turn_index:
            break
        role = turn.get("role")
        text = turn.get("text") or ""

        if role == "user":
            if i == 0:
                continue
            if not text:
                continue
            content_hash = segment_hash(text)
            if content_hash in declared_outputs:
                state_id = declared_outputs[content_hash]
            else:
                state_id = f"tool_output_{sid}_{len(declared_outputs):03d}"
                declared_outputs[content_hash] = state_id
                state_tokens = approx_tokens(text)
                state_meta[state_id] = {"hash": content_hash, "tokens": state_tokens}
                emit(v_events.STATE_DECLARE, None, {
                    "state_id": state_id,
                    "content_hash": content_hash,
                    "layer": "prompt_context",
                    "lifetime": "ephemeral",
                    "tokens": state_tokens,
                    "bytes": None,
                    "producer_node_id": None,
                })
            if state_id not in accumulated_output_state_ids:
                accumulated_output_state_ids.append(state_id)
            continue

        if role == "ai":
            if llm_call_index >= max_turns:
                break
            llm_call_index += 1
            node_id = f"{sid}_S{llm_call_index}"
            emit("add_subtask", node_id, {
                "description": f"llm_call_{sid}_{llm_call_index}",
                "parent_id": None,
                "weight": 1.0,
                "category": "product",
                "node_type": "llm_call",
                "workflow_id": workflow_id,
            })
            for state_id in ("system_prompt", issue_state_id, workspace_state_id,
                             *accumulated_output_state_ids):
                meta = state_meta[state_id]
                emit(v_events.STATE_READ, None, {
                    "state_id": state_id,
                    "content_hash": meta["hash"],
                    "consumer_node_id": node_id,
                    "tokens": meta["tokens"],
                })
            emit("update_status", node_id, {
                "status": "complete",
                "evidence": [f"{node_id} response emitted"],
            })


def _last_kept_ai_turn_index(turns: list[dict], max_turns: int) -> int:
    """Index (in `turns`) of the K-th ai turn, where K = max_turns. Used to
    drop trailing user turns whose ai-turn consumer is truncated away — those
    user turns would otherwise become tool_output state declarations with
    zero consumers."""
    ai_seen = 0
    for idx, turn in enumerate(turns):
        if turn.get("role") == "ai":
            ai_seen += 1
            if ai_seen >= max_turns:
                return idx
    return len(turns) - 1


def _load_traj(traj_path: str | Path) -> dict:
    raw = json.loads(Path(traj_path).read_text())
    turns = raw["trajectory"]
    if not turns or turns[0].get("role") != "system":
        raise ValueError(f"first turn must be system in {traj_path}")
    sys_prompt = turns[0].get("system_prompt") or ""
    if not sys_prompt:
        raise ValueError(f"system turn has no system_prompt in {traj_path}")
    if turns[1].get("role") != "user":
        raise ValueError(f"first non-system turn must be user (issue text) in {traj_path}")
    for turn in turns:
        if turn.get("role") == "user":
            issue = turn.get("text") or ""
            if not issue:
                raise ValueError(f"first user turn has empty text in {traj_path}")
            return {"sys": sys_prompt, "issue": issue, "turns": turns}
    raise ValueError(f"no user turn found in {traj_path}")


def generate_to_file(config: MultiSessionConfig, path: str | Path) -> list[dict]:
    ev = generate_events(config)
    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text("".join(json.dumps(e, separators=(",", ":")) + "\n" for e in ev))
    return ev
