"""Mobility episode schema (Workstream K1).

A `MobilityEpisode` is the input to agent_migrate's L3 abstraction: a batch of
stateful agentic workflows that must change placement and reconstitute
state at one or more destinations under finite resource budgets.

Per_workflow manifests stay on disk (each `Workflow.manifest_path` points
at a JSON serialization of `ServingGroupManifest`); the episode itself is
small JSON referencing them. This avoids inlining N×workflow_manifest
content into a single huge JSON for herd benchmarks.

`state_warmness` is the per_state_id set of sites already holding a warm
copy at episode trigger time — set to empty for cold starts. K2's
`WarmnessMap` is the in_memory representation; K1 just carries the JSON.
`capacities` is similarly a JSON_side representation of K3's `ResourceBudget`;
the simulator (K4) consumes a typed budget, not raw JSON.

Hard rule (per CLAUDE.md): episodes ride alongside the ledger, not on it.
No new event class.
"""
from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path


@dataclass(frozen=True)
class Workflow:
    """One stateful agentic workflow that must move during the episode.

    `manifest_path` references a JSON_serialized ServingGroupManifest
    (output of `agent_migrate_manifest build`). `src_site` is None for a cold
    start (no reconstitution from a source); set when the workflow is
    currently running at a known site. `deadline_s` is optional;
    when set, K7's policies may use it for prioritization, and the
    episode_level metric reports per_workflow on_time fraction.
    """
    workflow_id: str
    manifest_path: str
    src_site: str | None = None
    deadline_s: float | None = None


@dataclass(frozen=True)
class MobilityEpisode:
    """A batch event: N workflows reconstituting at >=1 destination(s).

    `source_sites` is the SET of sites workflows are currently running at:
      - exactly one entry => single_source_evacuation scenario (the
        dominant production motivation, per A2 audit)
      - >1 entry => distributed_origin scenario (existing H2/H5a/H5b
        fixtures fall here)
    `destination_sites` lists the candidate destinations (must be >=1; may
    overlap source_sites if some workflows can stay put).

    `state_warmness` is a dict[state_id -> tuple[site,...]] of sites
    already holding a warm copy of the state at trigger time.

    `capacities` carries per_site / per_link resource budgets in dict
    form (K3's ResourceBudget can be reconstructed from this).
    """
    episode_id: str
    source_sites: tuple[str, ...]
    destination_sites: tuple[str, ...]
    workflows: tuple[Workflow, ...]
    state_warmness: dict[str, tuple[str, ...]] = field(default_factory=dict)
    capacities: dict | None = None
    trigger_t_s: float = 0.0
    notes: str = ""

    def __post_init__(self) -> None:
        if len(self.source_sites) < 1:
            raise ValueError("episode requires >=1 source_sites (use the workflow's "
                             "current site, or list ('cold',) for cold_start episodes)")
        if len(self.destination_sites) < 1:
            raise ValueError("episode requires >=1 destination_sites")
        if not self.workflows:
            raise ValueError("episode requires >=1 workflows")
        wf_ids = [wf.workflow_id for wf in self.workflows]
        if len(wf_ids) != len(set(wf_ids)):
            raise ValueError(f"workflow_ids must be unique within an episode; got {wf_ids}")

    @property
    def scenario_class(self) -> str:
        """A2 audit's labels.

        Returns:
            "single_source_evacuation"   if len(source_sites) == 1
            "distributed_origin"          if len(source_sites) > 1
        """
        return "single_source_evacuation" if len(self.source_sites) == 1 else "distributed_origin"


# ---------------------------------------------------------------------------
# JSON serialization
# ---------------------------------------------------------------------------


def dump_episode(episode: MobilityEpisode, path: str | Path) -> None:
    """Write episode to JSON. Round_trips byte_deterministically with
    `load_episode`."""
    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(_to_json(episode))


def load_episode(path: str | Path) -> MobilityEpisode:
    raw = json.loads(Path(path).read_text())
    return _from_json(raw)


def _to_json(episode: MobilityEpisode) -> str:
    payload = {
        "episode_id": episode.episode_id,
        "source_sites": list(episode.source_sites),
        "destination_sites": list(episode.destination_sites),
        "workflows": [
            {
                "workflow_id": wf.workflow_id,
                "manifest_path": wf.manifest_path,
                "src_site": wf.src_site,
                "deadline_s": wf.deadline_s,
            }
            for wf in episode.workflows
        ],
        # Sort state_warmness keys for byte_determinism; sort each
        # site list for the same reason.
        "state_warmness": {
            sid: sorted(episode.state_warmness[sid]) for sid in sorted(episode.state_warmness)
        },
        "capacities": episode.capacities,
        "trigger_t_s": episode.trigger_t_s,
        "notes": episode.notes,
    }
    return json.dumps(payload, indent=2, sort_keys=False) + "\n"


def _from_json(raw: dict) -> MobilityEpisode:
    workflows = tuple(
        Workflow(
            workflow_id=wf["workflow_id"],
            manifest_path=wf["manifest_path"],
            src_site=wf.get("src_site"),
            deadline_s=wf.get("deadline_s"),
        )
        for wf in raw["workflows"]
    )
    state_warmness = {
        sid: tuple(sites) for sid, sites in (raw.get("state_warmness") or {}).items()
    }
    return MobilityEpisode(
        episode_id=raw["episode_id"],
        source_sites=tuple(raw["source_sites"]),
        destination_sites=tuple(raw["destination_sites"]),
        workflows=workflows,
        state_warmness=state_warmness,
        capacities=raw.get("capacities"),
        trigger_t_s=raw.get("trigger_t_s", 0.0),
        notes=raw.get("notes", ""),
    )


# ---------------------------------------------------------------------------
# Adapter: wrap one F2_style SWE_agent trace as a single_workflow episode
# ---------------------------------------------------------------------------


def linear_session_to_episode(
    trace_path: str | Path,
    manifest_path: str | Path,
    *,
    workflow_id: str = "wf_0",
    episode_id: str = "linear_session_episode",
    src_site: str = "phoenix",
    target_site: str = "phoenix",
    notes: str = "",
) -> MobilityEpisode:
    """Build a 1_workflow episode wrapping one already_built manifest.

    Caller is responsible for having generated `manifest_path` from
    `trace_path` via `agent_migrate_manifest build` (or equivalent). The trace
    path is recorded only in `notes` for traceability.

    By default, source = destination = phoenix → a no_op episode (used in
    K1 tests to verify the schema, not to drive interesting reconstitution).
    For a real reconstitution test, pass distinct src_site / target_site.
    """
    workflow = Workflow(
        workflow_id=workflow_id,
        manifest_path=str(manifest_path),
        src_site=src_site,
        deadline_s=None,
    )
    return MobilityEpisode(
        episode_id=episode_id,
        source_sites=(src_site,),
        destination_sites=(target_site,),
        workflows=(workflow,),
        state_warmness={},
        capacities=None,
        trigger_t_s=0.0,
        notes=notes or f"adapted from trace {trace_path}",
    )
