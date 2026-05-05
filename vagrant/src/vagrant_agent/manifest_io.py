"""JSON I/O for ServingGroupManifest. Lossless dataclass <-> dict <-> file."""
from __future__ import annotations

import json
from dataclasses import asdict
from pathlib import Path

from .manifest import ServingGroupManifest, StateEdge, StateObject, WorkNode


def to_dict(manifest: ServingGroupManifest) -> dict:
    return {
        "workflow_id": manifest.workflow_id,
        "root_task": manifest.root_task,
        "nodes": {nid: asdict(n) for nid, n in manifest.nodes.items()},
        "state_objects": {sid: asdict(s) for sid, s in manifest.state_objects.items()},
        "edges": [asdict(e) for e in manifest.edges],
    }


def from_dict(data: dict) -> ServingGroupManifest:
    return ServingGroupManifest(
        workflow_id=data["workflow_id"],
        root_task=data["root_task"],
        nodes={nid: WorkNode(**n) for nid, n in data["nodes"].items()},
        state_objects={sid: StateObject(**s) for sid, s in data["state_objects"].items()},
        edges=[StateEdge(**e) for e in data["edges"]],
    )


def write_json(manifest: ServingGroupManifest, path: str | Path) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    Path(path).write_text(json.dumps(to_dict(manifest), indent=2, sort_keys=False) + "\n")


def read_json(path: str | Path) -> ServingGroupManifest:
    return from_dict(json.loads(Path(path).read_text()))
