"""C2 — resume_package taxonomy.

A resume package is a manifest + claim about which state a destination would
need to continue an agent trajectory from a `CutPoint`. C2 enumerates five
package types and produces deterministic manifests; C3 (resume_validator.py)
checks whether a package's claims hold STATICALLY — no model calls, no tool
execution, no real harness.

Package types
-------------

prompt_only
    Only the model_context state declared up to the cut: system prompt, issue
    text, accumulated tool outputs. No harness state, no workspace, no diff.
    Useful as a lower_bound illustrative package; will fail on any cut where
    the next llm_call also needs harness/workspace state.

transcript_plus_harness_state
    `prompt_only` + a serialized harness config (e.g. open file, cwd, env).
    Still no workspace bytes.

transcript_plus_diff
    `transcript_plus_harness_state` + a `(base_commit, diff_blob)` pair so
    the destination can `git checkout base_commit && git apply diff`. C3
    validates the diff with `git apply --check`.

full_workspace_snapshot
    Transcript + harness + complete workspace tarball reference (file digests
    enumerated in the manifest). Upper bound on safety, upper bound on bytes.

agent_migrate_minimal
    Transcript + harness + S1 `must_move` layer subset (S3 will refine the
    must_materialize set with role classification). Layers like
    `dependency_cache` and `base_repo_checkout` are noted as
    `lazy_rehydrate` / `globally_available` rather than included as bytes.

Each package's `state_entries` are deterministic: ordered by `state_id` and
keyed by content hash, so two builds from the same trace produce
byte_identical manifests.

The package builders take only the trace events plus the cut point plus any
externally supplied context (workspace digests, diff blob, base commit,
harness config). They do NOT touch a filesystem on their own.
"""
from __future__ import annotations

import copy
import hashlib
import json
from dataclasses import asdict, dataclass, field, replace
from typing import Iterable

from .cut_points import CutPoint
from .state_layers import S1_LAYERS, materialization_for_role, role_for_layer


PACKAGE_TYPES: tuple[str, ...] = (
    "prompt_only",
    "transcript_plus_harness_state",
    "transcript_plus_diff",
    "full_workspace_snapshot",
    "agent_migrate_minimal",
)

MATERIALIZATIONS: tuple[str, ...] = (
    "included",
    "lazy_rehydrate",
    "globally_available",
)

VALIDATORS: tuple[str, ...] = (
    "digest",
    "diff_apply",
    "transcript_prefix",
    "harness_schema",
    "workspace_digest",
    "materialization_mode",
)

HARNESS_REQUIRED_KEYS: tuple[str, ...] = ("cwd", "open_file", "env")


@dataclass(frozen=True)
class StateEntry:
    state_id: str
    layer: str
    bytes: int
    content_hash: str
    materialization: str
    validator: str
    role_at_cut: str | None = None

    def __post_init__(self) -> None:
        if self.materialization not in MATERIALIZATIONS:
            raise ValueError(f"unknown materialization {self.materialization!r}")
        if self.validator not in VALIDATORS:
            raise ValueError(f"unknown validator {self.validator!r}")


@dataclass(frozen=True)
class WorkspaceFileEntry:
    rel_path: str
    bytes: int
    content_hash: str


@dataclass(frozen=True)
class ResumePackage:
    package_type: str
    cut_point: CutPoint
    state_entries: tuple[StateEntry, ...]
    transcript_prefix_hash: str
    harness_config: dict | None = None
    base_commit: str | None = None
    diff_blob: str | None = None
    workspace_files: tuple[WorkspaceFileEntry, ...] = field(default_factory=tuple)

    def __post_init__(self) -> None:
        if self.package_type not in PACKAGE_TYPES:
            raise ValueError(f"unknown package_type {self.package_type!r}")

    @property
    def included_bytes(self) -> int:
        state_bytes = sum(
            e.bytes
            for e in self.state_entries
            if e.materialization == "included" and not e.state_id.startswith("workspace_layer:")
        )
        ws_bytes = sum(f.bytes for f in self.workspace_files)
        diff_bytes = len((self.diff_blob or "").encode("utf_8"))
        harness_bytes = (
            len(json.dumps(self.harness_config, separators=(",", ":"), sort_keys=True).encode("utf_8"))
            if self.harness_config is not None else 0
        )
        return state_bytes + ws_bytes + diff_bytes + harness_bytes

    @property
    def lazy_rehydrate_bytes(self) -> int:
        return sum(e.bytes for e in self.state_entries if e.materialization == "lazy_rehydrate")

    def to_dict(self) -> dict:
        d = asdict(self)
        d["included_bytes"] = self.included_bytes
        d["lazy_rehydrate_bytes"] = self.lazy_rehydrate_bytes
        return d

    def metrics(self) -> dict:
        """Flat dict suitable for a CSV row. Stable across builds."""
        return {
            "package_type": self.package_type,
            "trace_id": self.cut_point.trace_id,
            "session_id": self.cut_point.session_id,
            "event_index": self.cut_point.event_index,
            "phase": self.cut_point.phase,
            "transcript_prefix_hash": self.transcript_prefix_hash,
            "n_state_entries": len(self.state_entries),
            "n_workspace_files": len(self.workspace_files),
            "included_bytes": self.included_bytes,
            "lazy_rehydrate_bytes": self.lazy_rehydrate_bytes,
            "has_diff": bool(self.diff_blob),
            "has_harness": self.harness_config is not None,
        }


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def transcript_prefix_hash(events: list[dict], cut_index: int) -> str:
    """SHA_256 of events[0:cut_index] serialized canonically (sort_keys, no whitespace)."""
    if cut_index < 0 or cut_index > len(events):
        raise ValueError(f"cut_index {cut_index} out of range for {len(events)} events")
    h = hashlib.sha256()
    for e in events[:cut_index]:
        h.update(json.dumps(e, separators=(",", ":"), sort_keys=True).encode("utf_8"))
        h.update(b"\n")
    return h.hexdigest()


def _collect_prompt_states(events: list[dict], cut_index: int) -> list[StateEntry]:
    """Collect all `prompt_context` state objects declared before the cut."""
    out: list[StateEntry] = []
    for e in events[:cut_index]:
        if e.get("event_type") != "state_declare":
            continue
        p = e.get("payload") or {}
        if p.get("layer") != "prompt_context":
            continue
        out.append(StateEntry(
            state_id=p.get("state_id", ""),
            layer="prompt_context",
            bytes=int(p.get("bytes") or 0),
            content_hash=p.get("content_hash") or "",
            materialization="included",
            validator="digest",
            role_at_cut=p.get("role_at_cut") or role_for_layer("prompt_context"),
        ))
    out.sort(key=lambda s: s.state_id)
    return out


def _collect_all_states(events: list[dict], cut_index: int) -> list[dict]:
    out: list[dict] = []
    for e in events[:cut_index]:
        if e.get("event_type") == "state_declare":
            out.append(e.get("payload") or {})
    return out


def _layer_mobility() -> dict[str, str]:
    return {layer.name: layer.mobility_class for layer in S1_LAYERS}


def _normalize_harness(config: dict) -> dict:
    """Deepcopy + recursively sort dict keys so determinism survives caller insertion order."""
    def _sort(o):
        if isinstance(o, dict):
            return {k: _sort(o[k]) for k in sorted(o.keys())}
        if isinstance(o, (list, tuple)):
            return [_sort(x) for x in o]
        return o
    return _sort(copy.deepcopy(config))


# ---------------------------------------------------------------------------
# Builders
# ---------------------------------------------------------------------------


def build_prompt_only(events: list[dict], cut_point: CutPoint) -> ResumePackage:
    state_entries = _collect_prompt_states(events, cut_point.event_index)
    return ResumePackage(
        package_type="prompt_only",
        cut_point=cut_point,
        state_entries=tuple(state_entries),
        transcript_prefix_hash=transcript_prefix_hash(events, cut_point.event_index),
    )


def build_transcript_plus_harness_state(
    events: list[dict], cut_point: CutPoint, *, harness_config: dict,
) -> ResumePackage:
    base = build_prompt_only(events, cut_point)
    return replace(base, package_type="transcript_plus_harness_state",
                   harness_config=_normalize_harness(harness_config))


def build_transcript_plus_diff(
    events: list[dict],
    cut_point: CutPoint,
    *,
    harness_config: dict,
    base_commit: str,
    diff_blob: str,
) -> ResumePackage:
    base = build_prompt_only(events, cut_point)
    return replace(base, package_type="transcript_plus_diff",
                   harness_config=_normalize_harness(harness_config),
                   base_commit=base_commit, diff_blob=diff_blob)


def build_full_workspace_snapshot(
    events: list[dict],
    cut_point: CutPoint,
    *,
    harness_config: dict,
    workspace_files: Iterable[WorkspaceFileEntry],
) -> ResumePackage:
    base = build_prompt_only(events, cut_point)
    files = tuple(sorted(workspace_files, key=lambda f: f.rel_path))
    state_entries = list(base.state_entries)
    if files:
        payload = json.dumps(
            [(f.rel_path, f.content_hash) for f in files],
            separators=(",", ":"),
        ).encode("utf_8")
        digest = hashlib.sha256(payload).hexdigest()[:16]
        state_entries.append(StateEntry(
            state_id="workspace_layer:full_workspace_snapshot",
            layer="workspace_snapshot",
            bytes=sum(f.bytes for f in files),
            content_hash="h_" + digest,
            materialization="included",
            validator="workspace_digest",
            role_at_cut="correctness_critical",
        ))
        state_entries.sort(key=lambda s: s.state_id)
    return replace(base, package_type="full_workspace_snapshot",
                   harness_config=_normalize_harness(harness_config),
                   state_entries=tuple(state_entries),
                   workspace_files=files)


def build_agent_migrate_minimal(
    events: list[dict],
    cut_point: CutPoint,
    *,
    harness_config: dict,
    workspace_files: Iterable[WorkspaceFileEntry] = (),
    workspace_layer_for_file: dict[str, str] | None = None,
) -> ResumePackage:
    """`agent_migrate_minimal` includes prompt states + S3 must_materialize
    workspace roles only. Other workspace layers are summarized via
    `state_entries` (not bytes).

    Role → materialization mapping:
      correctness_critical | external_side_effect → included
      performance_critical | reconstructable      → lazy_rehydrate
      diagnostic | disposable                     → SKIPPED ENTIRELY
    """
    base = build_prompt_only(events, cut_point)
    layer_mobility = _layer_mobility()

    included_files: list[WorkspaceFileEntry] = []
    summary_entries: list[StateEntry] = list(base.state_entries)

    if workspace_files:
        if workspace_layer_for_file is None:
            raise ValueError("agent_migrate_minimal requires workspace_layer_for_file when workspace_files is given")
        layer_to_files: dict[str, list[WorkspaceFileEntry]] = {}
        for wf in workspace_files:
            layer = workspace_layer_for_file.get(wf.rel_path)
            if layer is None:
                raise ValueError(f"workspace_layer_for_file missing entry for {wf.rel_path!r}")
            if layer not in layer_mobility:
                raise ValueError(f"unknown S1 layer {layer!r} for {wf.rel_path!r}")
            layer_to_files.setdefault(layer, []).append(wf)

        for layer, files in sorted(layer_to_files.items()):
            mobility = layer_mobility[layer]
            if mobility == "can_be_discarded":
                continue
            total_bytes = sum(f.bytes for f in files)
            role = role_for_layer(layer)
            materialization = materialization_for_role(role)
            if materialization == "skipped":
                continue
            payload = json.dumps(sorted([(f.rel_path, f.content_hash) for f in files]),
                                  separators=(",", ":")).encode("utf_8")
            digest = hashlib.sha256(payload).hexdigest()[:16]
            if materialization == "included":
                included_files.extend(files)
            if materialization == "lazy_rehydrate" and mobility == "globally_available":
                materialization = "globally_available"
            summary_entries.append(StateEntry(
                state_id=f"workspace_layer:{layer}",
                layer=layer,
                bytes=total_bytes,
                content_hash="h_" + digest,
                materialization=materialization,
                validator="workspace_digest",
                role_at_cut=role,
            ))

    summary_entries.sort(key=lambda s: s.state_id)
    return replace(base,
                   package_type="agent_migrate_minimal",
                   state_entries=tuple(summary_entries),
                   harness_config=_normalize_harness(harness_config),
                   workspace_files=tuple(sorted(included_files, key=lambda f: f.rel_path)))
