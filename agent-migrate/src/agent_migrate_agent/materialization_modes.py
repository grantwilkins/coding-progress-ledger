"""M1 — materialization mode registry keyed by (state_layer, role_at_cut).

The registry records structural representation choices; it does not claim that
rerunning commands, refetching documents, or replaying prompts is semantically
equivalent without validator evidence. Callers must pass both layer and role so
role-sensitive cases such as correctness-critical vs diagnostic tool outputs do
not collapse to a layer-only decision.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping

from .state_layers import ROLE_AT_CUTS, role_for_layer


VALIDATOR_IDS: tuple[str, ...] = (
    "commit_hash",
    "patch_and_file_digests",
    "lockfile_and_binary_digests",
    "model_profile_session",
    "stable_id",
    "ref_graph_reachability",
    "byte_digest",
    "none",
)


@dataclass(frozen=True)
class MaterializationModeSpec:
    layer: str
    role_at_cut: str
    mode: str
    validator: str
    conditional: bool = False

    def __post_init__(self) -> None:
        if self.validator not in VALIDATOR_IDS:
            raise ValueError(f"unknown validator {self.validator!r}")
        if self.role_at_cut not in ROLE_AT_CUTS:
            raise ValueError(f"unknown role_at_cut {self.role_at_cut!r}")


@dataclass(frozen=True)
class MaterializationValidationResult:
    valid: bool
    reasons: tuple[str, ...]
    checks_run: tuple[str, ...]


_REQUIRED_EVIDENCE: dict[str, tuple[str, ...]] = {
    "commit_hash": ("required_commit", "actual_commit"),
    "patch_and_file_digests": ("patch_applies", "file_digests_match"),
    "lockfile_and_binary_digests": ("lockfile_hash_match", "binary_digests_match"),
    "model_profile_session": ("model", "profile", "session_id", "cut_id"),
    "stable_id": ("stable_id_match",),
    "ref_graph_reachability": ("reachable",),
    "byte_digest": ("content_hash_match",),
    "none": (),
}


def _spec(layer: str, role_at_cut: str, mode: str, validator: str, *, conditional: bool = False) -> MaterializationModeSpec:
    return MaterializationModeSpec(layer, role_at_cut, mode, validator, conditional)


_REGISTRY: dict[tuple[str, str], tuple[MaterializationModeSpec, ...]] = {
    ("base_repo_checkout", "reconstructable"): (
        _spec("base_repo_checkout", "reconstructable", "clone-at-commit", "commit_hash"),
        _spec("base_repo_checkout", "reconstructable", "reuse-warm-clone", "commit_hash"),
    ),
    ("uncommitted_diff", "correctness_critical"): (
        _spec("uncommitted_diff", "correctness_critical", "transfer-diff", "patch_and_file_digests"),
        _spec("uncommitted_diff", "correctness_critical", "full-workspace", "byte_digest"),
    ),
    ("files_read", "reconstructable"): (
        _spec("files_read", "reconstructable", "clone-at-commit", "commit_hash"),
    ),
    ("files_touched", "correctness_critical"): (
        _spec("files_touched", "correctness_critical", "transfer-diff", "patch_and_file_digests"),
        _spec("files_touched", "correctness_critical", "full-workspace", "byte_digest"),
    ),
    ("tool_outputs", "correctness_critical"): (
        _spec("tool_outputs", "correctness_critical", "copy-bytes", "byte_digest"),
        _spec("tool_outputs", "correctness_critical", "rerun-cmd", "ref_graph_reachability", conditional=True),
    ),
    ("tool_outputs", "diagnostic"): (
        _spec("tool_outputs", "diagnostic", "copy-bytes", "byte_digest"),
        _spec("tool_outputs", "diagnostic", "discard", "none"),
    ),
    ("tool_outputs", "disposable"): (
        _spec("tool_outputs", "disposable", "discard", "none"),
    ),
    ("test_logs", "diagnostic"): (
        _spec("test_logs", "diagnostic", "copy-bytes", "byte_digest"),
        _spec("test_logs", "diagnostic", "discard", "none"),
    ),
    ("build_artifacts", "reconstructable"): (
        _spec("build_artifacts", "reconstructable", "rerun-build", "ref_graph_reachability", conditional=True),
        _spec("build_artifacts", "reconstructable", "copy-bytes", "byte_digest"),
    ),
    ("dependency_cache", "performance_critical"): (
        _spec("dependency_cache", "performance_critical", "transfer-bytes", "byte_digest"),
        _spec("dependency_cache", "performance_critical", "rerun-setup-cmd", "lockfile_and_binary_digests", conditional=True),
    ),
    ("retrieved_documents", "correctness_critical"): (
        _spec("retrieved_documents", "correctness_critical", "copy-bytes", "byte_digest"),
        _spec("retrieved_documents", "correctness_critical", "refetch-stable-uri", "stable_id", conditional=True),
    ),
    ("subagent_transcripts", "correctness_critical"): (
        _spec("subagent_transcripts", "correctness_critical", "copy-bytes", "byte_digest"),
    ),
    ("summaries_compaction", "correctness_critical"): (
        _spec("summaries_compaction", "correctness_critical", "copy-bytes", "byte_digest"),
    ),
    ("kv_cache", "performance_critical"): (
        _spec("kv_cache", "performance_critical", "transfer-kv", "model_profile_session"),
        _spec("kv_cache", "performance_critical", "replay-prompt", "model_profile_session"),
    ),
}


def materialization_registry() -> Mapping[tuple[str, str], tuple[MaterializationModeSpec, ...]]:
    return dict(_REGISTRY)


def lookup_materialization_modes(layer: str, role_at_cut: str) -> tuple[MaterializationModeSpec, ...]:
    if role_at_cut not in ROLE_AT_CUTS:
        raise ValueError(f"unknown role_at_cut {role_at_cut!r}")
    # `role_for_layer` makes unknown layers fail, while allowing documented
    # M1 special cases such as kv_cache below.
    if layer != "kv_cache":
        role_for_layer(layer)
    key = (layer, role_at_cut)
    if key not in _REGISTRY:
        raise ValueError(f"no materialization modes for layer={layer!r}, role_at_cut={role_at_cut!r}")
    return _REGISTRY[key]


def validate_materialization_mode(spec: MaterializationModeSpec, evidence: Mapping[str, object]) -> MaterializationValidationResult:
    checks = (spec.validator,)
    missing = tuple(k for k in _REQUIRED_EVIDENCE[spec.validator] if k not in evidence)
    reasons: list[str] = [f"missing_evidence:{k}" for k in missing]

    if spec.validator == "commit_hash" and not missing:
        if evidence["required_commit"] != evidence["actual_commit"]:
            reasons.append("commit_hash_mismatch")
    elif spec.validator == "patch_and_file_digests" and not missing:
        if not evidence["patch_applies"]:
            reasons.append("patch_does_not_apply")
        if not evidence["file_digests_match"]:
            reasons.append("file_digest_mismatch")
    elif spec.validator == "lockfile_and_binary_digests" and not missing:
        if not evidence["lockfile_hash_match"]:
            reasons.append("lockfile_hash_mismatch")
        if not evidence["binary_digests_match"]:
            reasons.append("binary_digest_mismatch")
    elif spec.validator == "stable_id" and not missing:
        if not evidence["stable_id_match"]:
            reasons.append("stable_id_mismatch")
    elif spec.validator == "ref_graph_reachability" and not missing:
        if not evidence["reachable"]:
            reasons.append("ref_graph_unreachable")
    elif spec.validator == "byte_digest" and not missing:
        if not evidence["content_hash_match"]:
            reasons.append("content_hash_mismatch")

    return MaterializationValidationResult(
        valid=not reasons,
        reasons=tuple(reasons),
        checks_run=checks,
    )
