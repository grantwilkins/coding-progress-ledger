"""
Claim:
M1's materialization-mode registry is a deterministic function of
(state_layer, role_at_cut), and each mode names a structural validator with
explicit evidence requirements. Conditional modes are not treated as proven
semantic equivalence by default.

Plausible wrong implementations:
- key the registry by layer only, so correctness-critical and diagnostic
  tool_outputs get the same modes;
- let free-form validator labels enter the registry, so C3-style validation
  cannot reason about them;
- accept discard for correctness-critical state consumed after the cut;
- validate KV transfer using state_id/content_hash only and ignore model,
  profile, session, or cut compatibility;
- accept warm clone reuse without checking the required commit hash.
"""

import pytest

from agent_migrate_agent.materialization_modes import (
    VALIDATOR_IDS,
    lookup_materialization_modes,
    materialization_registry,
    validate_materialization_mode,
)
from agent_migrate_agent.state_layers import S1_LAYERS


def _mode_names(layer: str, role: str) -> set[str]:
    return {m.mode for m in lookup_materialization_modes(layer, role)}


def test_registry_is_keyed_by_layer_and_role():
    critical = _mode_names("tool_outputs", "correctness_critical")
    diagnostic = _mode_names("tool_outputs", "diagnostic")
    assert "copy-bytes" in critical
    assert "discard" not in critical
    assert "discard" in diagnostic
    assert critical != diagnostic


def test_lookup_hard_fails_for_unknown_key_parts():
    with pytest.raises(ValueError):
        lookup_materialization_modes("not_a_layer", "correctness_critical")
    with pytest.raises(ValueError):
        lookup_materialization_modes("tool_outputs", "not_a_role")
    with pytest.raises(ValueError):
        lookup_materialization_modes("dependency_cache", "correctness_critical")


def test_every_s1_layer_has_at_least_one_registry_entry_for_its_canonical_role():
    registry = materialization_registry()
    covered = {layer for layer, _role in registry}
    assert {layer.name for layer in S1_LAYERS} <= covered


def test_registry_modes_are_deterministic_and_validator_ids_are_known():
    registry = materialization_registry()
    assert registry == materialization_registry()
    for specs in registry.values():
        names = [spec.mode for spec in specs]
        assert names == list(dict.fromkeys(names))
        assert {spec.validator for spec in specs} <= set(VALIDATOR_IDS)


def test_kv_cache_requires_model_profile_session_evidence():
    transfer = next(m for m in lookup_materialization_modes("kv_cache", "performance_critical") if m.mode == "transfer-kv")
    missing = validate_materialization_mode(transfer, {"model": "m"})
    assert not missing.valid
    assert "missing_evidence:profile" in missing.reasons
    ok = validate_materialization_mode(transfer, {
        "model": "m", "profile": "p", "session_id": "s", "cut_id": "c",
    })
    assert ok.valid


def test_warm_clone_reuse_checks_commit_hash():
    reuse = next(m for m in lookup_materialization_modes("base_repo_checkout", "reconstructable") if m.mode == "reuse-warm-clone")
    bad = validate_materialization_mode(reuse, {
        "required_commit": "abc", "actual_commit": "def",
    })
    assert not bad.valid
    assert "commit_hash_mismatch" in bad.reasons
    good = validate_materialization_mode(reuse, {
        "required_commit": "abc", "actual_commit": "abc",
    })
    assert good.valid
