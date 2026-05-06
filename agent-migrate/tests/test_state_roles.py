"""
Claim:
S3 adds a role_at_cut axis orthogonal to S1 mobility classes. Package and
byte-audit logic must decide what is correctness-critical, performance-critical,
reconstructable, diagnostic, disposable, or an external side effect at the cut,
not merely reuse the layer's origin/mobility label.

Plausible wrong implementations:
- use S1 mobility_class as role_at_cut, making dependency_cache look like a
  correctness payload or uncommitted_diff look lazily rehydratable;
- assign roles only to coarse manifest layers and leave S1 filesystem layers
  unmapped;
- aggregate role bytes at the wrong level so role totals do not conserve layer
  totals;
- treat diagnostic/disposable bytes as bytes that must materialize before
  resume;
- allow an unknown layer or role to silently fall through to a default.
"""

import pytest

from agent_migrate_agent.state_layers import (
    ROLE_AT_CUTS,
    S1_LAYERS,
    audit_workflow_directory,
    materialization_for_role,
    role_for_layer,
    write_audit_artifacts,
)
from test_state_layers import _build_synthetic_workflow_dir


def test_state_roles_canonical_assignments():
    assert role_for_layer("uncommitted_diff") == "correctness_critical"
    assert role_for_layer("files_touched") == "correctness_critical"
    assert role_for_layer("base_repo_checkout") == "reconstructable"
    assert role_for_layer("dependency_cache") == "performance_critical"
    assert role_for_layer("test_logs") == "diagnostic"
    assert role_for_layer("model_execution") == "performance_critical"


def test_every_s1_layer_has_a_role():
    roles = {role_for_layer(layer.name) for layer in S1_LAYERS}
    assert roles <= set(ROLE_AT_CUTS)
    assert {layer.name for layer in S1_LAYERS} == {
        layer.name for layer in S1_LAYERS if role_for_layer(layer.name)
    }


def test_materialization_for_role_is_role_not_mobility_class():
    assert materialization_for_role("correctness_critical") == "included"
    assert materialization_for_role("external_side_effect") == "included"
    assert materialization_for_role("performance_critical") == "lazy_rehydrate"
    assert materialization_for_role("reconstructable") == "lazy_rehydrate"
    assert materialization_for_role("diagnostic") == "skipped"
    assert materialization_for_role("disposable") == "skipped"


@pytest.mark.parametrize("bad", ["", "not_a_layer"])
def test_unknown_layer_role_hard_fails(bad):
    with pytest.raises(ValueError):
        role_for_layer(bad)


def test_role_projection_conserves_total_bytes(tmp_path):
    _build_synthetic_workflow_dir(tmp_path)
    report = audit_workflow_directory(tmp_path)
    assert sum(report.bytes_per_role_at_cut.values()) == report.total_bytes
    assert (
        report.bytes_must_materialize_before_resume
        + report.bytes_can_lazy_rehydrate
        + report.bytes_can_drop
        == report.total_bytes
    )


def test_role_bytes_sum_to_layer_bytes(tmp_path):
    _build_synthetic_workflow_dir(tmp_path)
    report = audit_workflow_directory(tmp_path)
    for layer in S1_LAYERS:
        layer_role_total = sum(
            n for (layer_name, _role), n in report.bytes_per_layer_role.items()
            if layer_name == layer.name
        )
        assert layer_role_total == report.bytes_per_layer[layer.name]


def test_write_audit_artifacts_includes_role_axis(tmp_path):
    _build_synthetic_workflow_dir(tmp_path)
    report = audit_workflow_directory(tmp_path)
    out = tmp_path / "out"
    write_audit_artifacts(report, out)
    assert (out / "audit_roles.csv").exists()
    assert (out / "audit_layer_roles.csv").exists()
    assert "bytes_must_materialize_before_resume" in (out / "audit.json").read_text()
