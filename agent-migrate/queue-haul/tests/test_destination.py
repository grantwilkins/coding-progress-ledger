"""
Claim:
Destination profiles convert context-conditioned demand and admit only within nested,
measured compatibility and migration domains.

Plausible wrong implementations:
- Use a single service rate for every context length.
- Check raw token mix instead of normalized prefill/decode work.
- Accept emergency capacity smaller than normal or stable capacity smaller than emergency.
- Permit replay across a tokenizer/log mismatch or KV transfer across an ABI mismatch.
- Treat a changed runtime baseline as load-induced slowdown.
- Interpolate one slowdown at the initial load instead of taking the worst measured value
  through the selected envelope boundary.
"""

from dataclasses import replace

import pytest

from destination import (CompatibilityFingerprint, ContextRate, DestinationType,
                         LoadedCoefficients, MigrationComponents)


def fingerprint(**changes):
    return CompatibilityFingerprint(**{
        "model": "m", "tokenizer": "t", "durable_log": "l", "kv_abi": "k",
        **changes,
    })


def loaded():
    return LoadedCoefficients((0, .5, 1), (1, 2, 1.5), (10, 100), (5, 20), "run")


def destination_type():
    return DestinationType(
        "q", fingerprint(), ContextRate((10, 100), (100, 50)),
        ContextRate((10, 100), (50, 25)), ((1, 1),),
        {"normal": (1,), "emergency": (1.5,), "stable": (2,)}, 1000,
        {"replay": loaded(), "kv_transfer": loaded()}, (0, 1), "run",
    )


def test_context_conditioned_service_work_matches_hand_calculation():
    q = destination_type()

    assert q.work(50, 25, 10).tolist() == [.5, .5]
    assert q.work(25, 12.5, 100).tolist() == [.5, .5]
    with pytest.raises(ValueError, match="outside measured range"):
        q.work(1, 1, 101)


def test_workload_direction_uses_normalized_service_work():
    q = replace(destination_type(), workload_prefill_fraction_range=(.49, .51))

    assert q.work(100, 50, 10).tolist() == [1, 1]
    with pytest.raises(ValueError, match="workload direction"):
        q.work(50, 50, 10)


@pytest.mark.parametrize("bounds", [
    {"normal": (1.1,), "emergency": (1,), "stable": (2,)},
    {"normal": (1,), "emergency": (2.1,), "stable": (2,)},
])
def test_destination_envelopes_must_be_nested(bounds):
    with pytest.raises(ValueError, match="nonnested"):
        replace(destination_type(), bounds=bounds)


def test_method_compatibility_has_the_required_boundary():
    source = fingerprint()

    assert fingerprint().supports(source, "replay")
    assert fingerprint(kv_abi="other").supports(source, "replay")
    assert not fingerprint(kv_abi="other").supports(source, "kv_transfer")
    assert not fingerprint(tokenizer="other").supports(source, "replay")


def test_loaded_coefficients_take_worst_slowdown_over_interval():
    curve = replace(loaded(), baseline_factor=.5)

    assert curve.worst(.25, .75, 50, 10) == 1
    with pytest.raises(ValueError, match="outside loaded-profile"):
        curve.worst(.25, .75, 101, 10)


def test_migration_evidence_marks_each_out_of_domain_quantity():
    components = MigrationComponents((16, 24), (5, 10), "hand", .5, 1)

    assert components.extrapolates(20, 7) == ()
    assert components.extrapolates(8, 20) == ("context", "bandwidth")
