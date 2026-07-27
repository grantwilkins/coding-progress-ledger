"""
Claim:
Evaluation operating points carry units and provenance, and assumed values
cannot be promoted to accepted evidence.

Plausible wrong implementations:
- Accept an assumed value as measured evidence.
- Permit a dimensionless route or deadline value.
- Omit the evidence needed to replace an assumption.
"""

import pytest

from evaluation_config import DEADLINES_S, EVALUATION_GRID, EvaluationValue


def test_assumption_cannot_support_accepted_evidence():
    with pytest.raises(ValueError, match="assumed"):
        DEADLINES_S.record("accepted")


@pytest.mark.parametrize("field", ("unit", "valid_range", "replacement_evidence"))
def test_provenance_fields_are_required(field):
    values = {
        "value": 1,
        "unit": "s",
        "provenance": "measured",
        "valid_range": "1",
        "replacement_evidence": "clock",
    }
    values[field] = ""
    with pytest.raises(ValueError, match="units, provenance, range, and evidence"):
        EvaluationValue(**values)


def test_every_canonical_assumption_is_visible_as_sensitivity():
    records = {name: value.record() for name, value in EVALUATION_GRID.items()}
    assert all(row["evidence_status"] == "sensitivity" for row in records.values())
    assert {row["provenance"] for row in records.values()} <= {
        "measured", "fitted", "assumed", "simulated",
    }
