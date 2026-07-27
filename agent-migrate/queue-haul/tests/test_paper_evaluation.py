"""
Claim:
The paper registry covers Q1-Q9, gives every plot a unique output, and refuses
tables that hide provenance or promote assumed inputs to accepted evidence.

Plausible wrong implementations:
- Omit one paper question from the executable pipeline.
- Overwrite two figures through a duplicate filename.
- Accept a table without the physical fields named by its plot.
- Label an assumed sweep as accepted measurement evidence.
"""

import json

import pytest

from paper_evaluation import PLOTS, validate_rows, write


def test_registry_covers_every_paper_question_once_or_more():
    assert {spec.question for spec in PLOTS} == {f"Q{i}" for i in range(1, 10)}
    assert len({spec.filename for spec in PLOTS}) == len(PLOTS)


def test_plot_rows_require_physical_fields_and_provenance():
    with pytest.raises(ValueError, match="lacks required fields"):
        validate_rows(PLOTS[0], [{"predicted_shed_w": 1}])


def test_assumed_rows_cannot_be_accepted():
    spec = PLOTS[0]
    row = {field: 1 for field in spec.required_fields}
    row.update(input_provenance="measured|assumed",
               result_provenance="simulated", evidence_status="accepted")
    with pytest.raises(ValueError, match="assumed"):
        validate_rows(spec, [row])


def test_manifest_writes_the_canonical_grid_and_plot_registry(tmp_path):
    write(tmp_path)
    manifest = json.loads((tmp_path / "evaluation-manifest.json").read_text())

    assert set(manifest) == {"grid", "plots"}
    assert manifest["grid"]["deadlines"]["provenance"] == "assumed"
    assert len(manifest["plots"]) == len(PLOTS)
    assert (tmp_path / "plot-specs.csv").exists()
