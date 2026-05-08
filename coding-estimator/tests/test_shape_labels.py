"""Shape labels (Workstream E6).

Claim:
    `shape_rows_for_source(source_id)` returns one row per
    (source, run_id) covering every run with a summary_by_category.json
    side file. Each row has:
      - identity: run_id, source
      - upstream pass-throughs: final_coding_progress, final_success,
        final_success_source, clean_success
      - one boolean column `shape_<tag>` for every tag in SHAPE_TAGS
    The boolean-tag values must agree EXACTLY with the upstream
    `label_run(run_dir)` snapshot's `tags` set.

Plausible wrong implementations:
    - drop a SHAPE_TAGS entry -> downstream slicing is silently
      missing a slice axis.
    - hard-code a tag list duplicated from upstream that drifts when
      SHAPE_TAGS gains an entry.
    - emit shape_<tag> = True for tags not in `rl.tags` but for which
      `rl.tags` is non-empty (set-membership inverted) — this is a
      classic copy-paste boolean inversion.
    - skip runs that DO have summary_by_category.json because of an
      unrelated missing file (e.g. requiring run_notes.md).
"""

from __future__ import annotations

from coding_estimator.ingest import paths
from coding_estimator.labels._upstream_shapes_snapshot import SHAPE_TAGS, label_run
from coding_estimator.labels.shapes import shape_rows_for_source


def test_emits_one_row_per_run_with_summary() -> None:
    """For a real source, the row count must equal the count of runs
    whose dir contains `summary_by_category.json`. A bug that silently
    skips eligible runs would land here as a count mismatch."""
    rows = shape_rows_for_source("swe_agent_pilot")
    eligible = sum(
        1
        for rid in paths.list_run_ids("swe_agent_pilot")
        if (paths.run_dir("swe_agent_pilot", rid) / "summary_by_category.json").is_file()
    )
    assert len(rows) == eligible
    assert eligible > 0


def test_every_row_has_one_boolean_column_per_shape_tag() -> None:
    """Tag-column coverage. A regression that drops a tag (e.g. if the
    upstream SHAPE_TAGS gains an entry) lands here."""
    rows = shape_rows_for_source("swe_agent_pilot")
    expected_cols = {f"shape_{t}" for t in SHAPE_TAGS}
    for row in rows:
        cols = {k for k in row if k.startswith("shape_")}
        assert cols == expected_cols, (row["run_id"], cols ^ expected_cols)
        for tag in SHAPE_TAGS:
            assert isinstance(row[f"shape_{tag}"], bool), (row["run_id"], tag)


def test_shape_columns_match_upstream_label_run_for_every_run() -> None:
    """Parity invariant: for every row, the True-set of `shape_<tag>`
    columns must equal `label_run(run_dir).tags`. This catches:
      - boolean inversion (tag in tags vs tag not in tags)
      - off-by-one when iterating SHAPE_TAGS
      - shadowing of the upstream snapshot by a re-implementation
    """
    rows = shape_rows_for_source("swe_agent_pilot")
    for row in rows:
        rd = paths.run_dir("swe_agent_pilot", row["run_id"])
        upstream = label_run(rd)
        ours_true = {t for t in SHAPE_TAGS if row[f"shape_{t}"]}
        assert ours_true == upstream.tags, (row["run_id"], ours_true ^ upstream.tags)


def test_clean_success_matches_upstream() -> None:
    """`clean_success` is a derived boolean defined by upstream as
    `final_success is True AND not (low_progress_success or
    submit_without_validation or no_validation_frontier or
    high_progress_failure)`. A regression that miscomputes the OR of
    the four exclusion tags lands here."""
    rows = shape_rows_for_source("swe_agent_pilot")
    seen_clean = False
    for row in rows:
        rd = paths.run_dir("swe_agent_pilot", row["run_id"])
        upstream = label_run(rd)
        assert row["clean_success"] is upstream.clean_success, row["run_id"]
        seen_clean = seen_clean or row["clean_success"]
    # Sanity: at least ONE pilot run must be a clean success — otherwise
    # the test exercises only the False branch.
    assert seen_clean, "no clean_success runs found in swe_agent_pilot"


def test_final_success_passthrough_matches_upstream() -> None:
    """`final_success` and `final_success_source` are pass-throughs of
    `upstream.resolve_final_success`. Any divergence (e.g. a different
    source-resolution path) means the shape table no longer agrees with
    upstream's published *_shape_labels.csv."""
    rows = shape_rows_for_source("swe_agent_pilot")
    for row in rows:
        rd = paths.run_dir("swe_agent_pilot", row["run_id"])
        upstream = label_run(rd)
        assert row["final_success"] is upstream.final_success, row["run_id"]
        assert row["final_success_source"] == upstream.final_success_source
        assert row["final_coding_progress"] == upstream.final_coding_progress
