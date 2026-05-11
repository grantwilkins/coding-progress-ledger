"""
Claim:
The online annotator opens and closes deterministic work units while marking a
stuck episode only when three identical observations carry meaningful repeated
failure evidence, and emits prefix-local v1.6 empirical-Bayes features.

Plausible wrong implementations:
- Treat empty tool acknowledgements as stuck evidence.
- Treat short success acknowledgements as stuck evidence.
- Suppress genuine repeated error messages because they are short.
- Suppress genuine repeated long observations because they do not contain an
  error keyword.
- Pad the recent-error denominator to 10 instead of using available observations.
- Let scratch-file writes flip the source-modification flag.
- Count observation turns in the investigation-overhead denominator.
"""

from observation_channel import Annotator, Category, Turn


def test_rows_are_emitted_for_all_turns_and_units_close_on_boundary() -> None:
    annotator = Annotator(instance_id="case")
    rows = [
        annotator.feed(Turn(step=1, kind="system", response="task")),
        annotator.feed(Turn(step=2, kind="action", tool="ls", command="ls")),
        annotator.feed(Turn(step=3, kind="observation", response="files")),
        annotator.feed(Turn(step=4, kind="action", tool="bash", command="cat <<'EOF' > a.py\nprint(1)\nEOF")),
        annotator.feed(Turn(step=5, kind="observation", response="")),
        annotator.feed(Turn(step=6, kind="action", tool="pytest", command="pytest")),
    ]
    summary = annotator.finalize()

    assert len(rows) == 6
    assert rows[0].total == 0
    assert rows[2].current_category == Category.INVESTIGATION.value
    assert rows[4].current_category == Category.PRODUCT.value
    assert rows[5].done == 2
    assert summary.final_total == 3
    assert summary.final_done == 3


def test_stuck_episode_is_done_on_finalize() -> None:
    annotator = Annotator(instance_id="stuck")
    annotator.feed(Turn(step=1, kind="action", tool="bash", command="pytest"))
    annotator.feed(Turn(step=2, kind="observation", response="same error"))
    annotator.feed(Turn(step=3, kind="observation", response="same error"))
    row = annotator.feed(Turn(step=4, kind="observation", response="same error"))
    summary = annotator.finalize()

    assert row.done == 0
    assert summary.had_stuck_episode is True
    assert summary.final_total == 1
    assert summary.final_done == 1


def test_empty_and_ack_observations_are_not_stuck_evidence() -> None:
    annotator = Annotator(instance_id="acks")
    annotator.feed(Turn(step=1, kind="action", tool="bash", command="pytest"))
    for step, response in enumerate(["", "", "", '{"success": true}', '{"success": true}', '{"success": true}'], start=2):
        annotator.feed(Turn(step=step, kind="observation", response=response))

    assert annotator.finalize().had_stuck_episode is False


def test_repeated_long_observation_is_stuck_without_error_keyword() -> None:
    annotator = Annotator(instance_id="long")
    repeated_body = "same diagnostic output " * 5
    annotator.feed(Turn(step=1, kind="action", tool="bash", command="pytest"))
    for step in range(2, 5):
        annotator.feed(Turn(step=step, kind="observation", response=repeated_body))

    assert annotator.finalize().had_stuck_episode is True


def test_product_target_change_splits_units_but_missing_target_abstains() -> None:
    annotator = Annotator()
    rows = [
        annotator.feed(Turn(step=1, kind="action", tool="bash", command="cat <<'EOF' > a.py\nx=1\nEOF")),
        annotator.feed(Turn(step=2, kind="action", tool="bash", command="python -c \"open('unknown','w')\"")),
        annotator.feed(Turn(step=3, kind="action", tool="bash", command="cat <<'EOF' > b.py\nx=2\nEOF")),
    ]

    assert rows[0].total == 1
    assert rows[1].total == 1
    assert rows[2].total == 2


def test_recent_error_bucket_uses_available_observations_in_step_window() -> None:
    annotator = Annotator()

    annotator.feed(Turn(step=1, kind="observation", response="ok"))
    row = annotator.feed(Turn(step=3, kind="observation", response="<returncode>1</returncode>"))

    assert row.recent_error_bucket == "moderate"


def test_recent_error_bucket_detects_reader_normalized_returncode() -> None:
    row = Annotator().feed(Turn(step=1, kind="observation", response="returncode=2\nfailed"))

    assert row.recent_error_bucket == "heavy"


def test_recent_error_bucket_detects_json_traceback_exception_and_error_markers() -> None:
    annotator = Annotator()
    rows = [
        annotator.feed(Turn(step=1, kind="observation", response='{"success": false}')),
        annotator.feed(Turn(step=2, kind="observation", response="Traceback (most recent call last):")),
        annotator.feed(Turn(step=3, kind="observation", response="Unhandled Exception")),
        annotator.feed(Turn(step=4, kind="observation", response="Error: bad")),
    ]

    assert [row.recent_error_bucket for row in rows] == ["heavy", "heavy", "heavy", "heavy"]


def test_touched_source_is_sticky_and_ignores_scratch_paths() -> None:
    annotator = Annotator()
    rows = [
        annotator.feed(Turn(step=1, kind="action", tool="bash", command="cat <<'EOF' > reproduce_case.py\nx=1\nEOF")),
        annotator.feed(Turn(step=2, kind="action", tool="bash", command="cat <<'EOF' > src/app.py\nx=2\nEOF")),
        annotator.feed(Turn(step=3, kind="action", tool="bash", command="cat <<'EOF' > test_tmp.py\nx=3\nEOF")),
    ]

    assert [row.touched_source for row in rows] == [False, True, True]


def test_investigation_ratio_uses_action_denominator_only() -> None:
    annotator = Annotator()
    rows = [
        annotator.feed(Turn(step=1, kind="action", tool="ls", command="ls")),
        annotator.feed(Turn(step=2, kind="observation", response="files")),
        annotator.feed(Turn(step=3, kind="action", tool="bash", command="cat <<'EOF' > src/app.py\nx=1\nEOF")),
    ]

    assert rows[0].investigation_ratio_bucket == "dominant"
    assert rows[1].investigation_ratio_bucket == "dominant"
    assert rows[2].investigation_ratio_bucket == "moderate"
