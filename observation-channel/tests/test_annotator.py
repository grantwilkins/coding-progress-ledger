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
