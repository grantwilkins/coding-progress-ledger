from observation_channel.models import Turn
from observation_channel.path_tracker import first_write_target


def test_first_write_target_uses_structured_path() -> None:
    turn = Turn(step=1, kind="action", tool="write_file", arguments={"path": "src/app.py"})
    assert first_write_target(turn) == "src/app.py"


def test_first_write_target_uses_first_redirect_only() -> None:
    turn = Turn(step=1, kind="action", tool="bash", command="echo one > a.py; echo two > b.py")
    assert first_write_target(turn) == "a.py"


def test_no_write_target_abstains() -> None:
    turn = Turn(step=1, kind="action", tool="bash", command="python script.py")
    assert first_write_target(turn) is None
