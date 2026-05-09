"""
Claim:
Path tracking extracts only real filesystem write targets so product units split
on file/path changes rather than on editor line ranges or shell flags.

Plausible wrong implementations:
- Treat SWE-Agent edit line ranges like `1:4` as filenames.
- Treat command flags like `mkdir -p` as write targets.
- Track the source operand of `cp`/`mv` instead of the destination.
- Stop detecting ordinary redirects after tightening shell parsing.
"""

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


def test_swe_agent_edit_line_range_is_not_a_path() -> None:
    turn = Turn(step=1, kind="action", tool="edit", command="edit 1:4 value > threshold")
    assert first_write_target(turn) is None


def test_shell_file_ops_skip_flags_and_use_destination_for_copy() -> None:
    assert first_write_target(Turn(step=1, kind="action", tool="bash", command="mkdir -p /app/src")) == "/app/src"
    assert first_write_target(Turn(step=2, kind="action", tool="bash", command="cp -R src backup")) == "backup"
