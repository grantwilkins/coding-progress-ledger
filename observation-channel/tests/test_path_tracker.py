"""
Claim:
Path tracking extracts only real filesystem write targets so product units split
on file/path changes rather than on editor line ranges or shell flags, and
distinguishes scratch paths from source paths for v1.6 prefix features.

Plausible wrong implementations:
- Treat SWE-Agent edit line ranges like `1:4` as filenames.
- Treat command flags like `mkdir -p` as write targets.
- Track the source operand of `cp`/`mv` instead of the destination.
- Stop detecting ordinary redirects after tightening shell parsing.
- Miss patch or sed in-place targets used by common edit tools.
- Treat root scratch scripts as source modifications.
"""

from observation_channel.models import Turn
from observation_channel.path_tracker import first_write_target, is_source_path


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


def test_patch_diff_header_and_sed_in_place_targets_are_extracted() -> None:
    patch = "--- a/src/app.py\n+++ b/src/app.py\n@@\n-old\n+new\n"

    assert first_write_target(Turn(step=1, kind="action", tool="patch", command=patch)) == "src/app.py"
    assert first_write_target(Turn(step=2, kind="action", tool="bash", command="sed -i 's/a/b/' /app/src/app.py")) == "/app/src/app.py"


def test_source_path_heuristic_separates_scratch_from_source() -> None:
    assert is_source_path("/app/src/app.py") is True
    assert is_source_path("tests/test_app.py") is True
    assert is_source_path("reproduce_failure.py") is False
    assert is_source_path("test_tmp.py") is False
    assert is_source_path("/app/scratch.py") is False
