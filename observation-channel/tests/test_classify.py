"""
Claim:
Bash and tool actions are mapped to the intended work-unit category using the
specified priority order.

Plausible wrong implementations:
- Classify write-and-run commands as validation because script execution appears first.
- Miss validation commands prefixed by `cd ... &&`.
- Treat submit markers as ordinary shell commands.
"""

from observation_channel.classify import classify_bash, classify_turn
from observation_channel.models import Category, Turn


def test_classifier_priority_state_change_outranks_script_execution() -> None:
    assert classify_bash("python - <<'PY'\nPath('x.py').write_text('x')\nPY") == Category.PRODUCT
    assert classify_bash("python script.py") == Category.VALIDATION


def test_classifier_priority_install_test_read_only() -> None:
    assert classify_bash("pip install pytest") == Category.ENVIRONMENT
    assert classify_bash("pytest tests") == Category.VALIDATION
    assert classify_bash('cd /app && R --slave -e "source(\'ars.R\'); test()"') == Category.VALIDATION
    assert classify_bash("rg TODO src") == Category.INVESTIGATION


def test_static_tool_maps_and_artifact_marker() -> None:
    assert classify_turn(Turn(step=1, kind="action", tool="create", command="create foo.py")) == Category.PRODUCT
    assert classify_turn(Turn(step=2, kind="action", tool="read_file", arguments={"path": "foo.py"})) == Category.INVESTIGATION
    assert classify_turn(Turn(step=3, kind="action", tool="bash", command="submit")) == Category.ARTIFACT
