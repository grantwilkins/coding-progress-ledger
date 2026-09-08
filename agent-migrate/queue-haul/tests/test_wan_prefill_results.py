import json

import numpy as np
import pytest

from plot_wan_prefill_results import pooled


def test_completion_pool_retains_late_failed_and_unselected_sessions():
    row = dict(campaign="wan", policy="greedy", not_moved_count="5", decisions=json.dumps([
        dict(session_id="a", action="replay", completion_s=10, error=None),
        dict(session_id="b", action="kv_transfer", completion_s=35, error=None),
        dict(session_id="c", action="replay", completion_s=None, error="failed"),
    ]))
    x, y, counts, total = pooled([row], "wan", "greedy")
    np.testing.assert_array_equal(x, [0, 10, 35])
    np.testing.assert_array_equal(y, [0, 1 / 8, 2 / 8])
    assert total == 8 and counts == dict(replay=2, kv_transfer=1, not_selected=5)
    with pytest.raises(ValueError, match="missing episodes"):
        pooled([row], "prefill", "greedy")
