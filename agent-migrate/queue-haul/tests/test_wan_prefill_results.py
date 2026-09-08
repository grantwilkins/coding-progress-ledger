import json

import numpy as np
import pytest

from plot_wan_prefill_results import pooled


def test_episode_ecdf_uses_last_completion_and_retains_incomplete_episodes():
    rows = []
    for times in ([10] * 8, [10] * 7 + [35], [10] * 7 + [None], [10] * 7):
        rows.append(dict(campaign="wan", policy="greedy", not_moved_count=8 - len(times),
                         decisions=json.dumps([
                             dict(session_id=str(i), action="replay", completion_s=t,
                                  error="failed" if t is None else None)
                             for i, t in enumerate(times)])))
    x, y, counts, total = pooled(rows, "wan", "greedy")
    np.testing.assert_array_equal(x, [0, 10, 35])
    np.testing.assert_array_equal(y, [0, 1 / 4, 2 / 4])
    assert total == 32 and counts == dict(replay=31, not_selected=1)
    with pytest.raises(ValueError, match="missing episodes"):
        pooled(rows, "prefill", "greedy")
