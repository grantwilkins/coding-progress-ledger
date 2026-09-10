import pytest

import json
from pathlib import Path

from quick_action_mix import calculate, schedule


def test_shared_link_can_leave_sessions_in_place():
    transfer = {'chunk_tokens': 1, 'object_groups': [{'sw_size_chunks': -1, 'chunk_bytes': 1}]}
    rows = schedule([9, 9], [0, 1], [(1, .01), (20, .01)], transfer, {'sink': .000008}, 10)
    assert [r['action'] for r in rows] == ['kv_transfer', 'not_moved']
    assert rows[0]['finish_s'] == 8


def test_replay_shares_compute_and_does_not_consume_the_other_link():
    transfer = {'chunk_tokens': 1, 'object_groups': [{'sw_size_chunks': -1, 'chunk_bytes': 1000000000}]}
    rows = schedule([10, 10, 10], [0, 1, 2], [(1, 10), (20, 10)], transfer,
                    {'east': 8000, 'germany': 8000}, 1.5)
    assert [r['action'] for r in rows] == ['replay', 'replay', 'not_moved']
    assert {r['destination'] for r in rows[:2]} == {'east', 'germany'}
    with pytest.raises(ValueError, match='support'):
        schedule([30], [0], [(1, 10), (20, 10)], transfer, {'east': 8000}, 30)


def test_frozen_inputs_reproduce_every_saved_action_and_summary():
    path = Path(__file__).resolve().parents[1] / 'outputs/quick-action-mix-20260910/report.json'
    original = path.read_text()
    assert calculate(json.loads(original)) == json.loads(original)
