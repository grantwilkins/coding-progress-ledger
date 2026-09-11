from types import SimpleNamespace

import pytest

from pool_shed_bandwidth_sweep import BASELINE, wave_requests
from pool_shed_resident_queue import simulate


def wave_case(action="replay"):
    fleet = SimpleNamespace(count=[1], metadata={"turn_offset": [0], "source_session_rps": .1,
        "source_phase_s": [1.], "sequence_cycle": True,
        "turn_sequences": [[{"context": 1024, "prompt": 16, "output": 4, "reset": True},
                            {"context": 1044, "prompt": 12, "output": 2, "reset": False}]]})
    wave = dict(counts=[1], action=action, phase_enter_s=[0., 20., 21., 22., 23., 25., 26., None],
                origin_context=[1024], quiesced_context=[1044], reset=[False], paused_turns=[1],
                pause_requested_s=21.)
    return fleet, wave


def test_offered_source_backlog_keeps_arrival_and_waits_for_migration():
    fleet, wave = wave_case()
    rows, gates = wave_requests(fleet, wave, 100., 256)
    first = next(r for r in rows if r["cohort"] == "incoming")
    assert first["arrival_s"] == BASELINE + 9
    assert first["release_s"] == BASELINE + 26
    assert first["cached_tokens"] == 1040
    assert gates == {"incoming-0-0-initial": {"end_s": BASELINE + 21},
                     "incoming-0-0-catchup": {"end_s": BASELINE + 26, "server_end_s": BASELINE + 25}}
    c = dict(prefill_step_s=.1, prefill_token_s=.001, prefill_attention_s=0.,
             decode_step_s=.01, decode_attention_s=0., endpoint_s=0.)
    outcomes = {r["request_id"]: r for r in simulate(rows, c, 100.)["requests"]}
    assert outcomes[first["request_id"]]["ttft_s"] > 17
    assert outcomes[first["request_id"]]["admitted_s"] >= outcomes["incoming-0-0-catchup"]["end_s"]


def test_replay_catchup_is_full_context_and_kv_only_replays_unsealed_tail():
    fleet, wave = wave_case()
    replay, _ = wave_requests(fleet, wave, 100., 256)
    assert [(r["prompt_tokens"], r["cached_tokens"], r["output_tokens"])
            for r in replay if r["cohort"] == "migration"] == [(1024, 0, 1), (1044, 0, 1)]
    wave["action"] = "kv_transfer"
    kv, _ = wave_requests(fleet, wave, 100., 256)
    assert [(r["prompt_tokens"], r["cached_tokens"]) for r in kv if r["cohort"] == "migration"] == [(1044, 1024)]


def test_uncommitted_wave_retains_blocked_offers_and_wrap_clears_cache():
    fleet, wave = wave_case()
    wave["phase_enter_s"][6] = None
    rows, _ = wave_requests(fleet, wave, 100., 256)
    incoming = [r for r in rows if r["cohort"] == "incoming"]
    assert incoming and all(r["release_s"] == 101. for r in incoming)
    assert incoming[1]["recorded_turn"] == 0 and incoming[1]["cached_tokens"] == 0
    wave["counts"] = [.5]
    with pytest.raises(ValueError, match="integer"):
        wave_requests(fleet, wave, 100., 256)


def test_unchanged_replay_has_no_duplicate_catchup_prefill():
    fleet, wave = wave_case()
    wave["origin_context"] = [1044]
    rows, _ = wave_requests(fleet, wave, 100., 256)
    assert sum(r["cohort"] == "migration" for r in rows) == 1
    wave["reset"] = [True]
    rows, _ = wave_requests(fleet, wave, 100., 256)
    assert sum(r["cohort"] == "migration" for r in rows) == 2
