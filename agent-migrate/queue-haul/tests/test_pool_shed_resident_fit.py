import json

import numpy as np
import pytest

from pool_shed_resident_fit import calibrate, prefill_seconds


def evidence(tmp_path):
    coefficients = dict(prefill_step_s=.04, prefill_token_s=.00004, prefill_attention_s=2.6e-9)
    unloaded = []
    for i, (context, q) in enumerate(((2048, 2048), (8192, 8192), (30000, 30000), (30032, 64))):
        unloaded.append(dict(seed=7101, width=1, phase_valid=True, episode=str(i), phase='initial',
            prompt_counts=json.dumps([context]), request_prefill_kv_computed_tokens_sum=q,
            request_prefill_time_seconds_sum=prefill_seconds(context, q, coefficients)))
    path = tmp_path / 'unloaded-analysis.json'
    path.write_text(json.dumps({'rows': unloaded}))
    path.with_name('runtime-launch.json').write_text(json.dumps({'cfg': {'max_num_batched_tokens': '8192'}}))
    rows = []
    for i, (context, q, output) in enumerate(((2000, 64, 8), (8000, 200, 12), (20000, 512, 16), (30000, 3000, 32))):
        ttft = .12 + prefill_seconds(context, q, coefficients)
        tpot = .025 + (context + output / 2) * 1e-8
        rows.append(dict(row_id=str(i), episode='control', seed=7101, arm='control', phase='service',
            serving_role='destination', done=True, status=200, completed_within_observation=True,
            exact_token_timestamps=True, start_s=i * 10., first_s=i * 10. + ttft,
            end_s=i * 10. + ttft + (output - 1) * tpot, output_tokens=output,
            prompt_tokens=context, cached_tokens=context - q, ttft_s=ttft, mean_tpot_s=tpot))
    return path, rows


def test_training_ignores_heldout_outcomes_and_unknown_cache(tmp_path):
    path, rows = evidence(tmp_path)
    result = calibrate(rows, request_sha256='0' * 64, unloaded_path=path)
    heldout = [{**r, 'episode': 'heldout', 'seed': 7102, 'mean_tpot_s': 1000., 'ttft_s': 1000.} for r in rows]
    unloaded = json.loads(path.read_text())
    unloaded['rows'] += [{**r, 'seed': 7102, 'request_prefill_time_seconds_sum': 1000.} for r in unloaded['rows']]
    path.write_text(json.dumps(unloaded))
    unknown = {**rows[0], 'row_id': 'unknown', 'episode': 'unknown', 'cached_tokens': None, 'output_tokens': 1, 'ttft_s': 1000.}
    repeated = calibrate(rows + heldout + [unknown], request_sha256='1' * 64, unloaded_path=path)
    assert result['coefficients'] == repeated['coefficients']
    assert repeated['selection']['isolated_unknown_cache'] == 1
    assert result['coefficients']['endpoint_s'] == pytest.approx(.12)
    assert result['coefficients']['decode_step_s'] == pytest.approx(.025)
    assert result['coefficients']['decode_attention_s'] == pytest.approx(1e-8)


def test_control_materialization_prevents_false_isolation(tmp_path):
    path, rows = evidence(tmp_path)
    probe = {**rows[0], 'row_id': 'control-probe', 'phase': 'control_initial', 'start_s': .01}
    result = calibrate(rows + [probe], request_sha256='0' * 64, unloaded_path=path)
    assert result['selection']['isolated_final_controls'] == 3
    assert '0' not in result['decode']['requests']
    assert 'control-probe' not in result['endpoint']['requests']
    assert all(np.isfinite(list(result['coefficients'].values())))
    bad = json.loads(path.read_text())
    bad['rows'][0]['request_prefill_kv_computed_tokens_sum'] = None
    path.write_text(json.dumps(bad))
    with pytest.raises(ValueError, match='invalid singleton'):
        calibrate(rows, request_sha256='0' * 64, unloaded_path=path)
