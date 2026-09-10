"""Frozen training split for the small, GPU-local resident queue pilot."""

import hashlib
import json
import math
from pathlib import Path

import numpy as np
from scipy.optimize import nnls


ROOT = Path(__file__).parent
UNLOADED = ROOT / 'outputs/a100-replay-live-20260909T1920/unloaded-analysis.json'
TOKEN_BUDGET = 8192


def prefill_seconds(context, uncached, coefficients):
    if not 0 < uncached <= context:
        raise ValueError('prefill requires a positive, known uncached token count')
    return (math.ceil(uncached / TOKEN_BUDGET) * coefficients['prefill_step_s']
            + uncached * coefficients['prefill_token_s']
            + uncached * (2 * context - uncached) * coefficients['prefill_attention_s'])


def errors(observed, predicted):
    delta = np.asarray(predicted) - np.asarray(observed)
    return {'samples': len(delta), 'rmse_s': float(np.sqrt(np.mean(delta ** 2))),
            'median_absolute_error_s': float(np.median(abs(delta))),
            'max_absolute_error_s': float(max(abs(delta)))}


def calibrate(rows, *, request_sha256, unloaded_path=UNLOADED):
    """Fit 7101 only; warmed final controls cannot identify batched decode costs."""
    if len(request_sha256) != 64 or any(c not in '0123456789abcdef' for c in request_sha256):
        raise ValueError('verified request-table SHA-256 required')
    unloaded_path = Path(unloaded_path)
    source = unloaded_path.read_bytes()
    runtime = unloaded_path.with_name('runtime-launch.json').read_bytes()
    if str(json.loads(runtime)['cfg']['max_num_batched_tokens']) != str(TOKEN_BUDGET):
        raise ValueError('prefill evidence uses a different scheduler token budget')
    training = [r for r in json.loads(source)['rows']
                if r['seed'] == 7101 and r['width'] == 1 and r['phase_valid']]
    x, y, contexts, uncached = [], [], [], []
    for r in training:
        counts = json.loads(r['prompt_counts'])
        if len(counts) != 1:
            raise ValueError('server prefill histogram must contain one isolated request')
        context, q = counts[0], r['request_prefill_kv_computed_tokens_sum']
        duration = r['request_prefill_time_seconds_sum']
        if q is None or duration is None or not all(math.isfinite(z) for z in (context, q, duration)) or not 0 < q <= context or duration <= 0:
            raise ValueError('invalid singleton server prefill evidence')
        x.append([math.ceil(q / TOKEN_BUDGET), q / 1000, q * (2 * context - q) / 1e8])
        y.append(duration); contexts.append(context); uncached.append(q)
    if len(x) < 3 or np.linalg.matrix_rank(x) != 3:
        raise ValueError('prefill training does not identify the three service terms')
    prefill, _ = nnls(np.asarray(x), np.asarray(y))
    c = {'prefill_step_s': float(prefill[0]), 'prefill_token_s': float(prefill[1] / 1000),
         'prefill_attention_s': float(prefill[2] / 1e8)}
    destination = [r for r in rows if r['serving_role'] == 'destination'
                   and r['start_s'] is not None and r['end_s'] is not None]
    eligible = [r for r in destination if r['seed'] == 7101 and r['arm'] == 'control'
                and r['phase'] == 'service' and r['done'] and r['status'] == 200
                and r['completed_within_observation'] and r['exact_token_timestamps']]
    isolated = [r for r in eligible if not any(z['row_id'] != r['row_id'] and z['episode'] == r['episode']
                and z['start_s'] < r['end_s'] and z['end_s'] > r['start_s'] for z in destination)]
    decode = [r for r in isolated if r['output_tokens'] > 1 and r['mean_tpot_s'] is not None]
    dx = np.asarray([[1., (r['prompt_tokens'] + r['output_tokens'] / 2) / 32768] for r in decode])
    dy = np.asarray([r['mean_tpot_s'] for r in decode])
    if len(decode) < 2 or np.linalg.matrix_rank(dx) != 2 or not np.isfinite([*dx.flat, *dy]).all() or np.any(dy <= 0):
        raise ValueError('isolated final controls do not identify decode timing')
    dc, _ = nnls(dx, dy)
    c.update(decode_step_s=float(dc[0]), decode_attention_s=float(dc[1] / 32768))
    endpoint = [{**r, 'cached_tokens': r.get('effective_cached_tokens', r['cached_tokens'])} for r in isolated]
    endpoint = [r for r in endpoint if r['cached_tokens'] is not None]
    if not endpoint:
        raise ValueError('endpoint calibration requires known native-cache observations')
    phase = [prefill_seconds(r['prompt_tokens'], r['prompt_tokens'] - r['cached_tokens'], c) for r in endpoint]
    observed = [r['ttft_s'] for r in endpoint]
    if not np.isfinite(observed).all() or any(t <= 0 for t in observed):
        raise ValueError('invalid isolated client TTFT evidence')
    c['endpoint_s'] = float(np.median(np.asarray(observed) - phase))
    if c['endpoint_s'] < 0 or min(c['prefill_step_s'], c['decode_step_s']) <= 0:
        raise ValueError('nonnegative endpoint latency and positive iteration costs required')
    source_name = str(unloaded_path.relative_to(ROOT) if unloaded_path.is_relative_to(ROOT) else unloaded_path)
    return {'coefficients': c, 'token_budget': TOKEN_BUDGET, 'training_seed': 7101,
            'heldout_seed': 7102, 'input_sha256': {'requests.csv': request_sha256,
                source_name: hashlib.sha256(source).hexdigest(),
                str(Path(source_name).with_name('runtime-launch.json')): hashlib.sha256(runtime).hexdigest()},
            'prefill': {'requests': [r['episode'] + '/' + r['phase'] for r in training],
                'context_tokens': [min(contexts), max(contexts)], 'uncached_tokens': [min(uncached), max(uncached)],
                'training_error': errors(y, np.asarray(x) @ prefill)},
            'decode': {'requests': [r['row_id'] for r in decode],
                'context_tokens': [min(r['prompt_tokens'] for r in decode), max(r['prompt_tokens'] for r in decode)],
                'training_error': errors(dy, dx @ dc)},
            'endpoint': {'requests': [r['row_id'] for r in endpoint],
                'uncached_tokens': [min(r['prompt_tokens'] - r['cached_tokens'] for r in endpoint),
                                    max(r['prompt_tokens'] - r['cached_tokens'] for r in endpoint)],
                'training_error': errors(observed, np.asarray(phase) + c['endpoint_s'])},
            'selection': {'eligible_final_controls': len(eligible), 'isolated_final_controls': len(isolated),
                          'isolated_unknown_cache': len(isolated) - len(endpoint)},
            'scope': ['Prefill costs transfer singleton server histogram timing from the earlier runtime; width-eight histogram sums are not GPU time.',
                'Final decode costs fit isolated client mean TPOT, which includes buffered delivery and is not measured GPU iteration time. Batched decode and mixed prefill/decode interference are held-out predictions, not identified coefficients.',
                'A shared iteration pays the larger prefill/decode base once, then adds token and attention terms; this composition is a declared model assumption.',
                'Endpoint residual is client-visible latency, not GPU work or measured server queue time; its placement around execution is unidentified.',
                'Native cached tokens condition the pilot. No ex-ante cache hit model, fleet SLO certification, or policy ranking is fitted.']}
