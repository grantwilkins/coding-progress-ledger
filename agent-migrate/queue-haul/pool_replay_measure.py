"""Bounded local-A100 acquisition on the existing reference destination stack."""
from __future__ import annotations

import argparse
import copy
import hashlib
import gzip
import json
import os
import subprocess
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from dataclasses import asdict, replace
from pathlib import Path

import destination_runner as serving
import migration_profiler as p
import migration_testbed as b


def write(path, value):
    path.write_text(json.dumps(value, indent=2, allow_nan=False) + '\n')


def cache_integrity(cold, warm, before, after):
    queries = after['vllm:prefix_cache_queries_total'] - before['vllm:prefix_cache_queries_total']
    hits = after['vllm:prefix_cache_hits_total'] - before['vllm:prefix_cache_hits_total']
    external = after['vllm:external_prefix_cache_hits_total'] - before['vllm:external_prefix_cache_hits_total']
    return {'cold_usage_cached_tokens': cold.get('cached_tokens'), 'warm_usage_cached_tokens': warm.get('cached_tokens'),
            'isolated_cold_engine_queries': queries, 'isolated_cold_engine_hits': hits,
            'isolated_cold_external_hits': external, 'cold_state_basis': 'independent isolated engine counter deltas',
            'passed': queries == cold['prompt_tokens'] and hits == external == 0
            and warm.get('cached_tokens') is not None and warm['cached_tokens'] > 0}


class Acquisition:
    def __init__(self, out):
        self.out, self.cfg = out, replace(b.Config(), src_port=b.Config().sink_port)
        self.launch = json.loads((out / 'runtime-launch.json').read_text())
        self.deadline = time.monotonic() + self.launch['start_wall_ns']/1e9 + 9000 - time.time()
        self.lock = threading.Lock()
        self.boot_id = Path('/proc/sys/kernel/random/boot_id').read_text().strip()
        suffix = '-recovered' if (out/'node-recovery.json').exists() else ''
        self.requests = (out / f'requests{suffix}.jsonl').open('a', buffering=1)
        self.events = (out / f'request-events{suffix}.jsonl').open('a', buffering=1)

    def remaining(self):
        seconds = self.deadline - time.monotonic()
        if seconds <= 0:
            raise TimeoutError('150-minute acquisition limit')
        return seconds

    def record(self, handle, row):
        with self.lock:
            handle.write(json.dumps({'boot_id':self.boot_id,**row}, separators=(',', ':')) + '\n')

    def probe(self, messages, code):
        session = p.LiveSession.__new__(p.LiveSession)
        session.state_code = code
        return session.probe(messages)

    def render(self, messages, code):
        tokens = b.mp_chat_tokens(self.cfg, self.probe(messages, code))
        if len(tokens) + p.PROBE_MAX_TOKENS > self.cfg.max_model_len:
            raise ValueError('full rendered migration prompt plus generation exceeds runtime limit')
        return tokens

    def history(self, label, target):
        row = {'id': label, 'state_code': hashlib.sha256(label.encode()).hexdigest()[:10]}
        return p.exact_calibration_messages(self.cfg, row, target), row['state_code']

    def append(self, messages, code, count):
        target = len(self.render(messages, code)) + count
        result = copy.deepcopy(messages)
        original = result[-1]['content']
        words = count
        for _ in range(4):
            result[-1]['content'] = original + ' x' * words
            delta = target - len(self.render(result, code))
            if not delta:
                return result
            words += delta
        raise ValueError('cannot render exact updated context without changing retained history')

    def chat(self, messages, code, tags, salt, timeout=90):
        tokens = self.render(messages, code)
        start = time.monotonic_ns()
        row = {**tags, 'dispatch_ns': start, 'messages': self.probe(messages, code),
               'rendered_prompt_token_ids': tokens, 'prompt_tokens': len(tokens),
               'context_hash': p.messages_hash(messages), 'cache_salt': salt,
               'max_tokens': p.PROBE_MAX_TOKENS, 'clock': 'client_monotonic',
               'server_queue_start_ns': None, 'executed_tokens': None, 'recomputed_tokens': None}
        self.record(self.events, {**tags, 'kind': 'dispatch', 'monotonic_ns': start})
        try:
            result, text = p.stream_chat(self.cfg, self.cfg.sink_port, self.probe(messages, code),
                p.PROBE_MAX_TOKENS, row['context_hash'], min(timeout, self.remaining()),
                bypass_lmcache=True, cache_salt=salt,
                event_sink=lambda event: self.record(self.events, {**tags, **event}))
            row.update(asdict(result), response_text=text, status='complete' if result.status_code == 200 else 'http_error',
                       derived_prompt_minus_cache_tokens=None if result.cached_tokens is None else result.prompt_tokens-result.cached_tokens,
                       native_cached_tokens=result.cached_tokens, external_retrieved_tokens=0,
                       external_retrieval_basis='request explicitly bypasses LMCache; engine metrics retained')
            if result.prompt_tokens != len(tokens):
                row['status'] = 'render_mismatch'
            row.update(expected_state_code=code, state_code_valid=code in text)
            if not row['state_code_valid']:
                row['status'] = 'validation_failed'
            if result.first_byte_ns is not None:
                row['ttft_s'] = (result.first_byte_ns-result.start_ns)/1e9
            row['mean_tpot_s'] = ((result.last_token_ns-result.first_byte_ns)/1e9/(result.output_tokens-1)
                if result.exact_token_timestamps and result.output_tokens > 1 else None)
        except Exception as exc:
            row.update(status='failed', error=f'{type(exc).__name__}: {exc}', end_ns=time.monotonic_ns())
            self.record(self.requests, row)
            raise
        self.record(self.requests, row)
        return row

    def engine_metrics(self):
        return serving.parse_metrics(b.http_text(self.cfg.host, self.cfg.sink_port, 'GET', '/metrics'))

    def unloaded(self, plan):
        previous = self.out / 'unloaded'
        root = self.out / ('unloaded-continuation' if (self.out/'unloaded-matrix').exists() else 'unloaded-matrix' if previous.exists() else 'unloaded')
        root.mkdir(exist_ok=False)
        metrics = serving.MetricsSampler(self.cfg.host, self.cfg.sink_port, root/'engine.csv', .5)
        power = p.PowerSampler(root/'power.csv', .5)
        metrics.start(); power.start()
        try:
            if previous != root:
                rows = [json.loads(line) for line in (self.out/'requests.jsonl').read_text().splitlines()]
                cold, warm = [next(r for r in rows if r.get('episode') == 'cache-integrity' and r['phase'] == phase)
                              for phase in ('cold', 'shared_prefix')]
                samples = [json.loads(line) for line in gzip.open(previous/'engine.prom.jsonl.gz', 'rt')]
                before = serving.parse_metrics(samples[0]['prometheus_text'])
                after = serving.parse_metrics(max((r for r in samples if r['monotonic_ns'] < warm['start_ns']),
                                                  key=lambda r:r['monotonic_ns'])['prometheus_text'])
            else:
                messages, code = self.history('cache-verification-7101', 2048)
                before = self.engine_metrics()
                cold = self.chat(messages, code, {'episode':'cache-integrity','phase':'cold'}, 'integrity-7101')
                after = self.engine_metrics()
                warm = self.chat(messages, code, {'episode':'cache-integrity','phase':'shared_prefix'}, 'integrity-7101')
            check = cache_integrity(cold, warm, before, after)
            write(root/'cache-integrity.json', check)
            if not check['passed']:
                raise RuntimeError('cold/shared-prefix cache telemetry verification failed')
            attempted = {json.loads(line).get('episode') for line in (self.out/'requests.jsonl').read_text().splitlines()}
            for index, trial in enumerate(plan['unloaded_trials']):
                started = time.monotonic()
                episode = f"unloaded-{index:02d}-s{trial['seed']}"
                if episode in attempted:
                    print(episode, 'retained prior attempt; no retry', flush=True)
                    continue
                histories = [self.history(f'{episode}-lane{lane}', trial['retained_tokens']) for lane in range(trial['width'])]
                cold_histories = [self.history(f'{episode}-cold-lane{lane}', trial['retained_tokens']+trial['append_tokens']) for lane in range(trial['width'])]
                results = []
                for phase in trial['order']:
                    pairs = cold_histories if phase == 'cold_updated' else histories
                    if phase == 'catch_up':
                        pairs = [(self.append(messages, code, trial['append_tokens']), code) for messages, code in histories]
                    before_phase = self.engine_metrics()
                    phase_start = time.monotonic_ns()
                    with ThreadPoolExecutor(max_workers=trial['width']) as executor:
                        futures = [executor.submit(self.chat, messages, code,
                            {**trial, 'episode':episode, 'phase':phase, 'session':lane, 'cohort':'migration', 'method':'replay'},
                            f'{episode}-lane{lane}' + ('-cold' if phase == 'cold_updated' else ''),
                            max(.1, 90-(time.monotonic()-started))) for lane,(messages,code) in enumerate(pairs)]
                        rows = []
                        for future in futures:
                            try:
                                rows.append(future.result())
                            except Exception as exc:
                                rows.append({'status':'failed','error':f'{type(exc).__name__}: {exc}'})
                    results.append({'phase':phase,'start_ns':phase_start,'end_ns':time.monotonic_ns(),
                        'request_ids':[r.get('request_id') for r in rows], 'cached_tokens':[r.get('cached_tokens') for r in rows],
                        'statuses':[r['status'] for r in rows], 'engine_before':before_phase, 'engine_after':self.engine_metrics()})
                status = 'complete' if all(v == 'complete' for phase in results for v in phase['statuses']) else 'failed'
                write(root/f'{episode}.json', {'trial':trial,'phases':results,'elapsed_s':time.monotonic()-started,'status':status})
                print(episode, status, round(time.monotonic()-started,2), flush=True)
        finally:
            metrics.close(); power.close()


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('stage',choices=['unloaded'])
    parser.add_argument('--out',type=Path,required=True)
    parser.add_argument('--plan',type=Path,required=True)
    args=parser.parse_args()
    plan=json.loads(args.plan.read_text())
    acquisition=Acquisition(args.out)
    b.wait_health(acquisition.cfg.host,acquisition.cfg.sink_port,min(600,acquisition.remaining()))
    write(args.out/f'{args.stage}-launch.json', {'argv':__import__('sys').argv,
        'commit':subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip(),
        'dirty':subprocess.check_output(['git','status','--porcelain'],text=True),
        'source_sha256':p.file_hash(Path(__file__)), 'plan_sha256':p.file_hash(args.plan),
        'gpu':subprocess.check_output(['nvidia-smi','--query-gpu=name,uuid,memory.total','--format=csv'],text=True)})
    acquisition.unloaded(plan)


if __name__=='__main__':
    main()
