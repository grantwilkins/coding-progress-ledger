"""Archive completed matched controls and expose finite-window interference."""
import argparse
import csv
import gzip
import hashlib
import json
import statistics
from pathlib import Path


def read_events(path):
    return [json.loads(line) for line in path.read_text().splitlines()]


def stats(values):
    return {'count': len(values), 'mean': statistics.mean(values),
            'median': statistics.median(values), 'p90': statistics.quantiles(values, n=10, method='inclusive')[8]} if len(values) > 1 else {'count': len(values), 'mean': values[0] if values else None}


p = argparse.ArgumentParser(description=__doc__)
p.add_argument('source', type=Path)
p.add_argument('output', type=Path)
a = p.parse_args()
assert json.loads((a.source / 'complete.json').read_text())['status'] == 'complete'
metadata = json.loads((a.source / 'metadata.json').read_text())
node = metadata['cluster']['destinations'][0]['id']
source_power = read_events(a.source / 'source-power.jsonl')
with (a.source / 'nodes' / node / 'power.csv').open() as f:
    destination_power = list(csv.DictReader(f))
summary = {'metadata': metadata, 'arms': {}, 'limitations': [
    'One repeat; synthetic resident tokens; eager timing runtime.',
    'Finite arrivals followed to completion; not sustained SLO capacity.',
    'Source histories are idle after export: no source-drain savings claim.',
    'Power windows use sensor averages and recorded host clock bounds; polls are not independent subsecond power measurements.']}
for arm in metadata.get('arms', ('none', 'replay', 'kv', 'mixed')):
    root = a.source / arm
    done = json.loads((root / 'complete.json').read_text())
    assert done['status'] == 'complete'
    events = read_events(root / 'events.jsonl')
    schedule = json.loads((root / 'schedule.json').read_text())
    epoch, wall = schedule['epoch_monotonic_ns'], schedule['epoch_wall_ns']
    end = next(r['monotonic_ns'] for r in events if r['event'] == 'all_completed')
    row = {'wire_evidence': done, 'offered_rps': metadata['resident_rps'],
           'arrival_window_s': metadata['seconds'], 'completion_window_s': (end - epoch) / 1e9}
    for kind in ('resident', 'migration'):
        requests = [r for r in events if r.get('kind') == kind and r['event'] == 'completion']
        assert len(requests) == sum(r.get('kind') == kind and r['event'] == 'arrival' for r in events)
        assert all(r['status'] == 200 and r['done'] and not r['error'] for r in requests)
        row[kind] = {'completed': len(requests), 'ttft_s': stats([r['ttft_s'] for r in requests]),
                     'request_mean_tpot_s': stats([r['mean_tpot_s'] for r in requests if r['mean_tpot_s'] is not None]),
                     'latency_s': stats([(r['end_ns'] - r['start_ns']) / 1e9 for r in requests]),
                     'finite_completion_rps': len(requests) / row['completion_window_s']}
    with (root / 'engine.csv').open() as f:
        metrics = list(csv.DictReader(f))
    before = [r for r in metrics if int(r['monotonic_ns']) < epoch + int(metadata['warmup_s'] * 1e9)]
    row['queue_at_migration'] = {k: float(before[-1][k]) for k in ('vllm:num_requests_running', 'vllm:num_requests_waiting')}
    row['peak_waiting'] = max(float(r['vllm:num_requests_waiting']) for r in metrics)
    row['power_windows'] = {}
    for name, start, finish in [('common_arrival_window', epoch, epoch + int(metadata['seconds'] * 1e9)), ('completion_tail', epoch + int(metadata['seconds'] * 1e9), end)]:
        if finish <= start: continue
        src = [r['average_power_mw']['value'] / 1000 for r in source_power if start <= r['query_start_monotonic_ns'] < finish and r['average_power_mw']['status'] == 'ok']
        dst = [float(r['power_w']) for r in destination_power if wall + start - epoch <= int(r['wall_ns']) < wall + finish - epoch and r['valid'] == '1']
        assert src and dst
        row['power_windows'][name] = {'duration_s': (finish - start) / 1e9, 'source_mean_w': statistics.mean(src), 'destination_mean_w': statistics.mean(dst), 'source_polls': len(src), 'destination_polls': len(dst)}
    summary['arms'][arm] = row

a.output.mkdir(parents=True, exist_ok=False)
manifest = []
for path in sorted(a.source.rglob('*')):
    if not path.is_file(): continue
    data = path.read_bytes(); relative = path.relative_to(a.source)
    target = a.output / 'raw' / (str(relative) if path.suffix == '.gz' else str(relative) + '.gz')
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_bytes(data if path.suffix == '.gz' else gzip.compress(data, mtime=0))
    manifest.append({'source': str(path), 'bytes': len(data), 'sha256': hashlib.sha256(data).hexdigest(), 'archive': str(target.relative_to(a.output))})
(a.output / 'summary.json').write_text(json.dumps(summary, indent=2) + '\n')
(a.output / 'input_manifest.json').write_text(json.dumps(manifest, indent=2) + '\n')
print(json.dumps(summary, indent=2))
