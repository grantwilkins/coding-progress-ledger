"""Archive completed control traces and summarize finite serving/idle windows."""
import argparse
import gzip
import hashlib
import json
import statistics
from pathlib import Path

p = argparse.ArgumentParser(description=__doc__)
p.add_argument('source', type=Path)
p.add_argument('output', type=Path)
a = p.parse_args()
assert json.loads((a.source / 'complete.json').read_text())['status'] == 'complete'
a.output.mkdir(parents=True, exist_ok=False)
manifest = []
for path in sorted(a.source.iterdir()):
    if path.is_file() and path.suffix in ('.json', '.jsonl', '.csv', '.log', '.gz'):
        data = path.read_bytes()
        target = a.output / (path.name if path.suffix == '.gz' else path.name + '.gz')
        target.write_bytes(data if path.suffix == '.gz' else gzip.compress(data, mtime=0))
        manifest.append({'source': str(path), 'sha256': hashlib.sha256(data).hexdigest(), 'bytes': len(data), 'archive': target.name})
power = [json.loads(line) for line in (a.source / 'power.jsonl').read_text().splitlines()]
events = [json.loads(line) for line in (a.source / 'phases.jsonl').read_text().splitlines()]
windows = []
for repeat, phase in sorted({(r['repeat'], r['phase']) for r in events}):
    times = {r['event']: r['monotonic_ns'] for r in events if (r['repeat'], r['phase']) == (repeat, phase)}
    requests = [json.loads(line) for line in (a.source / f'{phase}-r{repeat}.jsonl').read_text().splitlines() if json.loads(line)['event'] == 'completion']
    for label, start, end in [('idle_before', 'idle_before', 'active_start'), ('active', 'active_start', 'active_drained'), ('idle_after', 'idle_after', 'idle_after_end')]:
        rows = [r for r in power if times[start] <= r['query_start_monotonic_ns'] < times[end]]
        energy = [r for r in rows if r['total_energy_mj']['status'] == 'ok']
        duration = (times[end] - times[start]) / 1e9
        row = {'repeat': repeat, 'phase': phase, 'window': label, 'duration_s': duration, 'polls': len(rows),
               'mean_reported_average_w': statistics.mean(r['average_power_mw']['value'] / 1000 for r in rows if r['average_power_mw']['status'] == 'ok'),
               'peak_sampled_framebuffer_bytes': max(r['framebuffer_used_bytes']['value'] for r in rows)}
        if len(energy) > 1:
            joules = (energy[-1]['total_energy_mj']['value'] - energy[0]['total_energy_mj']['value']) / 1000
            span = (energy[-1]['query_start_monotonic_ns'] - energy[0]['query_start_monotonic_ns']) / 1e9
            assert joules >= 0 and span > 0
            row.update(energy_counter_j=joules, energy_counter_span_s=span, energy_counter_mean_w=joules / span)
        if label == 'active':
            row.update(completed_requests=len(requests), finite_completion_rps=len(requests) / duration,
                       input_tokens=sum(r['prompt_tokens'] for r in requests), output_tokens=sum(r['output_tokens'] for r in requests),
                       request_mean_tpot_s=statistics.mean(r['mean_tpot_s'] for r in requests if r['mean_tpot_s'] is not None) if any(r['mean_tpot_s'] is not None for r in requests) else None,
                       requests_with_tpot=sum(r['mean_tpot_s'] is not None for r in requests),
                       request_mean_delivery_tpot_s=statistics.mean((r['last_token_ns'] - r['first_ns']) / 1e9 / (r['output_tokens'] - 1) for r in requests if r['output_tokens'] > 1) if any(r['output_tokens'] > 1 for r in requests) else None,
                       request_mean_ttft_s=statistics.mean(r['ttft_s'] for r in requests))
        windows.append(row)
summary = {'metadata': json.loads((a.source / 'metadata.json').read_text()),
           'limitations': ['One repeat; synthetic token contents.', 'Delivery TPOT uses first/last client token events; coalesced events are not exact per-token execution timing.', 'NVML average fields are one-second averages; polling frequency is not independent sensor resolution.', 'Finite eight-concurrency controls do not establish sustained SLO capacity or net migration power relief.'],
           'windows': windows}
(a.output / 'summary.json').write_text(json.dumps(summary, indent=2) + '\n')
(a.output / 'input_manifest.json').write_text(json.dumps(manifest, indent=2) + '\n')
print(json.dumps(windows, indent=2))
