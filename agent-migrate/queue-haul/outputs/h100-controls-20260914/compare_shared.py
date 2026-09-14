"""Compare validated fixed arms; deadline counts are post-hoc, not policy choices."""
import csv
import gzip
import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).parent
SOURCES = ('shared-qwen', 'shared-gpt-partial', 'shared-gpt-mixed', 'shared-gemma')
rows, deadlines, manifest, seen = [], [], [], set()
for name in SOURCES:
    source = ROOT / name
    path = source / 'summary.json'
    data = json.loads(path.read_text()); meta = data['metadata']
    manifest.append({'path': str(path.relative_to(ROOT)), 'sha256': hashlib.sha256(path.read_bytes()).hexdigest()})
    for arm, r in data['arms'].items():
        key = meta['model'], arm
        assert key not in seen
        seen.add(key)
        rows.append({'model': meta['model'], 'arm': arm, 'evidence': name,
                     'offered_rps': meta['resident_rps'], 'scheduler_token_budget': meta['config']['max_num_batched_tokens'],
                     'resident_requests': r['resident']['completed'], 'migration_requests': r['migration']['completed'],
                     'resident_mean_ttft_s': r['resident']['ttft_s']['mean'], 'migration_mean_ttft_s': r['migration']['ttft_s']['mean'],
                     'resident_mean_tpot_s': r['resident']['request_mean_tpot_s']['mean'],
                     'completion_window_s': r['completion_window_s'], 'peak_waiting': r['peak_waiting'],
                     'kv_source_to_destination_bytes': r['wire_evidence']['wire_bytes']['kv/germany/target_to_client'],
                     'kv_destination_to_source_bytes': r['wire_evidence']['wire_bytes']['kv/germany/client_to_target']})
        if arm == 'none': continue
        with gzip.open(source / 'raw' / arm / 'schedule.json.gz', 'rt') as f: schedule = json.load(f)
        with gzip.open(source / 'raw' / arm / 'events.jsonl.gz', 'rt') as f: events = [json.loads(line) for line in f]
        requests = [r for r in events if r.get('kind') == 'migration' and r['event'] == 'completion']
        assert len(requests) == 8 and all(r['done'] and r['status'] == 200 for r in requests)
        release = schedule['epoch_monotonic_ns'] + int(meta['warmup_s'] * 1e9)
        for deadline in (5, 10, 20, 30, 60, 90, 120, 240, 300):
            end = release + int(deadline * 1e9)
            deadlines.append({'model': meta['model'], 'arm': arm, 'deadline_s': deadline,
                              'first_token_by_deadline': sum(r['first_ns'] <= end for r in requests),
                              'request_completed_by_deadline': sum(r['end_ns'] <= end for r in requests), 'requests': 8})
assert len(seen) == 12 and len({m for m, _ in seen}) == 3
assert sum(r['resident_requests'] for r in rows) == 360
assert sum(r['migration_requests'] for r in rows) == 72
for name, values in [('shared_comparison.csv', rows), ('fixed_arm_deadline_counts.csv', deadlines)]:
    with (ROOT / name).open('w') as f:
        writer = csv.DictWriter(f, fieldnames=list(values[0])); writer.writeheader(); writer.writerows(values)
(ROOT / 'comparison_manifest.json').write_text(json.dumps({'inputs': manifest, 'selected_arms': 12,
    'resident_requests': 360, 'migration_requests': 72,
    'limits': ['One repeat; fixed offered rate, unequal utilization.', 'GPT mixed is a separate recovery run.',
               'Interrupted attempts are preserved outside the selected arms.',
               'Deadline counts threshold recorded fixed-arm timings; no deadline-dependent planner decisions were executed here.']}, indent=2) + '\n')
