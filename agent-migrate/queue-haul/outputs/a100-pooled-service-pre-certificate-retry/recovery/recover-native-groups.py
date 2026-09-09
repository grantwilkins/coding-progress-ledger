"""Local-only completion after the original publisher is stopped; default is a read-only preview."""
import argparse
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor
import hashlib
import json
import os
from pathlib import Path
import shutil
import signal
import subprocess
import sys
import threading
import time

p = argparse.ArgumentParser(description=__doc__)
p.add_argument('--out', type=Path, default=Path('outputs/a100-pooled-service'))
p.add_argument('--coordinator-log', type=Path, default=Path('/tmp/qh-fresh-distributed-campaign.log'))
p.add_argument('--coordinator', type=Path, default=Path('/tmp/qh_native_group_coordinator.py'))
p.add_argument('--outage', type=Path, default=Path('/tmp/qh-remote-outage.json'))
p.add_argument('--run', action='store_true')
a = p.parse_args()
for key in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS', 'VECLIB_MAXIMUM_THREADS', 'NUMEXPR_NUM_THREADS'):
    os.environ[key] = '1'
sys.path.insert(0, str(Path.cwd()))
import pool_shed_campaign as c
out = a.out.resolve(); plan = c.load_plan(out); grid = c.cells(plan['config'])
sha = lambda path: hashlib.sha256(path.read_bytes()).hexdigest()
script_sha = sha(Path(__file__))
variants = len(plan['config']['wan_gbps']) * len(plan['config']['deadlines'])
groups = defaultdict(set)
for i in range(len(grid)):
    groups[i // ((plan['config']['draws'] + 1) * variants) * variants + i % variants].add(f'{i:06d}.json.gz')
assert len(groups) == 1500 and all(len(names) == 9 for names in groups.values())
recovery = out / 'recovery'
journals = sorted((out / 'cpu-staging').glob('publish-*.json'))
completed = {p.name for p in (out / 'cells').glob('*.json.gz')}
partial = [g for g, names in groups.items() if names & completed and not names <= completed]
print(json.dumps({'run': a.run, 'published_cells': len(completed), 'complete_groups': sum(names <= completed for names in groups.values()),
                  'partial_groups': partial, 'publication_journals': [p.name for p in journals]}), flush=True)
if not a.run:
    raise SystemExit
processes = subprocess.check_output(['ps', '-axo', 'pid=,command='], text=True).splitlines()
live = [line for line in processes if ('qh_native_group_coordinator.py' in line or 'pool_shed_campaign.py run' in line
        or ('10.1.0.4' in line and ('ssh ' in line or 'scp ' in line)))]
if live:
    raise RuntimeError('old publisher or native/SSH/SCP process is still live: ' + '\n'.join(live))
if recovery.exists():
    raise RuntimeError('recovery evidence already exists; inspect it before another recovery attempt')
started, epoch = time.monotonic(), time.time()
recovery.mkdir()
for source, name in ((a.coordinator_log, 'original-coordinator.log'), (a.coordinator, 'original-coordinator.py'), (a.outage, 'outage.json')):
    shutil.copyfile(source, recovery / name)
original = list((out / 'cells').glob('*.json.gz')) + list((out / 'native-runtimes').glob('*.json')) + list(out.glob('runtime-*.json')) + list((out / 'group-logs').glob('*.log'))
original += [out / name for name in ('plan.json', 'cpu-assignments.json', 'cpu-initial-checkpoints.json')]
inventory = {str(path.relative_to(out)): sha(path) for path in original}
replayed = []
for journal in journals:
    record = json.loads(journal.read_text()); g = record['group']; runtime = record['runtime']
    if (record['identity'] != plan['identity'] or set(record['files']) != groups[g] or runtime['identity'] != plan['identity']
            or runtime['shard'] != g or runtime['shards'] != len(groups)):
        raise ValueError('invalid publication journal')
    for name, digest in record['files'].items():
        target = out / 'cells' / name; source = target if target.exists() else Path(record['stage']) / name
        if sha(source) != digest:
            raise ValueError('publication checksum mismatch')
        c.read_checkpoint(source, plan, grid[int(name.split('.')[0])])
    shutil.copyfile(journal, recovery / journal.name)
    for name in record['files']:
        target = out / 'cells' / name
        if not target.exists():
            (Path(record['stage']) / name).replace(target)
    target = out / 'native-runtimes' / f'runtime-{g}.json'
    if target.exists():
        if json.loads(target.read_text()) != runtime:
            raise ValueError('publication runtime conflict')
    else:
        c.write_json(target, runtime)
    journal.unlink(); replayed.append(g)
completed = {p.name for p in (out / 'cells').glob('*.json.gz')}
if any(names & completed and not names <= completed for names in groups.values()):
    raise ValueError('partly published group without a valid publication journal')
accepted = {g for g, names in groups.items() if names <= completed}; missing = sorted(set(groups) - accepted)
if any((out / f'runtime-{g}.json').exists() or (out / 'native-runtimes' / f'runtime-{g}.json').exists() for g in missing):
    raise ValueError('unpublished group already has a runtime; inspect before overwriting evidence')
ownership = json.loads((out / 'cpu-assignments.json').read_text())
if ownership['identity'] != plan['identity'] or ownership['workers'] != 21:
    raise ValueError('unexpected original assignment provenance')
assignments = {int(g): w for g, w in ownership['assignments'].items()}
initial = json.loads((out / 'cpu-initial-checkpoints.json').read_text())
if (initial['identity'] != plan['identity'] or initial['cells_sha256'] or not set(assignments) <= set(groups)
        or any(not isinstance(w, int) or not 0 <= w < 21 for w in assignments.values())):
    raise ValueError('invalid fresh-run initial inventory or worker assignment')
success = [json.loads(line) for line in a.coordinator_log.read_text().splitlines() if line.startswith('{"worker":')]
if len({r['group'] for r in success}) != len(success):
    raise ValueError('duplicate original completion record')
for row in success:
    if row['complete'] is not True or assignments[row['group']] != row['worker'] or row['host'] != ('local' if row['worker'] < 9 else 'remote'):
        raise ValueError('invalid original completion ownership')
if {r['group'] for r in success} | set(replayed) != accepted:
    raise ValueError('published group and completion log disagree')
for g in accepted:
    runtime = json.loads((out / 'native-runtimes' / f'runtime-{g}.json').read_text())
    if runtime['identity'] != plan['identity'] or runtime['shard'] != g or runtime['shards'] != 1500:
        raise ValueError('invalid accepted original runtime')
    for name in groups[g]:
        c.read_checkpoint(out / 'cells' / name, plan, grid[int(name.split('.')[0])])
inventory.update({str(path.relative_to(out)): sha(path) for g in accepted for path in
    [*(out / 'cells' / name for name in groups[g]), out / 'native-runtimes' / f'runtime-{g}.json']})
manifest = {'identity': plan['identity'], 'status': 'running', 'original_files_sha256': inventory,
    'original_coordinator_log_sha256': sha(recovery / 'original-coordinator.log'), 'original_coordinator_sha256': sha(recovery / 'original-coordinator.py'),
    'outage_sha256': sha(recovery / 'outage.json'), 'outage_time_scope': 'Observation time does not establish when the VM stopped.',
    'original_accepted_groups': {str(g): {'worker': assignments[g], 'host_role': 'local' if assignments[g] < 9 else 'remote'} for g in sorted(accepted)},
    'publication_journals_replayed': replayed, 'publication_journals_sha256': {f'recovery/publish-{g}.json': sha(recovery / f'publish-{g}.json') for g in replayed},
    'publisher_stop_check': {'checked_at_epoch_s': epoch, 'matching_live_processes': []}, 'failed_remote_attempts': [{'group': g, 'worker': assignments[g]} for g in missing if assignments.get(g, -1) >= 9],
    'unpublished_local_attempts': [g for g in missing if 0 <= assignments.get(g, -1) < 9],
    'unstarted_groups': [g for g in missing if g not in assignments], 'replacement_groups': {}, 'recovery_workers': 9,
    'original_computed_cells': 9 * len(accepted), 'recovery_computed_cells': 9 * len(missing), 'reused_cells': 0,
    'stage1_wall_s_approx': max(0., epoch - (out / 'cpu-initial-checkpoints.json').stat().st_mtime), 'script_sha256': script_sha,
    'timing_scope': 'Stage1 starts approximately at original initial-checkpoint manifest mtime; stage2 uses monotonic wall time and includes verification.'}
c.write_json(out / 'recovery-manifest.json', manifest)
lock, stop = threading.Lock(), threading.Event()
for sig in (signal.SIGINT, signal.SIGTERM):
    signal.signal(sig, lambda *_: stop.set())
def worker(index):
    for g in missing[index::9]:
        if stop.is_set():
            return
        with lock:
            manifest['replacement_groups'][str(g)] = index
            c.write_json(out / 'recovery-manifest.json', manifest)
        try:
            with (recovery / f'group-{g}.log').open('w') as log:
                subprocess.run([sys.executable, 'pool_shed_campaign.py', 'run', '--out', str(out), '--shard', str(g), '--shards', '1500'], check=True, stdout=log, stderr=subprocess.STDOUT)
            for name in groups[g]:
                c.read_checkpoint(out / 'cells' / name, plan, grid[int(name.split('.')[0])])
            source, target = out / f'runtime-{g}.json', out / 'native-runtimes' / f'runtime-{g}.json'
            if target.exists():
                raise ValueError('recovery would overwrite an accepted native runtime')
            shutil.copyfile(source, target)
        except BaseException:
            stop.set()
            raise
        print(json.dumps({'recovery_worker': index, 'group': g, 'complete': True}), flush=True)
with ThreadPoolExecutor(max_workers=9) as pool:
    list(pool.map(worker, range(9)))
if stop.is_set():
    raise RuntimeError('recovery stopped after current groups; evidence retained')
if c.load_plan(out)['identity'] != plan['identity'] or sha(Path(__file__)) != script_sha or any(sha(out / p) != h for p, h in inventory.items()):
    raise ValueError('original evidence or frozen source changed')
if set(map(int, manifest['replacement_groups'])) != set(missing) or 9 * (len(accepted) + len(missing)) != len(grid):
    raise ValueError('incomplete or overlapping group accounting')
runtime_files = [f'native-runtimes/runtime-{g}.json' for g in range(1500)]
if {p.name for p in (out / 'native-runtimes').glob('*.json')} != {Path(n).name for n in runtime_files}:
    raise ValueError('missing or unexpected native runtime')
if {p.name for p in (out / 'cells').glob('*.json.gz')} != set().union(*groups.values()):
    raise ValueError('missing or unexpected final cell')
for g, name in enumerate(runtime_files):
    r = json.loads((out / name).read_text())
    if r['identity'] != plan['identity'] or r['shard'] != g or r['shards'] != 1500 or not 0 <= r['wall_s'] < float('inf'):
        raise ValueError('invalid final native runtime')
manifest.update(status='complete', stage2_wall_s_exact=time.monotonic() - started)
c.write_json(out / 'recovery-manifest.json', manifest)
bridge_path = c.ROOT / 'outputs/a100-pooled-service-validation/stable-greedy/platform-comparison.json'; bridge = json.loads(bridge_path.read_text())
wall = manifest['stage1_wall_s_approx'] + manifest['stage2_wall_s_exact']
c.write_json(out / 'performance.json', {'identity': plan['identity'], 'workers': 21, 'local_workers': 9, 'remote_workers': 12, 'recovery_workers': 9,
    'logical_shards': 1500, 'runtime_files': runtime_files, 'compute_wall_s': wall, 'prior_stage_elapsed_s': 0., 'total_wall_s': wall,
    'scenarios': len(grid), 'policy_evaluations': len(grid) * len(c.POLICIES), 'preexisting_completed_cells': 0,
    'bridge_report_sha256': sha(bridge_path), 'platform_bridge': {k: bridge[k] for k in ('identity', 'versions', 'python', 'platform', 'remote_python', 'remote_platform')},
    'coordinator_sha256': manifest['original_coordinator_sha256'], 'recovery_manifest_sha256': sha(out / 'recovery-manifest.json'),
    'runtime_scope': '21 workers initially allocated; remote VM deallocated; completed original groups plus 9 local recovery workers. Whole elapsed approximate; stage2 exact. No claim all 21 workers completed.'})
shutil.copyfile(recovery / 'original-coordinator.py', out / 'native-group-coordinator.py')
shutil.copyfile(Path(__file__), recovery / 'recover-native-groups.py')
c.reduce(out)
