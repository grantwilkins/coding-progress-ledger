"""Local coordinator only: approved native CLI jobs, SSH commands and downloads; no uploads."""
import argparse
from collections import defaultdict, deque
from concurrent.futures import ThreadPoolExecutor
import gzip
import hashlib
import json
from importlib.metadata import version
import os
from pathlib import Path
import shlex
import signal
import subprocess
import sys
import threading
import time
import uuid

p = argparse.ArgumentParser(description=__doc__)
p.add_argument('--repo', type=Path, default=Path.cwd())
p.add_argument('--out', type=Path, default=Path('outputs/a100-pooled-service'))
p.add_argument('--bridge-report', type=Path, default=Path('/tmp/qh-current-platform-comparison.json'))
p.add_argument('--host')
p.add_argument('--remote-repo')
p.add_argument('--remote-python')
p.add_argument('--key', type=Path, default=Path('~/.ssh/azrs').expanduser())
p.add_argument('--local-workers', type=int, default=9)
p.add_argument('--remote-workers', type=int, default=12)
p.add_argument('--prior-elapsed-s', type=float)
p.add_argument('--run', action='store_true')
a = p.parse_args()
a.repo = a.repo.resolve()
script_sha = hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
if a.run and a.out.is_absolute():
    raise ValueError("--out must be relative to each host repository")
out = a.out if a.out.is_absolute() else a.repo / a.out
for name in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS', 'VECLIB_MAXIMUM_THREADS', 'NUMEXPR_NUM_THREADS'):
    os.environ[name] = '1'
sys.path.insert(0, str(a.repo))
import pool_shed_campaign as c
plan = c.load_plan(out)
grid = json.loads(json.dumps(c.cells(plan['config'])))
variants = len(plan['config']['wan_gbps']) * len(plan['config']['deadlines'])
groups = defaultdict(list)
for i in range(len(grid)):
    groups[i // ((plan['config']['draws'] + 1) * variants) * variants + i % variants].append(i)
if set(groups) != set(range(len(groups))) or min(a.local_workers, a.remote_workers) < 1:
    raise ValueError('invalid scenario partition or worker counts')
stage, logs, native = out / 'cpu-staging', out / 'group-logs', out / 'native-runtimes'

def read_cell(path, index):
    return c.read_checkpoint(path, plan, grid[index])

def publish(journal):
    record = json.loads(journal.read_text())
    if (record['identity'] != plan['identity'] or set(record['files']) != {f'{i:06d}.json.gz' for i in groups[record['group']]}
            or record['runtime']['identity'] != plan['identity'] or record['runtime']['shard'] != record['group']
            or record['runtime']['shards'] != len(groups)):
        raise ValueError('stale staged publication')
    for name, digest in record['files'].items():
        source, target = Path(record['stage']) / name, out / 'cells' / name
        if target.exists():
            if hashlib.sha256(target.read_bytes()).hexdigest() != digest:
                raise ValueError(f'checkpoint overwrite conflict: {target}')
        else:
            if hashlib.sha256(source.read_bytes()).hexdigest() != digest:
                raise ValueError('staged cell changed before publication')
            source.replace(target)
    c.write_json(native / f"runtime-{record['group']}.json", record['runtime'])
    journal.unlink()

if a.run:
    bridge = json.loads(a.bridge_report.read_text())
    required = {'status', 'identity', 'versions', 'python', 'platform', 'remote_python', 'remote_platform'}
    if (not required <= bridge.keys() or bridge['status'] != 'pass' or bridge['identity'] != plan['identity']
            or bridge['versions'] != {name: version(name) for name in ('numpy', 'scipy', 'highspy')}
            or not all((a.host, a.remote_repo, a.remote_python))):
        raise ValueError('verified bridge and explicit remote connection are required')
    for directory in (stage, logs, native):
        directory.mkdir(parents=True, exist_ok=True)
    for journal in sorted(stage.glob('publish-*.json')):
        publish(journal)
completed, initial_hashes = set(), {}
for path in sorted((out / 'cells').glob('*.json.gz')):
    i = int(path.name.split('.')[0]); read_cell(path, i); completed.add(i)
    initial_hashes[path.name] = hashlib.sha256(path.read_bytes()).hexdigest()
local = deque(g for g, ids in groups.items() if any(i in completed for i in ids) and not all(i in completed for i in ids))
shared = deque(g for g, ids in groups.items() if not any(i in completed for i in ids))
print(json.dumps({'run': a.run, 'completed_cells': len(completed), 'touched_groups_local': len(local), 'untouched_groups_shared': len(shared)}), flush=True)
if not a.run:
    raise SystemExit
started, epoch = time.monotonic(), time.time()
prior = a.prior_elapsed_s if a.prior_elapsed_s is not None else epoch - (out / 'plan.json').stat().st_mtime
stop, lock = threading.Event(), threading.Lock()
for sig in (signal.SIGINT, signal.SIGTERM):
    signal.signal(sig, lambda *_: stop.set())
workers = a.local_workers + a.remote_workers
stats = [{'identity': plan['identity'], 'shard': i, 'shards': workers, 'host_role': 'local' if i < a.local_workers else 'remote',
          'groups': [], 'computed_cells': 0, 'reused_cells': 0} for i in range(workers)]
assignments = {}
c.write_json(out / 'cpu-initial-checkpoints.json', {'identity': plan['identity'], 'cells_sha256': initial_hashes})

def job(worker, group):
    ids, remote = groups[group], worker >= a.local_workers
    cli = ['pool_shed_campaign.py', 'run', '--out', str(Path(a.remote_repo) / a.out) if remote else str(out), '--shard', str(group), '--shards', str(len(groups))]
    with (logs / f'group-{group}.log').open('a') as log:
        if remote:
            threads = ['env', *[f'{k}=1' for k in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS', 'VECLIB_MAXIMUM_THREADS', 'NUMEXPR_NUM_THREADS')]]
            verify = shlex.join([*threads, a.remote_python, 'verify_bundle.py'])
            command = 'cd ' + shlex.quote(a.remote_repo) + ' && ' + verify + ' && ' + shlex.join([*threads, a.remote_python, *cli]) + ' && ' + verify
            subprocess.run(['ssh', '-i', str(a.key), '-o', 'BatchMode=yes', a.host, command], check=True, stdout=log, stderr=subprocess.STDOUT)
            directory = stage / f'{group}-{uuid.uuid4().hex}'; directory.mkdir()
            remote_out = Path(a.remote_repo) / a.out
            paths = [remote_out / 'cells' / f'{i:06d}.json.gz' for i in ids] + [remote_out / f'runtime-{group}.json']
            subprocess.run(['scp', '-i', str(a.key), '-o', 'BatchMode=yes', *[f'{a.host}:{x}' for x in paths], str(directory)], check=True, stdout=log, stderr=subprocess.STDOUT)
            files = {}
            for i in ids:
                path = directory / f'{i:06d}.json.gz'; read_cell(path, i); files[path.name] = hashlib.sha256(path.read_bytes()).hexdigest()
            runtime = json.loads((directory / f'runtime-{group}.json').read_text())
        else:
            subprocess.run([sys.executable, *cli], cwd=a.repo, check=True, stdout=log, stderr=subprocess.STDOUT)
            for i in ids:
                read_cell(out / 'cells' / f'{i:06d}.json.gz', i)
            runtime = json.loads((out / f'runtime-{group}.json').read_text())
        if runtime['identity'] != plan['identity'] or runtime['shard'] != group or runtime['shards'] != len(groups):
            raise ValueError('native runtime does not match assigned group')
        if remote:
            journal = stage / f'publish-{group}.json'
            c.write_json(journal, {'identity': plan['identity'], 'group': group, 'stage': str(directory), 'files': files, 'runtime': runtime})
            publish(journal)
        else:
            c.write_json(native / f'runtime-{group}.json', runtime)
    with lock:
        stats[worker]['groups'].append(group)
        stats[worker]['computed_cells'] += sum(i not in completed for i in ids)
        stats[worker]['reused_cells'] += sum(i in completed for i in ids)
    print(json.dumps({'worker': worker, 'group': group, 'host': 'remote' if remote else 'local', 'complete': True}), flush=True)

def worker(index):
    began = time.monotonic()
    while not stop.is_set():
        with lock:
            queue = local if index < a.local_workers and local else shared
            if not queue:
                break
            group = queue.popleft(); assignments[str(group)] = index
            c.write_json(out / 'cpu-assignments.json', {'identity': plan['identity'], 'workers': workers, 'assignments': assignments, 'scope': 'Local-only scenario ownership; no files uploaded.'})
        try:
            job(index, group)
        except BaseException:
            stop.set()
            raise
    stats[index].update(wall_s=time.monotonic() - began, complete=not stop.is_set())

with ThreadPoolExecutor(max_workers=workers) as pool:
    list(pool.map(worker, range(workers)))
if stop.is_set():
    raise RuntimeError('coordinator stopped after current groups; checkpoints preserved')
if (hashlib.sha256(Path(__file__).read_bytes()).hexdigest() != script_sha
        or any(hashlib.sha256((out / 'cells' / p).read_bytes()).hexdigest() != h for p, h in initial_hashes.items())
        or any(hashlib.sha256((a.repo / p).read_bytes()).hexdigest() != h for p, h in plan['sources'].items())):
    raise ValueError('frozen local inputs changed during the run')
c.write_json(out / 'cpu-workers.json', stats)
wall = time.monotonic() - started
c.write_json(out / 'performance.json', {'identity': plan['identity'], 'workers': workers, 'local_workers': a.local_workers, 'remote_workers': a.remote_workers,
    'compute_wall_s': wall, 'logical_shards': len(groups),
    'runtime_files': [f'native-runtimes/runtime-{g}.json' for g in sorted(map(int, assignments))],
    'prior_stage_elapsed_s': prior, 'prior_stage_time_scope': 'caller supplied' if a.prior_elapsed_s is not None else 'approximate plan modification time to coordinator start',
    'total_wall_s': prior + wall, 'scenarios': len(grid), 'policy_evaluations': len(grid) * len(c.POLICIES),
    'preexisting_completed_cells': len(completed), 'bridge_report_sha256': hashlib.sha256(a.bridge_report.read_bytes()).hexdigest(),
    'coordinator_sha256': script_sha, 'platform_bridge': {k: bridge[k] for k in ('identity', 'versions', 'python', 'platform', 'remote_python', 'remote_platform')}, 'runtime_scope': 'Coordinator monotonic interval including native jobs, SSH and downloads; prior stage separate.'})
(out / 'native-group-coordinator.py').write_bytes(Path(__file__).read_bytes())
c.reduce(out)
