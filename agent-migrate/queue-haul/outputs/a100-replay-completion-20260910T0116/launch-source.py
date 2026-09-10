"""Own only the supplemental source engine, cache and locally timed wire proxy."""
import hashlib
import json
import os
import signal
import subprocess
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path.cwd()))
import migration_testbed as b

out = Path(__file__).resolve().parent
root = out / 'stack'
plan = json.loads((out / 'supplemental-plan.json').read_text())
os.environ.update(QH_RUNTIME='native', QH_LMCACHE_MODE='mp', QH_PREFIX_CACHING='on', QH_PORT_OFFSET='10000',
    HF_HOME='/datadrive', QH_CACHE_ROOT='/tmp/qh-replay-completion-cache', CUDA_VISIBLE_DEVICES='0')
cfg = b.Config(hf_home=Path('/datadrive'), cache_root=Path('/tmp/qh-replay-completion-cache'),
    src_port=18100, sink_port=18200, lmc_port=15655, kv_proxy_port=18300, api_proxy_port=18400,
    src_lmc_port=15557, sink_lmc_port=15556, src_lmc_http_port=18080, sink_lmc_http_port=18081)
started = json.loads((out / 'supplemental-start.json').read_text())['start_wall_ns'] if (out / 'supplemental-start.json').exists() else time.time_ns()
(out / 'supplemental-start.json').write_text(json.dumps({'start_wall_ns': started,
    'deadline_wall_ns': started + int(plan['max_supplemental_wall_s'] * 1e9),
    'pid': os.getpid(), 'argv': sys.argv}, indent=2) + '\n')
stopped = False
def stop(*_):
    global stopped
    stopped = True
signal.signal(signal.SIGTERM, stop)
signal.signal(signal.SIGINT, stop)
procs, launches = [], []
def launch(name, command):
    p = b.start_logged(command, root / (name + '.log'))
    procs.append(p)
    launches.append({'name': name, 'pid': p.pid, 'command': [str(x) for x in command], 'wall_ns': time.time_ns()})
    (out / 'source-launches.json').write_text(json.dumps(launches, indent=2) + '\n')
    return p
try:
    patch = Path(sys.executable).parent.parent / 'lib/python3.12/site-packages/vllm/v1/engine/output_processor.py'
    previous = out.parent / 'a100-replay-live-20260909T1920'
    evidence = json.loads((previous / 'stream-patch.json').read_text())
    assert hashlib.sha256(patch.read_bytes()).hexdigest() in (evidence['before_sha256'], evidence['after_sha256'])
    patch.write_bytes((previous / 'output_processor.patched.py').read_bytes())
    (out / 'source-stream-patch.json').write_text(json.dumps(evidence | {'path': str(patch)}, indent=2) + '\n')
    redis = launch('redis', b.redis_cmd(cfg))
    b.wait_tcp_process(cfg.host, cfg.lmc_port, 30, redis, root / 'redis.log')
    routes = [b.Route('kv', '0.0.0.0', cfg.kv_proxy_port, cfg.host, cfg.lmc_port, 'resp'),
              b.Route('api', cfg.host, cfg.api_proxy_port, cfg.host, cfg.sink_port)]
    proxy = launch('proxy', [sys.executable, str(Path(b.__file__)), 'proxy', '--routes-json',
        json.dumps([r.__dict__ for r in routes]),
        '--aggregate-mbps', '1000', '--log', str(root / 'proxy_bytes.csv')])
    b.wait_tcp_process(cfg.host, cfg.kv_proxy_port, 30, proxy, root / 'proxy.log')
    cache = launch('lmcache-source', b.mp_server_cmd(cfg, 'source'))
    b.wait_tcp_process(cfg.host, cfg.src_lmc_port, 180, cache, root / 'lmcache-source.log')
    source = launch('source', b.vllm_cmd(cfg, 'source', gpu_index=0, sleep_mode=False))
    b.wait_health_process(cfg.host, cfg.src_port, 420, source, root / 'source.log')
    (out / 'source-ready.json').write_text(json.dumps({'wall_ns': time.time_ns(), 'port': cfg.src_port}) + '\n')
    while not stopped and time.time_ns() < started + int(plan['max_supplemental_wall_s'] * 1e9):
        if any(p.poll() is not None for p in procs):
            raise RuntimeError('an owned source-stack process exited')
        time.sleep(.5)
finally:
    for p in reversed(procs):
        b.stop_proc(p)
    (out / 'source-stop.json').write_text(json.dumps({'wall_ns': time.time_ns(), 'elapsed_s': (time.time_ns()-started)/1e9,
        'returncodes': [p.poll() for p in procs]}, indent=2) + '\n')
