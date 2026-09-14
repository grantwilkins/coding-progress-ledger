import json, os, signal, subprocess, sys, time
from pathlib import Path

root, code, role = Path(sys.argv[1]), Path(sys.argv[2]), sys.argv[3]
signal.signal(signal.SIGTERM,lambda *_: sys.exit('supervisor terminated'))
signal.signal(signal.SIGINT,lambda *_: sys.exit('supervisor interrupted'))
sys.path.insert(0, str(code))
os.environ.update(QH_RUNTIME='native', QH_LMCACHE_MODE='mp', QH_NATIVE_RUNTIME_VERSIONS='0.22.0,0.5.1',
    QH_PREFIX_CACHING='on', QH_PORT_OFFSET='10000', QH_LMCACHE_L1_GB='32', HF_HOME='/datadrive',
    QH_CACHE_ROOT='/tmp/qh-server-cache', CUDA_VISIBLE_DEVICES='0', VLLM_SYSTEM_START_DATE='2026-09-06',
    VLLM_USE_FLASHINFER_SAMPLER='0', VLLM_ATTENTION_BACKEND='TRITON_ATTN', VLLM_WORKER_MULTIPROC_METHOD='spawn',
    QH_SERVER_TIMING_MODE_FILE=str(root/'timing-mode'), QH_SERVER_TIMING_DIR=str(root/'timing'))
import migration_testbed as b
import migration_profiler as p
import destination_runner as serving
root.mkdir(parents=True, exist_ok=True)
clock=json.loads((root/'runtime-launch.json').read_text())
cfg=b.Config(hf_home=Path('/datadrive'),cache_root=Path('/tmp/qh-server-cache'),src_port=18100,sink_port=18200,
    lmc_port=15655,kv_proxy_port=18300,api_proxy_port=18400,src_lmc_port=15557,sink_lmc_port=15556,src_lmc_http_port=18080,sink_lmc_http_port=18081)
children=[]
def launch(name, command):
    child=b.start_logged(command,root/(name+'.log'))
    children.append((name,child))
    with (root/'launches.jsonl').open('a') as f:f.write(json.dumps({'name':name,'pid':child.pid,'argv':command,'wall_ns':time.time_ns()})+'\n')
    return child
power=p.PowerSampler(root/'power.csv',.5)
metrics=None
try:
    power.start()
    launch('gpu-clocks',['nvidia-smi','--query-gpu=timestamp,uuid,power.draw,power.limit,utilization.gpu,memory.used,clocks.sm,clocks.mem','--format=csv','-lms','500'])
    if role=='source':
        redis=launch('redis',b.redis_cmd(cfg));b.wait_tcp_process(cfg.host,cfg.lmc_port,20,redis,root/'redis.log')
        routes=[b.Route('kv','0.0.0.0',cfg.kv_proxy_port,cfg.host,cfg.lmc_port,'resp'), b.Route('api',cfg.host,cfg.api_proxy_port,cfg.host,cfg.sink_port)]
        proxy=launch('proxy',[sys.executable,str(code/'migration_testbed.py'),'proxy','--routes-json',json.dumps([r.__dict__ for r in routes]),'--aggregate-mbps','1000','--log',str(root/'proxy_bytes.csv')])
        b.wait_tcp_process(cfg.host,cfg.kv_proxy_port,20,proxy,root/'proxy.log')
    cache=launch('lmcache-'+role,b.mp_server_cmd(cfg,role,l2_host='127.0.0.1' if role=='source' else '10.0.0.4'))
    cache_port=cfg.src_lmc_port if role=='source' else cfg.sink_lmc_port
    b.wait_tcp_process(cfg.host,cache_port,120,cache,root/('lmcache-'+role+'.log'))
    server=launch(role,b.vllm_cmd(cfg,role,gpu_index=0,sleep_mode=False))
    port=cfg.src_port if role=='source' else cfg.sink_port
    b.wait_health_process(cfg.host,port,240,server,root/(role+'.log'))
    metrics=serving.MetricsSampler(cfg.host,port,root/'engine.csv',.5);metrics.start()
    (root/'ready.json').write_text(json.dumps({'wall_ns':time.time_ns(),'pid':os.getpid()})+'\n')
    while time.time_ns()<clock['deadline_wall_ns']-110_000_000_000:
        if any(child.poll() is not None for _,child in children):raise RuntimeError('owned stack process exited')
        time.sleep(.2)
finally:
    errors=[]
    for sampler in (metrics,power):
        if sampler is not None:
            try:sampler.close()
            except Exception as exc:errors.append(f'{type(exc).__name__}: {exc}')
    forced=[]
    already_exited=[]
    def terminate(name,child,sig):
        try:os.killpg(child.pid,sig)
        except ProcessLookupError:already_exited.append(name)
        except OSError as exc:errors.append(f'{name}: {type(exc).__name__}: {exc}')
    try:
        for name,child in reversed(children):
            if child.poll() is None:terminate(name,child,signal.SIGTERM)
        deadline=time.monotonic()+45
        while any(child.poll() is None for _,child in children) and time.monotonic()<deadline:time.sleep(.2)
        for name,child in children:
            if child.poll() is None:forced.append(name);terminate(name,child,signal.SIGKILL)
            try:child.wait(timeout=2)
            except subprocess.TimeoutExpired:errors.append(f'{name}: child did not exit after shutdown')
    finally:
        (root/'stop.json').write_text(json.dumps({'wall_ns':time.time_ns(),'forced_kills':forced,
            'telemetry_errors':errors,'already_exited':already_exited,'returncodes':{name:child.poll() for name,child in children}},indent=2)+'\n')
    if errors:raise RuntimeError('telemetry shutdown failed: '+str(errors))
