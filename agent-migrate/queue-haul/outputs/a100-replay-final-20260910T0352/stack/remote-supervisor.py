import json, os, signal, subprocess, time, threading, urllib.request
from pathlib import Path
root = Path('/tmp/qh-replay-final-20260910')
env = os.environ | dict(QH_LMCACHE_MODE='mp', QH_RUNTIME='native', QH_MODEL='openai/gpt-oss-20b', PYTHONHASHSEED='0', VLLM_EXECUTE_MODEL_TIMEOUT_SECONDS='900', VLLM_SERVER_DEV_MODE='1', VLLM_USE_FLASHINFER_SAMPLER='0', TMPDIR=str(root/'rpc'), VLLM_RPC_BASE_PATH=str(root/'rpc'), HF_HOME='/datadrive', HF_HUB_OFFLINE='1', TRANSFORMERS_OFFLINE='1', LMCACHE_REMOTE_URL='lm://10.0.0.4:18300', LMCACHE_REMOTE_SERDE='naive', LMCACHE_LMCACHE_INSTANCE_ID='completion_sink', LMCACHE_CHUNK_SIZE='256', LMCACHE_LOCAL_CPU='False', LMCACHE_MAX_LOCAL_CPU_SIZE='4', QH_KV_GEOMETRY_EVIDENCE='0', QH_LMCACHE_SEPARATE_OBJECT_GROUPS='0', PYTHONPATH=f'{root}/overlay:{root}/lmcache_compat', XDG_CACHE_HOME=str(root/'cache'))
commands = json.loads((root/'remote-commands.json').read_text())
started=time.time(); deadline=json.loads((root/'runtime-launch.json').read_text())['deadline_wall_ns']/1e9
processes={}
def sample():
    with (root/'engine-metrics.jsonl').open('w') as output:
        while time.time()<deadline-6:
            stamp=time.monotonic(); row={'wall_ns':time.time_ns(),'monotonic_ns':time.monotonic_ns(),'host':'germany'}
            try:
                with urllib.request.urlopen('http://127.0.0.1:18200/metrics',timeout=.8) as response: row['metrics']=response.read().decode()
            except Exception as error: row['error']=repr(error)
            output.write(json.dumps(row)+'\n'); output.flush(); time.sleep(max(0,1-(time.monotonic()-stamp)))
def stop(signum, frame): raise SystemExit(f'signal {signum}')
signal.signal(signal.SIGTERM,stop); signal.signal(signal.SIGINT,stop)
try:
    threading.Thread(target=sample,daemon=True).start()
    with (root/'power.csv').open('w') as log:
        processes['power']=subprocess.Popen(['nvidia-smi','--query-gpu=timestamp,uuid,power.draw,utilization.gpu,memory.used','--format=csv','-l','1'],stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
    for name, command in commands.items():
        with (root/f'{name}.log').open('w') as log:
            processes[name]=subprocess.Popen(command,env=env|{'CUDA_VISIBLE_DEVICES':'0' if name=='sink' else ''},stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
        (root/'owned-processes.json').write_text(json.dumps({'start_wall':started,'deadline_wall':deadline,'supervisor':os.getpid(),'processes':{k:v.pid for k,v in processes.items()},'commands':commands},indent=2))
        if name=='lmcache-sink': time.sleep(3)
    while time.time()<deadline-6:
        for name,process in processes.items():
            if process.poll() is not None: raise RuntimeError(f'{name} exited {process.returncode}')
        time.sleep(1)
finally:
    for process in processes.values():
        if process.poll() is None: os.killpg(process.pid,signal.SIGTERM)
    time.sleep(5)
    for process in processes.values():
        if process.poll() is None: os.killpg(process.pid,signal.SIGKILL)
    (root/'cleanup.json').write_text(json.dumps({'stop_wall':time.time(),'elapsed_seconds':time.time()-started,'processes':{k:v.pid for k,v in processes.items()}},indent=2))
