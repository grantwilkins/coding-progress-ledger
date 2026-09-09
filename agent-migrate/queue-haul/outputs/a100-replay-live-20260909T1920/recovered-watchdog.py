import time,os,signal,json,sys
from pathlib import Path
out=Path(sys.argv[1]);start=json.loads((out/'runtime-launch.json').read_text())['start_wall_ns']/1e9
time.sleep(max(0,start+9000-time.time()))
for name in ('vllm','cache','redis'):
 pid=int((out/f'{name}.pid').read_text())
 try:os.killpg(pid,signal.SIGTERM)
 except ProcessLookupError:pass
print('Original wall-clock acquisition cap reached',flush=True)
