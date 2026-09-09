import json,os,signal,time,sys
from pathlib import Path
out=Path(sys.argv[1]);launch=json.loads((out/'runtime-launch.json').read_text())
time.sleep(max(0,launch['start_monotonic_ns']/1e9+9000-time.monotonic()))
for name in ['vllm','cache','redis']:
 pid=int((out/f'{name}.pid').read_text())
 try:os.killpg(pid,signal.SIGTERM)
 except ProcessLookupError:pass
print('Acquisition hard cap reached; owned runtime process groups stopped.',flush=True)
