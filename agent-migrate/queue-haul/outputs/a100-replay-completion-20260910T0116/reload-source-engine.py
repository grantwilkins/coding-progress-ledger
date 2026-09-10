"""Reload the measured export fix while retaining the working cache connections."""
import json
import os
from pathlib import Path
import signal
import subprocess
import time

out = Path(__file__).resolve().parent
os.environ.update(QH_RUNTIME='native',QH_LMCACHE_MODE='mp',QH_PORT_OFFSET='10000',HF_HOME='/datadrive')
start = json.loads((out/'supplemental-start.json').read_text())
old = json.loads((out/'source-launches.json').read_text())
source = next(r for r in old if r['name']=='source')
if Path(f"/proc/{start['pid']}/cmdline").exists():
    assert 'launch-source.py' in Path(f"/proc/{start['pid']}/cmdline").read_bytes().decode()
    os.kill(start['pid'],signal.SIGKILL)
if Path(f"/proc/{source['pid']}/cmdline").exists():
    os.killpg(source['pid'],signal.SIGKILL)
if Path('/proc/32085/cmdline').exists():
    os.kill(32085,signal.SIGTERM)
time.sleep(1)
(out/'stack/source.log').rename(out/f'stack/source-before-reload-{time.time_ns()}.log')
with (out/'stack/source.log').open('w') as log:
    child = subprocess.Popen(source['command'],stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
owned = [r for r in old if r['name']!='source'] + [{'name':'source-export-fixed','pid':child.pid,'command':source['command']}]
(out/'source-engine-reload.json').write_text(json.dumps({'wall_ns':time.time_ns(),'pid':os.getpid(),'owned':owned,
    'deadline_wall_ns':start['deadline_wall_ns'],'reason':'native prefix hit lost on external miss; source cache and proxy connections retained'},indent=2)+'\n')
stop = False
def finish(*_):
    global stop
    stop = True
signal.signal(signal.SIGTERM,finish)
signal.signal(signal.SIGINT,finish)
try:
    while not stop and time.time_ns() < start['deadline_wall_ns']-30_000_000_000:
        if child.poll() is not None:
            raise RuntimeError('corrected source engine exited')
        time.sleep(.5)
finally:
    for r in reversed(owned):
        if Path(f"/proc/{r['pid']}").exists():
            os.killpg(r['pid'],signal.SIGTERM)
    time.sleep(3)
    for r in reversed(owned):
        if Path(f"/proc/{r['pid']}").exists():
            os.killpg(r['pid'],signal.SIGKILL)
    (out/'source-final-stop.json').write_text(json.dumps({'wall_ns':time.time_ns(),
        'elapsed_s':(time.time_ns()-start['start_wall_ns'])/1e9,'owned':owned},indent=2)+'\n')
