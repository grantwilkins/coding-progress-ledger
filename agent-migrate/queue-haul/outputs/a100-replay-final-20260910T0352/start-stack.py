"""Launch the frozen two-GPU stack once, with one shared acquisition deadline."""
import json, shlex, socket, subprocess, time
from pathlib import Path
out=Path(__file__).resolve().parent
remote='/tmp/qh-replay-final-20260910'
ssh=['ssh','-i','/home/azureuser/.ssh/azrs','-o','BatchMode=yes','-o','StrictHostKeyChecking=yes','-o','ExitOnForwardFailure=yes','-o','ServerAliveInterval=15','-o','ServerAliveCountMax=2']
host='azureuser@10.3.0.4'; launches=[]
def launch(name,command,path):
    with path.open('w') as log:
        child=subprocess.Popen(command,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
    launches.append({'name':name,'pid':child.pid,'command':command,'wall_ns':time.time_ns()})
    (out/'controller-launch.json').write_text(json.dumps(launches,indent=2)+'\n')
    return child
assert (out/'plan.json').is_file()
if (out/'runtime-launch.json').exists():raise FileExistsError('preserve existing acquisition; do not restart its clock')
source=launch('source-supervisor',['/tmp/qh-replay-completion-runtime/bin/python',str(out/'launch-source.py')],out/'source-supervisor.log')
ready=False
for _ in range(60):
    if source.poll() is not None:raise RuntimeError('source stack startup failed')
    try:
        with socket.create_connection(('127.0.0.1',18300),timeout=1):ready=True
    except OSError:time.sleep(.5)
    if ready:break
if not ready:raise TimeoutError('source proxy did not become ready before destination connection')
subprocess.run(['scp','-q','-i','/home/azureuser/.ssh/azrs','-o','BatchMode=yes','-o','StrictHostKeyChecking=yes',str(out/'runtime-launch.json'),host+':'+remote+'/runtime-launch.json'],check=True)
command=ssh+[host,'nohup /home/azureuser/coding-progress-ledger/agent-migrate/.venv/bin/python '+remote+'/remote-supervisor.py > '+remote+'/supervisor.log 2>&1 < /dev/null & echo $!']
result=subprocess.run(command,capture_output=True,text=True,check=True)
(out/'destination-launch.json').write_text(json.dumps({'command':command,'supervisor_pid':int(result.stdout.splitlines()[-1]),'wall_ns':time.time_ns(),'remote_root':remote},indent=2)+'\n')
launch('destination-tunnels',ssh+['-N','-L','18200:127.0.0.1:18200','-L','18081:127.0.0.1:18081',host],out/'tunnels.log')
for name,target in [('lmcache-sink.log',out/'stack/lmcache-sink.log'),('sink.log',out/'stack/sink.log'),('power.csv',out/'destination-power.csv'),('engine-metrics.jsonl',out/'destination-engine-metrics.jsonl')]:
    launch('mirror-'+name,ssh+[host,'tail -n +1 -F '+shlex.quote(remote+'/'+name)],target)
clock=json.loads((out/'runtime-launch.json').read_text())
inventory=json.loads((out/'inventory.json').read_text());inventory['deadline_wall_ns']=clock['deadline_wall_ns']-45_000_000_000
(out/'inventory.json').write_text(json.dumps(inventory,indent=2)+'\n')
print(json.dumps(clock))
