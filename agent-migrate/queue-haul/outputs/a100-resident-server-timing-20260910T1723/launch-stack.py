import json, os, shlex, signal, subprocess, sys, time
from pathlib import Path

out=Path(__file__).resolve().parent
code=out.parent.parent
sys.path.insert(0,str(code))
runtime='/tmp/qh-server-runtime/bin/python'
remote='/tmp/qh-server-destination'
ssh=['ssh','-i','/home/azureuser/.ssh/azrs','-o','BatchMode=yes','-o','StrictHostKeyChecking=yes','-o','ExitOnForwardFailure=yes','-o','ConnectTimeout=8','-o','ServerAliveInterval=15','-o','ServerAliveCountMax=2']
host='azureuser@10.1.0.4'
assert not (out/'runtime-launch.json').exists(), 'Never restart the acquisition clock'
start=time.time_ns()
clock={'start_wall_ns':start,'deadline_wall_ns':start+5400_000_000_000,'acquisition_cap_s':5400,'argv':sys.argv}
processes={'tunnels_and_mirrors':[]}
plan=json.loads((out/'plan.json').read_text())
cells=[{'id':e['spec']['episode'],'kind':'episode','status':'unstarted'} for e in plan['agentic_contract']['episodes']]
cells += [{'id':c['id'],'kind':'warm','status':'unstarted'} for c in plan['warm_decode_contract']['cells']]
report={'status':'starting','cells':cells,'start_wall_ns':start,'deadline_wall_ns':clock['deadline_wall_ns']}
def save_owned():
    (out/'owned-processes.json').write_text(json.dumps(processes,indent=2)+'\n')
def launch(command,log):
    with log.open('w') as f:child=subprocess.Popen(command,stdout=f,stderr=subprocess.STDOUT,start_new_session=True)
    with (out/'commands.jsonl').open('a') as f:f.write(json.dumps({'argv':command,'pid':child.pid,'wall_ns':time.time_ns()})+'\n')
    return child
signal.signal(signal.SIGTERM,lambda *_: sys.exit('launcher terminated'))
signal.signal(signal.SIGINT,lambda *_: sys.exit('launcher interrupted'))
acquisition=None
save_owned()
try:
    (out/'startup-report.json').write_text(json.dumps(report,indent=2)+'\n')
    for root in (out,out/'stack'):
        root.mkdir(exist_ok=True)
        (root/'runtime-launch.json').write_text(json.dumps(clock,indent=2)+'\n')
    (out/'stack/timing-mode').write_text('off')
    source=launch([runtime,str(out/'node-supervisor.py'),str(out/'stack'),str(code),'source'],out/'source-supervisor.log')
    processes['source']=source.pid;save_owned()
    subprocess.run(['scp','-q','-i','/home/azureuser/.ssh/azrs','-o','StrictHostKeyChecking=yes',str(out/'runtime-launch.json'),host+':'+remote+'/runtime-launch.json'],check=True,timeout=15)
    command='printf off > '+remote+'/timing-mode; nohup '+shlex.join([runtime,remote+'/node-supervisor.py',remote,'/tmp/qh-timing-code','sink'])+' > '+remote+'/supervisor.log 2>&1 < /dev/null & pid=$!; printf %s "$pid" > '+remote+'/supervisor.pid; echo "$pid"'
    processes['destination_launch_attempted']=True;save_owned()
    result=subprocess.run(ssh+[host,command],check=True,capture_output=True,text=True,timeout=15)
    processes['destination']=int(result.stdout.splitlines()[-1]);save_owned()
    tunnel=launch(ssh+['-N','-L','18200:127.0.0.1:18200','-L','18081:127.0.0.1:18081',host],out/'tunnels.log')
    processes['tunnels_and_mirrors'].append(tunnel.pid);save_owned()
    for name in ('lmcache-sink.log','sink.log'):
        child=launch(ssh+[host,'tail -n +1 -F '+remote+'/'+name],out/'stack'/name)
        processes['tunnels_and_mirrors'].append(child.pid);save_owned()
    inventory=json.loads((out/'inventory-prepared.json').read_text())
    inventory.update(deadline_wall_ns=clock['deadline_wall_ns'],stack_root=str(out/'stack'),
        timing_mode_commands={mode:[runtime,str(out/'control-stack.py'),'mode',mode] for mode in ('off','on')},
        cleanup_command=[runtime,str(out/'control-stack.py'),'cleanup'])
    (out/'inventory.json').write_text(json.dumps(inventory,indent=2)+'\n')
    import migration_testbed as b
    b.wait_health_process('127.0.0.1',18100,270,source,out/'source-supervisor.log')
    b.wait_health_process('127.0.0.1',18200,max(1,300-(time.time_ns()-start)/1e9),tunnel,out/'stack/sink.log')
    report['status']='ready'
    (out/'startup-report.json').write_text(json.dumps(report,indent=2)+'\n')
    command=[runtime,str(code/'pool_replay_server_acquire.py'),'--out',str(out),'--inventory',str(out/'inventory.json')]
    acquisition=subprocess.Popen(command,start_new_session=True)
    result=acquisition.wait()
    if result:raise RuntimeError(f'acquisition exited {result}')
except BaseException as exc:
    report.update(status='failed',error=f'{type(exc).__name__}: {exc}')
    if acquisition is None:(out/'startup-report.json').write_text(json.dumps(report,indent=2)+'\n')
    if not (out/'acquisition-report.json').exists():
        (out/'acquisition-report.json').write_text(json.dumps(report,indent=2)+'\n')
    raise
finally:
    if acquisition is not None and acquisition.poll() is None:
        os.killpg(acquisition.pid,signal.SIGTERM)
        try:acquisition.wait(timeout=5)
        except subprocess.TimeoutExpired:os.killpg(acquisition.pid,signal.SIGKILL);acquisition.wait()
    if not (out/'cleanup-result.json').exists():
        subprocess.run([runtime,str(out/'control-stack.py'),'cleanup'],check=True,timeout=120)
