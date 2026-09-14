import json, os, shlex, signal, subprocess, sys, tarfile, time
from pathlib import Path

out=Path(__file__).resolve().parent
remote='/tmp/qh-server-destination'
ssh=['ssh','-i','/home/azureuser/.ssh/azrs','-o','BatchMode=yes','-o','StrictHostKeyChecking=yes','-o','ConnectTimeout=8','azureuser@10.1.0.4']
if sys.argv[1]=='mode':
    mode=sys.argv[2]
    assert mode in ('off','on')
    temporary=out/'stack/timing-mode.new'
    temporary.write_text(mode);temporary.replace(out/'stack/timing-mode')
    subprocess.run(ssh+['printf %s '+shlex.quote(mode)+' > '+remote+'/timing-mode.new && mv '+remote+'/timing-mode.new '+remote+'/timing-mode'],check=True,timeout=10)
else:
    assert sys.argv[1]=='cleanup'
    launches=json.loads((out/'owned-processes.json').read_text())
    deadline=time.monotonic()+115
    errors=[]
    result={'status':'running','started_wall_ns':time.time_ns(),'errors':errors}
    def run(command,cap,**kwargs):
        try:
            return subprocess.run(command,check=True,timeout=max(.001,min(cap,deadline-time.monotonic())),**kwargs)
        except (OSError,subprocess.SubprocessError) as exc:
            errors.append(f'{type(exc).__name__}: {exc}')
    try:
        if launches.get('destination_launch_attempted') and 'destination' not in launches:
            recovered=run(ssh+['cat '+remote+'/supervisor.pid'],10,capture_output=True,text=True)
            if recovered is not None:launches['destination']=int(recovered.stdout.strip())
        if 'source' in launches and not (out/'stack/stop.json').exists():
            try:os.kill(launches['source'],signal.SIGTERM)
            except ProcessLookupError:errors.append('source supervisor exited without stop.json')
        if 'destination' in launches:
            run(ssh+['test -f '+remote+'/stop.json || kill -TERM '+str(launches['destination'])],10)
        limit=min(deadline,time.monotonic()+60)
        if 'source' in launches:
            while not (out/'stack/stop.json').exists() and time.monotonic()<limit:time.sleep(.2)
            if not (out/'stack/stop.json').exists():errors.append('source shutdown not recorded')
        if 'destination' in launches:
            run(ssh+['while ! test -f '+remote+'/stop.json; do sleep 0.2; done'],max(.001,limit-time.monotonic()))
        for pid in launches['tunnels_and_mirrors']:
            try:os.killpg(pid,signal.SIGTERM)
            except ProcessLookupError:result.setdefault('already_exited_tunnels_and_mirrors',[]).append(pid)
        if 'destination' in launches:
            with (out/'destination-raw.tar.gz').open('wb') as f:
                fetched=run(ssh+['tar -C '+remote+' -czf - .'],45,stdout=f)
            if fetched is not None:
                with tarfile.open(out/'destination-raw.tar.gz') as archive:archive.extractall(out/'destination',filter='data')
        for role,root in (('source',out/'stack'),('destination',out/'destination')):
            if role in launches and (root/'stop.json').exists():
                stop=json.loads((root/'stop.json').read_text())
                if stop['forced_kills'] or stop['telemetry_errors']:
                    errors.append(f'{role} shutdown incomplete: {stop}')
        if errors:raise RuntimeError('owned stack cleanup failed: '+str(errors))
        result['status']='complete'
        print('Owned source and destination stopped; available remote raw records retrieved.')
    finally:
        if result['status']!='complete':result['status']='failed'
        result['end_wall_ns']=time.time_ns()
        (out/'cleanup-result.json').write_text(json.dumps(result,indent=2)+'\n')
