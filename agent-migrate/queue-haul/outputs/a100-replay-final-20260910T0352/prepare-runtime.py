"""Record the two attached reference runtimes without starting serving engines."""
import hashlib, json, shlex, subprocess, time
from pathlib import Path
out = Path(__file__).resolve().parent
ssh = ['ssh','-i','/home/azureuser/.ssh/azrs','-o','BatchMode=yes','-o','StrictHostKeyChecking=yes','azureuser@10.3.0.4']
script = """import hashlib,json,subprocess,time
from pathlib import Path
from importlib.metadata import version
root=Path('/tmp/qh-replay-final-20260910')
paths=[root/'overlay/vllm/v1/engine/output_processor.py',*sorted((root/'lmcache_compat').glob('*.py'))]
print(json.dumps({'identity':{'host':'10.3.0.4','wall_ns':time.time_ns(),'gpu':subprocess.check_output(['nvidia-smi','--query-gpu=name,uuid,memory.total,driver_version','--format=csv'],text=True)},'runtime':{k:version(k) for k in ('vllm','lmcache','torch','transformers')},'sha256':{str(p):hashlib.sha256(p.read_bytes()).hexdigest() for p in paths}}))
"""
command=ssh+['/home/azureuser/coding-progress-ledger/agent-migrate/.venv/bin/python -c '+shlex.quote(script)]
start=time.time_ns(); result=subprocess.run(command,capture_output=True,text=True,check=True);end=time.time_ns()
row=json.loads(result.stdout.splitlines()[-1]); row['identity'].update(controller_send_wall_ns=start,controller_receive_wall_ns=end,clock_note='Own-host monotonic timestamps only; SSH roundtrip bounds wall alignment')
for name,value in [('destination-identity',row['identity']),('destination-runtime',row['runtime']),('destination-runtime-sha256',row['sha256'])]:
    (out/(name+'.json')).write_text(json.dumps(value,indent=2)+'\n')
local=json.loads((out/'source-runtime.json').read_text())
assert all(local[k]==row['runtime'][k] for k in ('vllm','lmcache'))
source=json.loads((out/'source-runtime-sha256.json').read_text())
for name in ('connector_patch.py','sitecustomize.py','server_info_middleware.py','output_processor.py'):
    assert next(v for k,v in source.items() if k.endswith('/'+name))==next(v for k,v in row['sha256'].items() if k.endswith('/'+name)),name
(out/'runtime-comparison.json').write_text(json.dumps({'source':local,'destination':row['runtime'],'differences':{k:[v,row['runtime'][k]] for k,v in local.items() if v!=row['runtime'][k]},'connector_and_stream_patch_identical':True,'command':command,'interpretation':'Reference vLLM and LMCache match. Torch build and Transformers versions differ; actual rendered tokens and state validation required.'},indent=2)+'\n')
print('Reference versions and corrected connector/stream hashes verified on both nodes')
