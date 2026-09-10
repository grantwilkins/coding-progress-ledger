"""Hash the exact loaded model files on both nodes without GPU requests."""
import concurrent.futures,json,shlex,subprocess
from pathlib import Path
out=Path(__file__).resolve().parent
script="""import hashlib,json
from pathlib import Path
root=Path('/datadrive/hub/models--openai--gpt-oss-20b/snapshots/6cee5e81ee83917806bbde320786a8fb61efebee')
rows={}
for p in sorted(root.rglob('*')):
 if p.is_file():
  digest=hashlib.sha256()
  with p.open('rb') as f:
   for block in iter(lambda:f.read(8*1024*1024),b''):digest.update(block)
  rows[str(p.relative_to(root))]={'sha256':digest.hexdigest(),'bytes':p.stat().st_size}
print(json.dumps(rows))
"""
commands={'source':['python3','-c',script], 'destination':['ssh','-i','/home/azureuser/.ssh/azrs','-o','BatchMode=yes','-o','StrictHostKeyChecking=yes','azureuser@10.3.0.4','python3 -c '+shlex.quote(script)]}
def measure(role):
 result=subprocess.run(commands[role],capture_output=True,text=True,check=True)
 rows=json.loads(result.stdout.splitlines()[-1]);(out/(role+'-model-sha256.json')).write_text(json.dumps(rows,indent=2)+'\n');return rows
with concurrent.futures.ThreadPoolExecutor(max_workers=2) as pool:results=dict(zip(commands,pool.map(measure,commands)))
(out/'model-comparison.json').write_text(json.dumps({'same_files_and_hashes':results['source']==results['destination'],'commands':commands},indent=2)+'\n')
if results['source']!=results['destination']:raise RuntimeError('source/destination model input bytes differ')
print('Identical model files verified on both nodes:',len(results['source']))
