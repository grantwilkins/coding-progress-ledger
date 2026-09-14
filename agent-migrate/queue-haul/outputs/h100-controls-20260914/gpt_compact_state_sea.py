import json
import os
from pathlib import Path
os.environ.pop('QH_FULL_KV_CONTROL', None)
import network_campaign as n
n.configure_handoff_environment('openai/gpt-oss-20b')
c=n.Cluster.load(Path('/datadrive/sea-controls-cluster-20260914.json'))
r=Path('/datadrive/gpt32-compact-state-sea-20260914')
k=Path('/home/azureuser/.ssh/azrs')
f=n.freeze_contract(json.loads(Path('/datadrive/c12.json').read_text()))
f['paths']={'east':f['paths']['east']}
f['aggregate']={key:f['paths']['east'][key] for key in ('natural_mbps','controlled_mbps')}
n.host_check(c,k)
s=n.start_cluster(c,k,f,'controlled_40',r,model='openai/gpt-oss-20b',literal_token_timing=True)
try:
    n._network_state_equivalence(s,32256,Path('/datadrive/gpt32-full-state-sea-20260914/full_state_control.json'))
finally:
    n.stop_cluster(s)
n.write_checkpoint(r/'complete.json',{'status':'complete'})
