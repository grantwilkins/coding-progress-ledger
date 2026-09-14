import os,sys,json,time
from pathlib import Path
root=Path('/datadrive/qh0912/queue-haul');os.chdir(root);sys.path.insert(0,str(root))
os.environ.update(QH_MODEL_PROFILE='profiles/network-gpt-h100.json',QH_RUNTIME='native',QH_LMCACHE_MODE='mp',QH_NATIVE_RUNTIME_VERSIONS='0.24.0,0.5.1',HF_HOME='/datadrive',QH_CACHE_ROOT='/datadrive/queue-haul-cache',QH_FULL_KV_CONTROL='1')
os.environ['PATH']='/home/azureuser/.local/bin:/datadrive/qh0912/.venv/bin:'+os.environ['PATH']
import network_campaign as n
from destination_runner import completion_payload
from model_architecture_campaign import _json_markers
n.configure_handoff_environment('openai/gpt-oss-20b')
c=n.Cluster.load(Path('/datadrive/sea-controls-cluster-20260914.json'))
contract=n.freeze_contract(json.load(open('/datadrive/c12.json')))
contract['paths']={'east':contract['paths']['east']}
contract['aggregate']={k:contract['paths']['east'][k] for k in ('natural_mbps','controlled_mbps')}
s=n.start_cluster(c,Path('~/.ssh/azrs').expanduser(),contract,'controlled_40',Path('/datadrive/gpt32-full-state-sea-20260914'),model='openai/gpt-oss-20b',literal_token_timing=True)
rows=[]
try:
 geometry=_json_markers(s.run_root/'source.log','QH_KV_GEOMETRY ')[-1]
 assert all(g['sw_size_chunks']==-1 for g in geometry['object_groups'])
 for node in c.destinations:
  for tokens in (32239,32256):
   n._clear_cluster(s)
   session={'id':f'state-{tokens}','state_code':f'QHS{tokens}'}
   messages=n.profiler.exact_calibration_messages(s.cfg,session,tokens,max_tokens=128)
   prompt=n.testbed.mp_chat_tokens(s.cfg,n._probe(s.cfg,messages,session['state_code']),max_tokens=128)
   warm=n._warm(s,messages,session['state_code'],600,prompt)
   requests={};before=n.testbed.proxy_counts(s.run_root/'proxy_bytes.csv')
   for method in ('kv','replay','replay_control'):
    if method!='kv':
     n.testbed.http_text(node.host,s.cfg.sink_lmc_http_port,'POST','/cache/clear')
     n.testbed.http_text(node.host,s.cfg.sink_port,'POST','/reset_prefix_cache')
    payload=completion_payload(s.cfg.model,prompt,32,None,method!='kv');payload['ignore_eos']=False
    if method!='kv': payload['kv_transfer_params']={'qh_bypass_lmcache':True,'lmcache.skip_save':True}
    requests[method]=n._completion(s.cfg.host,s.ports[node.id]['api'],s.cfg.model,prompt,32,None,600,method!='kv',prepared_body=json.dumps(payload))
    if method=='kv':
     time.sleep(1)
     wire=n.testbed.count_delta(before,n.testbed.proxy_counts(s.run_root/'proxy_bytes.csv'))
   row={'destination':node.id,'context_tokens':tokens,'warm':warm,**requests,
        'kv_wire_bytes':wire.get(f'kv/{node.id}/target_to_client',0),
        'expected_wire_bytes':sum(g['chunk_bytes']*(tokens//geometry['chunk_tokens']) for g in geometry['object_groups'])}
   row['kv_replay_equal']=requests['kv']['token_ids']==requests['replay']['token_ids']
   row['replay_control_equal']=requests['replay']['token_ids']==requests['replay_control']['token_ids']
   rows.append(row)
   n.write_checkpoint(s.run_root/'full_state_control.json',{'diagnostic_only':True,'ignore_eos':False,'forced_token':None,'model':s.cfg.model,'geometry':geometry,'rows':rows})
   print(json.dumps({k:row[k] for k in ('destination','context_tokens','kv_replay_equal','replay_control_equal','kv_wire_bytes')}),flush=True)
finally:n.stop_cluster(s)
for node in c.destinations:
 geometry=_json_markers(s.run_root/'nodes'/node.id/'sink.log','QH_KV_GEOMETRY ')[-1]
 assert all(g['sw_size_chunks']==-1 for g in geometry['object_groups'])
n.full_state_reference(s.run_root/'full_state_control.json',s.cfg.model,geometry['chunk_tokens'])
n.write_checkpoint(s.run_root/'complete.json',{'status':'complete','scope':'full-history restoration diagnostic; not cold replay equivalence'})
