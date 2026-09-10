"""Offline service-episode RESP payload accounting; GET traffic is not automatically WAN traffic."""
import argparse
import csv
import hashlib
import importlib.util
import io
import json
import re
from pathlib import Path

spec=importlib.util.spec_from_file_location('keyed_wire',Path(__file__).with_name('keyed-wire.py'))
keyed=importlib.util.module_from_spec(spec);spec.loader.exec_module(keyed)


def connection_origins(root, transfers):
    files=[root/'paired/paired-7101-8192-kv_transfer/events.jsonl',root/'paired/paired-plan.json',
        root/'stack/proxy_connections.csv',root/'stack/resp_transfers.csv',root/'stack/lmcache-source.log',root/'stack/lmcache-sink.log',
        root/'source-launches.json',root/'stack/remote-commands.json',root/'check-cache.py']
    files += sorted((root/'key-formula').rglob('*'))
    files=[p for p in files if p.is_file()]
    hashes={str(p.relative_to(root)):hashlib.sha256(p.read_bytes()).hexdigest() for p in files}
    base=(root/'key-formula/csrc/storage_backends/connector_base.h').read_text()
    cpp=(root/'key-formula/csrc/storage_backends/redis/connector.cpp').read_text()
    assert 'ConnectorBase(num_workers, WorkerPoolConfig{})' in base and 'for (int i = 0; i < num_workers_; i++)' in base
    loop=base[base.index('void worker_loop_for_queue('):base.index('void handle_tile_completion(')]
    assert loop.count('create_connection()')==1 and loop.index('create_connection()')<loop.index('for (;;)')
    assert ': ConnectorBase(num_workers)' in cpp
    scm=json.loads((root/'key-formula/lmcache-scm-version.json').read_text());assert scm['node']=='g979719d7' and not scm['dirty']
    for role in ('source','sink'):
        assert len(re.findall(r'Created RESP L2 adapter:.*\(workers=8\)',(root/f'stack/lmcache-{role}.log').read_text()))==1
    events=[json.loads(x) for x in files[0].read_text().splitlines()]
    start=min(e['monotonic_ns'] for e in events if e['event']=='request_start' and e['request_id']=='source_warm')
    end=min(e['monotonic_ns'] for e in events if e['event']=='copy_start')
    exclusive=[r for r in transfers if start<=int(r['start_ns']) and int(r['end_ns'])<=end]
    source={r['connection_id'] for r in exclusive};all_ids={r['connection_id'] for r in transfers}
    assert len(source)==8 and len(all_ids)==16 and {r['command'] for r in exclusive}<={'SET','EXISTS'}
    lifetimes=list(csv.DictReader((root/'stack/proxy_connections.csv').open()))
    closed=[r for r in lifetimes if r['connection_id'] in all_ids]
    assert all(int(r['end_ns'])>=max(int(t['end_ns']) for t in transfers if t['connection_id']==r['connection_id']) for r in closed)
    proof={'verified':True,'source_connection_ids':sorted(source),'destination_connection_ids':sorted(all_ids-source),
        'exclusive_source_warm_window_ns':[start,end],'exclusive_source_commands':dict(__import__('collections').Counter(r['command'] for r in exclusive)),
        'source_warm_requests':[{'session':e['session_id'],'route_port':e['route_port'],'event':e['event'],'monotonic_ns':e['monotonic_ns'],
            'request_id':e['request_id'],'context_hash':e['context_hash']} for e in events if e['event'] in ('request_start','request_end') and start<=e['monotonic_ns']<=end],
        'observed_pool_connections':len(all_ids),'closed_pool_connections':closed,
        'logic':'Cache-integrity requests bypassed LMCache. The first paired source-warm phase precedes any destination copy and identifies all eight source connections. The installed Redis backend creates exactly eight workers, one persistent TCP connection per worker before its operation loop, without reconnect. Both once-started pools have eight workers and all sixteen observed IDs remain distinct; the other eight therefore belong to Germany. Source-local and Germany GET traffic both traverse the billed proxy.',
        'source_commit':scm,'source_urls':['https://raw.githubusercontent.com/LMCache/LMCache/979719d7/'+p for p in ('csrc/storage_backends/connector_base.h','csrc/storage_backends/redis/connector.cpp','csrc/storage_backends/redis/connector.h')],
        'input_sha256':hashes,'physical_wire_limit':'Destination RESP request/response bytes exclude TCP/IP, TLS/SSH tunnel and link-layer overhead.'}
    binary=Path('/tmp/qh-replay-completion-runtime/lib/python3.12/site-packages/lmcache/lmcache_redis.cpython-312-x86_64-linux-gnu.so')
    if binary.exists():proof['installed_redis_binary_sha256']=hashlib.sha256(binary.read_bytes()).hexdigest()
    (root/'attribution-proof.json').write_text(json.dumps(proof,indent=2)+'\n')
    return proof


def episode(result, requests, transfers, model, origins=None):
    name=result['spec']['episode'];start,end=result['epoch_ns'],result['boundary_ns']
    selected=[r for r in requests if r.get('episode')==name]
    events=[]
    for row in selected:
        if 'full_prompt_token_ids' not in row:continue
        sid=f"{row['cohort'] if row['cohort']=='resident' else 'incoming'}-{row['session']}"
        label='kv_transfer_'+row['phase'] if result['spec']['arm']=='kv_transfer' and row['phase'] in ('initial','catch_up') else row['phase']
        events.append({'event':'rendered_request','session_id':sid,'request_label':label,'token_ids':row['full_prompt_token_ids']})
    for row in result.get('migration_events',[]):
        if row['kind'] not in ('initial_start','initial_end','catch_up_start','catch_up_end'):continue
        phase,edge=row['kind'].rsplit('_',1)
        events.append({'event':'copy_'+edge,'session_id':f"incoming-{row['session']}",'phase':phase,'monotonic_ns':row['monotonic_ns']})
    observed=[r for r in transfers if int(r['start_ns'])<end and int(r['end_ns'])>start]
    totals=keyed.attribute(events,observed,model)
    sets=[r for r in observed if r['command']=='SET']
    destination=[r for r in observed if origins and r['connection_id'] in origins['destination_connection_ids']]
    source=[r for r in observed if origins and r['connection_id'] in origins['source_connection_ids']]
    return {'episode':name,'arm':result['spec']['arm'],'workload':result['spec']['workload'],'seed':result['spec']['seed'],
        'window_ns':[start,end],**totals,'set_records':len(sets),
        'set_request_body_bytes':sum(int(r['request_body_bytes']) for r in sets),
        'set_wire_bytes':sum(int(r['request_wire_bytes'])+int(r['response_wire_bytes']) for r in sets),
        'all_resp_wire_bytes':sum(int(r['request_wire_bytes'])+int(r['response_wire_bytes']) for r in observed),
        'request_cache_tiers':[e for e in result.get('migration_events',[]) if e['kind']=='cache_request_tiers'],
        'destination_get':keyed.get_bytes(destination) if origins else None,
        'source_local_get':keyed.get_bytes(source) if origins else None,
        'destination_phase_attribution':keyed.attribute(events,destination,model) if origins else None,
        'physical_wan_get_payload_bytes':sum(int(r['payload_bytes']) for r in destination if r['command']=='GET') if origins else None,
        'connection_origin_proof':'attribution-proof.json' if origins else None,
        'origin_limit':('Persistent connection origins identified by attribution-proof.json; destination RESP bytes exclude TCP/IP and tunnel overhead.' if origins else 'The shared proxy records key/command/time, not client peer identity. Both source and destination LMCache can issue GETs; byte totals are observed RESP GET traffic, not a verified physical-WAN measurement.'),
        'attribution_limit':'Copy windows include source-export probes and destination prefetch/validation. The destination_phase_attribution view uses the proven Germany connection subset; top-level phase views retain both origins. Unmatched service-period GETs and ambiguous/incomplete phases remain explicit. Origin GET summaries include zero-payload response protocol; top-level keyed views describe positive-payload GETs.',
        'window_limit':'Whole completed RESP records intersecting observation are counted; no prorating. SET body bytes include command/key/value and are not counted as transferred destination KV payload.'}


def reduce(root):
    hashes={};partial=[]
    def raw(path):
        data=path.read_bytes();hashes[str(path.relative_to(root))]=hashlib.sha256(data).hexdigest();return data
    data=raw(root/'requests.jsonl')
    if data and not data.endswith(b'\n'):
        offset=data.rfind(b'\n')+1;partial.append({'path':'requests.jsonl','unparsed_final_bytes':len(data)-offset});data=data[:offset]
    requests=[json.loads(r) for r in data.splitlines()]
    data=raw(root/'stack/resp_transfers.csv')
    if data and not data.endswith(b'\n'):
        offset=data.rfind(b'\n')+1;partial.append({'path':'stack/resp_transfers.csv','unparsed_final_bytes':len(data)-offset});data=data[:offset]
    transfers=list(csv.DictReader(io.StringIO(data.decode())))
    log=raw(root/'stack/lmcache-source.log').decode();models=set(re.findall(r'\(model=(.*?), world_size=(\d+)\)',log))
    assert len(models)==1 and next(iter(models))[1]=='1' and 'chunk_size=256, hash_algorithm=blake3' in log
    model=next(iter(models))[0]
    raw(root/'keyed-wire.py')
    for path in (root/'key-formula').rglob('*.py'):raw(path)
    origins=connection_origins(root,transfers)
    raw(root/'attribution-proof.json')
    episodes=[episode(json.loads(raw(p)),requests,transfers,model,origins) for p in sorted(root.glob('episodes-*/result.json'))]
    return {'episodes':episodes,'input_sha256':hashes,'partial_input_tails':partial,
        'model':model,'key_formula':'Archived keyed-wire.py: blake3 rolling256-token chunks; TP1 rank01000100, group0, empty salt; source code archived in key-formula.',
        'source_sha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),'simulator_fitting':False}


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('root',type=Path);args=parser.parse_args()
    value=reduce(args.root);(args.root/'service-wire-analysis.json').write_text(json.dumps(value,indent=2)+'\n')
    print(json.dumps([{'episode':r['episode'],**r['scenario'],'unmapped':r['ownership_unmapped_records'],'multiple_owners':r['ownership_multiple_session_records']} for r in value['episodes']],indent=2))
