"""Reconstruct recorded RESP key ownership offline; no engine or GPU access."""
import argparse
import collections
import csv
import hashlib
import json
import re
import struct
from pathlib import Path

import blake3


def keys(tokens, model, salt=''):
    prefix = blake3.blake3(struct.pack('>qI', 0, 0)).digest()
    for offset in range(0, len(tokens)//256*256, 256):
        prefix = blake3.blake3(prefix+struct.pack('>256I', *tokens[offset:offset+256])).digest()
        wire = f'{model}@01000100@0@{prefix.hex()}' + (f'@{salt}' if salt else '')
        yield hashlib.sha256(wire.encode()).hexdigest()


def payload(rows):
    unique = {}
    for row in rows:
        key, size = row['key_hashes'], int(row['payload_bytes'])
        if key in unique and unique[key] != size:raise ValueError('inconsistent payload size for identical key')
        unique[key] = size
    actual = sum(int(r['payload_bytes']) for r in rows)
    wire = sum(int(r['request_wire_bytes'])+int(r['response_wire_bytes']) for r in rows)
    return {'get_records':len(rows),'unique_keys':len(unique),'unique_payload_bytes':sum(unique.values()),
        'total_payload_bytes':actual,'retransferred_payload_bytes':actual-sum(unique.values()),
        'protocol_bytes':wire-actual,'wire_bytes':wire,'payload_sizes':sorted(set(unique.values()))}


def get_bytes(rows):
    rows=[r for r in rows if r['command']=='GET'];summary=payload([r for r in rows if int(r['payload_bytes'])>0])
    request=sum(int(r['request_wire_bytes']) for r in rows);response=sum(int(r['response_wire_bytes']) for r in rows)
    return {**summary,'all_get_records':len(rows),'zero_payload_get_records':sum(int(r['payload_bytes'])==0 for r in rows),
        'request_protocol_bytes':request,'response_protocol_bytes':response-summary['total_payload_bytes'],
        'protocol_bytes':request+response-summary['total_payload_bytes'],'wire_bytes':request+response}


def attribute(events, transfers, model):
    owners, expected = collections.defaultdict(set), {}
    for event in events:
        if event['event']=='rendered_request':
            request_keys=set(keys(event['token_ids'],model))
            for key in request_keys:owners[key].add(event['session_id'])
            if event.get('request_label') in ('kv_transfer_initial','kv_transfer_catch_up'):
                expected[(event['session_id'],event['request_label'].removeprefix('kv_transfer_'))]=request_keys
    windows = []
    for event in events:
        if event['event']!='copy_start':continue
        ends = [e['monotonic_ns'] for e in events if e['event']=='copy_end' and e['session_id']==event['session_id'] and e['phase']==event['phase'] and e['monotonic_ns']>=event['monotonic_ns']]
        if ends:windows.append((event['session_id'],event['phase'],event['monotonic_ns'],min(ends)))
    gets = [r for r in transfers if r['command']=='GET' and int(r['payload_bytes'])>0]
    groups, ambiguous = collections.defaultdict(list), []
    for index,row in enumerate(gets):
        sessions = owners[row['key_hashes']]
        phases = [(s,p) for s,p,a,z in windows if s in sessions and a<=int(row['start_ns'])<=int(row['end_ns'])<=z]
        if len(sessions)!=1 or len(phases)!=1:
            ambiguous.append({'get_index':index,'key_hash':row['key_hashes'],'candidate_sessions':sorted(sessions),
                'candidate_phases':phases,'payload_bytes':int(row['payload_bytes'])})
        if len(sessions)==1:
            sid = next(iter(sessions));groups[(sid,'all')].append(row)
            if len(phases)==1:groups[phases[0]].append(row)
    lanes=[]
    for (sid,phase),rows in sorted(groups.items()):
        wanted=expected.get((sid,phase));observed={r['key_hashes'] for r in rows}
        starts=[a for s,p,a,z in windows if s==sid and p==phase]
        previous={r['key_hashes'] for r in gets if starts and int(r['end_ns'])<=min(starts)}
        lanes.append({'session':sid,'phase':phase,**payload(rows),
            'crossphase_previously_seen_payload_bytes':sum(int(r['payload_bytes']) for r in rows if r['key_hashes'] in previous) if starts else None,
            'expected_full_prefix_keys':len(wanted) if wanted is not None else None,
            'matching_full_prefix_keys_transferred':len(wanted&observed) if wanted is not None else None,
            'unexpected_keys_transferred':sorted(observed-wanted) if wanted is not None else None,
            'initial_full_rounded_prefix_transfer':wanted==observed if phase=='initial' and wanted is not None else None})
    return {'scenario':payload(gets),'lanes':lanes,
        'unresolved_get_records':ambiguous,'ownership_unmapped_records':sum(not owners[r['key_hashes']] for r in gets),
        'ownership_multiple_session_records':sum(len(owners[r['key_hashes']])>1 for r in gets),
        'phase_attribution_complete':not ambiguous,
        'scope':'Each positive GET counted once in scenario totals. Lane all totals and phase totals are separate views; never sum both. Same keys may legitimately appear across phases; total-minus-unique counts retransferred payload. Unknown ownership or incomplete/ambiguous phase windows retained explicitly.'}


def reduce(root, package):
    provenance = {}
    def raw(path):
        data=path.read_bytes();provenance[str(path)]=hashlib.sha256(data).hexdigest();return data
    sources = [package/'lmcache/v1/multiprocess/token_hasher.py',package/'lmcache/v1/distributed/l2_adapters/native_connector_l2_adapter.py',package/'lmcache/v1/distributed/api.py']
    code = [raw(p).decode() for p in sources]
    assert 'prefix_hash.to_bytes(8, byteorder="big", signed=True)' in code[0] and 'struct.pack(f">{len(tokens)}I", *tokens)' in code[0]
    assert 'key.kv_rank:08x' in code[1] and 'key.object_group_id:x' in code[1] and 'key.chunk_hash.hex()' in code[1]
    log=raw(root.parent/'stack/lmcache-source.log').decode()
    assert 'chunk_size=256, hash_algorithm=blake3' in log
    models=set(re.findall(r'\(model=(.*?), world_size=(\d+)\)',log));assert len(models)==1 and next(iter(models))[1]=='1'
    model=next(iter(models))[0];scenarios=[]
    proof=root.parent/'attribution-proof.json';origins=json.loads(raw(proof)) if proof.exists() else None
    for folder in sorted(root.glob('paired-*')):
        if not folder.is_dir() or not (folder/'resp_transfers.csv').exists():continue
        data=raw(folder/'events.jsonl')
        if data and not data.endswith(b'\n'):raise ValueError(f'partial event record: {folder}')
        events=[json.loads(x) for x in data.splitlines()]
        rows=list(csv.DictReader(raw(folder/'resp_transfers.csv').decode().splitlines()))
        value={'scenario_id':folder.name,**attribute(events,rows,model)}
        if origins:
            source=set(origins['source_connection_ids']);destination=set(origins['destination_connection_ids'])
            assert origins['verified'] and not source&destination and {r['connection_id'] for r in rows}<=(source|destination)
            dest_rows=[r for r in rows if r['connection_id'] in destination]
            value.update(destination_get=get_bytes(dest_rows),source_local_get=get_bytes([r for r in rows if r['connection_id'] in source]),
                destination_phase_attribution=attribute(events,dest_rows,model),connection_origin_proof=str(proof))
        scenarios.append(value)
    return {'model':model,'chunk_tokens':256,'world_size':1,'kv_rank_hex':'01000100','object_group_id':0,'salt':'',
        'method':'Frozen LMCache blake3 rolling full-token chunks, native ObjectKey serialization, proxy SHA256. All observed GETs checked against recorded rendered prompts; no GPU requests.',
        'scenarios':scenarios,'input_sha256':provenance,'source_code_sha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest()}


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('root',type=Path)
    parser.add_argument('--package',type=Path,default=Path(__file__).resolve().parent/'key-formula')
    args=parser.parse_args();result=reduce(args.root,args.package)
    (args.root/'keyed-wire-analysis.json').write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps([{k:s[k] for k in ('scenario_id','scenario','ownership_unmapped_records','ownership_multiple_session_records','phase_attribution_complete')} for s in result['scenarios']],indent=2))
