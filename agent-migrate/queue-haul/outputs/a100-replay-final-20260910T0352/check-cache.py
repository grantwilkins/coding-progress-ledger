"""Verify native cache and both renderers through the original full chat probe."""
import json,time
from dataclasses import replace
from pathlib import Path
import sys
sys.path.insert(0,str(Path.cwd()))
from pool_replay_measure import Acquisition,cache_integrity,write
import migration_testbed as b
out=Path(__file__).resolve().parent
a=Acquisition(out);inventory=json.loads((out/'inventory.json').read_text());a.cfg=b.Config(**inventory['config']);a.deadline=time.monotonic()+(inventory['deadline_wall_ns']-time.time_ns())/1e9
messages,code=a.history('final-native-cache-integrity-7101',8192)
source=b.mp_chat_tokens(a.cfg,a.probe(messages,code));destination=b.mp_chat_tokens(replace(a.cfg,src_port=a.cfg.sink_port),a.probe(messages,code))
write(out/'render-comparison.json',{'same_token_ids':source==destination,'source_tokens':source,'destination_tokens':destination})
if source!=destination:raise RuntimeError('source and destination full chat rendering differ')
before=a.engine_metrics();cold=a.chat(messages,code,{'episode':'cache-integrity','phase':'cold'},'final-native-7101');after=a.engine_metrics()
warm=a.chat(messages,code,{'episode':'cache-integrity','phase':'shared_prefix'},'final-native-7101')
check=cache_integrity(cold,warm,before,after)
write(out/'cache-integrity.json',check|{'engine_before':before,'engine_after_cold':after,'engine_after_warm':a.engine_metrics()})
if not check['passed']:raise RuntimeError('cold/shared-prefix cache telemetry verification failed')
print(json.dumps(check))
