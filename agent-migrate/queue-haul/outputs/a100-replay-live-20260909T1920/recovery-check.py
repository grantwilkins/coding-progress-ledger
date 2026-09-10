import json,sys,time
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[2]))
from pool_replay_measure import Acquisition,cache_integrity,write
import migration_testbed as b
import destination_runner as d
out=Path(__file__).resolve().parent;a=Acquisition(out)
b.wait_health(a.cfg.host,a.cfg.sink_port,min(360,a.remaining()))
messages,code=a.history('recovered-cache-integrity',2048)
before=a.engine_metrics();cold=a.chat(messages,code,{'episode':'recovered-integrity','phase':'cold'},'recovered-integrity');after=a.engine_metrics()
warm=a.chat(messages,code,{'episode':'recovered-integrity','phase':'shared_prefix'},'recovered-integrity')
check=cache_integrity(cold,warm,before,after)
prompt=d.deterministic_tokens('recovered-event-integrity',9391,200000,7101)
body=d.completion_payload(a.cfg.model,prompt,10,None,True);body['cache_salt']='recovered-event-integrity'
events=d._completion(a.cfg.host,a.cfg.sink_port,a.cfg.model,prompt,10,None,60,True,prepared_body=json.dumps(body))
write(out/'recovered-stream-integrity.json',{'cache':check,'completion':events,'wall_ns':time.time_ns()})
assert check['passed'] and events['exact_token_timestamps'] and events['output_tokens']==10
print('Recovered cold/warm cache and exact token events verified',flush=True)
