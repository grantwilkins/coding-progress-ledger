"""Assemble the bounded completion evidence without additional measurements."""
import hashlib
import json
import subprocess
from pathlib import Path

out = Path(__file__).resolve().parent
read = lambda name: json.loads((out/name).read_text())
write = lambda name, value: (out/name).write_text(json.dumps(value,indent=2,allow_nan=False)+'\n')
prior = out.parent/'a100-replay-live-20260909T1920'
facts = read('paired-final-facts.json')
start = read('supplemental-start.json')
source_stop = read('source-final-stop.json')
assert source_stop['elapsed_s'] < 1800
assert facts['counts']['paired']['scenario_statuses'] == {'failed':5,'complete':3}
assert facts['counts']['paired-corrected']['scenario_statuses'] == {'timeout':1,'unmeasured_after_timeout':1}
report = {
    'campaign_ready':False,
    'recommendation':'Do not launch full simulations as validated SLO-feasible shedding. The bounded acquisition and corrective work are complete; the missing measurements remain explicit.',
    'scope':'Initial evidence is preserved. This user-authorized continuation adds bounded controlled-source measurements, not a replacement agentic resident campaign.',
    'initial_acquisition':{'report':str(prior.relative_to(Path.cwd())/'report.json'),'elapsed_s':8914.842082686,'cap_s':9000},
    'supplemental_budget':{'plan':'supplemental-plan.json','start':start,'source_stop':source_stop,
        'destination_stop':'destination-reconnected-cleanup.json','cap_s':1800,
        'scope':'Wall time includes startup corrections and cleanup. Both GPUs released before cutoff; this is not GPU busy time.'},
    'observations':facts,
    'corrections':[
        {'commit':'7a4ce911','file':'migration_profiler.py','change':'Reject failed source activity before returning idle state and allowing ownership commit.',
         'evidence':'paired/paired-7101-8192-replay/events.jsonl: invalid source final response was followed by a stale-generation switch; entire old episode remains invalid.',
         'validation':'80 migration/driver tests passed; wait-idle-fix.patch and regression evidence retained.'},
        {'commit':'e0f197bd','file':'lmcache_compat/connector_patch.py','change':'Preserve aligned native-prefix hit count before external lookup can return zero.',
         'evidence':'A 2048-token source request with64native hits exported1792tokens. native-prefix-export-diagnosis.json reproduces the missing256-token tail. Corrected source block exported256initialchunks plus8appendchunks.',
         'validation':'9 compatibility tests passed; corrected destination reported8192external cached tokens. Completed KV handoff/timing validation remains open.'}],
    'simulator_changes':[],
    'simulator_verification':{'evaluations':20,'rerun':False,'reason':'No simulator coefficients or policy rules changed; reuse the completed bounded comparison and existing holdouts.',
        'table':str(prior.relative_to(Path.cwd())/'policy-comparison.csv'),'source_gpus':6666,'destination_gpus_each':6666,
        'gpus_per_node':8,'shared_wan_gbps':1000,'effective_wire_bytes_per_32768':800000000,
        'native_geometry_bytes_per_32768':1610612736,'resident_memory_accounting':'unchanged','latency_validated':False},
    'wire_interpretation':'Observed256-token storage chunks contain12582912payload bytes. This agrees with49152native bytes/token. Completed GET payloads, RESP protocol responses and flushed proxy buckets remain separate; incomplete buckets are not exact wire accounting. No successful complete migration validates the800000000-byte effective-wire anchor.',
    'timing_interpretation':'Use per-request exact token events including reasoning. The corrected block has16exact complete requests of17started (94.12%); its unfinished destination request cannot support a99%-coverage latency conclusion. Controller/proxy timestamps share the source clock; destination engine/power samples retain their own clock.',
    'gaps':[
        {'gap':'Full-context replay and naturally warm catch-up','status':'verified locally, including valid source-evolved controlled append cases','missing':'Matched recorded agentic service with physical inventory and cache placement.'},
        {'gap':'Failed source state reaching ownership switch','status':'corrected and regression-tested','missing':None},
        {'gap':'Native-prefix hit lost during KV export','status':'corrected, unit-tested and live export/retrieval observed','missing':'Completed paired migration repeats after the fix.'},
        {'gap':'Source quiescence under continuing traffic','status':'source activity overlapped initial migration; valid measured pauses were already idle','missing':'Positive in-flight wait and preserved queued arrivals using evolving recorded agentic demand.'},
        {'gap':'Paired KV completion and0.80GBwire anchor','status':'partial native retrieval observed; zero validated complete KV handoffs','missing':'Clean paired KV repeats with full validation responses, payload/protocol accounting and no censoring; both corrected long cases and second short repeat remain unmeasured.'},
        {'gap':'Resident service normalization and physical cache placement','status':'open; initial scout rates retained as tested points, not capacity fractions','missing':'Inventory-matched agentic rate bracket and shared-service control/replay/KV episodes.'},
        {'gap':'Divisible GPU work, latency and recovery mapping','status':'local packing/queue counterexamples preserved; no fleet mapping validated','missing':'Matched physical placement and longer paired recovery/latency windows with at least99%usable exact timing.'},
        {'gap':'Full SLO-feasible five-policy campaign','status':'not ready','missing':'Resolve the preceding service, placement and paired-KV gaps; a policy handoff fraction is not latency feasibility.'}],
    'prerequisite_failures':['Wrong source cache port at first startup; preserved in invalid-source-port-startup.',
        'Immutable configuration assignment failed before requests; old plan retained.',
        'Cold response omitted cache field; it stays unknown, with independent engine cache counters.',
        'Destination Redis pool lost connections across source startup correction; identical-config restart restored8connections; native adapter discarded the historical error string.',
        'First source reload omitted MP environment and failed initialization; log retained and environment corrected within the same cutoff.',
        'Corrected acquisition reached its global cutoff after14.6seconds; timeout remains failure and second repeat unmeasured.'],
    'runtime':{'source':'source-runtime.json','destination':'destination-runtime.json','identities':'inventory.json',
        'differences':'Both vLLM0.22.0/LMCache0.5.1; sourceTransformers5.17.0 versus destination5.15.1. Tested prompt rendered identical8192token IDs. Same FIFO event patch; unpatched performance equivalence is not established.',
        'model_hashes':'destination-model-comparison.json','exact_launches':['source-launches.json','source-engine-reload.json','stack/remote-commands.json','stack/remote-reconnect-launch.json','paired/paired-launch.json','paired-corrected/paired-launch.json']},
    'tests':{'migration_guard':'80passed: wait-idle-fix-tests.log','connector_export':'9passed: native-prefix-export-host-tests.log',
        'package':'14passed: final-package-tests.log','initial_focused_simulator':'181passed,3deselected in previous acquisition; original instrumentation timing failure remains retained there.',
        'additional':'Per-task driver/reducer/FIFO test records retained; sandbox-only permissions or stalled attempts are not claimed as passes.'},
    'reproduce':['python3 outputs/a100-replay-completion-20260910T0116/reduce-paired.py outputs/a100-replay-completion-20260910T0116/paired',
        'python3 outputs/a100-replay-completion-20260910T0116/reduce-paired.py outputs/a100-replay-completion-20260910T0116/paired-corrected',
        'python3 outputs/a100-replay-completion-20260910T0116/combine-paired.py',
        'python3 outputs/a100-replay-completion-20260910T0116/finalize.py']}
write('report.json',report)
write('provenance-final.json',{'commit_before_package_commit':subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip(),
    'dirty':subprocess.check_output(['git','status','--porcelain'],text=True),
    'baseline':'e68f1d26','initial_completion_commit':'a2177eae','autostash_fix':'35be3279'})
write('artifact-sha256.json',{str(p.relative_to(out)):hashlib.sha256(p.read_bytes()).hexdigest()
    for p in sorted(out.rglob('*')) if p.is_file() and p.name!='artifact-sha256.json' and '__pycache__' not in p.parts})
print('Final report and artifact hashes written.')
