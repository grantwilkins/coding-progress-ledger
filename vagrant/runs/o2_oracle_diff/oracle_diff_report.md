# O2 — oracle-vs-policy plan diff

## tiny_prefill_pressure
- cell: n=4, scale=tiny, prefill=tight, link=100 Gbps
- gaps: oracle vs mixed = 35.0%; oracle vs random = 50.2%; strong vs random = -13.7%
- p50 (s): oracle=0.0613, mixed=0.0943, strong=0.14, random=0.123

Per-policy bottleneck fractions (time-weighted; `attr` = fraction of makespan with an attributed bottleneck — low values mean the breakdown describes a small slice):
- `oracle`: network=6%, prefill=79%, workspace=15%, kv_memory=0% (attr=100%)
- `mixed_min_pressure`: network=65%, prefill=0%, workspace=35%, kv_memory=0% (attr=100%)
- `random_mode`: network=8%, prefill=79%, workspace=13%, kv_memory=0% (attr=100%)
- `strong_reuse`: network=81%, prefill=0%, workspace=19%, kv_memory=0% (attr=100%)

Per-workflow oracle vs mixed differences: dst=2/4, prompt_mode=2/4, workspace_mode=3/4

## medium_multi_resource
- cell: n=4, scale=medium, prefill=tight, link=5 Gbps
- gaps: oracle vs mixed = 50.0%; oracle vs random = 80.3%; strong vs random = -1.7%
- p50 (s): oracle=0.735, mixed=1.47, strong=3.8, random=3.73

Per-policy bottleneck fractions (time-weighted; `attr` = fraction of makespan with an attributed bottleneck — low values mean the breakdown describes a small slice):
- `oracle`: network=67%, prefill=7%, workspace=26%, kv_memory=0% (attr=100%)
- `mixed_min_pressure`: network=14%, prefill=39%, workspace=47%, kv_memory=0% (attr=100%)
- `random_mode`: network=96%, prefill=4%, workspace=0%, kv_memory=0% (attr=100%)
- `strong_reuse`: network=75%, prefill=25%, workspace=0%, kv_memory=0% (attr=100%)

Per-workflow oracle vs mixed differences: dst=2/4, prompt_mode=2/4, workspace_mode=1/4

## monorepo_workspace_pressure
- cell: n=4, scale=monorepo, prefill=loose, link=100 Gbps
- gaps: oracle vs mixed = 0.7%; oracle vs random = 33.7%; strong vs random = -33.6%
- p50 (s): oracle=9.98, mixed=10, strong=20.1, random=15

Per-policy bottleneck fractions (time-weighted; `attr` = fraction of makespan with an attributed bottleneck — low values mean the breakdown describes a small slice):
- `oracle`: network=0%, prefill=0%, workspace=100%, kv_memory=0% (attr=100%)
- `mixed_min_pressure`: network=0%, prefill=1%, workspace=99%, kv_memory=0% (attr=100%)
- `random_mode`: network=0%, prefill=0%, workspace=100%, kv_memory=0% (attr=100%)
- `strong_reuse`: network=0%, prefill=0%, workspace=100%, kv_memory=0% (attr=100%)

Per-workflow oracle vs mixed differences: dst=2/4, prompt_mode=2/4, workspace_mode=0/4

## slow_link_network_pressure
- cell: n=4, scale=medium, prefill=loose, link=1 Gbps
- gaps: oracle vs mixed = 49.9%; oracle vs random = 93.5%; strong vs random = -97.4%
- p50 (s): oracle=0.533, mixed=1.06, strong=16.1, random=8.13

Per-policy bottleneck fractions (time-weighted; `attr` = fraction of makespan with an attributed bottleneck — low values mean the breakdown describes a small slice):
- `oracle`: network=92%, prefill=0%, workspace=7%, kv_memory=0% (attr=100%)
- `mixed_min_pressure`: network=0%, prefill=9%, workspace=91%, kv_memory=0% (attr=100%)
- `random_mode`: network=99%, prefill=1%, workspace=0%, kv_memory=0% (attr=100%)
- `strong_reuse`: network=99%, prefill=1%, workspace=0%, kv_memory=0% (attr=100%)

Per-workflow oracle vs mixed differences: dst=2/4, prompt_mode=2/4, workspace_mode=0/4

