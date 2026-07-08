# Queue-Haul Stage 1b/1c Source-Sink Proof

## Current implementation

Stage 1b/1c is now a YAGNI proof, not the full experiment driver. It proves that
the old Apptainer vLLM sandbox can store KV from a source instance, later retrieve
that KV from a sink instance through LMCache, and let a controller dispatch replay
and KV-transfer sessions in a chosen order under a deadline.

Implemented files:

- `queue-haul/stage1b_drain_sink.py`: preflight, 1-GPU smoke, 2-GPU source/sink smoke, and stdlib throttle proxy.
- `queue-haul/stage1c_controller.py`: tiny MILP-backed controller proof that emits and checks `controller_manifest.json`.
- `queue-haul/tests/test_stage1b_drain_sink.py` and `queue-haul/tests/test_stage1c_controller.py`: pure-python checks for commands, config, proxy accounting, and manifest invariants.

## Working runtime path

Use `/scratch/users/gfw/ptsim/vllm-openai-v0.10.1.1.sandbox`. Do not use
`lmcache/vllm-openai:latest-cu129` for `openai/gpt-oss-20b` on A100; local tests
hit MXFP4 Marlin/MoE failures there.

The current node does not allow privileged kernel `tc`. The implemented traffic
control is a user-space TCP proxy with one shared token bucket. It bills only the
source-egress directions used by this proof:

- replay: API proxy `client_to_target`
- KV transfer: KV proxy `target_to_client`

A diagnosis run on 2026-07-08 showed live source+sink works with the current
launcher settings. The monitor ruled out cgroup OOM (`memory.failcnt=0`), GPU
cross-contamination (source PID only on GPU 0, sink PID only on GPU 1), Ray, and
matching `/dev/shm` collisions. The proven live flow is:

1. Start LMCache server on `127.0.0.1:5655`.
2. Start the shared 1Gbps proxy on KV `8300 -> 5655` and API `8400 -> 8200`.
3. Start source vLLM on GPU 0 / port 8100.
4. Start sink vLLM on GPU 1 / port 8200 while source remains healthy.
5. Warm selected prompts on source and store KV while sink is live.
6. Dispatch replay and KV-transfer sessions serially through the controller.

The important vLLM/LMCache settings are hard-coded or tested: `LMCacheConnectorV1`,
short per-role `VLLM_RPC_BASE_PATH`, role-specific `TMPDIR`, distinct `kv_port`,
distinct `lmcache_rpc_port`, `LMCACHE_LMCACHE_INSTANCE_ID`, `--enforce-eager`, and
no `--async-scheduling`.

## Commands

```bash
module load gcc/14.2.0 openblas/0.3.28
PY=.venv/bin/python

$PY queue-haul/stage1b_drain_sink.py preflight --required-gpus 2
$PY queue-haul/stage1b_drain_sink.py smoke2-live --mbps 1000 --run-root /tmp/qh-smoke2-live

$PY queue-haul/stage1c_controller.py plan
$PY queue-haul/stage1c_controller.py proof --mbps 1000 --run-root /tmp/qh-proof
$PY queue-haul/stage1c_controller.py check --run-root /tmp/qh-proof
```

`uv` is not on PATH on the current A100 node. Use `.venv/bin/python -m pytest`
after loading the modules above.

## Acceptance

Stage 1b smoke hard-fails unless the source stores KV, the sink retrieves KV via
the proxy, replay does not pull large KV bytes, and the proxy byte log proves the
1Gbps-shaped directions.

Stage 1c proof hard-fails unless the solver is feasible with zero shortfall, the
fixture yields both replay and KV actions, dispatch ranks execute serially, every
session returns HTTP 200 before the deadline, replay has API bytes, and KV transfer
has KV bytes.

A successful controller manifest has this shape:

- `schema = queue-haul-stage1c-v1`
- `acceptance.ok = true`
- one `R` session before one `S` session for the default fixture
- replay bytes on `api/client_to_target`
- KV bytes on `kv/target_to_client`
- `smoke2.acceptance.ok = true`, covered by the controller sessions

## Not in this stage

The workload generator, power logger/reducer, multi-destination routing, real
kernel `tc`/network namespaces, and
full queue-haul experiment driver are later work. The current code is only the
minimum proof that the controller can choose an order and that live replay/KV execution
paths are observable over a shaped link.
