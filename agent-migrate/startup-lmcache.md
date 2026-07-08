# Guidebook: live vLLM source-sink with LMCache

Use this path on the current two-A100 node to start live source and sink vLLM+LMCache instances and prove both online KV transfer and online full-context replay through the 1Gbps proxy. Do not use the `latest-cu129` LMCache image for `openai/gpt-oss-20b` on A100; it failed in MXFP4 Marlin/alternate MoE backends. The working path is the older vLLM sandbox with LMCache wrapped around it.

## On-demand command path

```bash
cd /home/groups/ramr/gfw/coding-progress-ledger/agent-migrate
module load gcc/14.2.0 openblas/0.3.28
PY=.venv/bin/python

$PY queue-haul/stage1b_drain_sink.py preflight --required-gpus 2
$PY queue-haul/stage1b_drain_sink.py smoke2-live --mbps 1000 --run-root /tmp/qh-smoke2-live

$PY queue-haul/stage1c_controller.py plan
$PY queue-haul/stage1c_controller.py proof --mbps 1000 --run-root /tmp/qh-proof-live
$PY queue-haul/stage1c_controller.py check --run-root /tmp/qh-proof-live
```

Passing those commands proves the full end-to-end live path: source and sink are alive together, source stores KV, sink retrieves KV through the shaped KV proxy, sink replays full context through the shaped API proxy, and the controller orders replay and KV-transfer sessions under deadline.

## Fixed paths

```bash
OLD=/scratch/users/gfw/ptsim/vllm-openai-v0.10.1.1.sandbox
HF_HOME=/scratch/users/gfw/ptsim/hf
VLLM_CACHE=/scratch/users/gfw/ptsim/cache/old-vllm-lmcache
```

Expected versions inside `OLD`:

```bash
apptainer exec --nv --bind /scratch/users/gfw:/scratch/users/gfw "$OLD" \
  bash -lc 'python3 - <<PY
import vllm, lmcache, torch
print("vllm", vllm.__version__)
print("lmcache", lmcache.__file__)
print("cuda", torch.cuda.is_available(), torch.cuda.get_device_name(0))
PY'
```

Known-good output shape:

```text
vllm 0.10.1.1
lmcache /usr/local/lib/python3.12/dist-packages/lmcache/__init__.py
cuda True NVIDIA A100-SXM4-80GB
```

## Manual single-instance startup

The commands below are for manual debugging of one vLLM+LMCache instance. The on-demand proof commands above start LMCache and vLLM automatically. Use port `5655`; port `5555` may already be occupied.

```bash
apptainer exec --nv --bind /scratch/users/gfw:/scratch/users/gfw "$OLD" \
  bash -lc 'python3 -m lmcache.v1.server 127.0.0.1 5655 cpu'
```

Expected LMCache server log:

```text
Initializing cpu-only cache server
Server started at 127.0.0.1:5655
```

Leave this process running.

## Start vLLM with LMCache

These knobs are required:

- `--enforce-eager`: avoids the failing compile/JIT service path.
- `VLLM_USE_FLASHINFER_SAMPLER=0`: matches the prior successful old-sandbox runs.
- `TMPDIR=/tmp/t`: keeps LMCache ZMQ IPC paths under the Linux 107-byte Unix socket limit.
- `engine_id=e0`: also keeps LMCache IPC paths short.
- `LMCACHE_MAX_LOCAL_CPU_SIZE=0.25`: avoids `cudaHostRegister` failures seen with larger local CPU pools.

```bash
mkdir -p "$VLLM_CACHE"/{xdg,torch_extensions,torchinductor,triton,tmp}

apptainer exec --nv --bind /scratch/users/gfw:/scratch/users/gfw "$OLD" bash -lc '
mkdir -p /tmp/t
export PYTHONHASHSEED=0
export VLLM_EXECUTE_MODEL_TIMEOUT_SECONDS=900
export VLLM_USE_FLASHINFER_SAMPLER=0
export TORCH_CUDA_ARCH_LIST=8.0
export XDG_CACHE_HOME=/scratch/users/gfw/ptsim/cache/old-vllm-lmcache/xdg
export TORCH_EXTENSIONS_DIR=/scratch/users/gfw/ptsim/cache/old-vllm-lmcache/torch_extensions
export TORCHINDUCTOR_CACHE_DIR=/scratch/users/gfw/ptsim/cache/old-vllm-lmcache/torchinductor
export TRITON_CACHE_DIR=/scratch/users/gfw/ptsim/cache/old-vllm-lmcache/triton
export TMPDIR=/tmp/t
export HF_HOME=/scratch/users/gfw/ptsim/hf
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export LMCACHE_REMOTE_URL=lm://127.0.0.1:5655
export LMCACHE_REMOTE_SERDE=naive
export LMCACHE_CHUNK_SIZE=256
export LMCACHE_MAX_LOCAL_CPU_SIZE=0.25
vllm serve openai/gpt-oss-20b \
  --host 127.0.0.1 \
  --port 8120 \
  --served-model-name openai/gpt-oss-20b \
  --tensor-parallel-size 1 \
  --max-model-len 32768 \
  --max-num-seqs 256 \
  --max-num-batched-tokens 8192 \
  --kv-cache-dtype auto \
  --enable-chunked-prefill \
  --enforce-eager \
  --kv-transfer-config "{\"kv_connector\":\"LMCacheConnectorV1\",\"engine_id\":\"e0\",\"kv_role\":\"kv_both\",\"kv_connector_extra_config\":{\"discard_partial_chunks\":false}}"
'
```

Expected startup milestones:

```text
Creating v1 connector with name: LMCacheConnectorV1 and engine_id: e0
Connected to remote storage at lm://127.0.0.1:5655
Loading safetensors checkpoint shards: 100% Completed
Model loading took 13.7194 GiB
Starting vLLM API server 0 on http://127.0.0.1:8120
```

## Validate

```bash
curl -fsS http://127.0.0.1:8120/health
curl -fsS http://127.0.0.1:8120/v1/models
curl -fsS http://127.0.0.1:8120/v1/chat/completions \
  -H 'Content-Type: application/json' \
  -d '{"model":"openai/gpt-oss-20b","messages":[{"role":"user","content":"Reply with exactly: OK"}],"max_tokens":128,"temperature":0}'
```

The final response should include:

```json
"content":"OK"
```

LMCache proof in vLLM logs:

```text
Stored 76 out of total 76 tokens
LMCache hit tokens: 76
Retrieved 76 out of 76 out of total 76 tokens
```


## Source-sink proof runbook

This is the proven end-to-end path for the Queue-Haul Stage 1b/1c proof. It uses
two live vLLM+LMCache instances on one A100 node. Source runs on GPU 0, sink runs
on GPU 1, and the sink retrieves KV written by the source instance through the
shared 1Gbps proxy.

What this proves:

- Source vLLM on GPU 0 can store KV for `openai/gpt-oss-20b` into LMCache.
- Sink vLLM on GPU 1 can retrieve that stored KV through the proxy and continue.
- The sink can also receive full context through the API proxy and replay/prefill.
- A controller can choose a serial dispatch order containing both replay and KV
  transfer actions.
- The proof uses a user-space 1Gbps token-bucket delay function shared by replay
  request bytes and KV-transfer bytes. This is not privileged kernel `tc`.

What this does not prove yet: multi-destination routing, workload realism,
power-down traces, or kernel/network-namespace traffic shaping.

A 2026-07-08 live diagnostic ruled out the likely false explanations for earlier
failures on this allocation: cgroup OOM (`memory.failcnt=0`), sink allocation on
GPU 0, Ray, and matching `/dev/shm` artifacts.

### 1. Start from a clean node

```bash
cd /home/groups/ramr/gfw/coding-progress-ledger/agent-migrate
module load gcc/14.2.0 openblas/0.3.28
PY=.venv/bin/python

pgrep -af 'stage1|vllm|lmcache|proxy_bytes|api_server|EngineCore' || true
ss -ltnp | rg '8100|8120|8200|8300|8400|5655' || true
nvidia-smi --query-gpu=index,name,memory.used --format=csv,noheader
```

Expected before a run: no matching vLLM/LMCache/proxy processes, no listeners on
`8100/8120/8200/8300/8400/5655`, and both GPUs near idle memory.

If a previous run left only this proof's proxy or vLLM process behind, stop that
specific PID. Do not kill unrelated user processes.

### 2. Preflight the host and sandbox

```bash
$PY queue-haul/stage1b_drain_sink.py preflight --required-gpus 2
```

This hard-fails if the old sandbox, HF cache, Apptainer, required imports, GPU
count, or ports are not usable.

### 3. Run live Stage 1b smoke

```bash
$PY queue-haul/stage1b_drain_sink.py smoke2-live \
  --mbps 1000 \
  --run-root /tmp/qh-smoke2-live
```

The driver starts LMCache on `5655`, the proxy on `8300` and `8400`, source vLLM
on GPU 0 / port `8100`, starts sink vLLM on GPU 1 / port `8200`, warms a long
prompt on the live source, then performs one KV resume and one replay request on
the live sink. It also checks that source still answers after sink work.

Expected output is the run directory path. Check the manifest:

```bash
$PY -c "import json; m=json.load(open('/tmp/qh-smoke2-live/smoke2_manifest.json')); print(json.dumps(m['acceptance'], indent=2, sort_keys=True)); print(json.dumps(m['evidence'], indent=2, sort_keys=True))"
```

Required evidence:

- `acceptance.ok` is `true`.
- `source_warmed_before_sink` is `false` for the live proof.
- `live.source_poll` and `live.sink_poll` are null.
- `source_stored` and `sink_retrieved` are `true`.
- `kv_proxy_bytes` is positive on `kv/target_to_client`.
- `api_proxy_bytes` is positive on `api/client_to_target`.
- `kv_observed_bytes_per_s` is at or below the 1Gbps envelope, allowing test
  tolerance.

### 4. Run Stage 1c controller proof

```bash
$PY queue-haul/stage1c_controller.py plan
$PY queue-haul/stage1c_controller.py proof \
  --mbps 1000 \
  --run-root /tmp/qh-proof-live
$PY queue-haul/stage1c_controller.py check --run-root /tmp/qh-proof-live
```

The default fixture intentionally has one replay-cheaper session and one
KV-cheaper session. The proof starts source and sink, prewarms only the KV session
on the live source, then dispatches sessions in controller order through the live
sink.

Inspect the result:

```bash
$PY -c "import json; m=json.load(open('/tmp/qh-proof-live/controller_manifest.json')); print('schema', m['schema']); print('acceptance', m['acceptance']); [print(s['dispatch_rank'], s['id'], s['action'], s['http_status'], s['deadline_met'], s['proxy_delta']) for s in m['sessions']]"
```

Successful proof shape:

- `schema` is `queue-haul-stage1c-v1`.
- `acceptance.ok` is `true`.
- The solver is feasible with `shortfall_w = 0`.
- Dispatch ranks are contiguous and serial in time.
- Both actions appear: `R` and `S`.
- The replay session has positive `api/client_to_target` bytes.
- The KV session has positive `kv/target_to_client` bytes.
- Every session returns HTTP 200 before the fixture deadline.

The successful live local run on 2026-07-08 used `/tmp/qh-proof-live`; it dispatched
`r0` as replay first and `k0` as KV second, and the KV action pulled
`264243456` bytes over `kv/target_to_client`.

### 5. Run the production-shaped live controller

Build the session manifest from a local pinned TraceLab JSONL/JSONL.gz artifact,
then run the LP-ranked live drain with 4 Hz `nvidia-smi` telemetry:

```bash
$PY queue-haul/stage1c_controller.py make-manifest \
  --source tracelab \
  --input /path/to/syfi_coding_trace.jsonl.gz \
  --out /tmp/qh-live-sessions.json \
  --sessions 8 \
  --seed 0
$PY queue-haul/stage1c_controller.py live-drain \
  --manifest /tmp/qh-live-sessions.json \
  --mbps 1000 \
  --nvsmi-ms 250 \
  --run-root /tmp/qh-live
$PY queue-haul/stage1c_controller.py check-live --run-root /tmp/qh-live
```

The live controller writes `gpu_power.csv`, `events.jsonl`,
`controller_manifest.json`, `power_summary.csv`, `power_trace.png`,
`source_power.png`, `sink_power.png`, `delay_summary.csv`, and
`delay_summary.png`. Power deltas are reported, not threshold-gated.

### 6. Clean up and verify the node is idle

The scripts stop their own children on normal exit. Verify anyway:

```bash
pgrep -af 'stage1|vllm|lmcache|proxy_bytes|api_server|EngineCore' || true
ss -ltnp | rg '8100|8120|8200|8300|8400|5655' || true
nvidia-smi --query-gpu=index,name,memory.used --format=csv,noheader
```

### 6. Run tests

The repo standard command is:

```bash
uv run pytest
```

On the current A100 node `uv` is not on PATH, so use:

```bash
module load gcc/14.2.0 openblas/0.3.28
.venv/bin/python -m pytest
```

The latest verification after adding the live proof command was `214 passed`.

## Failure signatures

- `ipc path ... is longer than 107 characters`: `TMPDIR` is too long or `engine_id` is too long. Use `TMPDIR=/tmp/t` and `engine_id=e0`.
- `cudaHostRegister failed: invalid argument`: LMCache local CPU pool is too large for this startup path. Use `LMCACHE_MAX_LOCAL_CPU_SIZE=0.25`.
- vLLM dies after `Model loading took ...` with Torch extension warnings: set `VLLM_USE_FLASHINFER_SAMPLER=0`, pin cache dirs to scratch, and use `--enforce-eager`.
- New `lmcache/vllm-openai:latest-cu129` image fails for this model on A100 with MXFP4 Marlin/MoE backend errors. Stay on `vllm-openai-v0.10.1.1.sandbox`.

## Ports and cleanup

```bash
ss -ltnp | rg '8100|8120|8200|8300|8400|5655'
nvidia-smi
```

The proof scripts stop their own children. If manual cleanup is needed, stop only this proof's vLLM, proxy, and LMCache PIDs. Do not kill unrelated user processes.
