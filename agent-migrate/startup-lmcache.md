# Startup: old vLLM sandbox + LMCache

Use this path on the current A100 node. Do not use the `latest-cu129` LMCache image for `openai/gpt-oss-20b` on A100; it failed in MXFP4 Marlin/alternate MoE backends. The working path is the older vLLM sandbox with LMCache wrapped around it.

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

## Start LMCache server

Use port `5655`; port `5555` may already be occupied.

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

## Failure signatures

- `ipc path ... is longer than 107 characters`: `TMPDIR` is too long or `engine_id` is too long. Use `TMPDIR=/tmp/t` and `engine_id=e0`.
- `cudaHostRegister failed: invalid argument`: LMCache local CPU pool is too large for this startup path. Use `LMCACHE_MAX_LOCAL_CPU_SIZE=0.25`.
- vLLM dies after `Model loading took ...` with Torch extension warnings: set `VLLM_USE_FLASHINFER_SAMPLER=0`, pin cache dirs to scratch, and use `--enforce-eager`.
- New `lmcache/vllm-openai:latest-cu129` image fails for this model on A100 with MXFP4 Marlin/MoE backend errors. Stay on `vllm-openai-v0.10.1.1.sandbox`.

## Ports and cleanup

```bash
ss -ltnp | rg '5655|8120'
nvidia-smi
```

Stop the vLLM process first, then the LMCache server. Do not kill unrelated user processes.
