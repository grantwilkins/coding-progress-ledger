# Queue-Haul Stage 1b — Two-Instance Drain/Sink on Sherlock (Planner + Dispatcher + Power-Down)

## Summary

End-to-end capability demo on one Sherlock A100 node: two old-sandbox vLLM+LMCache instances (source GPU0, sink GPU1), the existing `dispatch.solve` MILP picks replay-vs-KV per session, an operational dispatcher drains ALL sessions from source to sink over a user-space bandwidth-limited link, the source vLLM is killed (power-down), and the run is scored against deadline D with per-GPU power traces. Model: `openai/gpt-oss-20b`. One new Python file, zero changes to existing solver modules.

## Why this supersedes the Docker/tc plan

1. **No Docker on Sherlock.** Apptainer only: sandboxes built from Docker Hub images at `$SCRATCH/ptsim/*.sandbox` (`apptainer build --sandbox`; mksquashfs segfaults, so no SIF), run via `apptainer exec --nv --bind $SCRATCH` inside sbatch. Apptainer shares the host netns — no bridge networks, isolation by port only.
2. **tc is impossible** (no root / CAP_NET_ADMIN on compute nodes). Replacement: a userspace throttling TCP proxy/orchestrator (stdlib asyncio relay: shared token bucket + timestamped per-chunk byte log). Strictly better for measurement — exact per-route bytes vs `eth0` counters, which on a shared node are polluted by NFS/Lustre/Slurm traffic.
3. **KV store-timing hole.** With the LMCache server sink-side and `kv_role=kv_both`, KV likely streams to the server at prefill time — during warmup, before t0 — making the drain-time "transfer" a free local retrieve; `c_transfer = η·T/λ + …` would never be exercised inside the deadline window. Fixed by inverting the topology (below).
4. **Added vs old plan** (per 7526-queuehaul-nextsteps.md locked decisions): per-GPU power traces, the planner in the loop, and an explicit power-down event.

Kept from the old plan: two instances on separate GPUs/ports; gpt-oss-20b; hard-fail reducer; `max(2s, 20%)` acceptance.

Changed by local validation on 2026-07-07:

- Use `/scratch/users/gfw/ptsim/vllm-openai-v0.10.1.1.sandbox`, not `lmcache/vllm-openai:latest-cu129`.
- Use `LMCacheConnectorV1`, not `LMCacheMPConnector`.
- Start LMCache with `python3 -m lmcache.v1.server 127.0.0.1 5655 cpu`, not `lmcache server --host ...`.
- Use `--enforce-eager`; do not use `--async-scheduling` for this path.
- Required env: `VLLM_USE_FLASHINFER_SAMPLER=0`, `TORCH_CUDA_ARCH_LIST=8.0`, short `TMPDIR=/tmp/t`, short connector `engine_id`, and `LMCACHE_MAX_LOCAL_CPU_SIZE=0.25`.

## Verified environment

- `ramr` partition: 1 node (sh03-11n16), 4× A100 SXM4 80GB (CC 8.0), 64 cores, 512 GB, 7-day walltime. Practical alternative: `--partition=owners --constraint=GPU_SKU:A100_SXM4` — partition is a runbook flag, default `ramr`.
- **gpt-oss-20b on A100 is proven in-house**: Stage 1a served it from `/scratch/users/gfw/ptsim/vllm-openai-v0.10.1.1.sandbox` (vLLM 0.10.1.1, mxfp4 loads with only a "not fully optimized" warning, bf16, TP1). The LMCache-wrapped single-instance path also served and generated on 2026-07-07.
- Mirror the proven flags: `--served-model-name … --tensor-parallel-size 1 --max-num-seqs 256 --max-num-batched-tokens 8192 --kv-cache-dtype auto --max-model-len 32768 --enable-chunked-prefill --enforce-eager`.
- sbatch hygiene from `smoke_agentic_a100.sbatch`: `ulimit -n 65536`, `set -Eeuo` (no pipefail — diagnostic `| head` pipes would SIGPIPE), `source /etc/profile.d/modules.sh`, preflight sandbox/weights checks + `$APP python3 -c "import vllm"` before starting servers.
- `openai/gpt-oss-20b` and `Qwen3-8B` (fallback) are staged in `/scratch/users/gfw/ptsim/hf` (`HF_HUB_OFFLINE=1`).
- The `lmcache/vllm-openai:latest-cu129` Apptainer sandbox imports, but `openai/gpt-oss-20b` failed on A100 with MXFP4 Marlin/MoE backend errors. Do not use it for this plan.

## Design

### Topology: LMCache server source-side; sink pulls through the throttled link

- Warmup: source vLLM uses `kv_role="kv_producer"`, `engine_id="s0"`, and `LMCACHE_REMOTE_URL=lm://127.0.0.1:5655`; it stores KV to the source-side LMCache server locally and fast — pre-event normal operation, uncounted.
- Drain: sink vLLM uses `kv_role="kv_consumer"`, `engine_id="d0"`, and `LMCACHE_REMOTE_URL=lm://127.0.0.1:8300`; port `8300` is the throttled proxy route to the source-side LMCache server at `5655`. The sink retrieves η·T bytes **through the λ-throttled proxy, inside [t0, D], on the resume-TTFT critical path**. Consumer role prevents the sink storing replay prefills back across the link.
- Replay sessions: driver POSTs the full conversation through a second throttled route in front of the sink's OpenAI port, with a per-session **nonce prefix** (~5 tokens) so LMCache chunk hashes miss (prefix hashing is positional) and a true full prefill runs.
- Both routes live in ONE proxy process with ONE shared token bucket ⇒ a single shared source-egress link — exactly the MILP's one coupling constraint.
- Power-down: after the last session resumes (or at D), driver SIGKILLs the source vLLM process group; GPU0 → idle in the power trace. The CPU-side lmcache server dies in job cleanup (state must outlive the GPU workload only until drain completes — same as reality).
- Fallback if the smoke gate falsifies ZMQ-retrieve-through-relay: revert to sink-side server and accept warmup-time λ crossing (weaker but coherent).

### In-job orchestration (one sbatch, `--gres=gpu:2 --cpus-per-task=16 --mem=128GB`)

Order, each behind a bounded gate, all PGIDs recorded, `trap cleanup EXIT`:
1. Preflight (<1 min, pre-GPU): sandbox + weights exist; `$APP python3 -c "import vllm, lmcache"`; driver env `python -c "import cvxpy, numpy"`.
2. Power logger (native, bg): `nvidia-smi --query-gpu=index,timestamp,power.draw,clocks.sm,utilization.gpu,memory.used --format=csv,nounits -lms 250 -i 0,1 > power.csv`.
3. LMCache server: `$APP python3 -m lmcache.v1.server 127.0.0.1 5655 cpu`; gate: TCP connect.
4. Throttle proxy/orchestrator (native python3, stdlib-only): `--route kv:8300:5655 --route api:8400:8200 --rate-file rate.txt --log proxy_bytes.csv`; gate: TCP connect both proxy ports.
5. Source vLLM: `APPTAINERENV_CUDA_VISIBLE_DEVICES=0`, port `8100`, `engine_id=s0`, `kv_role=kv_producer`, `LMCACHE_REMOTE_URL=lm://127.0.0.1:5655`; gate: `/health` (≤30 min first time).
6. Sink vLLM **serially after** source healthy (avoids shared compile/cache races): `APPTAINERENV_CUDA_VISIBLE_DEVICES=1`, port `8200`, `engine_id=d0`, `kv_role=kv_consumer`, `LMCACHE_REMOTE_URL=lm://127.0.0.1:8300`; gate: `/health` — the handshake through the relay is itself a smoke check.
7. Driver (foreground). 8. `sleep 60` (post-kill power-decay capture). 9. cleanup; `echo ALL_STAGE1B_OK`.

Every vLLM process must set:

```bash
export PYTHONHASHSEED=0
export VLLM_EXECUTE_MODEL_TIMEOUT_SECONDS=900
export VLLM_USE_FLASHINFER_SAMPLER=0
export TORCH_CUDA_ARCH_LIST=8.0
export TMPDIR=/tmp/t
export HF_HOME=/scratch/users/gfw/ptsim/hf
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export LMCACHE_REMOTE_SERDE=naive
export LMCACHE_CHUNK_SIZE=256
export LMCACHE_MAX_LOCAL_CPU_SIZE=0.25
```

Use separate scratch cache roots per instance, e.g. `/scratch/users/gfw/ptsim/cache/stage1b-src/{xdg,torch_extensions,torchinductor,triton}` and `/scratch/users/gfw/ptsim/cache/stage1b-sink/{...}`.

### Driver (drain mode)

1. `rate.txt ← inf`. Preflight session s00: warm on source, resume on sink; measures **η** (kv-route payload bytes ÷ T; config arithmetic brackets it — gpt-oss-20b = 24 layers alternating sliding-128/full, 8 KV heads × 64 head-dim, bf16 ⇒ 24 KiB/tok if only full-attention layers scale with T, 48 KiB/tok if all stored full — assert 15 ≤ η ≤ 60 KiB/tok, measurement resolves it), **μ** (bytes/retrieve-duration unthrottled), and proves cross-instance reuse (`ttft_resume < 0.5·ttft_warm`) — the go/no-go gate.
2. Warm s01..sN sequentially on source (deterministic seeded prompts cycled over `--t-ladder`, nonce-distinct); record `T_j` from `usage.prompt_tokens`, `ttft_warm_j` → `ρ_meas = median(T_j/ttft_warm_j)`, `mfu_eff`.
3. Build pop/Impact (below); `plan = solve(…, s_star=N, integer=True)`; hard-fail if infeasible, printing `deadline_infeasible` bans + predicted lane busy times (raise D or λ and rerun). Write per-session action + `c_pred` to manifest.
4. `rate.txt ← λ`; sleep 2 (rate epoch lands in the byte log); **t0**. Execute as two concurrent lanes (2 threads): S-lane = sequential KV resumes (link-bound), R-lane = sequential replay resumes (sink-GPU-bound) — mirroring the MILP's two budget rows (egress ≤ D, W=1 prefill ≤ D); measured makespan ≈ max(lane busy) ≤ D is the headline correspondence. Per session: `t_submit`, streaming TTFT (stdlib http.client, first SSE content chunk), `t_done`.
5. `t_drain_end`; `os.killpg(source_pgid, SIGKILL)`; `t_kill`; write `manifest.json`. Exit 0 iff every session returned 200 with a nonempty first token — a deadline miss is a *result* the reducer scores, not an error.

Smoke mode has two levels:

- `--mode smoke1` (1 GPU): old-sandbox single vLLM+LMCache with `kv_role=kv_both`, `engine_id=e0`, `LMCACHE_REMOTE_URL=lm://127.0.0.1:5655`; verifies health, generation, LMCache store, and same-instance retrieval. This is already proven manually and should become an automated gate.
- `--mode smoke2` (2 GPU): source producer + sink consumer through the proxy; store a ~4k prompt on source, resume on sink, then throttle `rate.txt ← 5e7` and require retrieve duration ≈ bytes/rate ± 20%. This retires cross-instance/proxy risks before drain mode.

### Planner mapping (zero solver changes)

- Single-destination `solve(pop, pool, imp, s_star, event, move, integer=True)` with `fleet=None` takes its objective from the caller's Impact (`cost = imp.c_replay @ y_R + imp.c_transfer @ y_S`, dispatch.py:336) and its egress/ingest rows from `imp.b_replay`/`imp.b_transfer`. The only module-constant leakage is `rho_replay(T, pop.mfu)` in the prefill row + `deadline_infeasible`; calibrate `mfu_eff = mfu0 · ρ_meas / ρ_model(T_med, mfu0)` (ρ is linear in mfu).
- **Full evacuation without patching:** build `Impact` directly with `dp_certified = ones(N)`, `s_star = N` — the shed row with pairing `y_R+y_S ≤ 1` forces every session to move; the MILP's remaining freedom is exactly replay-vs-KV per session under the budgets. (`impact.compute()` is NOT used — it hardwires Qwen's `ETA_BYTES_PER_TOK=188KiB`.) `c_replay = 4·T/λ + T/ρ_meas`; `c_transfer = η·T/λ + η·T/μ`.
- `JobPopulation` constructed directly from the manifest: `state="active"`, `f=g=0`, `ell_pre=ell_dec=0` (idle-warm sessions), measured `T`, `m=η·T`, `mfu=mfu_eff`, `precision="bf16"`; `Event(D, dest_nodes=1, spare_frac=1.0, tau_*=0)`, `Movement(lambda_src=λ, mu_in=μ, dest_prefill_util=0, dest_ingest_util=0)`, `pool = PoolPower(mean_context_tokens=T.mean())`. Driver hard-asserts the `held`/`load` budget rows are slack at N (defaults give `s_node≈15`; N=12 default).

### Reducer

`reduce --run-dir …` joins manifest + proxy_bytes.csv + power.csv → `queue-haul/outputs/stage1b_gpt_oss_20b_drain_sink.csv` + one two-panel PNG (per-GPU power with t0/t_kill markers; link byte-rate). Columns: `session_id, action, T, rate_Bps, deadline_s, c_pred_s, ttft_meas_s, complete_s, bytes_pred, bytes_meas, eta_meas_Bpt, egress_hit, rebuild_hit, resume_within_tol, makespan_s, deadline_met, gpu0_prekill_w, gpu0_postkill_w, power_down_ok`.

- Byte attribution: routes disjoint, lanes sequential ⇒ bytes in `[t_submit, t_done]` per route → that session.
- `egress_hit`: S-session bytes within 25% of η·T **and** duration ≈ bytes/λ ± 20% (catches shaper drift AND LMCache silently falling back to recompute). `rebuild_hit`: R-session TTFT ≈ T/ρ_meas within `max(2s, 20%)`. Acceptance per row: `|ttft_meas − c_pred| ≤ max(2s, 0.2·c_pred)`.
- `power_down_ok`: GPU0 mean over `[t_kill+10, t_kill+40]` < 100 W and < 0.5× pre-kill drain-window mean; GPU1 still serving.
- Hard fails, no try/except: missing session in any input, no rate epochs, nonpositive rate, no post-kill power samples.

## Files

| File | Content |
|---|---|
| `queue-haul/stage1b_drain_sink.py` (new, one file) | argparse subcommands `runbook` / `proxy` / `driver` / `reduce`. Top-level imports stdlib-only so `proxy` runs under bare `ml python/3.12.1`; numpy/cvxpy/matplotlib imported inside driver/reduce. |
| `queue-haul/tests/test_stage1b_drain_sink.py` (new) | Pure-python, no GPU. |
| Generated per run | `runs/stage1b/<run_id>/{job.sbatch, commands.sh, rate.txt}` → at runtime `{proxy_bytes.csv, manifest.json, power.csv, *.pgid, logs}`. |
| Reused untouched | `dispatch.py`, `power.py`, `instance.py`; `stage1_curves.py` runbook pattern; `campaign.sbatch` env block; `server_lifecycle.sh` setsid/PGID pattern (re-emitted — it hardcodes port 8000); `build_container.sh`. |

Runbook CLI (mirrors stage1_curves.py; prints the sbatch path, user submits):

```
uv run python queue-haul/stage1b_drain_sink.py runbook \
  --mode {smoke1,smoke2,drain} --rate 1.25e8 --deadline 120 --sessions 12 \
  [--model openai/gpt-oss-20b] [--max-model-len 32768] [--t-ladder 1024 … 24576] \
  [--sandbox /scratch/users/gfw/ptsim/vllm-openai-v0.10.1.1.sandbox] [--partition ramr] \
  [--src-port 8100 --sink-port 8200 --lmc-port 5655 \
   --kv-proxy-port 8300 --api-proxy-port 8400] [--seed 0] [--run-id ID] [-- extra vllm args]
```

## Phases (each ends with `uv run pytest` + commit)

- **P0a — locked environment preflight (no repo code):** verify `/scratch/users/gfw/ptsim/vllm-openai-v0.10.1.1.sandbox` imports `vllm`, `lmcache`, and sees A100. Do not build or use `latest-cu129`.
- **P1 — code, no GPU:** `proxy` + `runbook` (both modes) + smoke driver path + tests.
- **P0b — smoke1 gate (1 GPU, ~30 min):** pass = old-sandbox gpt-oss-20b healthy on A100 + generation works + LMCache stores and retrieves same-instance KV.
- **P0c — smoke2 gate (2 GPU, ~30 min):** pass = source producer and sink consumer start on separate GPUs + sink retrieves through proxy + throttled retrieve ≈ bytes/rate.
- **P2 — code:** drain driver (warmup → solve → two lanes → kill) + planner-mapping tests.
- **P3 — main run (2 GPU, ~30 min):** λ near measured crossover `λ* = η·ρ_meas` (e.g. λ*/1.5 so the MILP picks a genuine R/S mix), D ≈ 1.3× predicted max lane busy time.
- **P4 — code:** `reduce` + tests; produce CSV/PNG.
- **P5 — planner-sensitivity rerun + README:** rerun at λ*/5 — the MILP should flip most S→R; the cheapest "planner responds to the link" evidence. Update README, commit.

## Tests (pure python, claim-docstring style like test_stage1_curves.py)

(a) runbook: old sandbox path, source pinned CVD=0/8100/producer→5655, sink pinned CVD=1/8200/consumer→8300, `python3 -m lmcache.v1.server 127.0.0.1 5655 cpu`, `--enforce-eager`, `VLLM_USE_FLASHINFER_SAMPLER=0`, `TMPDIR=/tmp/t`, `LMCACHE_MAX_LOCAL_CPU_SIZE=0.25`, short `engine_id`s, and separate cache dirs; power logger has `index` + `-i 0,1`; duplicate vLLM flags rejected. (b) proxy: real relay on ephemeral localhost ports at a small rate — duration ≈ bytes/rate; log rows sum to bytes sent; bucket shared across two routes. (c) planner mapping: manifest fixture → pop/Impact → `solve(integer=True)` moves all N; λ→tiny ⇒ all R; η·μ≫λ regime ⇒ all S; costs match hand formulas; infeasible (λ,D) hard-fails. (d) reducer: hard-fails on missing session/timing/bytes/post-kill power; fixture yields correct hits; `power_down_ok` false when GPU0 stays hot.

## Risks, ranked, each with its cheapest kill-switch

1. **Cross-instance `LMCacheConnectorV1` producer/consumer with gpt-oss has not yet been proven.** Kill: P0c smoke2. Fallback: same-instance Stage 1b capability table first, then staged `Qwen3-8B` for the two-instance drain.
2. **ZMQ/LMCache through a slow TCP relay** (timeout or silent recompute fallback). Kill: P0c throttled-retrieve check; reducer `egress_hit` re-checks in the main run. Fallback: sink-side-server topology, explicitly marked weaker because transfer can occur before t0.
3. **Pinned CPU memory regression.** Kill: startup gate checks for `cudaHostRegister`; keep `LMCACHE_MAX_LOCAL_CPU_SIZE=0.25`. Fallback: smaller pool; do not raise it without a fresh startup test.
4. **IPC path collision/overflow.** Kill: startup gate checks logs for `ipc path ... longer than 107`; keep `TMPDIR=/tmp/t` and short engine IDs.
5. **Store timing assumption** (write-through at prefill). The source-side-server topology is robust to it by construction; P0c's byte log confirms empirically.
6. **cvxpy/uv missing on compute node.** Kill: preflight import gate before any server starts.
7. **Planner infeasible at chosen (λ, D).** Kill: driver prints bans + predicted lane busy times pre-execution.
8. **Compile-cache race between two vLLM startups.** Designed out with serial startup and separate cache dirs.

## What proves end-to-end capability (two GPU jobs total)

(i) `manifest.json`: the MILP chose a nontrivial R/S mix from measured (T, η, ρ, μ, λ); (ii) `proxy_bytes.csv`: η·T KV bytes cross only inside `[t0, t_drain_end]` at ≤ λ; (iii) per-session resume TTFT within `max(2s, 20%)` of `c_pred`; (iv) makespan vs D = the MILP's two budget rows realized as the two lanes; (v) `power.csv`: GPU0 falls from serving power to <100 W within seconds of `t_kill` while GPU1 keeps serving; (vi) the λ/5 rerun flips actions. That is planner → dispatcher → drain → power-down → deadline, end to end.
