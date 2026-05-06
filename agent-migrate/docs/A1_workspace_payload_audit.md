# A1 — workspace-payload decomposition audit

**Status:** done, 2026-05-05
**Scope:** decompose the H5b "workspace bytes" into 8 candidate payload layers; re-run the H5b H1-vs-D2 calculation under each layer (and selected combinations); produce a sensitivity table that informs the K7 gauntlet's design.
**Headline finding:** **No measurable single-layer or measurable-combo interpretation flips the H5b regime.** D2 ≡ H1 to numerical noise (gap < 1e-9) under every payload definition we can compute from the cached trajectories + shallow HEAD clones. The 8 layers are listed below; the 4 layers that would be largest in a *running* agent (dependency_cache, build_artifact, test_log, and persistent KV state) are all empty or near-empty in our fresh clones, which is itself the load-bearing finding: the H5b "negative result" is a finding about *fresh, dependency-free, build-artifact-free, test-log-free* shallow checkouts, not about real running agents.

## Why this audit

Collaborator 2 flagged that `compute_repo_bytes(workspace_path)` summing full-tree bytes (excluding `.git`) may not represent the mobile state of a running agent. The mobile payload is layered: some layers are globally cacheable (don't move), some are mobile-and-large, some are mobile-and-small. If H5b's "33 MB total" is dominated by globally-cacheable repo tree and missing the actually-mobile layers (uncommitted diffs, dependency caches, build outputs, test logs, KV resident state), the headline negative finding might be "we measured the wrong thing."

This audit decomposes the payload into 8 candidate layers and re-runs the H5b H1-vs-D2 calculation under each.

## The 8 layers

| Layer | What it represents | Production magnitude | Currently mobile? |
| ----- | ------------------ | -------------------- | ----------------- |
| `repo_tree_bytes` | working tree files (excl. `.git`) | 100KB–10GB | partially — base often re-cloned from origin |
| `git_diff_bytes` | uncommitted changes | 1KB–1MB | yes — must move |
| `touched_file_bytes` | files the agent edited | 1KB–10MB | yes |
| `read_file_bytes` | files the agent observed | 1KB–10MB | as KV/prompt context, yes |
| `tool_output_bytes` | stdout/stderr the agent has accumulated into context | 1KB–1MB | yes — already in trace |
| `test_log_bytes` | test runner output (subset of tool_output where prior cmd was a test) | 100KB–100MB+ | yes if agent re-runs from log |
| `build_artifact_bytes` | `__pycache__`, `*.pyc`, `.pytest_cache`, `*.egg-info`, `build/`, `dist/`, `*.so` | 10MB–500MB | rebuildable; mobile-or-not is a policy choice |
| `dependency_cache_bytes` | `.venv`, `node_modules`, `site-packages`, `vendor`, `.tox` | 100MB–5GB+ | rebuildable from manifest; mobile-or-not is a policy choice |

The bottom four (test_log, build_artifact, dependency_cache, plus persistent KV memory) are the layers most likely to push a fixture into Regime B. They are also the layers our shallow HEAD clones do **not** contain.

## Per-repo bytes by layer (snapshot 2026-05-05)

| sid | repo_tree | git_diff | touched | read_file | tool_out | test_log | build | dep_cache |
| --- | --------: | -------: | ------: | --------: | -------: | -------: | ----: | --------: |
| cog |    21,922 |        0 |   1,485 |    12,130 |   46,096 |   11,570 |     0 |         0 |
| pok | 21,588,279 |       0 |   2,613 |    20,267 |   27,133 |        0 |     0 |         0 |
| dcj |   301,091 |        0 |     587 |    25,563 |   33,755 |        0 |     0 |         0 |
| ice | 11,568,017 |       0 |     564 |    16,030 |   12,427 |        0 |     0 |         0 |
| scf |    57,062 |        0 |     560 |    15,139 |   76,616 |    3,030 |     0 |         0 |

Notes:
- `git_diff_bytes = 0` everywhere because clones are at HEAD with no uncommitted edits (by construction).
- `touched_file_bytes` is parsed from each pilot trajectory's `generated_patch` field — the agent's final diff. It is uniformly tiny (560–2,613 bytes per session).
- `read_file_bytes` is heuristic: grepping `cat`/`view`/`open`/`grep`/`find` invocations from the agent's tool calls and summing the sizes of files that exist locally. May undercount.
- `tool_output_bytes` sums every `user`-role text in the trajectory after the issue. This is the bytes the agent has accumulated into its prompt context.
- `test_log_bytes` is a subset of tool_output: only outputs that immediately follow an `ai` turn invoking pytest/unittest/tox/nox/nosetests. Most pilot agents in this corpus did not run tests at all (zeroes for pok/dcj/ice).
- `build_artifact_bytes` and `dependency_cache_bytes` are both zero in our shallow clones: no `pip install` was run, so no `.venv`, no `__pycache__`, no `.pytest_cache`. **This is the key gap.**

## Sensitivity table — H1 vs D2 under each interpretation

Each row replaces H5b's per-session `workspace_bytes = 1_000_000_000` with the corresponding layer's bytes from the table above, holds H5a homes constant (phoenix, seattle, phoenix, seattle, phoenix), and runs `request_level_with_site_cache` (H1) and `shared_state_aware, tau=1` (D2) against the canonical `compact_kv × sites_2site.yaml` config.

| Interpretation                | H1 (s)    | D2 (s)    | D2−H1 (s)    | Verdict |
| ----------------------------- | --------: | --------: | -----------: | ------- |
| repo_tree                     |  0.148675 |  0.148675 |     0.000000 | ≈ collapse |
| git_diff                      |  0.148067 |  0.148067 |     0.000000 | ≈ collapse |
| touched_file                  |  0.148071 |  0.148071 |     0.000000 | ≈ collapse |
| read_file                     |  0.148151 |  0.148151 |     0.000000 | ≈ collapse |
| tool_output                   |  0.148317 |  0.148317 |     0.000000 | ≈ collapse |
| test_log                      |  0.148090 |  0.148090 |     0.000000 | ≈ collapse |
| build_artifact                |  0.148067 |  0.148067 |     0.000000 | ≈ collapse |
| dependency_cache              |  0.148067 |  0.148067 |     0.000000 | ≈ collapse |
| repo_tree + dep_cache         |  0.148675 |  0.148675 |     0.000000 | ≈ collapse |
| repo_tree + touched           |  0.148679 |  0.148679 |     0.000000 | ≈ collapse |
| touched + read + tool         |  0.148406 |  0.148406 |     0.000000 | ≈ collapse |

**No row flips the regime.** The H5b "0% gap survival" finding is robust to payload-interpretation choice: even the optimistic combo (`touched + read + tool`, the everything-the-agent-touched view) doesn't reach the regime-flip threshold (~50 MB cross-site bytes at 5 Gbps under the canonical config).

## What this audit cannot measure

The 8-layer decomposition is the right target, but our measurement is bounded by what fresh shallow clones contain. The four layers most likely to flip the regime in production are exactly the ones our clones lack:

- **`dependency_cache`.** A real running agent has `pip install`-ed its dependencies. For Python repos this commonly produces a 50–500 MB `.venv` (`scipy` alone is ~120 MB; `numpy` is ~60 MB; pyTorch is ~3 GB). For node it's `node_modules` at 100 MB–2 GB. Order-of-magnitude estimate: a **per-session dependency_cache of 100–500 MB is realistic**, which alone would push the seattle-minority bytes (currently 33.2 MB) up to 200–1000 MB — well past the 50 MB regime-flip threshold.
- **`build_artifact`.** `pytest`-style runs accumulate `__pycache__` directories at ~1× source size and a `.pytest_cache` that grows with test count. Build outputs (`*.so`, `*.pyd`, `dist/`) for compiled extensions are easily 10–100 MB.
- **`test_log`.** Long-running test suites can produce 10–500 MB of output, captured into the agent's tool_output. Our cached trajectories are short (max 2 ai turns retained per session in the H5b/H5a fixtures) and don't surface this.
- **Persistent KV state at the source.** Vagrant's cost model already captures `kv_transfer_s = 8 * T * kv_bytes_per_token / link_bps`. For a 100K-token agent context with `kv_bytes_per_token = 70656` (DeepSeek-V3 MLA), that's ~7 GB of KV state per session. **This dominates everything else when KV transfer is the chosen mode.**

A higher-fidelity audit would: (a) `pip install -e .` each repo, run its test suite, and re-measure all 8 layers post-run; (b) use longer trajectories (full pilot length, not 2 ai turns); (c) include resident KV bytes in the resource vector. (a) and (b) are out of scope for this audit; (c) is exactly what Workstream K's `ResourceCost` will introduce.

## Implications for K7 gauntlet design

1. **The H5b finding is robust within the regime we measured.** That regime is "fresh shallow clones, no dependencies installed, short pilot trajectories, distributed-origin scenario." None of these are obviously the production regime.

2. **Workstream K should generate herd fixtures with realistic per-session bytes**, not anchor on H5b's 33 MB total. Concretely:
   - K6's `gauntlet_t2_prefill_only.json` and `gauntlet_t3_multi_resource.json` should sample per-session workspace bytes from `LogNormal(μ=ln(200_000_000), σ=1.0)` to model real running-agent payloads (median ~200 MB, broad distribution). The K0 calibration writeup should justify this.
   - K6 should *also* include a small-bytes variant matching H5b's 33 MB total as the "Regime A anchor."
   - The `mobility_episode_usefulness_map` (K0) explicitly notes that for SWE-bench-class instances at HEAD-with-no-deps, mobility episodes don't matter — and that's *not* the production regime that motivates the project.

3. **The K3 resource vector must include `kv_resident_bytes`**, which is the layer most likely to dominate the mobile payload in production. Solving `8 * kv_resident_bytes / link_bps == prefill_savings` at our config gives a regime-flip threshold of ~50 MB of resident KV — a 100K-token context easily clears this 100×.

4. **A1's strongest message is not "the H5b finding was wrong."** It's "the H5b workload is very unlike the production workload that motivates the project." The K7 gauntlet must use parameter cells representative of the production regime, not the SWE-bench-pilot-trajectory-shallow-clone regime.

## What we did not change

H5b's numerical result is unchanged. The H5b test continues to pass with `D2 ≡ H1` at the canonical config — that's a true fact about that fixture. A1 reframes its scope without contradicting it.

`compute_repo_bytes(path)` continues to be the default workspace-byte computation. K6's herd adapter chooses bytes from a different distribution; A1 motivates that choice but does not change the H4/H5b plumbing.

## Pinning

`tests/test_a1_workspace_payload.py` pins:
- The 8-layer per-repo bytes within a 2× tolerance (catches HEAD drift like H5b's range check).
- The "no row flips the regime" finding (every layer's `|D2 − H1| < 1e-3`).
- The "build_artifact and dependency_cache are zero in fresh clones" expectation, asserted exactly. If a future maintainer runs `pip install` in the workspaces and re-runs the audit, the assertion fails loudly and forces an update of this writeup.
