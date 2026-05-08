# Harness Reassessment After GPT-5.4 / GPT-5.4-mini Mini-Pilot

## Correction

The 6/6 verifier failures should not be read as model quality evidence.
The clearer diagnosis is that the no-network agent policy was applied to
task images that were not agent-ready.

The old label `environment_network` was too broad. It hid two different
harness problems:

- `agent_image_missing_runtime`: the agent image lacks a runtime or import
  needed for local reasoning/validation.
- `no_network_install_mismatch`: the task expects solve-time package
  installation, but the agent sandbox has network disabled.

## Concrete Evidence

| task | observed issue | harness implication |
| --- | --- | --- |
| `classifier-debug` | `code.py` imports `numpy`/`torch`, but the agent image did not provide them. | Bake Python ML dependencies into the agent image or exclude the task. |
| `adaptive-rejection-sampler` | task requires R implementation/testing, but the agent image did not provide R. | Bake R into the agent image or exclude the task. |
| `nginx-request-logging` | task asks the agent to install/start Nginx, while agent network is disabled and Nginx is not present. | Either bake Nginx into the image and reinterpret the task as configure/start, or exclude it from no-network pilots. |

## Harness Changes

Implemented:

- First-class model-loop tools: `find_files`, `grep`, ranged `read_file`,
  and `apply_patch`.
- Head+tail transcript snippets instead of tail-only snippets.
- Observation events for truncation, repeated file inspection, chunked reads,
  network blocks, and missing dependencies.
- Agent readiness preflight in `scripts/run_model_agent_pilot.py`.

The preflight runs before model calls. If a provider run uses a no-network
agent image that lacks task-required runtimes or requires solve-time package
installation, the run is marked:

```text
run_status=environment_setup_failure
termination_reason=agent_readiness_preflight_failed
eligible_for_L_gate=false
```

This prevents API spend and prevents incompatible environments from entering
terminal-success or L-gate metrics as model failures.

## Policy Going Forward

Before another provider-backed pilot:

1. For each candidate task, run agent readiness preflight against the exact
   image with network disabled.
2. Exclude tasks with solve-time package install requirements unless the
   required dependency is baked into the agent image.
3. Keep task prompts faithful, but do not ask agents to install packages in a
   sandbox where package installation is impossible.
4. Treat setup-preflight failures as harness/task-selection failures, not
   model failures.
5. Only rerun model arms after the preflight report is clean.

## Current Mini-Pilot Reclassification

Updated report:

```text
reports/REAL_MODEL_MINI3_FAILURE_TRIAGE.md
reports/REAL_MODEL_MINI3_FAILURE_TRIAGE.csv
```

Current counts:

```text
agent_image_missing_runtime=3
no_network_install_mismatch=2
tool_affordance=1
```

Bottom line: this was not primarily a hard-task result. It was a harness
readiness failure caught late.
