#!/usr/bin/env bash
set -euo pipefail

REPO=${REPO:-/home/groups/ramr/gfw/coding-progress-ledger/agent-migrate}
PY=${PY:-$REPO/.venv/bin/python}
PLAN=${PLAN:?}
RUN_ROOT=${RUN_ROOT:?}
CHECK=${CHECK:?}

cd "$REPO"
module load gcc/14.2.0 openblas/0.3.28
export QH_APPTAINER_IMAGE=${QH_APPTAINER_IMAGE:-/scratch/users/gfw/ptsim/vllm-openai-v0.10.1.1.sandbox}
export QH_APPTAINER_GPU_MODE=${QH_APPTAINER_GPU_MODE:-nv}
export QH_PORT_OFFSET=${QH_PORT_OFFSET:-$((SLURM_JOB_ID % 40000 + 1000))}

$PY queue-haul/stage1b_drain_sink.py preflight --required-gpus 2
set --
if [[ -n ${RESUME_FROM_GIT_SHA:-} ]]; then
  set -- --resume-from-git-sha "$RESUME_FROM_GIT_SHA"
fi
status=0
$PY queue-haul/stage1c_controller.py run --plan "$PLAN" \
  --run-root "$RUN_ROOT" "$@" || status=$?
$PY queue-haul/stage1c_controller.py reduce --run-root "$RUN_ROOT" || status=$?
if ((status == 0)); then
  $PY queue-haul/stage1c_controller.py "$CHECK" --run-root "$RUN_ROOT"
fi
exit "$status"
