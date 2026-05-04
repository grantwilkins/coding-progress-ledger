#!/usr/bin/env bash
set -euo pipefail
agent_repo="$(cd "${1:?agent repo dir required}" && pwd)"
verifier_dir="$(cd "$(dirname "$0")" && pwd)"
project_root="$(cd "$verifier_dir/../../.." && pwd)"
cd "$project_root"
PYTHONPATH="$agent_repo/src" exec uv run pytest "$verifier_dir/verifier_tests" -q
