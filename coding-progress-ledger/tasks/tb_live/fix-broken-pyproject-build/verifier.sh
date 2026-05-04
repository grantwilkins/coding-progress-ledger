#!/usr/bin/env bash
set -euo pipefail
agent_repo="$(cd "${1:?agent repo dir required}" && pwd)"
tmpdir="$(mktemp -d)"
trap 'rm -rf "$tmpdir"' EXIT
rsync -a --exclude='.venv' --exclude='__pycache__' --exclude='*.egg-info' --exclude='.uv-cache' "$agent_repo/" "$tmpdir/"
cd "$tmpdir"
uv venv --clear .venv >/dev/null 2>&1
source .venv/bin/activate
uv pip install -e . >/dev/null 2>&1
out="$(python -c 'import smolpkg; print(smolpkg.entry())')"
expected="hello from smolpkg v0.1.0"
[[ "$out" == "$expected" ]]
