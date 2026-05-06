#!/usr/bin/env bash
# H5b workspace setup: shallow-clone the 5 upstream repos referenced by
# the H5a fixture's trajectories, so that compute_repo_bytes() can sum
# real working-tree bytes off disk.
#
# Usage:
#   scripts/h5b/clone_repos.sh                       # default /tmp/h5b_workspaces
#   scripts/h5b/clone_repos.sh /path/to/dest         # custom dest
#   VAGRANT_H5B_WORKSPACES=$DEST scripts/h5b/clone_repos.sh
#
# After this completes, run:
#   uv run pytest tests/test_h5b_real_bytes.py
# (the H5b tests skip themselves if any of the 5 sub-dirs is missing).
#
# Caveat: this clones HEAD, NOT the SWE-bench instance's pre-fix base_commit.
# The base_commit is not surfaced in the cached trajectory JSON or the
# inventory CSV; it lives in the SWE-bench dataset metadata, which we
# don't load locally. HEAD is defensible as "real bytes from the same
# upstream repo at a real commit"; the H1<D2 mechanism is bytes-layer
# (gap = 8*B/bps), so byte magnitude is what matters, not exact commit.
# The H5b numerical result is therefore "real bytes at HEAD on date X",
# not "real bytes at the trajectory's exact pre-fix state". Documented
# in TASKS.md.

set -euo pipefail

DEST="${1:-${VAGRANT_H5B_WORKSPACES:-/tmp/h5b_workspaces}}"
mkdir -p "$DEST"

declare -a SPECS=(
    "cog:Melevir/cognitive_complexity"
    "pok:hsahovic/poke-env"
    "dcj:lidatong/dataclasses-json"
    "ice:WIPACrepo/iceprod"
    "scf:asottile/setup-cfg-fmt"
)

for spec in "${SPECS[@]}"; do
    sid="${spec%%:*}"
    repo="${spec##*:}"
    target="$DEST/$sid"
    if [ -d "$target/.git" ]; then
        echo "$sid: already cloned at $target"
        continue
    fi
    echo "==> cloning $sid (https://github.com/$repo) -> $target"
    git clone --depth 1 --quiet "https://github.com/$repo.git" "$target"
done

echo
echo "Working-tree byte sizes (excludes .git, matches compute_repo_bytes default):"
for spec in "${SPECS[@]}"; do
    sid="${spec%%:*}"
    bytes=$(find "$DEST/$sid" -type f -not -path "*/.git/*" -exec stat -f%z {} + 2>/dev/null | awk '{s+=$1} END {print s+0}')
    printf "  %s: %12s bytes\n" "$sid" "$bytes"
done
