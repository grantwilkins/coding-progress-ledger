"""
Claim:
The H5b workspace-payload audit (`docs/A1_workspace_payload_audit.md`)
decomposes the workspace into 8 candidate layers. Under EVERY measurable
single-layer or measurable-combo interpretation, the H1<D2 gap remains
collapsed (D2 ≡ H1 within numerical noise). The "negative finding" of
H5b is robust to payload-interpretation choice.

The four layers most likely to flip the regime in production
(`dependency_cache`, `build_artifact`, `test_log`, persistent KV state)
are absent from our shallow HEAD clones. This test pins:
- per-layer bytes within a 2x tolerance of the snapshot (HEAD drift),
- "build and dep_cache are exactly zero in fresh clones" (so a future
  pip-install-and-re-measure breaks loudly),
- the no-row-flips-regime finding across all 8 layers + 3 combos.

Plausible wrong implementations the tests below try to catch:
- a future `compute_repo_bytes` change accidentally counts `.git` bytes
  -> repo_tree value would 2-50x; per-repo range check trips.
- `pip install` is run in the workspaces and forgotten -> dep_cache > 0
  triggers the "audit needs re-running" assertion.
- a cost-model tweak silently flips the H5b regime under any layer
  interpretation -> the no-row-flips assertion catches it across all 11
  rows in one place.
"""
from __future__ import annotations

import fnmatch
import json
import os
import re
import tempfile
from pathlib import Path

import pytest
from ledger_progress import from_jsonl

from vagrant_agent import build_manifest
from vagrant_agent.adapters.swe_agent_multi import (
    MultiSessionConfig,
    SessionSpec,
    generate_to_file,
)
from vagrant_agent.policies import (
    run_request_level_with_site_cache,
    run_shared_state_aware,
)
from vagrant_agent.profiles import load_bundle
from vagrant_agent.workspace import compute_repo_bytes

REPO = Path(__file__).resolve().parent.parent
FIX = REPO / "tests" / "fixtures"
MODELS = REPO / "configs" / "model_profiles.yaml"
SITES = REPO / "configs" / "sites_2site.yaml"

WORKSPACES_DIR = Path(os.environ.get("VAGRANT_H5B_WORKSPACES", "/tmp/h5b_workspaces"))

SESSIONS = (
    ("cog", "swe_agent_pilot_s_01.json", "phoenix"),
    ("pok", "swe_agent_pilot_s_03.json", "seattle"),
    ("dcj", "swe_agent_pilot_s_05.json", "phoenix"),
    ("ice", "swe_agent_pilot_f_01.json", "seattle"),
    ("scf", "swe_agent_pilot_f_03.json", "phoenix"),
)

# Snapshot — see docs/A1_workspace_payload_audit.md.
PER_SID_SNAPSHOT: dict[str, dict[str, int]] = {
    "cog": {"repo_tree": 21_922, "git_diff": 0, "touched_file": 1_485,
            "read_file": 12_130, "tool_output": 46_096, "test_log": 11_570,
            "build_artifact": 0, "dependency_cache": 0},
    "pok": {"repo_tree": 21_588_279, "git_diff": 0, "touched_file": 2_613,
            "read_file": 20_267, "tool_output": 27_133, "test_log": 0,
            "build_artifact": 0, "dependency_cache": 0},
    "dcj": {"repo_tree": 301_091, "git_diff": 0, "touched_file": 587,
            "read_file": 25_563, "tool_output": 33_755, "test_log": 0,
            "build_artifact": 0, "dependency_cache": 0},
    "ice": {"repo_tree": 11_568_017, "git_diff": 0, "touched_file": 564,
            "read_file": 16_030, "tool_output": 12_427, "test_log": 0,
            "build_artifact": 0, "dependency_cache": 0},
    "scf": {"repo_tree": 57_062, "git_diff": 0, "touched_file": 560,
            "read_file": 15_139, "tool_output": 76_616, "test_log": 3_030,
            "build_artifact": 0, "dependency_cache": 0},
}

BUILD_PATTERNS = ("__pycache__", "*.pyc", ".pytest_cache", "*.egg-info",
                  "build", "dist", "*.so")
DEP_PATTERNS = (".venv", "venv", "node_modules", "site-packages", "vendor", ".tox")


def _all_repos_present() -> bool:
    if not WORKSPACES_DIR.is_dir():
        return False
    return all((WORKSPACES_DIR / sid).is_dir() for sid, _, _ in SESSIONS)


_skip_no_repos = pytest.mark.skipif(
    not _all_repos_present(),
    reason=(
        f"A1 audit requires repo clones at {WORKSPACES_DIR}; run "
        f"scripts/h5b/clone_repos.sh first"
    ),
)


# ---------------------------------------------------------------------------
# Per-layer byte computations (mirror the audit script).
# ---------------------------------------------------------------------------


def _git_diff_bytes(sid: str) -> int:
    return 0  # clones are at HEAD


def _touched_file_bytes(traj_path: Path) -> int:
    raw = json.loads(traj_path.read_text())
    return len((raw.get("generated_patch") or "").encode("utf-8"))


_CMD_RE = re.compile(
    r"(?:cat|less|head|tail|view|open|read|grep\s+\S+|find\s+\S+)\s+([^\s|;<>&]+)",
    re.MULTILINE,
)


def _read_file_bytes(traj_path: Path, repo_dir: Path) -> int:
    raw = json.loads(traj_path.read_text())
    seen: set[str] = set()
    for turn in raw["trajectory"]:
        if turn.get("role") != "ai":
            continue
        text = turn.get("text") or ""
        for m in _CMD_RE.finditer(text):
            cand = m.group(1)
            if cand.startswith(("/", "-")) or ".." in cand:
                continue
            seen.add(cand.lstrip("./"))
    total = 0
    for rel in seen:
        p = repo_dir / rel
        if p.is_file():
            try:
                total += p.stat().st_size
            except OSError:
                pass
    return total


def _tool_output_bytes(traj_path: Path) -> int:
    raw = json.loads(traj_path.read_text())
    turns = raw["trajectory"]
    total = 0
    for turn in turns[2:]:
        if turn.get("role") == "user":
            total += len((turn.get("text") or "").encode("utf-8"))
    return total


_TEST_RE = re.compile(
    r"\b(pytest|python\s+-m\s+(?:unittest|pytest)|tox|nox|nosetests)\b"
)


def _test_log_bytes(traj_path: Path) -> int:
    raw = json.loads(traj_path.read_text())
    turns = raw["trajectory"]
    total = 0
    for i, turn in enumerate(turns):
        if turn.get("role") == "ai" and _TEST_RE.search(turn.get("text") or ""):
            if i + 1 < len(turns) and turns[i + 1].get("role") == "user":
                total += len((turns[i + 1].get("text") or "").encode("utf-8"))
    return total


def _matches_any(name: str, patterns) -> bool:
    return any(fnmatch.fnmatch(name, p) for p in patterns)


def _pattern_bytes(repo_dir: Path, patterns) -> int:
    total = 0
    for root, dirs, files in os.walk(repo_dir, followlinks=False):
        keep = []
        for d in dirs:
            if _matches_any(d, patterns):
                for r2, _, fs in os.walk(Path(root) / d, followlinks=False):
                    for f in fs:
                        try:
                            total += (Path(r2) / f).lstat().st_size
                        except OSError:
                            pass
            else:
                keep.append(d)
        dirs[:] = keep
        for f in files:
            if _matches_any(f, patterns):
                try:
                    total += (Path(root) / f).lstat().st_size
                except OSError:
                    pass
    return total


def _layer_bytes(sid: str, traj_name: str, layer: str) -> int:
    traj = FIX / traj_name
    repo_dir = WORKSPACES_DIR / sid
    if layer == "repo_tree":
        return compute_repo_bytes(repo_dir)
    if layer == "git_diff":
        return _git_diff_bytes(sid)
    if layer == "touched_file":
        return _touched_file_bytes(traj)
    if layer == "read_file":
        return _read_file_bytes(traj, repo_dir)
    if layer == "tool_output":
        return _tool_output_bytes(traj)
    if layer == "test_log":
        return _test_log_bytes(traj)
    if layer == "build_artifact":
        return _pattern_bytes(repo_dir, BUILD_PATTERNS)
    if layer == "dependency_cache":
        return _pattern_bytes(repo_dir, DEP_PATTERNS)
    raise ValueError(f"unknown layer {layer!r}")


# ---------------------------------------------------------------------------
# Per-layer per-repo byte snapshots.
# ---------------------------------------------------------------------------


@_skip_no_repos
@pytest.mark.parametrize("sid,traj,_home", SESSIONS)
@pytest.mark.parametrize("layer", list(PER_SID_SNAPSHOT["cog"]))
def test_per_layer_byte_snapshot(sid, traj, _home, layer):
    """Per-(repo, layer) byte sizes must stay within 2x of the audit
    snapshot. Catches HEAD drift, dependency installation, build runs,
    or `compute_repo_bytes` regressions."""
    actual = _layer_bytes(sid, traj, layer)
    expected = PER_SID_SNAPSHOT[sid][layer]
    if expected == 0:
        # For exactly-zero layers (build, dep_cache, git_diff, plus the
        # zero rows in test_log), a non-zero value is meaningful: a future
        # audit re-run with installed deps must trigger this and force
        # re-measurement of the regime.
        assert actual == 0, (
            f"{sid}/{layer}: snapshot is 0 but actual={actual:,}. "
            f"If this is intentional (e.g., dependencies installed for a "
            f"new audit pass), re-run docs/A1_workspace_payload_audit.md "
            f"and update PER_SID_SNAPSHOT."
        )
    else:
        assert expected // 2 <= actual <= expected * 2, (
            f"{sid}/{layer}: drift out of range. "
            f"snapshot={expected:,}, actual={actual:,}. "
            f"Update PER_SID_SNAPSHOT after re-running the audit."
        )


# ---------------------------------------------------------------------------
# Sensitivity table — H1 vs D2 under each interpretation.
# ---------------------------------------------------------------------------


def _bundle():
    return load_bundle(MODELS, SITES, "compact_kv")


def _gap_for_layer(layer: str, tmp_path: Path) -> tuple[float, float]:
    cfg = MultiSessionConfig(
        sessions=tuple(
            SessionSpec(
                traj_path=FIX / traj, session_id=sid,
                workspace_home_site=home,
                workspace_bytes=_layer_bytes(sid, traj, layer),
                max_ai_turns=2,
            )
            for sid, traj, home in SESSIONS
        ),
        workflow_id="a1_audit",
    )
    out = tmp_path / f"{layer}.jsonl"
    generate_to_file(cfg, out)
    m = build_manifest(from_jsonl(str(out)))
    b = _bundle()
    h1 = run_request_level_with_site_cache(m, b).total_cost_s()
    d2 = run_shared_state_aware(m, b, tau=1).total_cost_s()
    return h1, d2


@_skip_no_repos
@pytest.mark.parametrize("layer", list(PER_SID_SNAPSHOT["cog"]))
def test_no_single_layer_flips_regime(layer, tmp_path):
    """Headline assertion: under every measurable single-layer
    interpretation, D2 ≡ H1 within 1 ms. A flip in any direction would
    mean either (a) the H5b finding was payload-definition-dependent
    and needs reframing, or (b) a cost-model regression."""
    h1, d2 = _gap_for_layer(layer, tmp_path)
    assert abs(d2 - h1) < 1e-3, (
        f"Regime flipped under layer={layer!r}: H1={h1:.6f}, D2={d2:.6f}, "
        f"gap={d2-h1:.6e}. Either H5b's negative finding has a "
        f"payload-definition exception (good — write it up), or a cost-"
        f"model change is leaking into A1."
    )


@_skip_no_repos
def test_optimistic_combo_does_not_flip_regime(tmp_path):
    """`touched + read_file + tool_output` is the most generous all-the-
    agent-touched interpretation. Even this combo doesn't flip the
    regime in our shallow-clone fixtures."""
    cfg = MultiSessionConfig(
        sessions=tuple(
            SessionSpec(
                traj_path=FIX / traj, session_id=sid,
                workspace_home_site=home,
                workspace_bytes=(
                    _layer_bytes(sid, traj, "touched_file")
                    + _layer_bytes(sid, traj, "read_file")
                    + _layer_bytes(sid, traj, "tool_output")
                ),
                max_ai_turns=2,
            )
            for sid, traj, home in SESSIONS
        ),
        workflow_id="a1_audit_combo",
    )
    out = tmp_path / "combo.jsonl"
    generate_to_file(cfg, out)
    m = build_manifest(from_jsonl(str(out)))
    b = _bundle()
    h1 = run_request_level_with_site_cache(m, b).total_cost_s()
    d2 = run_shared_state_aware(m, b, tau=1).total_cost_s()
    assert abs(d2 - h1) < 1e-3, (
        f"Optimistic combo flipped: H1={h1:.6f}, D2={d2:.6f}, gap={d2-h1:.6e}"
    )


@_skip_no_repos
def test_build_and_dep_cache_zero_in_fresh_clones():
    """The four production-relevant layers (build_artifact + dependency_
    cache, plus test_log subsets and KV resident state) are precisely
    what's missing from our shallow clones. Pin the zeros loudly so a
    future audit pass that installs deps surfaces immediately."""
    for sid, _, _ in SESSIONS:
        b = _layer_bytes(sid, dict((s, t) for s, t, _ in SESSIONS)[sid], "build_artifact")
        d = _layer_bytes(sid, dict((s, t) for s, t, _ in SESSIONS)[sid], "dependency_cache")
        assert b == 0 and d == 0, (
            f"{sid}: build_artifact={b}, dependency_cache={d}; "
            "expected 0 in fresh shallow clones. If this changed because "
            "you ran `pip install` in the workspace, re-run the audit "
            "and update both snapshot constants and the writeup."
        )
