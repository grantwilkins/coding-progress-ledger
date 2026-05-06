"""
Claim:
`compute_repo_bytes(path)` returns the recursive sum of file sizes under
`path`, excluding directory names in `exclude_patterns` (default: `.git`).
When wired into `SessionSpec.workspace_path`, the resulting trace's
`workspace_<sid>` state has `bytes` equal to the value the helper returns
on the supplied path. This is how Workstream H2's mechanism is supposed
to graduate from synthetic to real workspaces — the trajectory text plus
a real rollout_dir snapshot.

Plausible wrong implementations the tests below try to catch:
- helper sums file sizes in the wrong unit (e.g., 4096_block_aligned
  apparent size vs. actual byte length). Catch with hand_checkable known-
  size files.
- helper double_counts symlink targets (followed=False on the os.walk
  call but file_stat does follow symlinks; the test pins the documented
  behavior).
- helper ignores hidden files unless we say so. Catch with a `.dotfile`
  test that asserts inclusion.
- `.git` is excluded only when explicitly named — i.e., a `git/` directory
  (no leading dot) is NOT excluded. Catch with both spellings.
- adapter precedence is wrong: workspace_bytes wins over workspace_path,
  or both compose. Catch with both set + assert the disk wins.
- adapter calls compute_repo_bytes once per session_emit, but the path
  doesn't exist, and the failure surfaces somewhere unhelpful. Catch via
  a missing_path test.
- adapter uses workspace_path as a string when the test passes a Path
  (or vice versa). Catch by passing both.
"""
from __future__ import annotations

from pathlib import Path

import pytest
from ledger_progress import from_jsonl

from agent_migrate_agent import build_manifest
from agent_migrate_agent.adapters.swe_agent_multi import (
    MultiSessionConfig,
    SessionSpec,
    generate_to_file,
)
from agent_migrate_agent.workspace import (
    DEFAULT_EXCLUDES,
    compute_repo_bytes,
)

REPO = Path(__file__).resolve().parent.parent
SWE_TRAJ = REPO / "tests" / "fixtures" / "swe_agent_pilot_s_07.json"


# ---------------------------------------------------------------------------
# compute_repo_bytes: hand_checkable cases.
# ---------------------------------------------------------------------------


def _write(p: Path, content: bytes | str) -> None:
    p.parent.mkdir(parents=True, exist_ok=True)
    if isinstance(content, str):
        p.write_text(content)
    else:
        p.write_bytes(content)


def test_empty_directory_zero_bytes(tmp_path: Path):
    assert compute_repo_bytes(tmp_path) == 0


def test_single_file_size_matches_byte_length(tmp_path: Path):
    _write(tmp_path / "a.txt", b"x" * 1234)
    assert compute_repo_bytes(tmp_path) == 1234


def test_recursive_sum_across_subdirs(tmp_path: Path):
    """Hand_checkable: 100 + 200 + 300 = 600."""
    _write(tmp_path / "a.txt", b"x" * 100)
    _write(tmp_path / "sub" / "b.txt", b"x" * 200)
    _write(tmp_path / "sub" / "deep" / "c.txt", b"x" * 300)
    assert compute_repo_bytes(tmp_path) == 600


def test_default_excludes_dot_git(tmp_path: Path):
    """`.git/` content is excluded by default — the most common reason
    for sums to be 5–50× larger than working_tree size."""
    _write(tmp_path / "src.py", b"x" * 100)
    _write(tmp_path / ".git" / "objects" / "00" / "abc", b"x" * 9000)
    assert compute_repo_bytes(tmp_path) == 100


def test_non_dot_git_is_not_excluded(tmp_path: Path):
    """Boundary: `git/` (no leading dot) is NOT in default excludes."""
    _write(tmp_path / "src.py", b"x" * 100)
    _write(tmp_path / "git" / "ignore_me.txt", b"x" * 50)
    assert compute_repo_bytes(tmp_path) == 150


def test_custom_exclude_pattern_drops_match(tmp_path: Path):
    _write(tmp_path / "src.py", b"x" * 100)
    _write(tmp_path / "node_modules" / "dep" / "x.js", b"x" * 9000)
    assert compute_repo_bytes(tmp_path, exclude_patterns=("node_modules",)) == 100


def test_empty_exclude_includes_dot_git(tmp_path: Path):
    """Pass `()` to opt out of all exclusions."""
    _write(tmp_path / ".git" / "config", b"x" * 1000)
    assert compute_repo_bytes(tmp_path, exclude_patterns=()) == 1000


def test_default_excludes_constant_is_dot_git_only():
    """Pin the default so a future expansion (adding `__pycache__`,
    `.venv`, etc.) is a deliberate decision visible in tests."""
    assert DEFAULT_EXCLUDES == (".git",)


def test_missing_path_raises():
    with pytest.raises(FileNotFoundError):
        compute_repo_bytes("/tmp/this_path_should_not_exist_xx_yy_zz")


def test_file_path_raises_not_a_directory(tmp_path: Path):
    f = tmp_path / "not_a_dir.txt"
    f.write_bytes(b"hi")
    with pytest.raises(NotADirectoryError):
        compute_repo_bytes(f)


def test_broken_symlink_contributes_only_its_own_size(tmp_path: Path):
    """`os.lstat` on a symlink returns the symlink's own metadata rather
    than the target's, so a broken symlink contributes a few bytes (the
    pathname stored in the inode) rather than failing or contributing 0."""
    _write(tmp_path / "real.txt", b"x" * 50)
    (tmp_path / "broken_link").symlink_to(tmp_path / "does_not_exist.txt")
    total = compute_repo_bytes(tmp_path)
    # Bound: 50 (real file) + small symlink size. Symlink size on most
    # filesystems is the byte length of the target path string; pin a
    # generous upper bound rather than asserting an exact number.
    assert 50 < total < 50 + 200


def test_symlinked_file_does_not_double_count_target(tmp_path: Path):
    """Symmetry with followlinks=False on directories: a symlinked file
    contributes the symlink's tiny inode size, NOT the target's content
    size. Otherwise a workspace with N symlinks to one big file would
    overstate transfer cost by Nx."""
    big = tmp_path / "big.bin"
    big.write_bytes(b"z" * 10_000)
    link_dir = tmp_path / "links"
    link_dir.mkdir()
    for i in range(5):
        (link_dir / f"link_{i}").symlink_to(big)
    total = compute_repo_bytes(tmp_path)
    # Tight bound: 10_000 (the real file) + 5 small symlink inodes.
    # Reject anything that would imply target_following (>= 50_000).
    assert 10_000 <= total < 12_000, \
        f"symlinks counted as targets? total={total}"


def test_symlinked_directory_is_not_traversed(tmp_path: Path):
    """`os.walk(followlinks=False)` should skip a symlinked directory.
    The link itself contributes only its inode bytes, not the target tree."""
    real_dir = tmp_path / "real_tree"
    _write(real_dir / "inside.txt", b"x" * 5000)
    (tmp_path / "link_to_real").symlink_to(real_dir, target_is_directory=True)
    # The walk visits real_tree once (because it's a real directory),
    # and visits link_to_real but does NOT recurse it. So we count 5000
    # + the link inode's tiny size.
    total = compute_repo_bytes(tmp_path)
    assert 5000 <= total < 5500, f"symlinked directory traversed? total={total}"


# ---------------------------------------------------------------------------
# Adapter wiring: SessionSpec.workspace_path.
# ---------------------------------------------------------------------------


def _two_session_cfg(tmp_path: Path, *, sa_workspace_path: Path | None = None,
                    sa_workspace_bytes: int = 0,
                    sb_workspace_path: Path | None = None,
                    sb_workspace_bytes: int = 0) -> MultiSessionConfig:
    return MultiSessionConfig(sessions=(
        SessionSpec(traj_path=SWE_TRAJ, session_id="sa",
                    workspace_home_site="phoenix",
                    workspace_bytes=sa_workspace_bytes,
                    max_ai_turns=2,
                    workspace_path=sa_workspace_path),
        SessionSpec(traj_path=SWE_TRAJ, session_id="sb",
                    workspace_home_site="seattle",
                    workspace_bytes=sb_workspace_bytes,
                    max_ai_turns=2,
                    workspace_path=sb_workspace_path),
    ))


def _build_manifest(cfg, out: Path):
    generate_to_file(cfg, out)
    return build_manifest(from_jsonl(str(out)))


def test_workspace_path_alone_yields_disk_bytes(tmp_path: Path):
    """When only workspace_path is set (workspace_bytes left at 0
    default), the resulting state's bytes equal compute_repo_bytes(path)."""
    repo = tmp_path / "repo"
    _write(repo / "code.py", b"x" * 4242)
    cfg = _two_session_cfg(
        tmp_path,
        sa_workspace_path=repo,
        sb_workspace_bytes=1,
    )
    m = _build_manifest(cfg, tmp_path / "trace.jsonl")
    assert m.state_objects["workspace_sa"].bytes == 4242
    assert m.state_objects["workspace_sb"].bytes == 1


def test_setting_both_workspace_path_and_bytes_hard_fails(tmp_path: Path):
    """Silent overrides are footguns. When a caller sets both, refuse
    rather than letting one win silently — the caller might think the
    integer is the source of truth."""
    repo = tmp_path / "repo"
    _write(repo / "code.py", b"x" * 100)
    cfg = _two_session_cfg(
        tmp_path,
        sa_workspace_path=repo, sa_workspace_bytes=99_999_999,
        sb_workspace_bytes=1,
    )
    with pytest.raises(ValueError, match="EITHER workspace_path"):
        generate_to_file(cfg, tmp_path / "trace.jsonl")


def test_workspace_path_accepts_string_or_path(tmp_path: Path):
    repo = tmp_path / "repo"
    _write(repo / "f.txt", b"x" * 7)
    cfg_str = _two_session_cfg(tmp_path, sa_workspace_path=str(repo),
                               sb_workspace_bytes=1)
    cfg_path = _two_session_cfg(tmp_path, sa_workspace_path=Path(repo),
                                sb_workspace_bytes=1)
    m_str = _build_manifest(cfg_str, tmp_path / "s.jsonl")
    m_path = _build_manifest(cfg_path, tmp_path / "p.jsonl")
    assert m_str.state_objects["workspace_sa"].bytes == 7
    assert m_path.state_objects["workspace_sa"].bytes == 7


def test_workspace_path_picks_up_dot_git_exclusion(tmp_path: Path):
    """End_to_end: a real_style workspace with a `.git` dir should not
    have the .git internals included in the workspace state's bytes.
    Otherwise H4 would silently overstate transfer costs by 5–50×."""
    repo = tmp_path / "repo"
    _write(repo / "src" / "main.py", b"x" * 200)
    _write(repo / ".git" / "objects" / "huge", b"x" * 50_000)

    cfg = _two_session_cfg(tmp_path, sa_workspace_path=repo,
                           sb_workspace_bytes=1)
    m = _build_manifest(cfg, tmp_path / "trace.jsonl")
    assert m.state_objects["workspace_sa"].bytes == 200


def test_workspace_path_empty_dir_yields_zero_bytes_state(tmp_path: Path):
    """Boundary: empty repo -> workspace state with bytes=0. The cost
    model handles bytes=0 as zero artifact_copy cost; this test pins
    the boundary so the adapter doesn't accidentally substitute a
    sentinel default."""
    empty = tmp_path / "empty_repo"
    empty.mkdir()
    cfg = _two_session_cfg(tmp_path, sa_workspace_path=empty,
                           sb_workspace_bytes=1)
    m = _build_manifest(cfg, tmp_path / "trace.jsonl")
    assert m.state_objects["workspace_sa"].bytes == 0


def test_workspace_path_missing_raises_at_generate(tmp_path: Path):
    """The error must surface at generate_to_file time, not buried
    inside replay or policy code, so users get a clear failure."""
    cfg = _two_session_cfg(tmp_path, sa_workspace_path=tmp_path / "nope",
                           sb_workspace_bytes=1)
    with pytest.raises(FileNotFoundError):
        generate_to_file(cfg, tmp_path / "trace.jsonl")


# ---------------------------------------------------------------------------
# Mechanism preserved: H1 < D2 still holds when workspace bytes come from
# disk rather than a synthetic int. This is the H4 = "real workspace bytes
# graduates the H2 finding" claim, in test form.
# ---------------------------------------------------------------------------


def _build_real_h2_with_workspace_size(tmp_path: Path, ws_bytes: int):
    sa_repo = tmp_path / "repo_sa"
    sb_repo = tmp_path / "repo_sb"
    sc_repo = tmp_path / "repo_sc"
    for repo in (sa_repo, sb_repo, sc_repo):
        repo.mkdir()
        big = repo / "data.bin"
        with open(big, "wb") as f:
            f.seek(ws_bytes - 1)
            f.write(b"\0")
        assert big.stat().st_size == ws_bytes
    cfg = MultiSessionConfig(sessions=(
        SessionSpec(traj_path=SWE_TRAJ, session_id="sa",
                    workspace_home_site="phoenix", workspace_bytes=0,
                    max_ai_turns=2, workspace_path=sa_repo),
        SessionSpec(traj_path=SWE_TRAJ, session_id="sb",
                    workspace_home_site="seattle", workspace_bytes=0,
                    max_ai_turns=2, workspace_path=sb_repo),
        SessionSpec(traj_path=SWE_TRAJ, session_id="sc",
                    workspace_home_site="phoenix", workspace_bytes=0,
                    max_ai_turns=2, workspace_path=sc_repo),
    ))
    out = tmp_path / "real_h2.jsonl"
    generate_to_file(cfg, out)
    return build_manifest(from_jsonl(str(out)))


def test_real_workspace_bytes_preserve_h1_lt_d2_direction(tmp_path: Path):
    """Direction test (NOT a snapshot of absolute costs) at a small
    workspace size. The H2 mechanism `H1 < D2` is bytes_layer and linear
    in workspace_bytes, so any non_trivial size proves the direction.
    10 MB stays sub_second on every reasonable filesystem; the 1 GB form
    is gated behind VAGRANT_SLOW_TESTS=1 so a CI move to a non_sparse-
    aware FS doesn't allocate 3 GB."""
    import os
    from agent_migrate_agent.policies import (
        run_request_level_with_site_cache,
        run_shared_state_aware,
    )
    from agent_migrate_agent.profiles import load_bundle

    m = _build_real_h2_with_workspace_size(tmp_path, ws_bytes=10_000_000)
    bundle = load_bundle(REPO / "configs" / "model_profiles.yaml",
                        REPO / "configs" / "sites_2site.yaml", "compact_kv")
    h1 = run_request_level_with_site_cache(m, bundle).total_cost_s()
    d2 = run_shared_state_aware(m, bundle, tau=1).total_cost_s()
    assert d2 > h1, \
        f"real_bytes H4 must preserve the H2 direction (H1 < D2); got h1={h1}, d2={d2}"


@pytest.mark.skipif(
    "VAGRANT_SLOW_TESTS" not in __import__("os").environ,
    reason="set VAGRANT_SLOW_TESTS=1 to allocate 3 sparse 1 GB files",
)
def test_real_workspace_bytes_preserve_h1_lt_d2_at_canonical_1gb_size(tmp_path: Path):
    """Pinned numerical version at 1 GB (the H2 canonical size). Gated
    behind an env var because 3 sparse 1 GB files require ~3 GB of free
    inode_allocated bytes on a non_sparse_aware filesystem."""
    from agent_migrate_agent.policies import (
        run_request_level_with_site_cache,
        run_shared_state_aware,
    )
    from agent_migrate_agent.profiles import load_bundle

    m = _build_real_h2_with_workspace_size(tmp_path, ws_bytes=1_000_000_000)
    bundle = load_bundle(REPO / "configs" / "model_profiles.yaml",
                        REPO / "configs" / "sites_2site.yaml", "compact_kv")
    h1 = run_request_level_with_site_cache(m, bundle).total_cost_s()
    d2 = run_shared_state_aware(m, bundle, tau=1).total_cost_s()
    assert d2 - h1 > 1.5, \
        f"real_bytes H4 at 1 GB must match H2 canonical (gap > 1.5 s); got gap={d2_h1:.4f}"


# ---------------------------------------------------------------------------
# Determinism: same path -> same bytes -> byte_identical trace.
# ---------------------------------------------------------------------------


def test_real_bytes_trace_is_deterministic_for_same_path(tmp_path: Path):
    repo = tmp_path / "repo"
    _write(repo / "a.txt", b"x" * 100)
    _write(repo / "sub" / "b.txt", b"y" * 200)

    cfg = _two_session_cfg(tmp_path, sa_workspace_path=repo,
                           sb_workspace_bytes=1)
    a = tmp_path / "a.jsonl"
    b = tmp_path / "b.jsonl"
    generate_to_file(cfg, a)
    generate_to_file(cfg, b)
    assert a.read_bytes() == b.read_bytes()
