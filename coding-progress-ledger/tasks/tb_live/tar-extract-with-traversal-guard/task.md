# tar-extract-with-traversal-guard

Build a Python package `safetar` that extracts tar archives while rejecting any member that could write outside the destination directory.

## What you must produce

A package importable as `safetar` with one public function and one public exception:

```python
def safe_extract(tar_path: str, dest_dir: str) -> None: ...

class UnsafeTarError(Exception): ...
```

Both must be importable directly from the package:

```python
from safetar import safe_extract, UnsafeTarError
```

## Threat model

Tar archives are untrusted. A malicious archive can contain members that, if extracted naively, would write files outside the intended destination directory. Your implementation must detect and reject four categories of attack:

**Path traversal via `..`**
A member name such as `../escape.txt` or `foo/../../etc/passwd` resolves to a path outside `dest_dir` when joined with it. Any member whose resolved path does not start with the resolved `dest_dir` is unsafe.

**Absolute paths**
A member name such as `/etc/passwd` is an absolute path. Python's `tarfile` module will strip the leading `/` by default, but a standards-compliant extractor must reject these explicitly.

**Symlink escape**
A member of type symlink whose target, when resolved relative to the symlink's location inside `dest_dir`, points outside `dest_dir`. For example, a symlink at `link` with target `../../outside` could be used to redirect subsequent writes or reads out of bounds.

**Hardlink escape**
A member of type hardlink whose `linkname` field references a path outside `dest_dir`. The hardlink target is evaluated relative to `dest_dir` (as tarfile does), but also must not be absolute.

## Behavior contract

- **Pre-flight check**: Before extracting any member, scan all members. If any member is unsafe, raise `safetar.UnsafeTarError` with a descriptive message. Do not extract anything — leave `dest_dir` clean.
- **Safe extraction**: If all members pass the check, extract them into `dest_dir`, preserving their relative paths.
- **Error type**: The exception must be `safetar.UnsafeTarError`. It must be a subclass of `Exception`.

## What "resolves outside" means

Use `pathlib.Path.resolve()` (or `os.path.realpath`) to compute the canonical absolute path of a candidate destination, then check that the resolved path starts with `str(dest_dir_resolved) + os.sep` or equals `str(dest_dir_resolved)`. A simpler equivalent: check that `dest_dir_resolved` is a prefix of the member's resolved destination.

For symlinks, compute the target by joining the symlink's containing directory (inside `dest_dir`) with the symlink target string, then resolve.

For hardlinks, `linkname` is the hardlink target stored in the tar header. Evaluate it relative to `dest_dir` and check the resolved path stays inside.

## Repository layout

Use the standard `src/` layout:

```
<agent_repo>/
  src/
    safetar/
      __init__.py     # exports safe_extract and UnsafeTarError
```

The verifier puts `<agent_repo>/src` on `PYTHONPATH` and runs `pytest` against hidden tests. You may add a `pyproject.toml` or anything else; only `src/safetar/` is load-bearing.

## Implementation notes

- Use Python's `tarfile` module from the standard library. No third-party dependencies are needed.
- `tarfile.open(tar_path)` opens in read mode by default — use that.
- Iterate `tf.getmembers()` for the pre-flight scan, then call `tf.extract(member, dest_dir)` (or `tf.extractall`) only after all members pass.
- For the symlink check: `member.issym()` is True for symlinks; the target is `member.linkname`.
- For the hardlink check: `member.islnk()` is True for hardlinks; the target is also `member.linkname`.
- Absolute path check: `member.name.startswith("/")` or `os.path.isabs(member.name)`.
- Keep the implementation tight — you do not need to handle special file types (device nodes, FIFOs) beyond what the threat model requires.

## How to track progress

You are running under the N_TB live ledger harness. After each meaningful action (subtask added, started, completed, blocked, etc.), emit one wire-format event with:

```bash
uv run python /Users/grantwilkins/houdini/coding-progress-ledger/scripts/tb_emit.py \
    /Users/grantwilkins/houdini/coding-progress-ledger/runs/tb_live/tar-extract-with-traversal-guard \
    <step_number> \
    '[{"op":"add","id":"s1","description":"...","category":"product"}]'
```

See the project's `docs/AGENT_USAGE.md` and `docs/TB_LIVE_TASK_FORMAT.md` for the protocol. Use `product` for code-that-ships, `validation` for tests / asserts / manual checks, `investigation` for reading / search / trace work. Add subtasks as you discover them, not as a plan up front. Mark complete only with concrete evidence.

## Done condition

You are done when `verifier.sh` exits 0 against your repo. The verifier is hidden — you cannot read it. Your fastest path to done is to write your own tests that cover each threat category above plus at least one benign extraction, run them, and only declare a leaf complete when the test passes.
