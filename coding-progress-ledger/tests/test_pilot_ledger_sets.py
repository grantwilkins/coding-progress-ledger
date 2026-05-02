"""
Claim:
- Each pilot's singleton set scores exactly the pilot's single-ledger
  coding-progress.
- The 20-member rollup score equals the unweighted mean of those 20
  per-pilot coding-progress scores (the claim made in
  PILOT_ANNOTATION_SUMMARY.md, modulo rounding).
- The wrapper writes nothing into source_trace.json or ledger.jsonl.

Plausible wrong implementations:
- Score the rollup using all categories instead of CODING.
- Resolve ledger_ref against CWD instead of the set file's parent.
- Mutate the underlying ledger.jsonl while writing the sibling set.jsonl.
"""

import hashlib
import importlib.util
import statistics
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from ledger_progress import from_jsonl, read_set_jsonl, score, score_set
from ledger_progress.queries import CODING_CATEGORIES


PILOT_DIR = ROOT / "runs" / "swe_agent_pilot"
EXPECTED_MEMBER_COUNT = 20


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _pilot_ids() -> list[str]:
    return sorted(p.name for p in PILOT_DIR.iterdir() if p.is_dir() and p.name.startswith("swe_agent_pilot_"))


def test_artifacts_present():
    ids = _pilot_ids()
    assert len(ids) == EXPECTED_MEMBER_COUNT
    for pid in ids:
        assert (PILOT_DIR / pid / "set.jsonl").exists(), f"missing singleton set for {pid}"
    assert (PILOT_DIR / "pilot_rollup_set.jsonl").exists()


def test_singleton_set_score_equals_single_ledger_coding_progress():
    for pid in _pilot_ids():
        ledger_path = PILOT_DIR / pid / "ledger.jsonl"
        expected = score(from_jsonl(str(ledger_path)), categories=CODING_CATEGORIES).progress
        s = read_set_jsonl(str(PILOT_DIR / pid / "set.jsonl"))

        assert len(s.members) == 1
        actual = score_set(s, base_dir=PILOT_DIR / pid)

        assert abs(actual - expected) < 1e-12, f"{pid}: singleton {actual} != single-ledger {expected}"


def test_rollup_score_matches_per_member_mean():
    rollup = read_set_jsonl(str(PILOT_DIR / "pilot_rollup_set.jsonl"))
    assert len(rollup.members) == EXPECTED_MEMBER_COUNT
    assert len({m.member_id for m in rollup.members}) == EXPECTED_MEMBER_COUNT
    assert all(m.weight == 1.0 for m in rollup.members)

    actual = score_set(rollup, base_dir=PILOT_DIR)

    per_pilot = []
    for m in rollup.members:
        ledger_path = PILOT_DIR / m.ledger_ref
        per_pilot.append(score(from_jsonl(str(ledger_path)), categories=CODING_CATEGORIES).progress)
    expected = statistics.fmean(per_pilot)

    assert abs(actual - expected) < 1e-12


def test_rollup_score_in_summary_band():
    """Sanity: the rollup mean should land between the failure-mean (~0.68)
    and the success-mean (~0.97) reported in PILOT_ANNOTATION_SUMMARY.md.
    Locks the value to the band a human would expect from the table."""
    rollup = read_set_jsonl(str(PILOT_DIR / "pilot_rollup_set.jsonl"))
    actual = score_set(rollup, base_dir=PILOT_DIR)

    assert 0.78 <= actual <= 0.86, f"rollup score {actual} outside expected band"


def test_writing_sets_does_not_mutate_ledger_files():
    """Idempotency: re-running the build script must not alter ledger.jsonl
    bytes. We snapshot, re-run, and compare."""
    spec = importlib.util.spec_from_file_location(
        "build_pilot_ledger_sets", ROOT / "scripts" / "build_pilot_ledger_sets.py"
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)

    ledger_paths = [PILOT_DIR / pid / "ledger.jsonl" for pid in _pilot_ids()]
    trace_paths = [PILOT_DIR / pid / "source_trace.json" for pid in _pilot_ids()]
    before = {p: _sha(p) for p in ledger_paths + trace_paths}

    module.write_singleton_sets(PILOT_DIR)
    module.write_rollup_set(PILOT_DIR)

    after = {p: _sha(p) for p in ledger_paths + trace_paths}
    assert before == after
