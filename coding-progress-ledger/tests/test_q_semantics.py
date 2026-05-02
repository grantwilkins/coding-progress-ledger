"""
Claim:
  build_q_labels.py and q_baselines.py implement the Q1–Q4 contract:
  - Horizon-dependent labels look at events in the half-open window
    (S, S+H], using categories resolved at event-time.
  - future_progress_drop fires only on a STRICT decrease from the
    checkpoint's coding_progress.
  - validation_exposes_new_work requires a VALIDATION transition in
    the window followed (in step order) by an ADD/REOPEN of
    PRODUCT/INVESTIGATION.
  - stuck_loop_next_window is masked to False when the row already
    has repeated_observation_loop_flag=True at step S.
  - submit_without_validation_state is constant per run (terminal).
  - q_baselines: AUROC averages tied ranks (constant predictor → 0.5);
    LORO holds out exactly the runs requested with no overlap.
  - always_mean predicts the *train* base rate, so predictions vary
    across LORO folds for per-run-constant targets.

Plausible wrong implementations:
  1. Closed window [S, S+H] — counts current step as future.
  2. Open window (S, S+H) — drops the boundary event.
  3. Drop comparison <= instead of < — labels constants as drops.
  4. Validation→discovery without temporal ordering — fires on
     discovery that happened *before* validation in the window.
  5. Stuck-loop mask flipped — fires on rows where loop is already
     known.
  6. Category resolved from final replay state instead of event-time
     state — wrong on subtasks deleted later.
  7. AUROC ties not averaged → returns 0 or 1 for constants.
  8. LORO leaks: train set contains held-out run.
  9. always_mean uses the test base rate (or global rate) instead of
     the train rate.
"""
from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from ledger_progress import EventType, LedgerEvent

_BL = ROOT / "scripts" / "build_q_labels.py"
_QB = ROOT / "scripts" / "q_baselines.py"


def _load(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


build_q_labels = _load("build_q_labels", _BL)
q_baselines = _load("q_baselines", _QB)


def _ev(step: int, event_type: str, subtask_id=None, payload=None, reason=None):
    return LedgerEvent(step, EventType(event_type), subtask_id, payload or {}, reason)


def _bare_run():
    return [
        _ev(0, "init", payload={"root_task": "T"}),
        _ev(1, "add_subtask", "S1", payload={
            "description": "P1", "category": "product", "weight": 1.0}),
        _ev(2, "update_status", "S1", payload={
            "status": "complete", "evidence": ["e"]}),
    ]


# ── future_progress_drop: window boundaries and strictness ──────────────────


def test_future_progress_drop_includes_upper_boundary():
    """Wrong-impl candidate: open-top window (S, S+H) drops the boundary."""
    events = _bare_run() + [
        _ev(5, "add_subtask", "S2", payload={
            "description": "P2", "category": "product", "weight": 1.0}),
    ]
    assert build_q_labels._label_future_progress_drop(
        events, checkpoint_step=2, horizon=3, current_progress=1.0) is True


def test_future_progress_drop_excludes_just_past_upper_boundary():
    events = _bare_run() + [
        _ev(6, "add_subtask", "S2", payload={
            "description": "P2", "category": "product", "weight": 1.0}),
    ]
    assert build_q_labels._label_future_progress_drop(
        events, checkpoint_step=2, horizon=3, current_progress=1.0) is False


def test_future_progress_drop_excludes_lower_boundary_event_at_S():
    """Wrong-impl candidate: closed-bottom window [S, S+H] counts step S."""
    assert build_q_labels._label_future_progress_drop(
        _bare_run(), checkpoint_step=2, horizon=3, current_progress=1.0) is False


def test_future_progress_drop_strict_not_equal():
    """Constant progress is NOT a drop."""
    events = _bare_run() + [
        _ev(3, "add_evidence", "S1", payload={"evidence": ["more"]}),
    ]
    assert build_q_labels._label_future_progress_drop(
        events, checkpoint_step=2, horizon=3, current_progress=1.0) is False


# ── product_reopened_after_completion: window + category-at-event-time ──────


def test_product_reopen_window_inclusive_at_upper_bound():
    events = _bare_run() + [_ev(5, "reopen_subtask", "S1", payload={})]
    assert build_q_labels._label_product_reopened(
        events, checkpoint_step=2, horizon=3) is True


def test_product_reopen_window_excludes_just_past():
    events = _bare_run() + [_ev(6, "reopen_subtask", "S1", payload={})]
    assert build_q_labels._label_product_reopened(
        events, checkpoint_step=2, horizon=3) is False


def test_product_reopen_ignores_non_product_categories():
    """A reopened VALIDATION subtask must NOT flip product_reopened."""
    events = [
        _ev(0, "init", payload={"root_task": "T"}),
        _ev(1, "add_subtask", "S1", payload={
            "description": "V", "category": "validation", "weight": 1.0}),
        _ev(2, "update_status", "S1", payload={
            "status": "complete", "evidence": ["e"]}),
        _ev(4, "reopen_subtask", "S1", payload={}),
    ]
    assert build_q_labels._label_product_reopened(
        events, checkpoint_step=2, horizon=3) is False


# ── validation_exposes_new_work: temporal ordering inside window ────────────


def test_validation_exposes_requires_validation_BEFORE_discovery():
    """Wrong-impl candidate: any (validation, add) pair in window fires —
    even if the add precedes validation. Should require strict ordering."""
    events = [
        _ev(0, "init", payload={"root_task": "T"}),
        _ev(1, "add_subtask", "S1", payload={
            "description": "V", "category": "validation", "weight": 1.0}),
        _ev(3, "add_subtask", "S2", payload={
            "description": "P", "category": "product", "weight": 1.0}),
        _ev(4, "update_status", "S1", payload={
            "status": "complete", "evidence": ["e"]}),
    ]
    assert build_q_labels._label_validation_exposes_new_work(
        events, checkpoint_step=2, horizon=5) is False


def test_validation_exposes_fires_when_discovery_after_validation():
    events = [
        _ev(0, "init", payload={"root_task": "T"}),
        _ev(1, "add_subtask", "S1", payload={
            "description": "V", "category": "validation", "weight": 1.0}),
        _ev(3, "update_status", "S1", payload={
            "status": "complete", "evidence": ["e"]}),
        _ev(5, "add_subtask", "S2", payload={
            "description": "P", "category": "product", "weight": 1.0}),
    ]
    assert build_q_labels._label_validation_exposes_new_work(
        events, checkpoint_step=2, horizon=5) is True


def test_validation_exposes_ignores_documentation_discovery():
    """ARTIFACT / DOCUMENTATION discoveries must NOT count."""
    events = [
        _ev(0, "init", payload={"root_task": "T"}),
        _ev(1, "add_subtask", "S1", payload={
            "description": "V", "category": "validation", "weight": 1.0}),
        _ev(3, "update_status", "S1", payload={
            "status": "complete", "evidence": ["e"]}),
        _ev(4, "add_subtask", "S2", payload={
            "description": "doc", "category": "documentation", "weight": 1.0}),
    ]
    assert build_q_labels._label_validation_exposes_new_work(
        events, checkpoint_step=2, horizon=5) is False


# ── stuck_loop_next_window: mask direction ──────────────────────────────────


def test_stuck_loop_masked_when_already_seen():
    events = [
        _ev(0, "init", payload={"root_task": "T"}),
        _ev(1, "add_subtask", "S1", payload={
            "description": "P", "category": "product", "weight": 1.0}),
        _ev(4, "update_status", "S1", payload={"status": "blocked"},
            reason="stuck loop in tool"),
    ]
    assert build_q_labels._label_stuck_loop_next_window(
        events, checkpoint_step=2, horizon=3, repeated_loop_already=True) is False


def test_stuck_loop_unmasked_fires_in_window():
    events = [
        _ev(0, "init", payload={"root_task": "T"}),
        _ev(1, "add_subtask", "S1", payload={
            "description": "P", "category": "product", "weight": 1.0}),
        _ev(4, "update_status", "S1", payload={"status": "blocked"},
            reason="stuck loop"),
    ]
    assert build_q_labels._label_stuck_loop_next_window(
        events, checkpoint_step=2, horizon=3, repeated_loop_already=False) is True


def test_stuck_loop_ignores_blocked_without_loop_keyword():
    events = [
        _ev(0, "init", payload={"root_task": "T"}),
        _ev(1, "add_subtask", "S1", payload={
            "description": "P", "category": "product", "weight": 1.0}),
        _ev(4, "update_status", "S1", payload={"status": "blocked"},
            reason="dependency unavailable"),
    ]
    assert build_q_labels._label_stuck_loop_next_window(
        events, checkpoint_step=2, horizon=3, repeated_loop_already=False) is False


# ── q_baselines: AUROC tie handling ─────────────────────────────────────────


def test_auroc_constant_predictor_returns_one_half():
    """Wrong-impl candidate: ties give 0 or 1 instead of 0.5."""
    auroc = q_baselines._auroc(
        labels=[0, 1, 0, 1, 0, 1],
        probs=[0.5, 0.5, 0.5, 0.5, 0.5, 0.5],
    )
    assert auroc == 0.5


def test_auroc_perfect_separation_is_one():
    auroc = q_baselines._auroc(
        labels=[0, 0, 1, 1],
        probs=[0.1, 0.2, 0.8, 0.9],
    )
    assert auroc == 1.0


def test_auroc_inverted_separation_is_zero():
    auroc = q_baselines._auroc(
        labels=[1, 1, 0, 0],
        probs=[0.1, 0.2, 0.8, 0.9],
    )
    assert auroc == 0.0


def test_auroc_single_class_returns_none():
    assert q_baselines._auroc([1, 1, 1], [0.1, 0.5, 0.9]) is None
    assert q_baselines._auroc([0, 0, 0], [0.1, 0.5, 0.9]) is None


# ── q_baselines: LORO splits ────────────────────────────────────────────────


def test_loro_splits_cover_each_run_exactly_once_as_test():
    splits = q_baselines._loro_splits(["a", "b", "c"])
    assert len(splits) == 3
    test_runs = [next(iter(test)) for _, test in splits]
    assert sorted(test_runs) == ["a", "b", "c"]


def test_loro_splits_train_test_disjoint():
    splits = q_baselines._loro_splits(["a", "b", "c", "d"])
    for train, test in splits:
        assert not (train & test)
        assert train | test == {"a", "b", "c", "d"}


# ── q_baselines: always_mean uses TRAIN base rate (LORO degeneracy) ─────────


def test_always_mean_predicts_train_base_rate_not_global():
    """Wrong-impl candidate: predicting the global (incl. test) rate
    would make held-out predictions equal the population rate.
    Correct LORO must exclude the held-out run from the mean."""
    train_rows = [
        {"label_target_x": "true"} for _ in range(8)
    ] + [{"label_target_x": "false"} for _ in range(2)]
    test_rows = [{"label_target_x": "false"}] * 3
    probs = q_baselines._fit_predict(
        train_rows, test_rows, features=(), target="x")
    assert probs == [0.8, 0.8, 0.8]


def test_always_mean_zero_train_positives_returns_zero():
    train_rows = [{"label_target_x": "false"}] * 5
    test_rows = [{"label_target_x": "true"}] * 2
    probs = q_baselines._fit_predict(
        train_rows, test_rows, features=(), target="x")
    assert probs == [0.0, 0.0]


# ── invariants on the actual pilot dataset ──────────────────────────────────


def test_horizon_monotone_in_horizon():
    """Direction test: extending the look-ahead can only ADD positives,
    never remove them. (One-step-ago information is still in the
    longer window.)"""
    rows_h3 = build_q_labels.build_labels(
        ROOT / "runs" / "swe_agent_pilot",
        ROOT / "datasets" / "swe_agent_estimator_checkpoints.csv",
        horizon=3)
    rows_h7 = build_q_labels.build_labels(
        ROOT / "runs" / "swe_agent_pilot",
        ROOT / "datasets" / "swe_agent_estimator_checkpoints.csv",
        horizon=7)
    by_key_h3 = {(r["run_id"], r["step"]): r for r in rows_h3}
    by_key_h7 = {(r["run_id"], r["step"]): r for r in rows_h7}
    horizon_dependent = (
        "future_progress_drop",
        "product_reopened_after_completion",
        "validation_exposes_new_work",
        "stuck_loop_next_window",
    )
    for key, h3 in by_key_h3.items():
        h7 = by_key_h7[key]
        for col in horizon_dependent:
            if h3[col]:
                assert h7[col], f"{col} at {key}: positive at h=3 lost at h=7"
