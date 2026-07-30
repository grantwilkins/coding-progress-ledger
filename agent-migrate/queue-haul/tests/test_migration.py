"""
Claim: initial copy runs while the source serves, each session pauses once, changed
state is copied once, and ordered moves start eagerly under one shared concurrency
limit.

Plausible wrong implementations:
- Pause before starting the initial copy.
- Switch the initial snapshot after a request changed the conversation.
- Retry catch-up silently or leave the source paused after a failure.
- Copy the original snapshot again after ordered append stages.
- Apply independent replay and KV concurrency limits.
- Wait for one move to commit before starting the next.
- Report initial-copy time as service pause time.
"""

from __future__ import annotations

import threading
import time

from migration import (
    AppendStageResult, MigrationController, Move, RequestResult, SessionState,
)


class FakeRuntime:
    def __init__(self, changed: set[str] = set(), fail: set[tuple[str, str]] = set()):
        self.changed = changed
        self.fail = fail
        self.paused: set[str] = set()
        self.calls: list[tuple] = []
        self.active = self.peak = 0
        self.lock = threading.Lock()

    @staticmethod
    def state(session_id: str, generation: int = 0) -> SessionState:
        return SessionState(session_id, generation, ({"role": "user", "content": str(generation)},), f"h{generation}")

    def snapshot(self, move: Move) -> SessionState:
        self.calls.append(("snapshot", move.session_id))
        return self.state(move.session_id)

    def prepare(self, move: Move, state: SessionState, phase: str) -> RequestResult:
        assert move.session_id not in self.paused or phase == "catch_up"
        with self.lock:
            self.calls.append(("prepare", move.session_id, move.method, phase, state.generation))
            self.active += 1
            self.peak = max(self.peak, self.active)
        time.sleep(0.01)
        with self.lock:
            self.active -= 1
        if (move.session_id, phase) in self.fail:
            raise RuntimeError("copy failed")
        now = time.monotonic_ns()
        return RequestResult(f"{move.session_id}-{phase}", 200, state.context_hash, now, now)

    def pause(self, session_id: str) -> None:
        assert session_id not in self.paused
        self.paused.add(session_id)
        self.calls.append(("pause", session_id))

    def background(self, move: Move, state: SessionState):
        return ()

    def wait_idle(self, session_id: str) -> SessionState:
        self.calls.append(("idle", session_id))
        return self.state(session_id, int(session_id in self.changed))

    def commit(self, move: Move, state: SessionState) -> None:
        assert move.session_id in self.paused
        self.calls.append(("commit", move.session_id, state.generation))
        self.paused.remove(move.session_id)

    def resume_source(self, session_id: str) -> None:
        self.calls.append(("resume", session_id))
        self.paused.remove(session_id)


def test_initial_copy_precedes_pause_and_changed_state_is_copied_once():
    runtime = FakeRuntime(changed={"s"})

    result = MigrationController(runtime, 1).run([Move("s", "replay", 0)])[0]

    assert result.succeeded
    assert [(call[0], call[-1]) for call in runtime.calls] == [
        ("snapshot", "s"), ("prepare", 0), ("pause", "s"), ("idle", "s"),
        ("prepare", 1), ("commit", 1),
    ]
    assert result.committed_state.generation == 1
    assert result.initial_end_ns <= result.pause_start_ns <= result.switch_end_ns


def test_unchanged_state_skips_catch_up():
    runtime = FakeRuntime()

    result = MigrationController(runtime, 1).run([Move("s", "kv_transfer", 0)])[0]

    assert result.succeeded
    assert result.catch_up is None
    assert [call[0] for call in runtime.calls].count("prepare") == 1


def test_ordered_append_stages_advance_the_prepared_state():
    class StagedRuntime(FakeRuntime):
        def background(self, move, state):
            rows = []
            for stage_index in range(4):
                state = self.state(move.session_id, stage_index + 1)
                request = self.prepare(move, state, "append")
                rows.append(AppendStageResult(
                    stage_index, request.start_ns, request.end_ns, state, request,
                    stage_index, stage_index + 1, 10,
                ))
            return tuple(rows)

        def wait_idle(self, session_id):
            self.calls.append(("idle", session_id))
            return self.state(session_id, 4)

    result = MigrationController(StagedRuntime(), 1).run(
        [Move("s", "kv_transfer", 0)]
    )[0]

    assert [row.stage_index for row in result.append_stages] == list(range(4))
    assert result.catch_up is None
    assert result.committed_state.generation == 4


def test_copy_failure_resumes_source_without_commit():
    runtime = FakeRuntime(changed={"s"}, fail={("s", "catch_up")})

    result = MigrationController(runtime, 1).run([Move("s", "replay", 0)])[0]

    assert not result.succeeded
    assert "copy failed" in result.error
    assert "s" not in runtime.paused
    assert [call[0] for call in runtime.calls][-1] == "resume"
    assert not any(call[0] == "commit" for call in runtime.calls)


def test_ordered_moves_start_eagerly_under_one_mixed_method_limit():
    runtime = FakeRuntime()
    moves = [Move("c", "replay", 2), Move("a", "replay", 0), Move("b", "kv_transfer", 1)]

    results = MigrationController(runtime, 2).run(moves)

    assert runtime.peak == 2
    assert [call[1] for call in runtime.calls if call[0] == "snapshot"] == ["a", "b", "c"]
    assert runtime.calls.index(("snapshot", "b")) < next(
        i for i, call in enumerate(runtime.calls) if call[0] == "commit"
    )
    assert [result.move.session_id for result in results] == ["a", "b", "c"]


def test_rejects_duplicate_sessions_and_orders():
    controller = MigrationController(FakeRuntime(), 1)

    for moves in ([Move("s", "replay", 0), Move("s", "kv_transfer", 1)],
                  [Move("a", "replay", 0), Move("b", "kv_transfer", 0)]):
        try:
            controller.run(moves)
        except ValueError:
            pass
        else:
            raise AssertionError("invalid plan was accepted")
