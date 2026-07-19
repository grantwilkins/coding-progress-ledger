from __future__ import annotations

import time
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from threading import Condition
from typing import Callable, Literal, Protocol


Method = Literal["replay", "kv_transfer"]
Phase = Literal["initial", "append", "catch_up"]


@dataclass(frozen=True)
class SessionState:
    session_id: str
    generation: int
    messages: tuple[dict, ...]
    context_hash: str


@dataclass(frozen=True)
class Move:
    session_id: str
    method: Method
    order: int


@dataclass(frozen=True)
class StreamChunk:
    monotonic_ns: int
    byte_count: int


@dataclass(frozen=True)
class RequestResult:
    request_id: str
    status_code: int
    context_hash: str
    start_ns: int
    end_ns: int
    first_byte_ns: int | None = None
    prompt_tokens: int = 0
    output_tokens: int = 0
    processed_tokens: int = 0
    logical_kv_chunks: int = 0
    logical_kv_bytes: int = 0
    wire_bytes: int = 0
    stream_chunks: tuple[StreamChunk, ...] = ()


@dataclass(frozen=True)
class AppendStageResult:
    stage_index: int
    start_ns: int
    end_ns: int
    source_state: SessionState
    destination_request: RequestResult


@dataclass(frozen=True)
class MoveResult:
    move: Move
    queued_ns: int
    initial_start_ns: int
    initial_end_ns: int
    pause_start_ns: int
    idle_ns: int
    catch_up_start_ns: int | None
    catch_up_end_ns: int | None
    switch_start_ns: int
    switch_end_ns: int
    initial: RequestResult | None
    append_stages: tuple[AppendStageResult, ...]
    catch_up: RequestResult | None
    committed_state: SessionState | None
    error: str | None

    @property
    def succeeded(self) -> bool:
        return self.error is None


class Runtime(Protocol):
    def snapshot(self, move: Move) -> SessionState: ...

    def prepare(self, move: Move, state: SessionState, phase: Phase) -> RequestResult: ...

    def background(self, move: Move, state: SessionState) -> tuple[AppendStageResult, ...]: ...

    def pause(self, session_id: str) -> None: ...

    def wait_idle(self, session_id: str) -> SessionState: ...

    def commit(self, move: Move, state: SessionState) -> None: ...

    def resume_source(self, session_id: str) -> None: ...


class MigrationController:
    def __init__(self, runtime: Runtime, concurrency: int, clock: Callable[[], int] = time.monotonic_ns):
        if concurrency < 1:
            raise ValueError("concurrency must be positive")
        self.runtime = runtime
        self.concurrency = concurrency
        self.clock = clock
        self._start = Condition()
        self._next_start = 0

    def run(self, moves: list[Move]) -> list[MoveResult]:
        ordered = sorted(moves, key=lambda move: move.order)
        if len({move.session_id for move in ordered}) != len(ordered):
            raise ValueError("a session may move only once")
        if len({move.order for move in ordered}) != len(ordered):
            raise ValueError("move order must be unique")
        self._next_start = 0
        queued = [self.clock() for _ in ordered]
        with ThreadPoolExecutor(max_workers=self.concurrency) as pool:
            futures = [pool.submit(self._run_one, move, rank, queued[rank]) for rank, move in enumerate(ordered)]
        return [future.result() for future in futures]

    def _run_one(self, move: Move, rank: int, queued_ns: int) -> MoveResult:
        with self._start:
            self._start.wait_for(lambda: rank == self._next_start)
            state = self.runtime.snapshot(move)
            initial_start_ns = self.clock()
            self._next_start += 1
            self._start.notify_all()

        initial = catch_up = committed_state = None
        append_stages: tuple[AppendStageResult, ...] = ()
        initial_end_ns = pause_start_ns = idle_ns = initial_start_ns
        catch_up_start_ns = catch_up_end_ns = None
        switch_start_ns = switch_end_ns = initial_start_ns
        paused = False
        try:
            initial = self.runtime.prepare(move, state, "initial")
            self._check(initial, state)
            initial_end_ns = self.clock()
            append_stages = self.runtime.background(move, state)
            for stage in append_stages:
                self._check(stage.destination_request, stage.source_state)
            prepared = append_stages[-1].source_state if append_stages else state
            pause_start_ns = self.clock()
            self.runtime.pause(move.session_id)
            paused = True
            current = self.runtime.wait_idle(move.session_id)
            idle_ns = self.clock()
            if current.generation != prepared.generation \
                    or current.context_hash != prepared.context_hash:
                catch_up_start_ns = self.clock()
                catch_up = self.runtime.prepare(move, current, "catch_up")
                self._check(catch_up, current)
                catch_up_end_ns = self.clock()
            switch_start_ns = self.clock()
            self.runtime.commit(move, current)
            switch_end_ns = self.clock()
            paused = False
            committed_state = current
            error = None
        except Exception as exc:
            if paused:
                self.runtime.resume_source(move.session_id)
            switch_end_ns = self.clock()
            error = f"{type(exc).__name__}: {exc}"
        return MoveResult(
            move, queued_ns, initial_start_ns, initial_end_ns, pause_start_ns, idle_ns,
            catch_up_start_ns, catch_up_end_ns, switch_start_ns, switch_end_ns,
            initial, append_stages, catch_up, committed_state, error,
        )

    @staticmethod
    def _check(result: RequestResult, state: SessionState) -> None:
        if result.status_code != 200:
            raise RuntimeError(f"request {result.request_id} returned HTTP {result.status_code}")
        if result.context_hash != state.context_hash:
            raise RuntimeError(f"request {result.request_id} prepared stale state")
