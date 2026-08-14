"""Event-driven feasibility repair without a planning tick."""

from __future__ import annotations

from dataclasses import dataclass, replace
from math import inf
from typing import Callable, Literal


Status = Literal["pending", "running", "committed", "failed", "cancelled"]
Outcome = Literal["applied", "rejected", "shadow"]


@dataclass(frozen=True)
class Assignment:
    method: str
    destination: str
    pool: str = ""


@dataclass(frozen=True)
class Attempt:
    session_id: str
    generation: int
    assignment: Assignment
    status: Status
    total_work: float
    completed_work: float
    observed_s: float
    planned_commit_s: float
    commit_overhead_s: float = 0.0
    rate: float | None = None
    soft_changed: bool = False
    repairable: bool = True


@dataclass(frozen=True)
class AttemptUpdate:
    session_id: str
    generation: int
    status: Status
    total_work: float
    completed_work: float


@dataclass(frozen=True)
class Failure:
    event_id: str
    kind: str
    sessions: tuple[str, ...] = ()


@dataclass(frozen=True)
class PrefillCapacity:
    """Observed pool prefill capacity at one measured context."""

    pool: str
    context_tokens: float
    tokens_per_s: float

    def __post_init__(self):
        if not self.pool or self.context_tokens <= 0 or self.tokens_per_s <= 0:
            raise ValueError("invalid observed prefill capacity")


@dataclass(frozen=True)
class ObservationBatch:
    sample_id: int
    now_s: float
    attempts: tuple[AttemptUpdate, ...] = ()
    route_rates: tuple[tuple[str, float], ...] = ()
    replay_rates: tuple[tuple[str, float], ...] = ()
    failures: tuple[Failure, ...] = ()
    prefill_capacities: tuple[PrefillCapacity, ...] = ()


@dataclass(frozen=True)
class LedgerSnapshot:
    now_s: float
    target_watts: float
    credit_deadline_s: float
    eta_guard_s: float
    plan_version: int
    budget_version: int
    committed: frozenset[str]
    source_sessions: frozenset[str]
    attempts: tuple[Attempt, ...]
    route_rates: tuple[tuple[str, float], ...]
    replay_rates: tuple[tuple[str, float], ...]
    prefill_capacities: tuple[PrefillCapacity, ...] = ()


@dataclass(frozen=True)
class RepairRequest:
    request_id: int
    trigger: str
    retry: int
    snapshot: LedgerSnapshot


@dataclass(frozen=True)
class RepairMove:
    session_id: str
    assignment: Assignment
    duration_s: float
    total_work: float
    commit_overhead_s: float = 0.0


@dataclass(frozen=True)
class RepairResult:
    request_id: int
    budget_version: int
    moves: tuple[RepairMove, ...]
    attainable_watts: float
    reaches_target: bool


@dataclass(frozen=True)
class PlanChange:
    session_id: str
    generation: int
    assignment: Assignment | None


@dataclass(frozen=True)
class ProposedDiff:
    proposal_id: int
    request_id: int
    trigger: str
    changes: tuple[PlanChange, ...]
    moves: tuple[RepairMove, ...]


@dataclass(frozen=True)
class RevisedMaximum:
    request_id: int
    attainable_watts: float


Decision = RepairRequest | ProposedDiff | RevisedMaximum | None


class FeasibilityRepairController:
    def __init__(self, attempts: tuple[Attempt, ...], source_sessions,
                 target_watts: float, credit_deadline_s: float, eta_guard_s: float,
                 watts: Callable[[frozenset[str]], float]):
        if target_watts < 0 or credit_deadline_s <= 0 or eta_guard_s < 0:
            raise ValueError("invalid repair target, deadline, or ETA guard")
        if len({attempt.session_id for attempt in attempts}) != len(attempts):
            raise ValueError("attempt sessions must be unique")
        self.attempts = {attempt.session_id: attempt for attempt in attempts}
        self.source_sessions = set(source_sessions)
        self.target_watts, self.credit_deadline_s = target_watts, credit_deadline_s
        self.eta_guard_s, self.watts = eta_guard_s, watts
        self.committed = {a.session_id for a in attempts if a.status == "committed"}
        self.source_sessions -= self.committed
        self.route_rates: dict[str, float] = {}
        self.replay_rates: dict[str, float] = {}
        self.prefill_capacities: dict[str, PrefillCapacity] = {}
        self.plan_version = self.budget_version = self.last_sample_id = 0
        self.plan_feasible, self.consecutive_misses = True, 0
        self.feasible_samples = self.feasibility_epoch = 0
        self.progress_gate: set[str] = set()
        self.handled_failures: set[str] = set()
        self.pending_failures: set[str] = set()
        self._next_id = 1
        self._active: RepairRequest | None = None
        self._proposal: ProposedDiff | None = None

    def snapshot(self, now_s: float) -> LedgerSnapshot:
        return LedgerSnapshot(
            now_s, self.target_watts, self.credit_deadline_s, self.eta_guard_s,
            self.plan_version, self.budget_version, frozenset(self.committed),
            frozenset(self.source_sessions),
            tuple(sorted(self.attempts.values(), key=lambda a: a.session_id)),
            tuple(sorted(self.route_rates.items())), tuple(sorted(self.replay_rates.items())),
            tuple(sorted(self.prefill_capacities.values(), key=lambda row: row.pool)),
        )

    def _eta(self, attempt: Attempt, now_s: float) -> float:
        if attempt.status == "committed":
            return attempt.observed_s
        if attempt.status not in {"pending", "running"}:
            return inf
        if attempt.rate is None:
            return attempt.planned_commit_s + self.eta_guard_s
        if attempt.rate <= 0:
            return inf
        progress_eta = now_s \
            + (attempt.total_work - attempt.completed_work) / attempt.rate \
            + attempt.commit_overhead_s
        return max(attempt.planned_commit_s, progress_eta) + self.eta_guard_s

    def _forecast(self, now_s: float, moves: tuple[RepairMove, ...] | None = None) -> float:
        on_time = set(self.committed)
        if moves is None:
            on_time |= {
                attempt.session_id for attempt in self.attempts.values()
                if self._eta(attempt, now_s) <= self.credit_deadline_s
            }
        else:
            for move in moves:
                current = self.attempts.get(move.session_id)
                same = current and current.assignment == move.assignment \
                    and current.status in {"pending", "running"}
                eta = self._eta(current, now_s) if same else (
                    now_s + move.duration_s + self.eta_guard_s
                )
                if eta <= self.credit_deadline_s:
                    on_time.add(move.session_id)
        return self.watts(frozenset(on_time))

    @staticmethod
    def _valid(attempt: Attempt, update: AttemptUpdate, now_s: float) -> None:
        if update.total_work <= 0 or not 0 <= update.completed_work <= update.total_work \
                or update.total_work != attempt.total_work \
                or update.completed_work < attempt.completed_work \
                or now_s <= attempt.observed_s:
            raise ValueError("invalid attempt progress")
        terminal = {"committed", "failed", "cancelled"}
        if attempt.status in terminal and update.status != attempt.status:
            raise ValueError("terminal attempt status changed")

    def _apply(self, batch: ObservationBatch) -> tuple[str, ...]:
        for update in batch.attempts:
            attempt = self.attempts.get(update.session_id)
            if attempt is None:
                raise ValueError(f"unknown attempt {update.session_id!r}")
            if update.generation < attempt.generation:
                continue
            if update.generation > attempt.generation:
                raise ValueError("future attempt generation")
            self._valid(attempt, update, batch.now_s)
            rate = (update.completed_work - attempt.completed_work) \
                / (batch.now_s - attempt.observed_s)
            self.attempts[update.session_id] = replace(
                attempt, status=update.status, completed_work=update.completed_work,
                observed_s=batch.now_s, rate=rate,
            )
            if update.completed_work > attempt.completed_work:
                self.progress_gate.discard(update.session_id)
            if update.status == "committed":
                self.committed.add(update.session_id)
                self.source_sessions.discard(update.session_id)
        changed = False
        for target, updates in ((self.route_rates, batch.route_rates),
                                (self.replay_rates, batch.replay_rates)):
            for key, value in updates:
                if value <= 0:
                    raise ValueError("observed rates must be positive")
                changed |= target.get(key) != value
                target[key] = value
        for value in batch.prefill_capacities:
            changed |= self.prefill_capacities.get(value.pool) != value
            self.prefill_capacities[value.pool] = value
        self.budget_version += int(changed)
        failures = []
        for failure in batch.failures:
            if failure.event_id in self.handled_failures:
                continue
            self.handled_failures.add(failure.event_id)
            failures.append(failure.event_id)
            for session_id in failure.sessions:
                attempt = self.attempts.get(session_id)
                if attempt and attempt.status in {"pending", "running"}:
                    self.attempts[session_id] = replace(
                        attempt, status="failed", observed_s=batch.now_s,
                    )
        return tuple(failures)

    def observe(self, batch: ObservationBatch) -> Decision:
        if batch.sample_id <= self.last_sample_id:
            return None
        if batch.now_s < 0:
            raise ValueError("observation time must be nonnegative")
        self.last_sample_id = batch.sample_id
        failures = self._apply(batch)
        feasible = self._forecast(batch.now_s) >= self.target_watts
        if feasible:
            self.consecutive_misses = 0
            if not self.plan_feasible:
                self.feasible_samples += 1
                if self.feasible_samples >= 2 and not self.progress_gate:
                    self.plan_feasible = True
            return None
        self.feasible_samples = 0
        if failures:
            if self._active or self._proposal:
                self.pending_failures.update(failures)
                return None
            return self._request(batch.now_s, "hard:" + ",".join(failures))
        if not self.plan_feasible or self.progress_gate or self._active or self._proposal:
            return None
        self.consecutive_misses += 1
        if self.consecutive_misses < 2:
            return None
        self.plan_feasible = False
        self.feasibility_epoch += 1
        return self._request(batch.now_s, f"soft:{self.feasibility_epoch}")

    def _request(self, now_s: float, trigger: str, retry: int = 0) -> RepairRequest:
        request = RepairRequest(self._next_id, trigger, retry, self.snapshot(now_s))
        self._next_id += 1
        self._active = request
        return request

    def _pending(self, now_s: float) -> RepairRequest | None:
        if not self.pending_failures or self._forecast(now_s) >= self.target_watts:
            self.pending_failures.clear()
            return None
        failures = ",".join(sorted(self.pending_failures))
        self.pending_failures.clear()
        return self._request(now_s, "hard:" + failures)

    def complete_repair(self, result: RepairResult, now_s: float) -> Decision:
        request = self._active
        if request is None or result.request_id != request.request_id:
            raise ValueError("repair result does not match the active request")
        self._active = None
        pending = self._pending(now_s)
        if pending:
            return pending
        if self._forecast(now_s) >= self.target_watts:
            return None
        stale = result.budget_version != self.budget_version
        reaches = result.reaches_target and self._forecast(now_s, result.moves) \
            >= self.target_watts
        if (stale or result.reaches_target and not reaches) and request.retry == 0:
            return self._request(now_s, request.trigger, 1)
        if not reaches:
            return RevisedMaximum(result.request_id, result.attainable_watts)
        moves = {move.session_id: move for move in result.moves}
        changes = []
        soft = request.trigger.startswith("soft:")
        for session_id, attempt in self.attempts.items():
            if attempt.status not in {"pending", "running"}:
                continue
            move = moves.get(session_id)
            if move is None or move.assignment != attempt.assignment:
                if soft and attempt.soft_changed:
                    raise ValueError("soft repair changed a session twice")
                changes.append(PlanChange(
                    session_id, attempt.generation + 1,
                    None if move is None else move.assignment,
                ))
        for session_id, move in moves.items():
            if session_id not in self.attempts:
                changes.append(PlanChange(session_id, 0, move.assignment))
        proposal = ProposedDiff(
            self._next_id, result.request_id, request.trigger,
            tuple(sorted(changes, key=lambda row: row.session_id)), result.moves,
        )
        self._next_id += 1
        self._proposal = proposal
        return proposal

    def acknowledge(self, proposal_id: int, outcome: Outcome, now_s: float) -> Decision:
        proposal = self._proposal
        if proposal is None or proposal.proposal_id != proposal_id:
            raise ValueError("unknown repair proposal")
        self._proposal = None
        if outcome == "shadow":
            return self._pending(now_s)
        if outcome == "rejected":
            return self._request(now_s, f"hard:proposal-{proposal_id}") \
                if self._forecast(now_s) < self.target_watts else None
        if outcome != "applied":
            raise ValueError("invalid proposal outcome")
        moves = {move.session_id: move for move in proposal.moves}
        soft = proposal.trigger.startswith("soft:")
        for change in proposal.changes:
            current, move = self.attempts.get(change.session_id), moves.get(change.session_id)
            if move is None:
                self.attempts[change.session_id] = replace(
                    current, generation=change.generation, status="cancelled",
                    observed_s=now_s, soft_changed=current.soft_changed or soft,
                )
                continue
            self.attempts[change.session_id] = Attempt(
                change.session_id, change.generation, move.assignment, "pending",
                move.total_work, 0, now_s, now_s + move.duration_s,
                move.commit_overhead_s, soft_changed=bool(current and soft),
            )
            self.progress_gate.add(change.session_id)
        self.plan_version += 1
        self.plan_feasible, self.consecutive_misses = False, 0
        return self._pending(now_s)
