"""Event simulation for profile-driven session migration."""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass, replace
import heapq
from typing import Literal

import numpy as np

from profiles import ModelProfile
from power_model import ExpectedPower


MoveMethod = Literal["replay", "kv_transfer", "replay_on_request"]
SessionState = Literal["active", "cold"]
FinalState = Literal["awake", "sleep", "off"]
MOVE_METHODS_BY_STATE: dict[SessionState, tuple[MoveMethod, ...]] = {
    "active": ("replay", "kv_transfer"),
    "cold": ("replay_on_request",),
}


@dataclass(frozen=True)
class NetworkLink:
    link_id: str
    bytes_per_s: float

    def __post_init__(self):
        if not self.link_id or self.bytes_per_s <= 0:
            raise ValueError("network links require an id and positive capacity")


@dataclass(frozen=True)
class PowerNode:
    node_id: str
    gpus: int = 8
    local: bool = True
    site_id: str = ""

    def __post_init__(self):
        if not self.node_id or self.gpus < 1:
            raise ValueError("power nodes require an id and at least one GPU")


@dataclass(frozen=True)
class ServingInstance:
    instance_id: str
    gpu_nodes: tuple[str, ...]

    def __post_init__(self):
        if not self.instance_id or not self.gpu_nodes:
            raise ValueError("serving instances require an id and assigned GPUs")


@dataclass(frozen=True)
class SimRequest:
    gap_s: float
    prompt_tokens: int
    output_tokens: int

    def __post_init__(self):
        if self.gap_s < 0 or self.prompt_tokens < 1 or self.output_tokens < 0:
            raise ValueError("invalid request")


@dataclass(frozen=True)
class SimSession:
    session_id: str
    source_instance: str
    context_tokens: int
    expected_f: float
    expected_g: float
    log_bytes: int
    requests: tuple[SimRequest, ...] = ()
    movable: bool = True
    wake_probability: float = 0.0
    state: SessionState = "active"
    expected_growth_tokens_per_s: float = 0.0

    def __post_init__(self):
        if not self.session_id or not self.source_instance or self.context_tokens < 1 \
                or min(self.expected_f, self.expected_g) < 0 or self.log_bytes < 1 \
                or not 0 <= self.wake_probability <= 1 or self.state not in MOVE_METHODS_BY_STATE \
                or self.expected_growth_tokens_per_s < 0 \
                or self.state == "cold" and (self.expected_f or self.expected_g) \
                or self.state == "active" and self.wake_probability:
            raise ValueError("invalid session")


@dataclass(frozen=True)
class PlannedMove:
    session_id: str
    destination_instance: str
    method: MoveMethod
    order: int
    path: tuple[str, ...]
    rate_limit_bytes_per_s: float | None = None
    quiesce_s: float | None = None
    destination_pool: str | None = None

    def __post_init__(self):
        if not self.session_id or not self.destination_instance or self.method not in {
            "replay", "kv_transfer", "replay_on_request"
        } or self.order < 0 or self.rate_limit_bytes_per_s is not None \
                and self.rate_limit_bytes_per_s <= 0 \
                or self.quiesce_s is not None and self.quiesce_s < 0:
            raise ValueError("invalid planned move")


@dataclass(frozen=True)
class ExecutionScenario:
    deadline_s: float
    end_s: float
    power_limit_w: float
    final_state: FinalState
    controller_delay_s: float
    nodes: tuple[PowerNode, ...]
    instances: tuple[ServingInstance, ...]
    sessions: tuple[SimSession, ...]
    links: tuple[NetworkLink, ...]
    assumed_shutdown_s: float | None = None

    def __post_init__(self):
        if self.deadline_s <= 0 or self.end_s < self.deadline_s or self.power_limit_w < 0 \
                or not 0 <= self.controller_delay_s <= self.deadline_s \
                or self.final_state not in {"awake", "sleep", "off"} \
                or self.assumed_shutdown_s is not None and self.assumed_shutdown_s < 0 \
                or self.final_state == "off" and self.assumed_shutdown_s is None:
            raise ValueError("invalid execution scenario")


@dataclass(frozen=True)
class ExecutionEvent:
    time_s: float
    event: str
    session_id: str = ""
    node_id: str = ""
    detail: str = ""


@dataclass(frozen=True)
class SessionExecution:
    session_id: str
    method: MoveMethod
    initial_start_s: float | None
    initial_ready_s: float | None
    pause_s: float | None
    idle_s: float | None
    catch_up_start_s: float | None
    catch_up_ready_s: float | None
    switch_s: float | None
    committed_s: float | None
    wake_start_s: float | None
    wake_ready_s: float | None
    initial_replay_start_s: float | None = None
    catch_up_replay_start_s: float | None = None


@dataclass(frozen=True)
class NetworkExecution:
    session_id: str
    phase: str
    bytes: int
    transferred_bytes: int
    remaining_bytes: int
    path: tuple[str, ...]
    start_s: float
    end_s: float | None


@dataclass
class QueueExecution:
    session_id: str
    phase: str
    destination_instance: str
    bytes: int
    arrival_s: float
    start_s: float | None
    end_s: float | None
    depth_at_arrival: int
    bytes_at_arrival: int


@dataclass(frozen=True)
class PoolServiceExecution:
    pool_id: str
    facet: int
    time_s: float
    demand_replicas: float
    stable_capacity_replicas: float
    queued_replica_s: float
    peak_queued_replica_s: float
    debt_budget_replica_s: float
    required_recovery_s: float
    within_contract: bool


@dataclass(frozen=True)
class ExecutionResult:
    events: tuple[ExecutionEvent, ...]
    sessions: tuple[SessionExecution, ...]
    requests: tuple["RequestExecution", ...]
    network: tuple[NetworkExecution, ...]
    queues: tuple[QueueExecution, ...]
    power: tuple[tuple[float, float, float], ...]
    modeled_source_power_at_deadline_w: float
    deadline_met: bool
    migration_makespan_s: float | None
    final_state_ready_s: float | None
    makespan_s: float
    pool_service: tuple[PoolServiceExecution, ...] = ()

    @property
    def completed_sessions(self) -> int:
        return sum(row.committed_s is not None for row in self.sessions)

    def requests_started_by(self, deadline_s: float) -> bool:
        arrived = {
            (event.session_id, int(event.detail)) for event in self.events
            if event.event == "request_arrival" and event.time_s <= deadline_s
        }
        started = {
            (request.session_id, request.request_index) for request in self.requests
            if request.start_s <= deadline_s
        }
        return arrived <= started


@dataclass
class _Flow:
    flow_id: int
    move_index: int
    phase: str
    remaining: float
    path: tuple[str, ...]
    rate_path: tuple[object, ...]
    bytes: int
    start: float


@dataclass(frozen=True)
class RequestExecution:
    session_id: str
    request_index: int
    instance_id: str
    arrival_s: float
    start_s: float
    prefill_end_s: float
    end_s: float
    prompt_tokens: int
    output_tokens: int


@dataclass
class _MoveState:
    move: PlannedMove
    snapshot_tokens: int = 0
    initial_start: float | None = None
    initial_ready: float | None = None
    pause: float | None = None
    idle: float | None = None
    catch_start: float | None = None
    catch_ready: float | None = None
    switch: float | None = None
    committed: float | None = None
    wake_start: float | None = None
    wake_ready: float | None = None
    initial_replay_start: float | None = None
    catch_replay_start: float | None = None
    copied_blocks: int = 0
    scheduled_blocks: int = 0
    append_pending: int = 0
    final_started: bool = False


def fair_link_rates(paths: dict[int, tuple[str, ...]], links: dict[str, float]) -> dict[int, float]:
    """Equal-share active flows at every bottleneck, redistributing unused capacity."""
    if not paths:
        return {}
    for path in paths.values():
        if not path or len(set(path)) != len(path) or any(link not in links for link in path):
            raise ValueError("every flow requires a valid path without duplicate links")
    rates, residual, active = {flow: 0.0 for flow in paths}, dict(links), set(paths)
    members = {link: set() for link in links}
    for flow, path in paths.items():
        for link in path:
            members[link].add(flow)
    counts = {link: len(flows) for link, flows in members.items() if flows}
    while active:
        share, bottleneck = min(
            ((residual[link] / count, link) for link, count in counts.items() if count),
            key=lambda item: item[0],
        )
        for flow in active:
            rates[flow] += share
        for link, count in counts.items():
            residual[link] -= share * count
        blocked = members[bottleneck] & active
        active -= blocked
        for flow in blocked:
            for link in paths[flow]:
                counts[link] -= 1
                if not counts[link]:
                    del counts[link]
    return rates


def step_average(points, end_s: float, window_s: float, column: int = 1) -> float:
    """Average a stepwise signal over [end_s - window_s, end_s]."""
    start = end_s - window_s
    if window_s <= 0 or not points or points[0][0] > start:
        raise ValueError("power points must cover a positive averaging window")
    area, value, cursor = 0.0, points[0][column], start
    for point in points:
        if point[0] <= start:
            value = point[column]
        elif point[0] <= end_s:
            area += (point[0] - cursor) * value
            cursor, value = point[0], point[column]
    return (area + (end_s - cursor) * value) / window_s


def fluid_service_completion(work, capacity, arrivals=None):
    """Return processor-sharing completion times for divisible work."""
    work = np.asarray(work, float)
    arrivals = np.zeros(len(work)) if arrivals is None else np.asarray(arrivals, float)
    if work.ndim != 1 or arrivals.shape != work.shape or capacity <= 0 \
            or np.any(work < 0) or np.any(arrivals < 0):
        raise ValueError("invalid fluid service workload")
    order = np.argsort(arrivals, kind="stable")
    completed, active, cursor, virtual = np.empty(len(work)), [], 0, 0.0
    time = 0.0
    while cursor < len(work) or active:
        arrival = arrivals[order[cursor]] if cursor < len(work) else np.inf
        finish = time + (active[0][0] - virtual) * len(active) / capacity \
            if active else np.inf
        next_time = min(arrival, finish)
        if active:
            virtual += (next_time - time) * capacity / len(active)
        time = next_time
        if active and finish <= arrival:
            virtual = max(virtual, active[0][0])
        while active and active[0][0] <= virtual + 1e-12:
            _, index = heapq.heappop(active)
            completed[index] = time
        while cursor < len(work) and arrivals[order[cursor]] <= time + 1e-12:
            index = int(order[cursor])
            if work[index] == 0:
                completed[index] = time
            else:
                heapq.heappush(active, (virtual + work[index], index))
            cursor += 1
    return completed


def fluid_pipeline_completion(work, starts, ends, capacity):
    """Completion of ordered divisible work streaming from an upstream service."""
    work, starts, ends = map(lambda value: np.asarray(value, float), (work, starts, ends))
    if work.ndim != 1 or starts.shape != work.shape or ends.shape != work.shape \
            or capacity <= 0 or np.any(work < 0) or np.any(starts < 0) \
            or np.any(ends < starts) or np.any(starts[1:] < ends[:-1]):
        raise ValueError("invalid fluid pipeline workload")
    completed, backlog, previous = np.empty(len(work)), 0.0, 0.0
    for i, (amount, begin, end) in enumerate(zip(work, starts, ends)):
        backlog = max(0.0, backlog - capacity * (begin - previous))
        backlog = max(0.0, backlog + amount - capacity * (end - begin))
        completed[i] = end + backlog / capacity
        previous = end
    return completed


class ExecutionSimulator:
    def __init__(self, scenario: ExecutionScenario, profile: ModelProfile,
                 moves: tuple[PlannedMove, ...], case_id: str = "central",
                 detailed: bool = True, destination=None):
        self.scenario, self.profile, self.case = scenario, profile, profile.case(case_id)
        self.detailed = detailed
        self.nodes = {n.node_id: n for n in scenario.nodes}
        self.instances = {i.instance_id: i for i in scenario.instances}
        self.sessions = {s.session_id: s for s in scenario.sessions}
        self.links = {link.link_id: link.bytes_per_s for link in scenario.links}
        self.kv_links = {
            instance: ("kv_destination", instance) for instance in self.instances
        }
        self.moves = tuple(sorted(moves, key=lambda m: m.order))
        self.migration_components = {}
        if destination:
            types = destination.type_by_id
            self.migration_components = {
                replica.replica_id: types[pool.type_id].migration or {}
                for pool in destination.pools for replica in pool.replicas
            }
        self.pace_links = {
            index: ("pace", index) for index, move in enumerate(self.moves)
            if move.rate_limit_bytes_per_s is not None
        }
        self.rate_links = {
            **self.links,
            **{link: (self.migration_components.get(instance, {})
                      .get("kv_transfer").kv_ingest_bytes_per_s
                      if self.migration_components.get(instance, {}).get("kv_transfer")
                      and self.migration_components[instance]["kv_transfer"]
                      .kv_ingest_bytes_per_s is not None
                      else self.case.kv_transfer.destination_bytes_per_s)
               for instance, link in self.kv_links.items()},
            **{
                self.pace_links[index]: self.moves[index].rate_limit_bytes_per_s
                for index in self.pace_links
            },
        }
        self._validate()
        self.time = 0.0
        self.heap: list[tuple[float, int, str, object]] = []
        self.sequence = 0
        self.flows: dict[int, _Flow] = {}
        self.next_flow = 0
        self.link_flows = {link: set() for link in self.rate_links}
        self.changed_links: set[object] = set()
        self.states = [_MoveState(m) for m in self.moves]
        self.move_index = {state.move.session_id: i for i, state in enumerate(self.states)}
        self.context = {s.session_id: s.context_tokens for s in scenario.sessions}
        self.active_request_end = {s.session_id: 0.0 for s in scenario.sessions}
        self.active_request_instance: dict[str, str] = {}
        self.serving_active: set[str] = set()
        self.serving_waiting: dict[str, deque[tuple[str, int, float]]] = {}
        self.quiescing = set()
        self.paused = set()
        self.power_model = ExpectedPower(scenario, profile, case_id)
        self.node_state = {n.node_id: "awake" for n in scenario.nodes}
        self.node_ready: dict[str, float] = {}
        self.active_actions: dict[object, tuple[str, bool, str]] = {}
        self.action_counts: dict[tuple[str, bool, str], int] = {}
        self.action_power_w = {True: 0.0, False: 0.0}
        self.action_group_counts = {True: 0, False: 0}
        self.deferred = set()
        self.waking = set()
        self.pending_requests: dict[str, tuple[int, float]] = {}
        self.endpoint_active: dict[tuple[str, str], int] = {}
        self.endpoint_waiting: dict[
            tuple[str, str], deque[tuple[int, str, tuple[int, int]]]
        ] = {}
        self.kv_active: dict[str, int] = {}
        self.kv_waiting: dict[
            str, deque[tuple[int, str, tuple[int, int], int, QueueExecution | None]]
        ] = {}
        self.kv_waiting_bytes: dict[str, int] = {}
        self.kv_records: dict[tuple[int, str], QueueExecution] = {}
        self.events: list[ExecutionEvent] = []
        self.power: list[tuple[float, float, float]] = []
        self.requests: list[RequestExecution] = []
        self.queues: list[QueueExecution] = []
        self.request_arrivals: set[tuple[str, int]] = set()
        self.network: list[NetworkExecution] = []

    def _validate(self):
        if self.scenario.deadline_s < self.profile.power_window_s:
            raise ValueError("deadline must cover the profile power window")
        if len(self.nodes) != len(self.scenario.nodes) or len(self.instances) != len(self.scenario.instances) \
                or len(self.sessions) != len(self.scenario.sessions):
            raise ValueError("node, instance, and session ids must be unique")
        used = {node: 0 for node in self.nodes}
        for instance in self.instances.values():
            if len(instance.gpu_nodes) != self.profile.tensor_parallel:
                raise ValueError("serving instance size must match profile tensor parallelism")
            for node in instance.gpu_nodes:
                if node not in used:
                    raise ValueError(f"unknown node {node!r}")
                used[node] += 1
        if any(used[n] > self.nodes[n].gpus for n in used):
            raise ValueError("serving instances exceed GPU capacity")
        if len({m.session_id for m in self.moves}) != len(self.moves) \
                or len({m.order for m in self.moves}) != len(self.moves):
            raise ValueError("moves require unique sessions and orders")
        for move in self.moves:
            if move.session_id not in self.sessions or move.destination_instance not in self.instances:
                raise ValueError("move references an unknown session or destination")
            session = self.sessions[move.session_id]
            if session.source_instance == move.destination_instance:
                raise ValueError("a move requires different source and destination instances")
            if not self.sessions[move.session_id].movable:
                raise ValueError(f"session {move.session_id!r} cannot move")
            if move.method not in MOVE_METHODS_BY_STATE[session.state]:
                raise ValueError(f"{move.method} is invalid for a {session.state} session")
            if move.method == "replay_on_request" and session.log_bytes <= 0:
                raise ValueError("replay_on_request requires a durable session log")
            if not move.path or any(link not in self.links for link in move.path):
                raise ValueError("move path contains an unknown link")
        resident = {instance: 0 for instance in self.instances}
        active = set()
        for session in self.sessions.values():
            if session.state == "active":
                resident[session.source_instance] += self.profile.kv_admission_tokens(
                    session.context_tokens)
                active.add(session.session_id)
        if any(tokens > self.profile.kv_capacity_tokens for tokens in resident.values()):
            raise ValueError("serving instance exceeds resident KV capacity")
        self.resident_tokens, self.resident_sessions = resident.copy(), active
        for move in self.moves:
            session = self.sessions[move.session_id]
            if session.state == "active":
                tokens = self.profile.kv_admission_tokens(session.context_tokens)
                resident[session.source_instance] -= tokens
                resident[move.destination_instance] += tokens
        if any(tokens > self.profile.kv_capacity_tokens for tokens in resident.values()):
            raise ValueError("serving instance exceeds resident KV capacity")

    def _schedule(self, when: float, kind: str, payload=None):
        self.sequence += 1
        heapq.heappush(self.heap, (when, self.sequence, kind, payload))

    def _event(self, name: str, session: str = "", node: str = "", detail: str = ""):
        if self.detailed:
            self.events.append(ExecutionEvent(self.time, name, session, node, detail))

    def _start_action(self, key, action: str, instance: str | None = None,
                      node: str | None = None):
        if (instance is None) == (node is None) or key in self.active_actions:
            raise RuntimeError("an action requires one unused key and one resource")
        resource = node or instance
        local = self.nodes[node].local if node else self.nodes[self.instances[instance].gpu_nodes[0]].local
        group = action, local, resource
        old_count = self.action_counts.get(group, 0)
        profile = self.case.action_power_w[action]
        old_power = profile.power(1, True) if local and old_count else (
            profile.power(old_count, False) if old_count else 0.0
        )
        new_power = old_power if local and old_count else profile.power(
            old_count + 1, local
        )
        self.action_counts[group] = old_count + 1
        self.action_power_w[local] += new_power - old_power
        if not old_count:
            self.action_group_counts[local] += 1
        self.active_actions[key] = action, local, resource

    def _stop_action(self, key):
        action, local, resource = self.active_actions.pop(key)
        group = action, local, resource
        old_count = self.action_counts[group]
        profile = self.case.action_power_w[action]
        old_power = profile.power(1, True) if local else profile.power(
            old_count, False
        )
        new_power = old_power if local and old_count > 1 else (
            profile.power(old_count - 1, False) if old_count > 1 else 0.0
        )
        if old_count > 1:
            self.action_counts[group] = old_count - 1
        else:
            del self.action_counts[group]
            self.action_group_counts[local] -= 1
        self.action_power_w[local] += new_power - old_power
        if not self.action_group_counts[local]:
            self.action_power_w[local] = 0.0

    def _action_power(self, local: bool) -> float:
        return self.action_power_w[local]

    def _node_power(self, local: bool) -> float:
        return self.power_model.power(local) + self._action_power(local)

    def _record_power(self, force: bool = False):
        point = (self.time, self._node_power(True), self._node_power(False))
        if not self.power or point[1:] != self.power[-1][1:] or force:
            self.power.append(point)

    def _start(self, index: int):
        state = self.states[index]
        state.snapshot_tokens = self.context[state.move.session_id]
        state.scheduled_blocks = self.case.kv_transfer.sealed_blocks(
            state.snapshot_tokens
        )
        state.initial_start = self.time
        self._event("initial_start", state.move.session_id, detail=state.move.method)
        setup = self.case.kv_transfer.setup_s if state.move.method == "kv_transfer" else 0.0
        self._schedule(self.time + setup, "prepare", (index, "initial"))

    def _payload(self, index: int, phase: str) -> tuple[int, int, int]:
        state, session = self.states[index], self.sessions[self.states[index].move.session_id]
        tokens = state.snapshot_tokens if phase == "initial" else self.context[session.session_id]
        if state.move.method == "replay":
            if phase == "catch_up":
                tokens -= state.snapshot_tokens
            ratio = session.log_bytes / session.context_tokens
            return max(1, round(tokens * ratio)), tokens, 0
        if state.move.method == "replay_on_request":
            byte_count = session.log_bytes if phase == "wake" else 0
            return byte_count, tokens if phase == "wake" else 0, 0
        blocks = self.case.kv_transfer.sealed_blocks(tokens)
        if phase == "catch_up":
            blocks -= state.copied_blocks
            tail = self.case.kv_transfer.tail_tokens(tokens)
            return max(0, blocks) * self.case.kv_transfer.block_bytes, tail, max(0, blocks)
        return self.case.kv_transfer.sealed_bytes(tokens), 0, blocks

    def _path(self, state: _MoveState, phase: str) -> tuple[str, ...]:
        return state.move.path

    def _prepare(self, index: int, phase: str,
                 payload: tuple[int, int, int] | None = None):
        state = self.states[index]
        byte_count, replay_tokens, blocks = payload or self._payload(index, phase)
        detail = (replay_tokens, blocks)
        if state.move.method == "kv_transfer":
            destination = state.move.destination_instance
            waiting = self.kv_waiting.setdefault(destination, deque())
            queued = self.kv_active.get(destination, 0) \
                >= self.profile.max_destination_kv_streams
            queued_bytes = self.kv_waiting_bytes.get(destination, 0) + byte_count
            record = QueueExecution(
                state.move.session_id, phase, destination, byte_count, self.time, None, None,
                len(waiting) + 1 if queued else 0, queued_bytes if queued else 0,
            ) if self.detailed else None
            if record:
                self.queues.append(record)
                self.kv_records[index, phase] = record
            if queued:
                waiting.append((index, phase, detail, byte_count, record))
                self.kv_waiting_bytes[destination] = queued_bytes
                self._event("kv_queued", state.move.session_id, detail=destination)
                return
            self._start_kv(index, phase, detail, byte_count, record)
            return
        self._transfer(index, phase, byte_count, detail)

    def _start_kv(self, index: int, phase: str, detail: tuple[int, int], byte_count: int,
                  record: QueueExecution | None):
        destination = self.states[index].move.destination_instance
        self.kv_active[destination] = self.kv_active.get(destination, 0) + 1
        if record:
            record.start_s = self.time
        self._transfer(index, phase, byte_count, detail)

    def _transfer(self, index: int, phase: str, byte_count: int, detail: tuple[int, int]):
        state = self.states[index]
        if byte_count:
            source = self.sessions[state.move.session_id].source_instance
            action = "catch_up" if phase == "catch_up" else state.move.method
            self._start_action((index, phase, "source"), action, source)
            path = self._path(state, phase)
            rate_path: tuple[object, ...] = path
            if state.move.method == "kv_transfer" and state.initial_start is not None:
                self._start_action(
                    (index, phase, "destination"), action,
                    state.move.destination_instance,
                )
                rate_path += (self.kv_links[state.move.destination_instance],)
                if index in self.pace_links and phase != "catch_up":
                    rate_path += (self.pace_links[index],)
            flow = _Flow(
                self.next_flow, index, phase, float(byte_count), path, rate_path,
                byte_count, self.time,
            )
            self.next_flow += 1
            self.flows[flow.flow_id] = flow
            self.changed_links.update(flow.rate_path)
            for link in flow.rate_path:
                self.link_flows[link].add(flow.flow_id)
            self._event("network_start", state.move.session_id, detail=f"{phase}:{byte_count}")
        else:
            self._endpoint(index, phase, detail)

    def _endpoint(self, index: int, phase: str, detail: tuple[int, int] | None = None):
        state = self.states[index]
        components = self.migration_components.get(
            state.move.destination_instance, {}).get(state.move.method)
        replay_tokens, blocks = detail or self._payload(index, phase)[1:]
        replay = state.move.method == "replay" or (
            state.move.method == "replay_on_request" and phase == "wake"
        )
        key = state.move.destination_instance, "replay"
        if replay:
            if self.endpoint_active.get(key, 0) \
                    >= self.profile.max_destination_replays:
                self.endpoint_waiting.setdefault(key, deque()).append(
                    (index, phase, (replay_tokens, blocks))
                )
                self._event("endpoint_queued", state.move.session_id, detail="replay")
                return
            self.endpoint_active[key] = self.endpoint_active.get(key, 0) + 1
        if replay:
            destination = state.move.destination_instance
            active = self.endpoint_active[key]
            # TODO(concurrency): update running replay rates when validated limits exceed one.
            # TODO(catch-up-rate): replace this full-context rate with measured incremental replay.
            rate_context = self.context[state.move.session_id] if phase == "catch_up" else replay_tokens
            duration = replay_tokens / self.case.replay.rate(
                rate_context, active,
            ) + self.case.replay_completion_s
            if components:
                duration *= components.compute_completion_factor
            if phase == "initial":
                state.initial_replay_start = self.time
            elif phase == "catch_up":
                state.catch_replay_start = self.time
            self._event("replay_start", state.move.session_id, detail=destination)
        elif state.move.method == "kv_transfer":
            duration = 0.0 if phase.startswith("append") else (
                self.case.kv_transfer.catch_up_fixed_s
                + replay_tokens / self.case.kv_transfer.tail_replay_tps
                if phase == "catch_up"
                else components.residual_s if components
                else self.case.kv_transfer.initial_completion_s
            )
        else:
            duration = 0.0
        if duration and (index, phase, "destination") not in self.active_actions:
            action = "catch_up" if phase == "catch_up" else state.move.method
            self._start_action(
                (index, phase, "destination"), action, state.move.destination_instance
            )
        self._schedule(self.time + duration, "ready", (index, phase))

    def _ready(self, index: int, phase: str):
        state, session_id = self.states[index], self.states[index].move.session_id
        action = index, phase, "destination"
        if action in self.active_actions:
            self._stop_action(action)
        replay = state.move.method == "replay" or (
            state.move.method == "replay_on_request" and phase == "wake"
        )
        if replay:
            key = state.move.destination_instance, "replay"
            self.endpoint_active[key] -= 1
            waiting = self.endpoint_waiting.get(key, [])
            if waiting:
                self._endpoint(*waiting.popleft())
        elif state.move.method == "kv_transfer":
            destination = state.move.destination_instance
            if self.detailed:
                self.kv_records[index, phase].end_s = self.time
            self.kv_active[destination] -= 1
            waiting = self.kv_waiting[destination]
            if waiting:
                queued = waiting.popleft()
                self.kv_waiting_bytes[destination] -= queued[3]
                self._start_kv(*queued)
        if state.move.method == "replay" or phase == "wake":
            self._event("replay_done", session_id, detail=state.move.destination_instance)
        if phase == "wake":
            destination = state.move.destination_instance
            self.resident_tokens[destination] += self.profile.kv_admission_tokens(
                self.context[session_id])
            self.resident_sessions.add(session_id)
            self._check_resident(destination)
            state.wake_ready = self.time
            self.deferred.remove(session_id)
            self.waking.remove(session_id)
            self._event("wake_ready", session_id)
            request_index, arrival = self.pending_requests.pop(session_id)
            self._schedule(self.time, "request_start", (session_id, request_index, arrival))
        elif phase == "initial":
            state.copied_blocks = state.scheduled_blocks
            state.initial_ready = self.time
            self._event("initial_ready", session_id)
            if state.move.method == "kv_transfer":
                self._append_available(index, "initial")
            self._schedule(
                max(self.time, state.move.quiesce_s or self.time),
                "quiesce", index,
            )
        elif phase.startswith("append"):
            state.append_pending -= 1
            if not state.append_pending:
                state.copied_blocks = state.scheduled_blocks
            self._event("append_ready", session_id)
            if state.idle is not None and not state.append_pending:
                self._start_final(index)
        else:
            state.catch_ready = self.time
            self._event("catch_up_ready", session_id)
            self._begin_switch(index)

    def _quiesce(self, index: int):
        state, session_id = self.states[index], self.states[index].move.session_id
        if state.committed is not None or session_id in self.quiescing:
            return
        state.pause = self.time
        self.quiescing.add(session_id)
        self._event("pause", session_id)
        self._schedule(max(self.time, self.active_request_end[session_id]), "idle", index)

    def _idle(self, index: int):
        state, session_id = self.states[index], self.states[index].move.session_id
        state.idle = self.time
        self.paused.add(session_id)
        self._event("idle", session_id)
        if not state.append_pending:
            self._start_final(index)

    def _start_final(self, index: int):
        state, session_id = self.states[index], self.states[index].move.session_id
        if state.final_started:
            return
        state.final_started = True
        if self.context[session_id] != state.snapshot_tokens or (
            state.move.method == "kv_transfer"
            and self.case.kv_transfer.tail_tokens(self.context[session_id])
        ):
            state.catch_start = self.time
            self._event("catch_up_start", session_id)
            self._prepare(index, "catch_up")
        else:
            self._begin_switch(index)

    def _begin_switch(self, index: int):
        state = self.states[index]
        state.switch = self.time
        self._event("switch_start", state.move.session_id)
        self._schedule(self.time + self.case.switch_s, "commit", index)

    def _commit(self, index: int):
        state, session_id = self.states[index], self.states[index].move.session_id
        state.committed = self.time
        source = self.sessions[session_id].source_instance
        if session_id in self.resident_sessions:
            tokens = self.profile.kv_admission_tokens(self.context[session_id])
            self.resident_tokens[source] -= tokens
            self.resident_tokens[state.move.destination_instance] += tokens
            self._check_resident(state.move.destination_instance)
        self.power_model.move(session_id, state.move.destination_instance)
        self.quiescing.discard(session_id)
        self.paused.discard(session_id)
        if state.move.method == "replay_on_request":
            self.deferred.add(session_id)
        self._event("commit", session_id)
        for instance, waiting in self.serving_waiting.items():
            kept = deque()
            while waiting:
                request = waiting.popleft()
                if request[0] == session_id:
                    self._schedule(self.time, "request_start", request)
                else:
                    kept.append(request)
            self.serving_waiting[instance] = kept
        if session_id in self.pending_requests:
            request_index, arrival = self.pending_requests.pop(session_id)
            self._schedule(self.time, "request_start", (session_id, request_index, arrival))
        for node_id in self.instances[source].gpu_nodes:
            if self.node_state[node_id] != "awake":
                continue
            dependents = self.power_model.dependents[node_id]
            if dependents <= self.power_model.removed and self.scenario.final_state != "awake":
                # TODO(transition-power): replace the step change with a measured trace shape.
                duration = self.case.sleep_s if self.scenario.final_state == "sleep" \
                    else self.scenario.assumed_shutdown_s
                self.node_state[node_id] = "transition"
                self._start_action(("node", node_id), self.scenario.final_state, node=node_id)
                self._event(f"{self.scenario.final_state}_start", node=node_id)
                self._schedule(self.time + duration, "node_state", node_id)

    def _request_start(self, session_id: str, request_index: int, arrival_s: float):
        session = self.sessions[session_id]
        request_key = session_id, request_index
        if request_key not in self.request_arrivals:
            self.request_arrivals.add(request_key)
            self._event("request_arrival", session_id, detail=str(request_index))
        if session_id in self.quiescing or session_id in self.paused:
            self.pending_requests[session_id] = request_index, arrival_s
            return
        if session_id in self.deferred:
            self.pending_requests[session_id] = request_index, arrival_s
            if session_id not in self.waking:
                self.waking.add(session_id)
                index = self.move_index[session_id]
                self.states[index].wake_start = self.time
                self._event("wake_start", session_id)
                self._prepare(index, "wake")
            return
        request = session.requests[request_index]
        instance = self.power_model.route[session_id]
        if instance in self.serving_active:
            self.serving_waiting.setdefault(instance, deque()).append(
                (session_id, request_index, arrival_s)
            )
            self._event("serving_queued", session_id, detail=instance)
            return
        context = self.context[session_id]
        # TODO(request-power): fit request-level power before varying power with sampled service.
        prefill_s = request.prompt_tokens / self.case.prefill.rate(context, 1)
        decode_s = request.output_tokens / self.case.decode.rate(context + request.prompt_tokens, 1) \
            if request.output_tokens else 0.0
        end = self.time + prefill_s + decode_s
        self.serving_active.add(instance)
        self.active_request_instance[session_id] = instance
        self.active_request_end[session_id] = end
        if self.detailed:
            self.requests.append(RequestExecution(
                session_id, request_index, instance, arrival_s, self.time,
                self.time + prefill_s, end, request.prompt_tokens, request.output_tokens,
            ))
        self._event("request_start", session_id)
        self._schedule(end, "request_done", (session_id, request_index))

    def _request_done(self, session_id: str, request_index: int):
        request = self.sessions[session_id].requests[request_index]
        added = request.prompt_tokens + request.output_tokens
        previous = self.profile.kv_admission_tokens(self.context[session_id])
        self.context[session_id] += added
        if session_id in self.move_index and session_id not in self.quiescing:
            index = self.move_index[session_id]
            state = self.states[index]
            if state.move.method == "kv_transfer" and state.initial_ready is not None:
                self._append_available(index, str(request_index))
        if session_id in self.resident_sessions:
            instance = self.active_request_instance[session_id]
            self.resident_tokens[instance] += (
                self.profile.kv_admission_tokens(self.context[session_id]) - previous)
            self._check_resident(instance)
        self.active_request_end[session_id] = self.time
        instance = self.active_request_instance.pop(session_id)
        self.serving_active.remove(instance)
        self._event("request_done", session_id)
        if request_index + 1 < len(self.sessions[session_id].requests):
            gap = self.sessions[session_id].requests[request_index + 1].gap_s
            arrival = self.time + gap
            self._schedule(arrival, "request_start", (session_id, request_index + 1, arrival))
        waiting = self.serving_waiting.get(instance, deque())
        while waiting and instance not in self.serving_active:
            self._request_start(*waiting.popleft())

    def _append_available(self, index: int, label: str):
        state = self.states[index]
        completed = self.case.kv_transfer.sealed_blocks(
            self.context[state.move.session_id]
        )
        blocks = completed - state.scheduled_blocks
        if blocks <= 0:
            return
        state.scheduled_blocks = completed
        state.append_pending += 1
        self._prepare(
            index, f"append_{label}",
            (blocks * self.case.kv_transfer.block_bytes, 0, blocks),
        )

    def _check_resident(self, instance: str):
        if self.resident_tokens[instance] > self.profile.kv_capacity_tokens:
            raise RuntimeError(f"serving instance {instance!r} exceeded resident KV capacity")

    def _advance(self, target: float, rates: dict[int, float]):
        elapsed = target - self.time
        for flow_id, rate in rates.items():
            self.flows[flow_id].remaining -= rate * elapsed
        self.time = target

    def _update_rates(self, rates: dict[int, float]) -> dict[int, float]:
        pending, links, affected = list(self.changed_links), set(self.changed_links), set()
        while pending:
            for flow_id in self.link_flows[pending.pop()] - affected:
                affected.add(flow_id)
                for link in self.flows[flow_id].rate_path:
                    if link not in links:
                        links.add(link)
                        pending.append(link)
        rates = {flow_id: rate for flow_id, rate in rates.items() if flow_id in self.flows}
        rates.update(fair_link_rates(
            {flow_id: self.flows[flow_id].rate_path for flow_id in sorted(affected)},
            {link: self.rate_links[link] for link in links},
        ))
        self.changed_links.clear()
        return rates

    def run(self) -> ExecutionResult:
        self._record_power()
        for session in self.sessions.values():
            if session.requests:
                arrival = session.requests[0].gap_s
                self._schedule(arrival, "request_start", (session.session_id, 0, arrival))
        self._schedule(self.scenario.controller_delay_s, "plan_ready")
        rates = {}
        while (self.heap or self.flows) and self.time <= self.scenario.end_s:
            before = self.time
            if self.changed_links:
                rates = self._update_rates(rates)
            finishes = {
                i: self.time + flow.remaining / rates[i] for i, flow in self.flows.items()
            }
            flow_time = min(finishes.values(), default=np.inf)
            event_time = self.heap[0][0] if self.heap else np.inf
            target = min(flow_time, event_time, self.scenario.end_s)
            if target < before:
                raise RuntimeError(f"simulated time moved backwards from {before} to {target}")
            self._advance(target, rates)
            if self.time != target:
                raise RuntimeError(f"simulator failed to advance from {before} to {target}")
            processed = target > before
            if flow_time <= event_time and flow_time <= self.scenario.end_s:
                done = [i for i, end in finishes.items() if end <= flow_time + 1e-12]
                processed |= bool(done)
                for flow_id in done:
                    flow = self.flows.pop(flow_id)
                    self.changed_links.update(flow.rate_path)
                    for link in flow.rate_path:
                        self.link_flows[link].remove(flow_id)
                    state = self.states[flow.move_index]
                    action = flow.move_index, flow.phase, "source"
                    if action in self.active_actions:
                        self._stop_action(action)
                    if self.detailed:
                        self.network.append(NetworkExecution(
                            state.move.session_id, flow.phase, flow.bytes, flow.bytes, 0,
                            flow.path, flow.start, self.time,
                        ))
                    self._event("network_done", state.move.session_id, detail=flow.phase)
                    self._endpoint(flow.move_index, flow.phase)
            else:
                events = 0
                while self.heap and self.heap[0][0] <= self.time + 1e-12:
                    events += 1
                    _, _, kind, payload = heapq.heappop(self.heap)
                    if kind == "prepare":
                        self._prepare(*payload)
                    elif kind == "ready":
                        self._ready(*payload)
                    elif kind == "idle":
                        self._idle(payload)
                    elif kind == "quiesce":
                        self._quiesce(payload)
                    elif kind == "commit":
                        self._commit(payload)
                    elif kind == "request_start":
                        self._request_start(*payload)
                    elif kind == "request_done":
                        self._request_done(*payload)
                    elif kind == "node_state":
                        self.node_state[payload] = self.scenario.final_state
                        self.node_ready[payload] = self.time
                        self.power_model.set_state(payload, self.scenario.final_state)
                        self._stop_action(("node", payload))
                        self._event(f"{self.scenario.final_state}_done", node=payload)
                    elif kind == "plan_ready":
                        self._event("plan_ready")
                        for index in range(len(self.states)):
                            self._start(index)
                    else:
                        raise RuntimeError(f"unknown event {kind!r}")
                processed |= bool(events)
            if not processed:
                raise RuntimeError(f"simulator made no progress at {self.time}")
            self._record_power()
            if target == self.scenario.end_s \
                    and (not self.heap or self.heap[0][0] > target + 1e-12):
                break
        if self.power[-1][0] != self.scenario.end_s:
            self.time = self.scenario.end_s
            self._record_power(force=True)
        sessions = tuple(
            SessionExecution(
                s.move.session_id, s.move.method, s.initial_start, s.initial_ready, s.pause,
                s.idle, s.catch_start, s.catch_ready, s.switch, s.committed,
                s.wake_start, s.wake_ready, s.initial_replay_start,
                s.catch_replay_start,
            ) for s in self.states
        )
        at_deadline = step_average(
            self.power, self.scenario.deadline_s, self.profile.power_window_s
        )
        migrations_complete = all(s.committed_s is not None for s in sessions)
        migration_makespan = max(
            (s.committed_s for s in sessions if s.committed_s is not None),
            default=self.scenario.controller_delay_s,
        ) if migrations_complete else None
        drained = {
            node_id for node_id, dependents in self.power_model.dependents.items()
            if dependents and dependents <= self.power_model.removed
        }
        state_complete = self.scenario.final_state == "awake" or drained <= self.node_ready.keys()
        final_ready = max(
            [migration_makespan or self.scenario.controller_delay_s]
            + ([] if self.scenario.final_state == "awake" else [
                self.node_ready[node] for node in drained
            ])
        ) if migrations_complete and state_complete else None
        deadline_met = at_deadline <= self.scenario.power_limit_w + 1e-8 \
            and migration_makespan is not None \
            and migration_makespan <= self.scenario.deadline_s \
            and final_ready is not None and final_ready <= self.scenario.deadline_s
        makespan = final_ready if final_ready is not None else (
            migration_makespan
            if migration_makespan is not None else self.scenario.controller_delay_s
        )
        network = self.network + ([
            NetworkExecution(
                self.states[flow.move_index].move.session_id, flow.phase, flow.bytes,
                round(flow.bytes - flow.remaining), round(flow.remaining), flow.path,
                flow.start, None,
            ) for flow in self.flows.values()
        ] if self.detailed else [])
        return ExecutionResult(
            tuple(self.events), sessions, tuple(self.requests), tuple(network),
            tuple(self.queues) if self.detailed else (),
            tuple(self.power), at_deadline, deadline_met,
            migration_makespan, final_ready, makespan,
        )


def execute(scenario: ExecutionScenario, profile: ModelProfile,
            moves: tuple[PlannedMove, ...], case_id: str = "central",
            destination=None) -> ExecutionResult:
    return _run(scenario, profile, moves, case_id, destination, True)


def predict(scenario: ExecutionScenario, profile: ModelProfile,
            moves: tuple[PlannedMove, ...], case_id: str = "central",
            destination=None) -> ExecutionResult:
    """Execute exactly without retaining audit records used only by experiments."""
    return _run(scenario, profile, moves, case_id, destination, False)


def _run_fluid(scenario, profile, moves, case_id, destination, detailed):
    if scenario.final_state != "awake" or any(
        session.requests or session.expected_growth_tokens_per_s
        for session in scenario.sessions
    ):
        raise ValueError("fluid migration execution requires an idle awake snapshot")
    moves = tuple(sorted(moves, key=lambda move: move.order))
    pools = {pool.pool_id: pool for pool in destination.pools}
    if any(move.destination_pool not in pools for move in moves):
        raise ValueError("fluid moves require a destination pool")
    paths = tuple(set(move.path for move in moves))
    if any(set(paths[i]) & set(paths[j]) for i in range(len(paths))
           for j in range(i + 1, len(paths))):
        raise ValueError("fluid execution requires identical or disjoint routes")
    sessions = {session.session_id: session for session in scenario.sessions}
    case, start = profile.case(case_id), scenario.controller_delay_s
    links = {link.link_id: link.bytes_per_s for link in scenario.links}
    route_bytes = np.array([
        sessions[move.session_id].log_bytes if move.method == "replay" else
        case.kv_transfer.sealed_bytes(sessions[move.session_id].context_tokens)
        for move in moves
    ], float)
    network_done = np.empty(len(moves))
    for path in paths:
        members = [i for i, move in enumerate(moves) if move.path == path]
        bandwidth = min((links[link] for link in path), default=np.inf)
        network_done[members] = start + route_bytes[members].sum() / bandwidth
    commits = np.empty(len(moves))
    for pool_id, pool in pools.items():
        service = pool.fluid_migration
        members = [i for i, move in enumerate(moves) if move.destination_pool == pool_id]
        if not members:
            continue
        if service is None:
            raise ValueError("fluid execution cannot mix legacy destination pools")
        replay = [i for i in members if moves[i].method == "replay"]
        kv = [i for i in members if moves[i].method == "kv_transfer"]
        q = destination.type_by_id[pool.type_id]
        for i in members:
            if q.migration is None:
                continue
            session = sessions[moves[i].session_id]
            components = q.migration[moves[i].method]
            bandwidth = min(links[link] for link in moves[i].path)
            extrapolated = components.extrapolates(
                session.context_tokens, bandwidth,
            )
            if extrapolated and not components.allow_extrapolation:
                raise ValueError(
                    "migration execution outside calibrated "
                    + "/".join(extrapolated) + " range"
                )

        def replay_rate(i):
            tokens = sessions[moves[i].session_id].context_tokens
            return case.replay.rate(tokens, 1) if q.migration is None else \
                case.replay.conservative_rate(tokens, 1)

        if service.coupling:
            replay_work = sum(
                sessions[moves[i].session_id].context_tokens / replay_rate(i)
                + case.replay_completion_s for i in replay
            ) / service.replay_speedup
            residual = q.migration["kv_transfer"].residual_s \
                if q.migration else case.kv_transfer.initial_completion_s
            kv_work = sum(route_bytes[i] / service.kv_ingest_bytes_per_s
                          + residual for i in kv)
            work = max(
                replay_work + service.coupling * kv_work,
                service.coupling * replay_work + kv_work,
            ) / len(pool.replicas)
            switches = len(members) * case.switch_s / len(pool.replicas)
            done = (max(network_done[members[0]], start + work)
                    if service.route_overlap else network_done[members[0]] + work)
            done += switches
            commits[members] = done
            continue
        if replay:
            work = np.array([
                sessions[moves[i].session_id].context_tokens / replay_rate(i)
                for i in replay
            ])
            capacity = len(pool.replicas) * service.replay_speedup
            streamed = np.full(len(replay), max(
                network_done[replay[0]], start + work.sum() / capacity,
            ))
            done = fluid_service_completion(
                np.full(len(replay), case.replay_completion_s), capacity, streamed,
            )
            commits[replay] = done + case.switch_s
        if kv:
            ingested = np.full(len(kv), max(
                network_done[kv[0]], start + route_bytes[kv].sum() / (
                    len(pool.replicas) * service.kv_ingest_bytes_per_s
                ),
            ))
            residual = q.migration["kv_transfer"].residual_s \
                if q.migration else case.kv_transfer.initial_completion_s
            commits[kv] = ingested + residual + case.switch_s

    power_model = ExpectedPower(scenario, profile, case_id)
    power = [(0.0, power_model.power(True), power_model.power(False))]
    by_time = {}
    for i, commit in enumerate(commits):
        if commit <= scenario.end_s:
            by_time.setdefault(float(commit), []).append(i)

    def action(active, local):
        total = 0.0
        for pool_id, pool in pools.items():
            service = pool.fluid_migration
            for method in ("replay", "kv_transfer"):
                members = [i for i in active if moves[i].destination_pool == pool_id
                           and moves[i].method == method]
                if not members:
                    continue
                replicas = len({sessions[moves[i].session_id].source_instance
                                for i in members}) if local else len(pool.replicas)
                values = service.source_power_w if local else service.destination_power_w
                total += replicas * values[method]
        return total

    active = set(range(len(moves)))
    if start:
        power.append((start, power_model.power(True), power_model.power(False)))
    if active:
        power.append((start, power_model.power(True) + action(active, True),
                      power_model.power(False) + action(active, False)))
    for time in sorted(by_time):
        for i in by_time[time]:
            power_model.remove(moves[i].session_id)
            active.remove(i)
        power.append((time, power_model.power(True) + action(active, True),
                      power_model.power(False) + action(active, False)))
    if power[-1][0] != scenario.end_s:
        power.append((scenario.end_s, power[-1][1], power[-1][2]))
    rows = tuple(SessionExecution(
        move.session_id, move.method, start,
        float(network_done[i]) if move.method == "replay" else float(commits[i]),
        float(commits[i]), float(commits[i]), None, None, float(commits[i]),
        float(commits[i]) if commits[i] <= scenario.end_s else None, None, None,
        float(network_done[i]) if move.method == "replay" else None,
    ) for i, move in enumerate(moves))
    at_deadline = step_average(power, scenario.deadline_s, profile.power_window_s)
    complete = all(commit <= scenario.end_s for commit in commits)
    makespan = float(max(commits, default=start)) if complete else None
    deadline_met = complete and makespan <= scenario.deadline_s \
        and at_deadline <= scenario.power_limit_w + 1e-8
    network = tuple(NetworkExecution(
        move.session_id, "initial", int(route_bytes[i]), int(route_bytes[i]), 0,
        move.path, start, float(network_done[i]),
    ) for i, move in enumerate(moves)) if detailed else ()
    events = tuple(ExecutionEvent(float(commits[i]), "commit", move.session_id)
                   for i, move in enumerate(moves)
                   if commits[i] <= scenario.end_s) if detailed else ()
    return ExecutionResult(
        events, rows, (), network, (), tuple(power), at_deadline, deadline_met,
        makespan, makespan, makespan or start,
    )


def _run(scenario, profile, moves, case_id, destination, detailed):
    if destination:
        from pool_planner import validate_destination_execution
        validate_destination_execution(scenario, destination, moves)
    fluid = destination and any(pool.fluid_migration for pool in destination.pools)
    result = _run_fluid(
        scenario, profile, moves, case_id, destination, detailed,
    ) if fluid else ExecutionSimulator(
        scenario, profile, moves, case_id, detailed, destination).run()
    if destination:
        from pool_planner import destination_service_execution
        rows = destination_service_execution(
            scenario, profile, destination, moves, result, detailed,
        )
        result = replace(
            result, pool_service=rows,
            deadline_met=result.deadline_met and all(row.within_contract for row in rows),
        )
    return result
