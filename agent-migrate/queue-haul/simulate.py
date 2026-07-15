"""Reconstruction DES (formulation.md §10.2 validation; replays a solved dispatch Plan).

Deterministic flow-shop checker: a solved Plan moves (j,ℓ) shipments over ONE shared egress
link (λ_src, serial — the multi-dest coupling) then K parallel rebuild clusters, each ℓ with
⌊spare_ℓ⌋ prefill servers (replay, T/ρ_ℓ) and ⌊spare_ℓ⌋ ingest channels (transfer, η·T/μ_in) —
the destination's own spare nodes, shared with serving headroom; no dedicated pool. No job
selection; replay the plan and report where realized shed (egress done by D), reconstruction
(rebuild done by D), and per-ℓ realized load (§6.2 admission) fall short of the LP certificate.
Stage-2 uses BARE rates — finite servers model the contention the LP folded into c_*'s (1+φ)
factor, so feeding c_* in would double-count the queue wait. A split shipment (y_R>0 AND y_S>0)
rebuilds BOTH pieces, each on its own resource. fleet=None ⇒ K=1, reproducing the single-dest DES.
"""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass
import heapq
from typing import Literal

import numpy as np

from dispatch import DestFleet, Event, Plan, bind_dp, movement_columns
from impact import Impact, Movement
from instance import JobPopulation
from profiles import ModelProfile
from power_model import ExpectedPower


@dataclass(frozen=True)
class SimResult:
    egress_start: np.ndarray  # per-job, NaN if not moved
    egress_done: np.ndarray
    rebuild_start: np.ndarray
    rebuild_done: np.ndarray
    realized_shed: float  # Σ dp·y over jobs whose egress completes by D (grid relief)
    reconstruction_shed: float  # Σ dp·y over jobs whose rebuild completes by D
    reconstruction_success_count: int
    makespan: float
    analytic_lb: float
    analytic_ub: float
    realized_load: np.ndarray  # (K,) Σ_j ℓ_j·y[j,ℓ] over shipments resident (rebuilt) by D
    certified_load: np.ndarray  # (K,) Σ_j ℓ_j·y[j,ℓ], LP-certified routing (D-independent)
    load_cap: np.ndarray  # (K,) L̄_dest,ℓ = spare_ℓ·ρ*; admission holds iff realized_load ≤ this
    discipline: str
    mode: str


def _source_node(pop):
    if pop.source_node is None:
        raise ValueError("node_marginal_pd requires pop.source_node")
    node = np.asarray(pop.source_node, int)
    if len(node) != len(pop) or np.any(node < 0):
        raise ValueError("source_node must assign every job to a nonnegative node")
    return node


def _node_marginal_order(pop, pool, Yf, p1, mv, K):
    node = _source_node(pop)
    resid = np.bincount(node, weights=pop.ell, minlength=int(node.max()) + 1)
    todo, order = list(mv), []
    while todo:
        f = np.asarray(todo)
        j = f // K
        load = pop.ell[j] * Yf[f]
        gain = pool.node_power(resid[node[j]]) - pool.node_power(resid[node[j]] - load)
        k = int(np.argmax(gain / np.maximum(p1[f], 1e-300)))
        pick = int(todo.pop(k))
        order.append(pick)
        resid[node[pick // K]] -= pop.ell[pick // K] * Yf[pick]
    return np.asarray(order, int)


def _order(mv, p1, p2, dens, discipline, pop=None, pool=None, Yf=None, K=1):
    """Order moving jobs at the shared egress link."""
    if discipline == "fifo":
        return mv
    if discipline == "lpt":
        return mv[np.argsort(-p1[mv])]
    if discipline in ("pd", "certified_pd"):  # certified active-work density
        return mv[np.argsort(-dens[mv])]
    if discipline == "node_marginal_pd":
        return _node_marginal_order(pop, pool, Yf, p1, mv, K)
    if discipline == "johnson":  # 2-machine makespan-optimal (exact only at W=1, single-action)
        a, b = mv[p1[mv] <= p2[mv]], mv[p1[mv] > p2[mv]]
        return np.concatenate([a[np.argsort(p1[a])], b[np.argsort(-p2[b])]])
    raise ValueError(f"unknown discipline {discipline!r}")


def _stage_lb(off, p2s, w):
    """P||Cmax lower bound for a parallel-server stage: off + max(longest job, total/W)."""
    return off + max(p2s.max(), p2s.sum() / w) if p2s.size and p2s.max() > 0 else 0.0


def simulate(pop: JobPopulation, pool, imp: Impact, plan: Plan, event: Event = Event(),
             move: Movement = Movement(), mode: str = "sf", discipline: str = "pd",
             fleet: DestFleet = None) -> SimResult:
    # fleet=None ⇒ the K=1 single-dest fleet (uses the y_R/y_S aggregates); an explicit fleet
    # routes the (n,K) Y_R/Y_S over K clusters with per-ℓ ρ_ℓ. Shipments are flat (j,ℓ) indices:
    # job = f//K, dest ℓ = f%K. One shared egress link serializes all of them.
    if mode not in ("sf", "cutthrough"):
        raise ValueError(f"unknown mode {mode!r}")
    if not 0 <= move.alpha_in < 1:
        raise ValueError(f"alpha_in must be in [0, 1), got {move.alpha_in}")
    n = len(pop)
    multidest = fleet is not None
    fleet = fleet or DestFleet.from_event(event, move, pool, pop)
    K, W = len(fleet), np.atleast_1d(fleet.W)
    YR, YS = (plan.Y_R, plan.Y_S) if multidest else (plan.y_R[:, None], plan.y_S[:, None])
    Y, dp = YR + YS, bind_dp(imp)
    cols = movement_columns(pop, pool, imp, fleet, move)
    p1 = (YR * cols["R"]["egress"] + YS * cols["S"]["egress"]) / move.lambda_src
    p2R = YR * cols["R"]["prefill"]  # bare prefill replay, 0 where not replayed
    p2S = YS * cols["S"]["ingest"] / move.mu_in  # bare KV ingest, 0 where not transferred
    p1f, p2Rf, p2Sf, YRf, YSf, Yf = (a.ravel() for a in (p1, p2R, p2S, YR, YS, Y))
    shedf = (dp[:, None] * Y).ravel()
    # value density dp·y/p1 → dp·λ/bytes for a mover (y cancels); floor only guards non-movers.
    densf = shedf / np.maximum(p1f, 1e-300)
    mv = np.flatnonzero(Y.ravel() > 1e-9)  # flat (j,ℓ) shipment indices

    order = _order(mv, p1f, p2Rf + p2Sf, densf, discipline, pop, pool, Yf, K)
    es, ed = np.full(n * K, np.nan), np.full(n * K, np.nan)
    t = event.tau_src  # link available once at τ_src, then continuous
    for f in order:  # serial link; egress_done is monotone along `order`
        es[f], ed[f] = t, t + p1f[f]
        t = ed[f]

    rs, rd = np.full(n * K, np.nan), np.full(n * K, np.nan)
    pf = [np.full(int(W[dest]), event.tau_pre) for dest in range(K)]
    ig = [np.full(int(W[dest]), event.tau_in) for dest in range(K)]
    # cut-through = optimistic earliest-overlap bound: rebuild may run from egress_start, but
    # still completes no sooner than full byte arrival ed (the outer max below). sf waits for ed.
    floor = es if mode == "cutthrough" else ed
    for f in order:  # split shipment rebuilds both pieces on dest ℓ's resources, in egress order
        dest, starts, done = f % K, [], ed[f]
        for srv, w, work, drag in ((pf[dest], YRf[f], p2Rf[f], move.alpha_in),
                                   (ig[dest], YSf[f], p2Sf[f], 0.0)):
            if w <= 1e-9:
                continue
            k = int(np.argmin(srv))
            st = max(floor[f], srv[k])
            if drag:  # first-order interference: ingest-busy fraction at st slows prefill
                # (sampled at start only; same-shipment ingest is assigned after R, so it never drags its own prefill)
                work /= 1.0 - drag * (ig[dest] > max(st, event.tau_in)).mean()
            srv[k] = max(ed[f], st + work)  # outer max = cut-through byte-arrival cap
            starts.append(st)
            done = max(done, srv[k])
        rs[f], rd[f] = (min(starts) if starts else ed[f]), done

    e_ok = np.where(np.isfinite(ed), ed, np.inf) <= event.D
    r_ok = np.where(np.isfinite(rd), rd, np.inf) <= event.D
    # TODO(dest-load): recompute ell per destination when fleet hardware/precision differs.
    ellY = pop.ell[:, None] * Y  # (n,K) load placed at each dest
    resident = (np.where(np.isfinite(rd), rd, np.inf) <= event.D).reshape(n, K)  # rebuilt by D
    realized_load = (ellY * resident).sum(0)  # §6.2 admission anchor: resident ⇒ consuming load
    certified_load = ellY.sum(0)
    load_cap = np.asarray(fleet.spare, float) * pool.rho_star  # L̄_dest,ℓ
    if mv.size:
        lb = max(event.tau_src + p1f[mv].sum(),
                 max(_stage_lb(event.tau_pre, p2R[:, dest], int(W[dest])) for dest in range(K)),
                 max(_stage_lb(event.tau_in, p2S[:, dest], int(W[dest])) for dest in range(K)))
        ub = max(event.tau_src + p1f[mv].sum(), event.tau_pre, event.tau_in) \
            + p2Rf[mv].sum() / (1 - move.alpha_in) + p2Sf[mv].sum()  # worst-case ingest drag
        makespan = float(np.nanmax(rd))
    else:
        lb = ub = makespan = 0.0
    sq = (lambda a: a.reshape(n, K)[:, 0]) if K == 1 else (lambda a: a.reshape(n, K))
    return SimResult(sq(es), sq(ed), sq(rs), sq(rd), float(shedf[e_ok].sum()),
                     float(shedf[r_ok].sum()), int(r_ok.sum()), makespan, float(lb), float(ub),
                     realized_load, certified_load, load_cap, discipline, mode)


MoveMethod = Literal["replay", "kv_transfer", "replay_on_request"]
FinalState = Literal["awake", "sleep", "off"]


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
    log_external: bool = True
    requests: tuple[SimRequest, ...] = ()
    movable: bool = True
    wake_probability: float = 0.0

    def __post_init__(self):
        if not self.session_id or not self.source_instance or self.context_tokens < 1 \
                or min(self.expected_f, self.expected_g) < 0 or self.log_bytes < 1 \
                or not 0 <= self.wake_probability <= 1:
            raise ValueError("invalid session")


@dataclass(frozen=True)
class PlannedMove:
    session_id: str
    destination_instance: str
    method: MoveMethod
    order: int
    path: tuple[str, ...]
    external_path: tuple[str, ...] = ()

    def __post_init__(self):
        if not self.session_id or not self.destination_instance or self.method not in {
            "replay", "kv_transfer", "replay_on_request"
        } or self.order < 0:
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

    def __post_init__(self):
        if self.deadline_s <= 0 or self.end_s < self.deadline_s or self.power_limit_w < 0 \
                or not 0 <= self.controller_delay_s <= self.deadline_s \
                or self.final_state not in {"awake", "sleep", "off"}:
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


@dataclass(frozen=True)
class ExecutionResult:
    events: tuple[ExecutionEvent, ...]
    sessions: tuple[SessionExecution, ...]
    requests: tuple["RequestExecution", ...]
    network: tuple[NetworkExecution, ...]
    power: tuple[tuple[float, float, float], ...]
    source_power_at_deadline_w: float
    deadline_met: bool
    makespan_s: float

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
    counts = {link: len(flows) for link, flows in members.items()}
    while active:
        share, bottleneck = min(
            (residual[link] / count, link) for link, count in counts.items() if count
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


class ExecutionSimulator:
    def __init__(self, scenario: ExecutionScenario, profile: ModelProfile,
                 moves: tuple[PlannedMove, ...], case_id: str = "central"):
        self.scenario, self.profile, self.case = scenario, profile, profile.case(case_id)
        self.nodes = {n.node_id: n for n in scenario.nodes}
        self.instances = {i.instance_id: i for i in scenario.instances}
        self.sessions = {s.session_id: s for s in scenario.sessions}
        self.links = {link.link_id: link.bytes_per_s for link in scenario.links}
        self.moves = tuple(sorted(moves, key=lambda m: m.order))
        self._validate()
        self.time = 0.0
        self.heap: list[tuple[float, int, str, object]] = []
        self.sequence = 0
        self.flows: dict[int, _Flow] = {}
        self.next_flow = 0
        self.states = [_MoveState(m) for m in self.moves]
        self.move_index = {state.move.session_id: i for i, state in enumerate(self.states)}
        self.by_source: dict[str, list[int]] = {}
        for i, state in enumerate(self.states):
            source = self.sessions[state.move.session_id].source_instance
            self.by_source.setdefault(source, []).append(i)
        self.running = {source: 0 for source in self.by_source}
        self.next_by_source = {source: 0 for source in self.by_source}
        self.context = {s.session_id: s.context_tokens for s in scenario.sessions}
        self.active_request_end = {s.session_id: 0.0 for s in scenario.sessions}
        self.active_request_instance: dict[str, str] = {}
        self.serving_active: set[str] = set()
        self.serving_waiting: dict[str, deque[tuple[str, int, float]]] = {}
        self.quiescing = set()
        self.paused = set()
        self.moved = set()
        self.route = {s.session_id: s.source_instance for s in scenario.sessions}
        self.power_model = ExpectedPower(scenario, profile, case_id)
        self.node_state = {n.node_id: "awake" for n in scenario.nodes}
        self.active_actions: dict[object, tuple[str, bool]] = {}
        self.action_power = {True: 0.0, False: 0.0}
        self.node_actions: dict[str, str] = {}
        self.deferred = set()
        self.waking = set()
        self.pending_requests: dict[str, tuple[int, float]] = {}
        self.endpoint_active: dict[tuple[str, str], int] = {}
        self.endpoint_waiting: dict[
            tuple[str, str], deque[tuple[int, str, tuple[int, int]]]
        ] = {}
        self.events: list[ExecutionEvent] = []
        self.power: list[tuple[float, float, float]] = []
        self.requests: list[RequestExecution] = []
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
            if move.method == "replay_on_request" and not self.sessions[move.session_id].log_external \
                    and self.sessions[move.session_id].log_bytes <= 0:
                raise ValueError("replay_on_request requires a durable session log")
            if not move.path or any(link not in self.links for link in move.path):
                raise ValueError("move path contains an unknown link")
            if session.log_external and move.method in {"replay", "replay_on_request"} \
                    and (not move.external_path
                         or any(link not in self.links for link in move.external_path)):
                raise ValueError("external replay requires an external-to-destination path")

    def _schedule(self, when: float, kind: str, payload=None):
        self.sequence += 1
        heapq.heappush(self.heap, (when, self.sequence, kind, payload))

    def _event(self, name: str, session: str = "", node: str = "", detail: str = ""):
        self.events.append(ExecutionEvent(self.time, name, session, node, detail))

    def _start_action(self, key, action: str, instance: str | None = None,
                      node: str | None = None):
        local = self.nodes[node].local if node else self.nodes[self.instances[instance].gpu_nodes[0]].local
        self.active_actions[key] = action, local
        self.action_power[local] += self.case.action_power_w[action][0 if local else 1]

    def _stop_action(self, key):
        value = self.active_actions.pop(key, None)
        if value:
            action, local = value
            self.action_power[local] -= self.case.action_power_w[action][0 if local else 1]

    def _node_power(self, local: bool) -> float:
        return self.power_model.power(local) + self.action_power[local]

    def _record_power(self, force: bool = False):
        point = (self.time, self._node_power(True), self._node_power(False))
        if not self.power or point[1:] != self.power[-1][1:] or force:
            self.power.append(point)

    def _start_available(self, source: str):
        queue = self.by_source[source]
        while self.running[source] < self.profile.max_parallel_moves \
                and self.next_by_source[source] < len(queue):
            index = queue[self.next_by_source[source]]
            self.next_by_source[source] += 1
            self.running[source] += 1
            state = self.states[index]
            state.snapshot_tokens = self.context[state.move.session_id]
            state.initial_start = self.time
            self._event("initial_start", state.move.session_id, detail=state.move.method)
            setup = self.case.kv_transfer.setup_s if state.move.method == "kv_transfer" else 0.0
            self._schedule(self.time + setup, "prepare", (index, "initial"))

    def _payload(self, index: int, phase: str) -> tuple[int, int, int]:
        state, session = self.states[index], self.sessions[self.states[index].move.session_id]
        tokens = state.snapshot_tokens if phase == "initial" else self.context[session.session_id]
        if state.move.method == "replay":
            ratio = session.log_bytes / session.context_tokens
            return max(1, round(tokens * ratio)), tokens, 0
        if state.move.method == "replay_on_request":
            byte_count = session.log_bytes if session.log_external == (phase == "wake") else 0
            return byte_count, tokens if phase == "wake" else 0, 0
        blocks = self.case.kv_transfer.blocks(tokens)
        if phase == "catch_up":
            blocks -= self.case.kv_transfer.blocks(state.snapshot_tokens)
        return max(0, blocks) * self.case.kv_transfer.block_bytes, 0, max(0, blocks)

    def _path(self, state: _MoveState, phase: str) -> tuple[str, ...]:
        session = self.sessions[state.move.session_id]
        external = session.log_external and (
            state.move.method == "replay" or
            state.move.method == "replay_on_request" and phase == "wake"
        )
        return state.move.external_path if external else state.move.path

    def _prepare(self, index: int, phase: str):
        state = self.states[index]
        byte_count, replay_tokens, blocks = self._payload(index, phase)
        detail = (replay_tokens, blocks)
        if byte_count:
            source = self.sessions[state.move.session_id].source_instance
            if state.move.method == "kv_transfer" or not self.sessions[state.move.session_id].log_external:
                self._start_action(
                    (index, phase, "source"),
                    "catch_up" if phase == "catch_up" else state.move.method, source,
                )
            flow = _Flow(
                self.next_flow, index, phase, float(byte_count), self._path(state, phase),
                byte_count, self.time,
            )
            self.next_flow += 1
            self.flows[flow.flow_id] = flow
            self._event("network_start", state.move.session_id, detail=f"{phase}:{byte_count}")
        else:
            self._endpoint(index, phase, detail)

    def _endpoint(self, index: int, phase: str, detail: tuple[int, int] | None = None):
        state = self.states[index]
        replay_tokens, blocks = detail or self._payload(index, phase)[1:]
        replay = state.move.method == "replay" or (
            state.move.method == "replay_on_request" and phase == "wake"
        )
        kind = "replay" if replay else "kv_transfer" if state.move.method == "kv_transfer" else ""
        key = state.move.destination_instance, kind
        if kind:
            limit = (self.profile.max_parallel_replay if replay
                     else self.profile.max_parallel_kv)
            if self.endpoint_active.get(key, 0) >= limit:
                self.endpoint_waiting.setdefault(key, deque()).append(
                    (index, phase, (replay_tokens, blocks))
                )
                self._event("endpoint_queued", state.move.session_id, detail=kind)
                return
            self.endpoint_active[key] = self.endpoint_active.get(key, 0) + 1
        if replay:
            destination = state.move.destination_instance
            active = self.endpoint_active[key]
            # TODO(concurrency): update running replay rates when validated limits exceed one.
            duration = replay_tokens / self.case.replay.rate(replay_tokens, active)
            self._event("replay_start", state.move.session_id, detail=destination)
        elif state.move.method == "kv_transfer":
            duration = blocks * self.case.kv_transfer.block_processing_s + self.case.kv_transfer.sync_s
        else:
            duration = 0.0
        if duration:
            action = "catch_up" if phase == "catch_up" else state.move.method
            self._start_action(
                (index, phase, "destination"), action, state.move.destination_instance
            )
        self._schedule(self.time + duration, "ready", (index, phase))

    def _ready(self, index: int, phase: str):
        state, session_id = self.states[index], self.states[index].move.session_id
        self._stop_action((index, phase, "destination"))
        replay = state.move.method == "replay" or (
            state.move.method == "replay_on_request" and phase == "wake"
        )
        kind = "replay" if replay else "kv_transfer" if state.move.method == "kv_transfer" else ""
        if kind:
            key = state.move.destination_instance, kind
            self.endpoint_active[key] -= 1
            waiting = self.endpoint_waiting.get(key, [])
            if waiting:
                self._endpoint(*waiting.popleft())
        if state.move.method == "replay" or phase == "wake":
            self._event("replay_done", session_id, detail=state.move.destination_instance)
        if phase == "wake":
            state.wake_ready = self.time
            self.deferred.remove(session_id)
            self.waking.remove(session_id)
            self._event("wake_ready", session_id)
            request_index, arrival = self.pending_requests.pop(session_id)
            self._schedule(self.time, "request_start", (session_id, request_index, arrival))
        elif phase == "initial":
            state.initial_ready = self.time
            self.quiescing.add(session_id)
            self._event("initial_ready", session_id)
            idle = max(self.time, self.active_request_end[session_id])
            self._schedule(idle, "idle", index)
        else:
            state.catch_ready = self.time
            self._event("catch_up_ready", session_id)
            self._begin_switch(index)

    def _idle(self, index: int):
        state, session_id = self.states[index], self.states[index].move.session_id
        state.pause = state.idle = self.time
        self.paused.add(session_id)
        self._event("pause", session_id)
        self._event("idle", session_id)
        if self.context[session_id] != state.snapshot_tokens:
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
        self.power_model.move(session_id, state.move.destination_instance)
        self.route[session_id] = state.move.destination_instance
        self.moved.add(session_id)
        self.quiescing.discard(session_id)
        self.paused.discard(session_id)
        if state.move.method == "replay_on_request":
            self.deferred.add(session_id)
        self._event("commit", session_id)
        if session_id in self.pending_requests:
            request_index, arrival = self.pending_requests.pop(session_id)
            self._schedule(self.time, "request_start", (session_id, request_index, arrival))
        source = self.sessions[session_id].source_instance
        self.running[source] -= 1
        self._start_available(source)
        for node_id in self.instances[source].gpu_nodes:
            if self.node_state[node_id] != "awake":
                continue
            dependents = self.power_model.dependents[node_id]
            if dependents <= self.moved and self.scenario.final_state != "awake":
                # TODO(transition-power): replace the step change with a measured trace shape.
                duration = self.case.sleep_s if self.scenario.final_state == "sleep" else self.case.shutdown_s
                self.node_state[node_id] = "transition"
                self.node_actions[node_id] = self.scenario.final_state
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
        instance = self.route[session_id]
        if instance in self.serving_active:
            self.serving_waiting.setdefault(instance, deque()).append(
                (session_id, request_index, arrival_s)
            )
            self._event("serving_queued", session_id, detail=instance)
            return
        context = self.context[session_id]
        prefill_s = request.prompt_tokens / self.case.prefill.rate(context, 1)
        decode_s = request.output_tokens / self.case.decode.rate(context + request.prompt_tokens, 1) \
            if request.output_tokens else 0.0
        end = self.time + prefill_s + decode_s
        self.serving_active.add(instance)
        self.active_request_instance[session_id] = instance
        self.active_request_end[session_id] = end
        self.requests.append(RequestExecution(
            session_id, request_index, instance, arrival_s, self.time,
            self.time + prefill_s, end, request.prompt_tokens, request.output_tokens,
        ))
        self._event("request_start", session_id)
        self._schedule(end, "request_done", (session_id, request_index))

    def _request_done(self, session_id: str, request_index: int):
        request = self.sessions[session_id].requests[request_index]
        self.context[session_id] += request.prompt_tokens + request.output_tokens
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

    def _advance(self, target: float, rates: dict[int, float]):
        elapsed = target - self.time
        for flow_id, rate in rates.items():
            self.flows[flow_id].remaining -= rate * elapsed
        self.time = target

    def run(self) -> ExecutionResult:
        self._record_power()
        self.time = self.scenario.controller_delay_s
        self._event("plan_ready")
        for session in self.sessions.values():
            if session.requests:
                arrival = self.time + session.requests[0].gap_s
                self._schedule(arrival, "request_start", (session.session_id, 0, arrival))
        for source in self.by_source:
            self._start_available(source)
        self._record_power()
        while (self.heap or self.flows) and self.time <= self.scenario.end_s:
            rates = fair_link_rates({i: f.path for i, f in self.flows.items()}, self.links)
            flow_time = min(
                (self.time + flow.remaining / rates[i] for i, flow in self.flows.items()),
                default=np.inf,
            )
            event_time = self.heap[0][0] if self.heap else np.inf
            target = min(flow_time, event_time, self.scenario.end_s)
            self._advance(target, rates)
            if flow_time <= event_time and flow_time <= self.scenario.end_s:
                done = [i for i, flow in self.flows.items() if flow.remaining <= 1e-7]
                for flow_id in done:
                    flow = self.flows.pop(flow_id)
                    state = self.states[flow.move_index]
                    self._stop_action((flow.move_index, flow.phase, "source"))
                    self.network.append(NetworkExecution(
                        state.move.session_id, flow.phase, flow.bytes, flow.bytes, 0, flow.path,
                        flow.start, self.time,
                    ))
                    self._event("network_done", state.move.session_id, detail=flow.phase)
                    self._endpoint(flow.move_index, flow.phase)
            else:
                while self.heap and self.heap[0][0] <= self.time + 1e-12:
                    _, _, kind, payload = heapq.heappop(self.heap)
                    if kind == "prepare":
                        self._prepare(*payload)
                    elif kind == "ready":
                        self._ready(*payload)
                    elif kind == "idle":
                        self._idle(payload)
                    elif kind == "commit":
                        self._commit(payload)
                    elif kind == "request_start":
                        self._request_start(*payload)
                    elif kind == "request_done":
                        self._request_done(*payload)
                    elif kind == "node_state":
                        self.node_state[payload] = self.scenario.final_state
                        self.power_model.set_state(payload, self.scenario.final_state)
                        self.node_actions.pop(payload)
                        self._stop_action(("node", payload))
                        self._event(f"{self.scenario.final_state}_done", node=payload)
                    else:
                        raise RuntimeError(f"unknown event {kind!r}")
            self._record_power()
            if target == self.scenario.end_s:
                break
        if self.power[-1][0] != self.scenario.end_s:
            self.time = self.scenario.end_s
            self._record_power(force=True)
        sessions = tuple(
            SessionExecution(
                s.move.session_id, s.move.method, s.initial_start, s.initial_ready, s.pause,
                s.idle, s.catch_start, s.catch_ready, s.switch, s.committed,
                s.wake_start, s.wake_ready,
            ) for s in self.states
        )
        at_deadline = step_average(
            self.power, self.scenario.deadline_s, self.profile.power_window_s
        )
        makespan = max((s.committed_s for s in sessions if s.committed_s is not None),
                       default=self.scenario.controller_delay_s)
        network = self.network + [
            NetworkExecution(
                self.states[flow.move_index].move.session_id, flow.phase, flow.bytes,
                round(flow.bytes - flow.remaining), round(flow.remaining), flow.path,
                flow.start, None,
            ) for flow in self.flows.values()
        ]
        return ExecutionResult(
            tuple(self.events), sessions, tuple(self.requests), tuple(network),
            tuple(self.power), at_deadline, at_deadline <= self.scenario.power_limit_w, makespan,
        )


def execute(scenario: ExecutionScenario, profile: ModelProfile,
            moves: tuple[PlannedMove, ...], case_id: str = "central") -> ExecutionResult:
    return ExecutionSimulator(scenario, profile, moves, case_id).run()
