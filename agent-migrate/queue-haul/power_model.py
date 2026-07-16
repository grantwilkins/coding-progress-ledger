"""Incremental expected power for planner and execution state."""

from __future__ import annotations

from profiles import ModelProfile


class ExpectedPower:
    def __init__(self, scenario, profile: ModelProfile, case_id: str = "central"):
        self.scenario, self.profile, self.case = scenario, profile, profile.case(case_id)
        self.nodes = {n.node_id: n for n in scenario.nodes}
        self.instances = {i.instance_id: i for i in scenario.instances}
        self.sessions = {s.session_id: s for s in scenario.sessions}
        self.ell = {
            s.session_id: s.expected_f / self.case.F + s.expected_g / self.case.G
            for s in scenario.sessions
        }
        self.instance_load = {i.instance_id: 0.0 for i in scenario.instances}
        for session in scenario.sessions:
            self.instance_load[session.source_instance] += self.ell[session.session_id]
        self.slots = {n.node_id: [0.0] * n.gpus for n in scenario.nodes}
        used = {n.node_id: 0 for n in scenario.nodes}
        self.instance_slots = {}
        for instance in scenario.instances:
            owned = []
            for node_id in instance.gpu_nodes:
                if node_id not in self.nodes or used[node_id] == len(self.slots[node_id]):
                    raise ValueError(f"serving instances exceed GPU capacity on {node_id!r}")
                owned.append((node_id, used[node_id]))
                used[node_id] += 1
            self.instance_slots[instance.instance_id] = owned
            for node_id, slot in owned:
                self.slots[node_id][slot] = self.instance_load[instance.instance_id] / len(owned)
        self.dependents = {node_id: set() for node_id in self.nodes}
        for session in scenario.sessions:
            for node_id in self.instances[session.source_instance].gpu_nodes:
                self.dependents[node_id].add(session.session_id)
        self.route = {s.session_id: s.source_instance for s in scenario.sessions}
        self.state = {node_id: "awake" for node_id in self.nodes}
        self.removed = set()
        self.slot_power = {
            node_id: [self.case.power_curve.power(load) for load in loads]
            for node_id, loads in self.slots.items()
        } if self.profile.power_scope == "gpu" else {}
        self.node_power = {node_id: self._power(node_id) for node_id in self.nodes}
        self.total = {
            local: sum(self.node_power[n.node_id] for n in scenario.nodes if n.local == local)
            for local in (True, False)
        }

    def _power(self, node_id: str, slots=None, state=None) -> float:
        node = self.nodes[node_id]
        state = self.state[node_id] if state is None else state
        if state == "off":
            return 0.0
        if state == "sleep":
            return self.case.sleep_power_w * (node.gpus if self.profile.power_scope == "gpu" else 1)
        if self.profile.power_scope == "gpu":
            if slots is None:
                return sum(self.slot_power[node_id])
            return sum(self.case.power_curve.power(load) for load in slots)
        slots = self.slots[node_id] if slots is None else slots
        return self.case.power_curve.power(sum(slots))

    def power(self, local: bool) -> float:
        return self.total[local]

    def _update(self, nodes, change) -> None:
        nodes = set(nodes)
        before = {local: sum(self.node_power[n] for n in nodes if self.nodes[n].local == local)
                  for local in (True, False)}
        change()
        for node_id in nodes:
            self.node_power[node_id] = self._power(node_id)
        for local in (True, False):
            self.total[local] += sum(
                self.node_power[n] for n in nodes if self.nodes[n].local == local
            ) - before[local]

    def _change_instance(self, instance_id: str, delta: float) -> None:
        owned = self.instance_slots[instance_id]
        self.instance_load[instance_id] += delta
        for node_id, slot in owned:
            self.slots[node_id][slot] += delta / len(owned)
            if self.profile.power_scope == "gpu":
                self.slot_power[node_id][slot] = self.case.power_curve.power(
                    self.slots[node_id][slot]
                )

    def remove(self, session_id: str) -> None:
        if session_id in self.removed:
            return
        source = self.route[session_id]
        nodes = {node_id for node_id, _slot in self.instance_slots[source]}
        def change():
            self._change_instance(source, -self.ell[session_id])
            self.route[session_id] = ""
            self.removed.add(session_id)
            for node_id in nodes:
                if self.dependents[node_id] <= self.removed:
                    self.state[node_id] = self.scenario.final_state
        self._update(nodes, change)

    def move(self, session_id: str, destination: str) -> None:
        source = self.route[session_id]
        nodes = [n for n, _slot in self.instance_slots[source] + self.instance_slots[destination]]
        def change():
            self._change_instance(source, -self.ell[session_id])
            self._change_instance(destination, self.ell[session_id])
        self._update(nodes, change)
        self.route[session_id] = destination
        self.removed.add(session_id)

    def set_state(self, node_id: str, state: str) -> None:
        self._update((node_id,), lambda: self.state.__setitem__(node_id, state))

    def marginal(self, session_id: str) -> float:
        source = self.route[session_id]
        owned = self.instance_slots[source]
        by_node, share = {}, self.ell[session_id] / len(owned)
        for node_id, slot in owned:
            by_node.setdefault(node_id, list(self.slots[node_id]))[slot] -= share
        return sum(
            self.node_power[node_id] - self._power(
                node_id, slots,
                self.scenario.final_state
                if self.dependents[node_id] - self.removed == {session_id} else "awake",
            ) for node_id, slots in by_node.items() if self.nodes[node_id].local
        )

    def drain_gain(self, session_ids) -> float:
        affected = {
            node_id for session_id in session_ids
            for node_id, _slot in self.instance_slots[self.route[session_id]]
            if self.nodes[node_id].local
        }
        return sum(
            self.node_power[node_id] - self._power(
                node_id, [0.0] * self.nodes[node_id].gpus, self.scenario.final_state
            ) for node_id in affected
        )
