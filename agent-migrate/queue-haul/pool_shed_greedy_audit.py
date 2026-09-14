"""Reproduce six frozen 120-second cells and inspect their exact admission choices."""

import hashlib
import inspect
import json
import sys
from dataclasses import replace
from pathlib import Path

import numpy as np

import pool_shed_campaign as q
import pool_shed_planner as planner
from pool_shed_execution import kv_transfer_bytes


OUT = q.ROOT / 'outputs/a100-power-frontier-20260910/greedy-audit.json'
REFERENCE = q.ROOT / 'outputs/a100-replay-queue-resolution-20260910/local-recovery/frontier.json'


def packs(table, chosen):
    rows = []
    for route in (0, 1):
        for action, counts in enumerate((table.replay, table.kv)):
            for width in range(1, 9):
                ids = (table.route == route) & (counts.sum(1) == width) & (chosen > 1e-10)
                mass = float(chosen[ids].sum())
                if mass:
                    work = float(chosen[ids] @ (counts[ids] @ table.fleet.demand))
                    rows.append(dict(route=route, action='replay' if action == 0 else 'kv', width=width,
                                     replicas=mass, sessions=mass * width, incoming_load_per_replica=work / mass,
                                     shed_fraction=work / (q.SOURCE_LOAD * table.fleet.gpus)))
    return rows


def run():
    reference = json.loads(REFERENCE.read_text())
    measured = q.calibration(0)
    timing = measured['timing'][0]
    sources = {**q.provenance(measured), str(REFERENCE.relative_to(q.ROOT)): hashlib.sha256(REFERENCE.read_bytes()).hexdigest(),
               Path(__file__).name: hashlib.sha256(Path(__file__).read_bytes()).hexdigest()}
    report = dict(scope='Six existing local-recovery 120s evaluations; instrumentation only, no coefficient or policy changes.',
                  sources=sources, policies=dict(queue_haul='LP over all action choices', greedy='dominant remaining-resource greedy',
                  replay_only='LP restricted to replay'), cells=[])
    choose, admit = planner._choose, planner.plan_admission
    source_lines, first_line = inspect.getsourcelines(choose)
    trace_line = first_line + next(i for i, line in enumerate(source_lines) if 'chosen[j] += take' in line)
    active = {}

    def choose_audit(matrix, capacity, gains, debt, fleet, greedy):
        caller = inspect.currentframe().f_back.f_locals
        original, starts, edges, table = [caller[k] for k in ('original', 'starts', 'edges', 'table')]
        labels = [f'source_cohort_{i}' for i in range(len(fleet.count))]
        labels += [f'{name}_route_{r}' for name in ('standing_service', 'memory', 'replicas') for r in (0, 1)]
        labels += [f'{name}_route_{r}_bin_{k}' for name in ('compute', 'network') for r in (0, 1) for k in range(len(edges) - 1)]
        labels += [f'shared_network_bin_{k}' for k in range(len(edges) - 1)]
        labels += [f'application_route_{r}_bin_{k}' for r in (0, 1) for k in range(len(edges) - 1)]
        assert len(labels) == len(capacity)
        decisions = []

        def trace(frame, event, arg):
            if frame.f_code is choose.__code__ and event == 'line' and frame.f_lineno == trace_line:
                state = frame.f_locals
                j, take = state['j'], float(state['take'])
                used = state['used']
                limits = state['remaining'][used] / state['normalized'][used, j]
                binding = np.flatnonzero(used)[np.isclose(limits, take, rtol=1e-8, atol=1e-9)]
                col = int(original[j])
                decisions.append(dict(column=col, start_s=float(edges[starts[j]]), mass=take,
                    gained_fraction=float(gains[j] * take), score=float(state['scores'][j]),
                    binding=[labels[i] for i in binding], packs=packs(table, np.eye(1, len(table.route), col)[0] * take)))
            return trace

        if greedy:
            sys.settrace(trace)
        try:
            chosen = choose(matrix, capacity, gains, debt, fleet, greedy)
        finally:
            sys.settrace(None)
        alternative = choose(matrix, capacity, gains, debt, fleet, False) if greedy else chosen
        assert np.max((matrix @ chosen - capacity) / np.maximum(capacity, 1.)) <= 1e-8
        assert float(gains @ alternative) + 1e-8 >= float(gains @ chosen)
        pending = capacity - matrix @ chosen
        active['plan']['solves'].append(dict(variables=len(gains), selected_variables=int(np.sum(chosen > 1e-10)),
            horizon_gain=float(gains @ chosen), same_matrix_lp_gain=float(gains @ alternative),
            immediate_gain=float(gains[starts == 0] @ chosen[starts == 0]),
            same_matrix_lp_immediate_gain=float(gains[starts == 0] @ alternative[starts == 0]),
            binding_rows=[labels[i] for i in np.flatnonzero(pending <= 1e-8 * np.maximum(capacity, 1.))],
            greedy_steps=decisions))
        return chosen

    def admission_audit(engine, table, policy, **kwargs):
        active['engine'] = engine
        active['plan'] = dict(time_s=engine.now, solves=[])
        chosen, until, audit = admit(engine, table, policy, **kwargs)
        active['plan'].update(next_decision_s=until, admitted_fraction=float(table.gains @ chosen), packs=packs(table, chosen))
        active['plans'].append(active['plan'])
        return chosen, until, audit

    planner._choose, planner.plan_admission = choose_audit, admission_audit
    try:
        endpoint = np.r_[np.median(q.network_samples()[:, :2], axis=0), 0.]
        endpoint[2] = endpoint[:2].sum()
        for workload in ('coding', 'coding_long'):
            fleet = q.replica_fleet(q.sample_fleet(workload, gpus=6666, gpus_per_node=8))
            fleet = replace(fleet, metadata={**fleet.metadata, 'planning_reference_s': 4., 'require_local_recovery': True,
                'kv_wire_scale': 800_000_000 / (32768 * 49152), 'replay_cached_tokens': [0] * len(fleet.count),
                'kv_shared_tokens': [0] * len(fleet.count)})
            fleet = replace(fleet, kv=kv_transfer_bytes(fleet, fleet.context, measured))
            sources.update(fleet.metadata['source_timing_inputs'])
            budgets = q.bandwidth(endpoint, fleet.nodes, 1000)
            replay, kv = q.include_isolated(*q.library(fleet), q.isolated_methods(fleet, .5, endpoint, budgets, timing))
            table = q.schedule_table(fleet, replay, kv, .5, 120., endpoint, budgets, timing)
            saved = next(c for c in reference['cells'] if c['workload'] == workload and c['deadline_s'] == 120)
            assert hashlib.sha256(replay.tobytes() + kv.tobytes()).hexdigest() == saved['candidate_sha256']
            for policy in ('queue_haul', 'greedy', 'replay_only'):
                active['plans'] = []
                result = q.execute_feedback(table, table, policy, timing, measured)
                for key, expected in saved['results'][policy].items():
                    if key != 'planning_s' and isinstance(expected, (int, float, list)):
                        assert np.allclose(result[key], expected, rtol=1e-8, atol=1e-8), (policy, key)
                engine = active['engine']
                remaining = fleet.count - (table.replay + table.kv).T @ engine.selected_total
                cell = dict(workload=workload, policy=policy, plans=active['plans'],
                    result={k: result[k] for k in ('shed_fraction', 'recovered_handoff_fraction', 'action_fractions',
                        'reserved_destination_replicas', 'final_destination_load', 'last_completion_s', 'planning_steps')},
                    packs=packs(table, engine.selected_total),
                    remaining=[dict(cohort=int(i), context_tokens=float(fleet.context[i]), demand=float(fleet.demand[i]),
                        sessions=float(remaining[i]), shed_fraction=float(remaining[i] * fleet.gain[i]))
                        for i in np.flatnonzero(remaining > 1e-6)])
                report['cells'].append(cell)
                print(workload, policy, result['shed_fraction'], 'initial admission', active['plans'][0]['admitted_fraction'], flush=True)
    finally:
        planner._choose, planner.plan_admission = choose, admit
    assert all(hashlib.sha256((q.ROOT / name).read_bytes()).hexdigest() == checksum for name, checksum in sources.items())
    q.write_json(OUT, report)
    return report


if __name__ == '__main__':
    run()
