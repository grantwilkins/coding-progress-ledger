"""QH Greedy and event-simulator deadline sweep for static, equally weighted sessions."""
import argparse
from collections import Counter
import csv
from dataclasses import replace
import hashlib
import json
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

import plot_style
from destination import DestinationPool, DestinationReplica, dedicated_sink_architecture
from planner import plan
from profiles import ModelProfile, PowerCurve, RateCurve
from simulate import ExecutionScenario, NetworkLink, PowerNode, ServingInstance, SimSession, predict

ROOT = Path(__file__).resolve().parent
ACTIONS = ('kv_transfer', 'replay', 'not_moved')


def private_bytes(context, geometry):
    chunks = max(0, context - 1) // geometry['chunk_tokens']
    return sum((chunks if row['sw_size_chunks'] == -1 else min(chunks, row['sw_size_chunks']))
               * row['chunk_bytes'] for row in geometry['object_groups'])


def profile_for(model, contexts):
    path = ROOT / model['frozen_profile']
    if hashlib.sha256(path.read_bytes()).hexdigest() != model['profile']['sha256']:
        raise ValueError('frozen measured profile hash mismatch')
    profile = ModelProfile.load(path)
    if (profile.power_window_s, profile.max_destination_kv_streams, profile.max_destination_replays) != (5, 8, 1):
        raise ValueError('requires 5s power window, eight KV streams and one replay server')
    if profile.model != model['model']:
        raise ValueError('profile model mismatch')
    curve = RateCurve.parse({'1': model['prefill_tps']})
    for context in contexts:
        curve.rate(context, 1)
    if not profile.kv_geometry.fits([max(contexts)] * 8):
        raise ValueError('eight largest contexts exceed measured KV group capacity')
    transfer = replace(profile.case().kv_transfer, block_tokens=1, block_bytes=1,
                       bytes_by_context=tuple((c, private_bytes(c, model['private_geometry']))
                                              for c in sorted(set(contexts))),
                       setup_s=0, destination_bytes_per_s=float('inf'),
                       initial_completion_s=0, catch_up_fixed_s=0)
    case = replace(profile.case(), prefill=curve, replay=curve, replay_completion_s=0,
                   switch_s=0, kv_transfer=transfer, phase_power=None,
                   power_curve=PowerCurve.parse([[0, 0], [profile.max_power_load, profile.max_power_load]]))
    return replace(profile, cases={'central': case})


def problem_for(contexts, order, profile, bandwidth, deadline):
    # The existing planner reserves this window; its migration budget is exactly D.
    end = deadline + profile.power_window_s
    nodes = ('source', *sorted(bandwidth))
    problem = ExecutionScenario(
        end, end, profile.case().power(0), 'awake', 0,
        tuple(PowerNode(node, 1, node == 'source') for node in nodes),
        tuple(ServingInstance(node, (node,)) for node in nodes),
        tuple(SimSession(str(j), 'source', contexts[j], 1., 0., 2 * contexts[j]) for j in order),
        tuple(NetworkLink(f'link/{node}', bandwidth[node] * 125000) for node in nodes[1:]))
    architecture = dedicated_sink_architecture(profile, nodes[1], (f'link/{nodes[1]}',))
    architecture = replace(architecture, pools=tuple(
        DestinationPool(f'pool/{node}', architecture.types[0].type_id,
                        (DestinationReplica(node),), f'route/{node}', (f'link/{node}',))
        for node in nodes[1:]))
    if sum(1 / profile.case().prefill.rate(c, 1) for c in contexts) >= .004:
        raise ValueError('equal-credit bookkeeping exceeds 0.4% destination service')
    return problem, architecture


def schedule(contexts, order, profile, bandwidth, deadline):
    problem, architecture = problem_for(contexts, order, profile, bandwidth, deadline)
    result = plan(problem, profile, {}, 'greedy', destination=architecture, admission_mode='normal')
    execution = predict(problem, profile, result.moves, destination=architecture)
    completed = {r.session_id: r.committed_s for r in execution.sessions}
    moves = {r.session_id: r for r in result.moves}
    rows = []
    for index in order:
        move, finish = moves.get(str(index)), completed.get(str(index))
        on_time = move is not None and finish is not None and finish <= deadline + 1e-8
        rows.append({'session': index, 'action': move.method if on_time else 'not_moved',
                     'selected_action': move.method if move else 'not_moved',
                     'destination': move.destination_instance if move else None, 'finish_s': finish})
    return rows


def plot(report, out):
    plot_style.apply()
    fig, axes = plt.subplots(1, 3, figsize=(11, 4), sharey=True)
    for ax, model in zip(axes, report['models']):
        rows = model['summary']
        bottom = np.zeros(len(rows))
        for action in ACTIONS:
            values = np.array([row['percent'][action] for row in rows])
            ax.bar(range(len(rows)), values, bottom=bottom, width=.7,
                   label=plot_style.ACTION_NAMES[action], color=plot_style.ACTION_COLORS[action],
                   edgecolor='white', linewidth=.6)
            for x, (value, base) in enumerate(zip(values, bottom)):
                if value >= 7:
                    ax.text(x, base + value / 2, f'{value:.1f}%', ha='center', va='center',
                            color='white' if action == 'kv_transfer' else 'black', fontsize=10)
            bottom += values
        ax.set(xticks=range(len(rows)), xticklabels=[f"{r['deadline_s']:g}" for r in rows],
               xlabel='Deadline (s)', ylim=(0, 100), title=plot_style.MODEL_NAMES[model['model']])
        ax.spines[['top', 'right']].set_visible(False)
    axes[0].set_ylabel('Session outcomes by deadline (%)')
    fig.suptitle('QH Greedy: static migration, equal session weights', fontsize=15, y=1.01)
    fig.legend(*axes[0].get_legend_handles_labels(), loc='upper center', ncol=3,
               bbox_to_anchor=(.5, .94), frameon=False)
    rates = report['bandwidth_mbps']
    fig.text(.5, -.01, f"{report['resamples']} matched eight-session draws; Germany {rates['germany'] / 1000:.2f} Gb/s, East {rates['east'] / 1000:.2f} Gb/s.\n"
             'Remain includes late moves. Timing estimate; excludes lookup, restoration and decode.',
             ha='center', fontsize=10)
    fig.tight_layout(rect=(0, .08, 1, .86))
    for suffix in ('png', 'pdf'):
        fig.savefig(out / f'action-mix.{suffix}', bbox_inches='tight')
    plt.close(fig)


def calculate(report):
    packs, orders, bandwidth = report['context_packs'], report['session_orders'], report['bandwidth_mbps']
    if (report['resamples'] != len(packs) or len(orders) != len(packs)
            or any(len(pack) != 8 for pack in packs)
            or any(sorted(order) != list(range(8)) for order in orders)
            or set(bandwidth) != {'east', 'germany'} or any(rate <= 0 for rate in bandwidth.values())):
        raise ValueError('requires matched eight-session draws and two positive measured links')
    report.update(schema='queue-haul-quick-timing-mix-v2', execution='offline QH planner and event simulation',
                  policy='planner.plan(solver=greedy, admission_mode=normal), then simulate.predict',
                  assumptions='Static idle destinations; each session has equal expected_f=1, expected_g=0 solely for equal selection credit under a synthetic linear power curve, below 0.4% service utilization. No ongoing requests or growth. Measured A100 single-request P50 prefill is used for replay on both destinations. Private KV/link only, 2-byte/token replay log, zero endpoint residuals and tail recomputation. Measured scalar memory limits; all eight largest contexts also checked against measured KV group budgets. Eight KV streams, one replay server. Planner deadline is D+5 with its existing 5s power window, yielding migration budget D; bars classify actual simulator commits by D. No claim of live readiness, full decode stream completion, original physical-demand power attainment, or optimality. Independent links; no shared source-egress cap.',
                  counting='Completed KV/replay by D; remain includes unselected or late sessions. Selected counts and late moves retained separately.')
    for model in report['models']:
        profile = profile_for(model, [c for pack in packs for c in pack])
        model['summary'], model['cases'] = [], []
        for deadline in report['deadlines_s']:
            total, selected, destinations, full, late = Counter(), Counter(), Counter(), 0, 0
            for index, (contexts, order) in enumerate(zip(packs, orders)):
                actions = schedule(contexts, order, profile, bandwidth, deadline)
                counts = Counter(r['action'] for r in actions)
                total.update(counts)
                selected.update(r['selected_action'] for r in actions)
                destinations.update(f"{r['destination']}/{r['action']}" for r in actions if r['action'] != 'not_moved')
                late += sum(r['selected_action'] != 'not_moved' and r['action'] == 'not_moved' for r in actions)
                full += counts['not_moved'] == 0
                model['cases'].append({'draw': index, 'deadline_s': deadline, 'actions': actions})
            counts = {a: total[a] for a in ACTIONS}
            row = {'deadline_s': deadline, 'counts': counts,
                   'percent': {a: 100 * count / (8 * len(packs)) for a, count in counts.items()},
                   'all_eight_fit_draws': full, 'late_selected_sessions': late,
                   'selected_counts': {a: selected[a] for a in ACTIONS},
                   'completed_destination_counts': dict(sorted(destinations.items())),
                   'session_attainment_percent': 100 * (1 - counts['not_moved'] / (8 * len(packs)))}
            model['summary'].append(row)
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--inputs', type=Path, default=ROOT / 'outputs/quick-action-mix-20260910/report.json')
    parser.add_argument('--out', type=Path, required=True)
    args = parser.parse_args()
    report = calculate(json.loads(args.inputs.read_text()))
    args.out.mkdir(parents=True, exist_ok=False)
    (args.out / 'report.json').write_text(json.dumps(report, indent=2) + '\n')
    with (args.out / 'summary.csv').open('w') as handle:
        writer = csv.writer(handle, lineterminator='\n')
        writer.writerow(['model', 'deadline_s', 'draws', 'action', 'count', 'percent'])
        for model in report['models']:
            for row in model['summary']:
                for action in ACTIONS:
                    writer.writerow([model['model'], row['deadline_s'], report['resamples'], action,
                                     row['counts'][action], row['percent'][action]])
    plot(report, args.out)


if __name__ == '__main__':
    main()
