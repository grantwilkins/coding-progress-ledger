"""Optimistic transport/prefill action estimates; not the Queue-Haul planner."""
import argparse
from collections import Counter
import csv
import json
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

import plot_style

ROOT = Path(__file__).resolve().parent
ACTIONS = ('kv_transfer', 'replay', 'not_moved')


def private_bytes(context, geometry):
    chunks = max(0, context - 1) // geometry['chunk_tokens']
    return sum((chunks if row['sw_size_chunks'] == -1 else min(chunks, row['sw_size_chunks']))
               * row['chunk_bytes'] for row in geometry['object_groups'])


def schedule(contexts, order, prefill, geometry, bandwidth, deadline):
    """Choose the earliest finishing action, sharing each link and replay server."""
    links, servers = dict.fromkeys(bandwidth, 0.), dict.fromkeys(bandwidth, 0.)
    rows = []
    for index in order:
        context = contexts[index]
        if not prefill[0][0] <= context <= prefill[-1][0]:
            raise ValueError('context outside measured prefill support')
        seconds = context / np.interp(context, *zip(*prefill))
        candidates = []
        for node, mbps in bandwidth.items():
            rate = mbps * 1e6 / 8
            kv_end = links[node] + private_bytes(context, geometry) / rate
            log_end = links[node] + 2 * context / rate
            candidates.extend(((kv_end, 'kv_transfer', node, kv_end),
                               (max(log_end, servers[node]) + seconds, 'replay', node, log_end)))
        finish, action, node, link_end = min(candidates)
        if finish > deadline:
            rows.append({'session': index, 'action': 'not_moved', 'destination': None, 'finish_s': None})
            continue
        links[node] = link_end
        if action == 'replay':
            servers[node] = finish
        rows.append({'session': index, 'action': action, 'destination': node, 'finish_s': finish})
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
    axes[0].set_ylabel('Share of session choices (%)')
    fig.suptitle('Optimistic timing-only estimate: earliest-finish heuristic', fontsize=15, y=1.01)
    fig.legend(*axes[0].get_legend_handles_labels(), loc='upper center', ncol=3,
               bbox_to_anchor=(.5, .94), frameon=False)
    rates = report['bandwidth_mbps']
    fig.text(.5, -.01, f"{report['resamples']} matched eight-session draws; Germany {rates['germany'] / 1000:.2f} Gb/s, East {rates['east'] / 1000:.2f} Gb/s.\n"
             'Not moved = remain. Excludes lookup, restoration, decode and ongoing-load admission.',
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
    for model in report['models']:
        model['summary'], model['cases'] = [], []
        for deadline in report['deadlines_s']:
            total, full = Counter(), 0
            for index, (contexts, order) in enumerate(zip(packs, orders)):
                actions = schedule(contexts, order, model['prefill_tps'], model['private_geometry'], bandwidth, deadline)
                counts = Counter(r['action'] for r in actions)
                total.update(counts)
                full += counts['not_moved'] == 0
                model['cases'].append({'draw': index, 'deadline_s': deadline, 'actions': actions})
            counts = {a: total[a] for a in ACTIONS}
            row = {'deadline_s': deadline, 'counts': counts,
                   'percent': {a: 100 * count / (8 * len(packs)) for a, count in counts.items()},
                   'all_eight_fit_draws': full}
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
        writer = csv.writer(handle)
        writer.writerow(['model', 'deadline_s', 'draws', 'action', 'count', 'percent'])
        for model in report['models']:
            for row in model['summary']:
                for action in ACTIONS:
                    writer.writerow([model['model'], row['deadline_s'], report['resamples'], action,
                                     row['counts'][action], row['percent'][action]])
    plot(report, args.out)


if __name__ == '__main__':
    main()
