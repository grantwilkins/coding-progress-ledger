"""Bounded 2/2 MW comparison with fixed destination replicas and causal source turns."""

import argparse
import csv
import hashlib
from dataclasses import replace
from pathlib import Path

import numpy as np

import pool_shed_campaign as q
import plot_style
from pool_shed_execution import kv_transfer_bytes
from pool_shed_replay_audit import FIELDS, fleet_summary
from pool_shed_resident_fit import calibrate_server_decode


def plot(report, out):
    import matplotlib.pyplot as plt

    plot_style.apply()
    workloads = list(report['fleets'])
    fig, axes = plt.subplots(2, len(workloads), figsize=(6 * len(workloads), 7), squeeze=False, sharex=True, sharey=True)
    for col, workload in enumerate(workloads):
        cells = [cell for cell in report['cells'] if cell['workload'] == workload]
        for row, metric in enumerate(('shed_fraction', 'recovered_handoff_fraction')):
            ax = axes[row, col]
            for policy in q.POLICIES:
                ax.plot([cell['deadline_s'] for cell in cells], [100 * cell['results'][policy][metric] for cell in cells],
                        **plot_style.policy_style(policy, names=plot_style.PAPER_POLICY_NAMES))
            ax.set(title=workload.replace('_', ' '), ylim=(0, 103), ylabel=('Handoff (%)' if row == 0 else 'Recovered handoff (%)'))
            ax.grid(alpha=.2)
            if row == 1:
                ax.set_xlabel('Deadline (s)')
    handles, labels = axes[0, 0].get_legend_handles_labels()
    fig.legend(handles, labels, loc='upper center', ncol=3, frameon=False)
    fig.text(.5, .015, report['planning_criterion'] + '\nConditional simulation; queue clearance is a fluid work measure, not resident SLO certification.', ha='center', fontsize=10)
    fig.tight_layout(rect=(0, .04, 1, .9))
    for extension in ('png', 'svg', 'pdf'):
        fig.savefig(out / f'frontier.{extension}', dpi=220)
    plt.close(fig)


def run(out, workloads, deadlines, require_local_recovery=False):
    if (out / 'frontier.json').exists():
        raise ValueError('use a fresh output directory; preserve the frozen comparison')
    calibration, decode = q.calibration(0), calibrate_server_decode()
    timing = calibration['timing'][0]
    sources = q.provenance(calibration)
    sources.update({name: hashlib.sha256((q.ROOT / name).read_bytes()).hexdigest() for name in
                    (Path(__file__).name, 'pool_shed_resident_fit.py', 'pool_shed_resident_queue.py')})
    report = {'scope': 'Bounded fixed-replica scenario, 2 MW source and 2 MW EACH destination; full-context replay; no GPU campaign.',
              'resident_latency_validated': False, 'campaign_ready': False, 'server_decode_fit': decode,
              'require_local_recovery': require_local_recovery,
              'planning_criterion': 'Require predicted local queues to clear by deadline' if require_local_recovery else 'Maximize handoff; report remaining queues separately',
              'shared_wan_gbps': 1000, 'wire_bytes_per_32768_tokens': 800_000_000, 'sources': sources, 'fleets': {}, 'cells': [],
              'limits': ['One incoming action pack per destination GPU; this is a declared placement scenario, not optimal bin packing.',
                         'Resident queues retain the measured throughput-loss fluid approximation; per-request resident TTFT is checked separately.',
                         'Source service uses warm-trained server cadence and sequential dependent turns, with no additional source GPU contention.',
                         'Historical regional replay factors, standing-service normalization and offered rates stay frozen.',
                         'Handoff is work-weighted ownership transfer; full handoff does not release the installed 2 MW nameplate.',
                         'Lines join evaluated deadlines only; no confidence bands or unmeasured deadline claims.']}
    endpoint = np.r_[np.median(q.network_samples()[:, :2], axis=0), 0.]
    endpoint[2] = endpoint[:2].sum()
    csv_rows = []
    for workload in workloads:
        fleet = q.replica_fleet(q.sample_fleet(workload, gpus=6666, gpus_per_node=8))
        fleet = replace(fleet, metadata={**fleet.metadata, 'planning_reference_s': 4.,
                        'require_local_recovery': require_local_recovery,
                        'kv_wire_scale': 800_000_000 / (32768 * 49152), 'replay_cached_tokens': [0] * len(fleet.count),
                        'kv_shared_tokens': [0] * len(fleet.count)})
        fleet = replace(fleet, kv=kv_transfer_bytes(fleet, fleet.context, calibration))
        sources.update(fleet.metadata['source_timing_inputs'])
        report['fleets'][workload] = {**fleet_summary(fleet, calibration),
            **{key: fleet.metadata[key] for key in ('source_timing_scope', 'source_timing_coefficients', 'placement_scope')},
            'maximum_source_turn_s': max(map(max, fleet.metadata['turn_duration_s']))}
        budgets = q.bandwidth(endpoint, fleet.nodes, 1000)
        replay, kv = q.include_isolated(*q.library(fleet), q.isolated_methods(fleet, .5, endpoint, budgets, timing))
        candidate_hash = hashlib.sha256(replay.tobytes() + kv.tobytes()).hexdigest()
        for deadline in deadlines:
            table = q.schedule_table(fleet, replay, kv, .5, deadline, endpoint, budgets, timing)
            cell = {'workload': workload, 'deadline_s': deadline, 'candidate_sha256': candidate_hash, 'results': {}}
            for policy in q.POLICIES:
                result = q.execute_feedback(table, table, policy, timing, calibration)
                generated, recovered, pending = [np.asarray(result[key]) for key in
                    ('resident_debt_generated_work_s', 'resident_debt_recovered_work_s', 'pending_resident_debt_work_s')]
                if (result['max_relative_residual'] > 1e-8 or result['last_completion_s'] > deadline + 1e-8
                        or not np.allclose(generated - recovered, pending, rtol=1e-8, atol=1e-6)
                        or np.any(result['resident_pool_compensation_work_s'])
                        or not 0 <= result['recovered_handoff_fraction'] <= result['shed_fraction'] + 1e-8
                        or np.any(np.asarray(result['transferred_bytes']) > budgets * deadline * (1 + 1e-8))):
                    raise ValueError('physical conservation or deadline check failed')
                cell['results'][policy] = {key: result[key] for key in (*FIELDS, 'recovered_handoff_fraction',
                    'recovered_action_fractions', 'reserved_destination_replicas', 'resident_displaced_work_s',
                    'resident_pool_compensation_work_s', 'service_recovery_scope', 'resident_debt_scope')}
                csv_rows.append({'workload': workload, 'deadline_s': deadline, 'policy': policy,
                    **{key: result[key] for key in ('shed_fraction', 'recovered_handoff_fraction', 'last_completion_s', 'pending_buffered_work_s')},
                    'pending_resident_work_s': float(pending.sum()), 'kv_shed_fraction': sum(result['action_fractions'][1::2]),
                    'source_power_proxy_mw': result['shed_fraction'] * report['fleets'][workload]['full_handoff_source_power_mw']})
                print(workload, deadline, policy, 'handoff', round(result['shed_fraction'], 5),
                      'local queues cleared', round(result['recovered_handoff_fraction'], 5), flush=True)
            report['cells'].append(cell)
            q.write_json(out / f'{workload}-{deadline:g}.json', cell)
    if any(hashlib.sha256((q.ROOT / path).read_bytes()).hexdigest() != checksum for path, checksum in sources.items()):
        raise ValueError('comparison inputs changed during execution')
    report['policy_evaluations'] = len(csv_rows)
    q.write_json(out / 'frontier.json', report)
    with (out / 'frontier.csv').open('w') as handle:
        writer = csv.DictWriter(handle, fieldnames=list(csv_rows[0]), lineterminator='\n')
        writer.writeheader()
        writer.writerows(csv_rows)
    plot(report, out)
    return report


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--out', type=Path, default=q.ROOT / 'outputs/a100-replay-queue-resolution-20260910')
    parser.add_argument('--workloads', choices=('coding', 'coding_long'), nargs='+', default=['coding', 'coding_long'])
    parser.add_argument('--deadlines', type=float, nargs='+', default=[30., 60., 120.])
    parser.add_argument('--require-local-recovery', action='store_true')
    args = parser.parse_args()
    if any(not np.isfinite(value) or value <= 0 for value in args.deadlines) or sorted(set(args.deadlines)) != args.deadlines:
        raise ValueError('deadlines must be positive, finite, unique and increasing')
    run(args.out, args.workloads, args.deadlines, args.require_local_recovery)
