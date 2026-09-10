"""Isolate network scaling in the frozen 2/2 MW, 30-second comparison."""

import argparse
import hashlib
import json
from dataclasses import replace
from pathlib import Path

import numpy as np

import pool_shed_campaign as q
from pool_shed_execution import kv_transfer_bytes, network_nodes


OUT = q.ROOT / 'outputs/a100-network-balance-audit-20260910'
REFERENCE = q.ROOT / 'outputs/a100-replay-queue-resolution-20260910/local-recovery/frontier.json'


def run(wan_gbps=None):
    if wan_gbps is not None and (not np.isfinite(wan_gbps) or wan_gbps <= 0):
        raise ValueError('WAN capacity must be positive and finite')
    caps = (1000., None) if wan_gbps is None else (wan_gbps,)
    output = OUT / ('network-sensitivity.json' if wan_gbps is None else f'network-{wan_gbps:g}gbps.json')
    if output.exists():
        raise ValueError('preserve the completed audit')
    measured = q.calibration(0)
    timing = measured['timing'][0]
    reference = json.loads(REFERENCE.read_text())
    endpoint = np.r_[np.median(q.network_samples()[:, :2], axis=0), 0.]
    endpoint[2] = endpoint[:2].sum()
    sources = {**q.provenance(measured), **{name: hashlib.sha256((q.ROOT / name).read_bytes()).hexdigest()
               for name in (Path(__file__).name, 'pool_shed_resident_fit.py', 'pool_shed_resident_queue.py')},
               str(REFERENCE.relative_to(q.ROOT)): hashlib.sha256(REFERENCE.read_bytes()).hexdigest()}
    report = dict(scope=f'{12 * len(caps)} CPU evaluations: two workloads, two endpoint-sharing widths, '
                  f'{len(caps)} shared WAN settings, three policies, D=30s. '
                  'Network counterfactuals, not new calibrated datacenter recommendations.', sources=sources,
                  controls='Same source and destination GPU counts, trajectories, phases, offered rates, full-context replay, '
                  '0.80GB/32K KV anchor, regional timing, local recovery requirement, 4s planning reference and candidate library.',
                  endpoint_bytes_per_s=endpoint.tolist(), cells=[])
    for workload in ('coding', 'coding_long'):
        base = q.replica_fleet(q.sample_fleet(workload, gpus=6666, gpus_per_node=8))
        base = replace(base, metadata={**base.metadata, 'planning_reference_s': 4., 'require_local_recovery': True,
            'kv_wire_scale': 800_000_000 / (32768 * 49152), 'replay_cached_tokens': [0] * len(base.count),
            'kv_shared_tokens': [0] * len(base.count)})
        base = replace(base, kv=kv_transfer_bytes(base, base.context, measured))
        sources.update(base.metadata['source_timing_inputs'])
        budgets = q.bandwidth(endpoint, base.nodes, 1000)
        replay, kv = q.include_isolated(*q.library(base), q.isolated_methods(base, .5, endpoint, budgets, timing))
        candidate_hash = hashlib.sha256(replay.tobytes() + kv.tobytes()).hexdigest()
        saved = next(c for c in reference['cells'] if c['workload'] == workload and c['deadline_s'] == 30)
        assert candidate_hash == saved['candidate_sha256']
        for width in (8, 1):
            fleet = replace(base, gpus_per_node=width)
            for cap in caps:
                budgets = q.bandwidth(endpoint, fleet.nodes, cap) if cap is not None else endpoint * network_nodes(fleet)
                table = q.schedule_table(fleet, replay, kv, .5, 30., endpoint, budgets, timing)
                cell = dict(workload=workload, gpus_per_network_endpoint=width, shared_wan_gbps=cap,
                    network_budgets_gbps=(8e-9 * budgets).tolist(), candidate_sha256=candidate_hash, results={})
                for policy in ('queue_haul', 'kv_only', 'replay_only'):
                    result = q.execute_feedback(table, table, policy, timing, measured)
                    assert result['max_relative_residual'] <= 1e-8
                    assert np.all(np.asarray(result['transferred_bytes']) <= budgets * 30 * (1 + 1e-8))
                    assert 0 <= result['recovered_handoff_fraction'] <= result['shed_fraction'] + 1e-8
                    if width == 8 and cap == 1000:
                        for key in ('shed_fraction', 'recovered_handoff_fraction', 'action_fractions', 'transferred_bytes'):
                            assert np.allclose(result[key], saved['results'][policy][key], rtol=1e-8, atol=1e-8)
                    cell['results'][policy] = {key: result[key] for key in ('shed_fraction', 'recovered_handoff_fraction',
                        'action_fractions', 'transferred_bytes', 'last_completion_s', 'reserved_destination_replicas')}
                    print(workload, width, f'{cap:g}Gbps' if cap is not None else 'replicated measured endpoints', policy,
                          round(result['recovered_handoff_fraction'], 6), flush=True)
                report['cells'].append(cell)
    assert all(hashlib.sha256((q.ROOT / name).read_bytes()).hexdigest() == checksum for name, checksum in sources.items())
    q.write_json(output, report)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--wan-gbps', type=float, help='Evaluate one shared WAN capacity at both endpoint-sharing widths')
    run(parser.parse_args().wan_gbps)
