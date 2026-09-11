"""Verify frozen initial matrices and portable historical scalar witnesses."""
import hashlib
import importlib.util
import json
from pathlib import Path
import sys
import numpy as np

OUT = Path(__file__).resolve().parent
ROOT = OUT.parents[3]
sys.path.insert(0, str(ROOT))
from pool_shed_priced_greedy import priced_greedy


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def array_sha(value):
    value = np.ascontiguousarray(value)
    return hashlib.sha256(str((value.dtype.str, value.shape)).encode() + value.tobytes()).hexdigest()


def run():
    baseline = ROOT / 'outputs/a100-greedy-quality-20260911/matrix'
    before = np.load(OUT / 'precompression.npz')
    binary = Path(importlib.util.find_spec('_queue_haul_native._queue_haul_native').origin)
    files = [Path(__file__), OUT / 'test_budget_zero.py', OUT / 'precompression.npz', OUT / 'feedback-regression.npz', OUT / 'convergence-regression.npz', binary,
             *(ROOT / name for name in ('native/src/packing.rs', 'native/src/lib.rs', 'pool_shed_priced_greedy.py',
               'pool_shed_planner.py', 'tests/test_pool_shed_priced_greedy.py')),
             *(baseline / f'{case}.npz' for case in before.files)]
    sources = {str(p.relative_to(ROOT)) if p.is_relative_to(ROOT) else str(p): sha(p) for p in files}
    records, witnesses = [], {}
    for case in sorted(before.files):
        with np.load(baseline / f'{case}.npz') as data:
            A, b, g, debt, lp = (data[k] for k in ('matrix', 'capacity', 'gains', 'debt', 'lp'))
        old = before[case]
        arrays = {name: array_sha(v) for name, v in zip(('matrix', 'capacity', 'gains', 'debt'), (A, b, g, debt))}
        x, info = priced_greedy(A, b, g, np.zeros(len(g)), debt, tolerance=1e-12, absolute_tolerance=.001, return_prices=True)
        dual = info.pop('dual_prices')
        usage, covered = A @ x, A.T @ dual
        residual = float(np.max(usage[b > 0] / b[b > 0] - 1, initial=0))
        deficit = float(np.max(1 - covered[g > 0] / g[g > 0], initial=0))
        drift = float(g @ (x - old))
        assert np.isfinite(x).all() and np.all(x >= 0) and np.isfinite(dual).all() and np.all(dual >= 0)
        assert not np.any(usage[b == 0] > 0) and residual <= 1e-8 and deficit <= 1e-10
        assert g @ x <= b @ dual * (1 + 1e-10) and g @ lp <= b @ dual + 1e-8 and b @ dual - g @ x <= .001
        assert np.all(old >= 0) and np.max((A @ old)[b > 0] / b[b > 0]) <= 1 + 1e-8
        assert g @ old <= b @ dual + 1e-8 and g @ old - g @ x <= .001
        records.append(dict(case=case, original_array_sha256=arrays,
            final_matrix_file=f'outputs/a100-native-greedy-20260911/matrix/{case}.npz',
            shape=list(A.shape), scalar_precompression_support=int((old > 0).sum()), current_support=int((x > 0).sum()), lp_support=int((lp > 0).sum()),
            before_source_rows_binding=int((b[:24] - A[:24] @ old <= 1e-8 * b[:24]).sum()),
            after_source_rows_binding=int((b[:24] - A[:24] @ x <= 1e-8 * b[:24]).sum()),
            objective_change_from_scalar_primal=drift, secondary_before=float(debt @ old), secondary_after=float(debt @ x),
            max_relative_primal_residual=residual, max_relative_dual_deficit=deficit, **info))
        witnesses[case + '_allocation'], witnesses[case + '_dual'] = x, dual
    assert len(records) == 12 and sources == {p: sha(ROOT / p) for p in sources}
    target = OUT / 'witnesses.npz'
    np.savez_compressed(target, **witnesses)
    report = dict(scope=[
        'Correctness and support checks on 12 frozen initial matrices, identified by case and exact array hashes; no datacenter execution or timing comparison.',
        'Every original capacity and the current primary dual certificate are checked. Historical scalar allocations are a different coordinate trajectory; their signed objective changes are descriptive. Direct Rust tests separately verify preservation of the current primary value during support cleanup.',
        'Secondary work is a greedy tie-break only; cleanup can change it.',
        'Positive columns are grouped by at most 128 touched resource rows; each dense inverse is at most 129×129 including the primary equality. Wider singleton columns remain unchanged.',
        'Budget zero and already-certified initial seeds retain their existing behavior. There is no resident TTFT/SLO or global execution-optimum certificate.'],
        source_sha256=sources, witness_file=str(target.relative_to(ROOT)), witness_sha256=sha(target), records=records)
    (OUT / 'review.json').write_text(json.dumps(report, indent=2) + '\n')
    print(json.dumps(dict(cases=len(records), max_gap=max(r['absolute_gap'] for r in records),
                         support_range=[min(r['current_support'] for r in records), max(r['current_support'] for r in records)])))


if __name__ == '__main__':
    run()
