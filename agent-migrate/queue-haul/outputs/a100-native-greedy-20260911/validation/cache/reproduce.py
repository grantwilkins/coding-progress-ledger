"""Verify the archive, or replay its two fixed-admission forecasts without a solver."""

import argparse
import cProfile
import gzip
import hashlib
import json
import pickle
import pstats
import subprocess
import sys
import tempfile
import time
import zipfile
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(',', ':'), allow_nan=False).encode()).hexdigest()


def result_summary(result):
    return {'result_sha256': digest(result), 'result_field_sha256': {k: digest(v) for k, v in result.items()},
            'wave_count': len(result['wave_schedules'])}


def verify():
    review = json.loads((ROOT / 'review.json').read_text())
    for name, expected in review['artifact_sha256'].items():
        assert hashlib.sha256((ROOT / name).read_bytes()).hexdigest() == expected, name
    with zipfile.ZipFile(ROOT / 'sources.zip') as archive:
        assert set(archive.namelist()) == set(review['source_member_sha256'])
        for name, expected in review['source_member_sha256'].items():
            assert hashlib.sha256(archive.read(name)).hexdigest() == expected, name
    return review


def forecast(variant, output):
    review = verify()
    output.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix='qh-cache-source-') as temporary:
        source = Path(temporary)
        with zipfile.ZipFile(ROOT / 'sources.zip') as archive:
            for name in archive.namelist():
                if name.startswith(('common/', variant + '/')):
                    (source / Path(name).name).write_bytes(archive.read(name))
        sys.path.insert(0, str(source))
        from pool_shed_execution import PooledExecution
        from pool_shed_planner import mandatory_profile, planning_grid
        # This owned pickle is loaded only after its archive checksum is verified.
        bundle = pickle.loads(gzip.decompress((ROOT / 'prepared.pkl.gz').read_bytes()))
        table, calibration = bundle['table'], bundle['calibration']
        engine = PooledExecution(table, calibration['timing'][0], calibration, chunks=bundle['chunks'])
        started = time.perf_counter()
        for entry in bundle['admissions']:
            engine.advance(entry['time_s'])
            chosen = np.zeros(len(table.route))
            for column, mass in entry['chosen']:
                chosen[column] = mass
            engine.admit(chosen)
        prefix_wall_s = time.perf_counter() - started
        assert engine.now == bundle['observation_s'] and engine.n == review['comparison']['wave_count']
        edges = planning_grid(engine.now, bundle['deadline_s'], np.array([4.]), .5)
        assert edges.tolist() == review['comparison']['planning_edges_s']
        forecasts, original = [], engine.nominal_continuation

        def capture(*args):
            clone = original(*args)
            forecasts.append(clone)
            return clone

        engine.nominal_continuation = capture
        profiler = cProfile.Profile()
        profiler.enable()
        arrays, finish = mandatory_profile(engine, table, calibration['timing'][0], calibration, edges)
        profiler.disable()
        assert len(forecasts) == 1
        arrays['finish'] = finish
        with np.load(ROOT / 'forecast.npz') as expected:
            assert set(arrays) == set(expected.files)
            for key, value in arrays.items():
                assert np.array_equal(value, expected[key]), key
        summary = result_summary(forecasts[0].result())
        assert summary == review['comparison']['result'], variant
        stats = pstats.Stats(profiler)
        summary['profile'] = {'prefix_wall_s': prefix_wall_s, 'cprofile_s': stats.total_tt,
                              'calls': {name: {'calls': values[1], 'cumulative_s': values[3]}
                                        for (_, _, name), values in stats.stats.items()
                                        if name in ('network_cap', 'network_caps', 'source_turns', '_buffered')}}
        (output / 'result.json').write_text(json.dumps(summary, indent=2, sort_keys=True) + '\n')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--reproduce', action='store_true', help='run two CPU-only forecasts sequentially, roughly 20 seconds')
    parser.add_argument('--variant', choices=('before', 'after'), help=argparse.SUPPRESS)
    parser.add_argument('--output', type=Path, help=argparse.SUPPRESS)
    args = parser.parse_args()
    review = verify()
    if args.variant:
        assert args.output is not None
        forecast(args.variant, args.output)
    elif args.reproduce:
        with tempfile.TemporaryDirectory(prefix='qh-cache-forecast-') as temporary:
            for variant in ('before', 'after'):
                output = Path(temporary) / variant
                subprocess.run([sys.executable, str(Path(__file__).resolve()), '--variant', variant, '--output', str(output)], check=True)
                print(json.dumps({'variant': variant, **json.loads((output / 'result.json').read_text())}, sort_keys=True))
    else:
        print(json.dumps({'archive_verified': True, 'artifacts': len(review['artifact_sha256']),
                          'source_members': len(review['source_member_sha256']),
                          'historical_comparison': review['comparison']['wave_count']}, sort_keys=True))
