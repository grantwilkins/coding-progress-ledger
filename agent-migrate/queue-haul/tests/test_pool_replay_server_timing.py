import ast
import asyncio
import json
import pickle
from pathlib import Path
from types import SimpleNamespace as NS

import pytest

import pool_replay_server_timing as timing


@pytest.fixture
def records(monkeypatch, tmp_path):
    mode = tmp_path / 'mode'
    mode.write_text('on')
    monkeypatch.setenv('QH_SERVER_TIMING_MODE_FILE', str(mode))
    result = []
    monkeypatch.setattr(timing, 'emit', lambda kind, **fields: result.append(dict(kind=kind, **fields)))
    monkeypatch.setattr(timing, '_ORDINALS', {})
    return result


def test_scheduler_snapshots_preupdate_cache_queue_and_serializable_iteration(records):
    request = NS(request_id='internal', num_prompt_tokens=8192, num_output_tokens=0,
                 num_computed_tokens=8160, status=NS(name='RUNNING'), num_preemptions=2,
                 prefill_stats=NS(num_local_cached_tokens=8160, num_external_cached_tokens=0))
    scheduler = NS(requests={'internal': request}, running=[request], waiting=[])
    output = NS(num_scheduled_tokens={'internal': 32})
    timing.schedule_start(scheduler)
    timing.scheduled(scheduler, output)
    request.num_computed_tokens += 32
    row = records[-1]['requests'][0]
    assert (row['computed_before'], row['scheduled_tokens'], row['native_cached_tokens'], row['external_cached_tokens']) == (8160, 32, 8160, 0)
    assert row['preemptions'] == 2
    assert pickle.loads(pickle.dumps(output)).qh_iteration == records[-1]['iteration']
    output.num_scheduled_tokens['internal'] = 8193
    with pytest.raises(ValueError, match='budget'):
        timing.scheduled(scheduler, output)


class Event:
    def __init__(self, complete=True):
        self.complete = complete

    def query(self):
        return self.complete

    def elapsed_time(self, other):
        assert self.complete and other.complete
        return 1.25


def batch(iteration):
    return dict(iteration=iteration, ready_ns=123, ordinals={},
                events={f'{phase}_{edge}': Event() for phase in ('forward', 'logits', 'sample') for edge in ('start', 'end')})


def test_async_batches_keep_their_iteration_and_ordinals_out_of_order(records):
    first, second = batch('first'), batch('second')
    timing.reserve(first, ['r', 'prefill'], {1})
    timing.reserve(second, ['r'], set())
    timing.generated(second, NS(req_ids=['r'], sampled_token_ids=[[11]]))
    timing.generated(first, NS(req_ids=['r', 'prefill'], sampled_token_ids=[[10], []]))
    assert [(r['iteration'], r['requests'][0]['ordinal_start']) for r in records] == [('second', 1), ('first', 0)]
    assert records[1]['requests'][1]['token_ids'] == []
    assert records[0]['forward_stream_ms'] == 1.25


def test_device_events_must_already_be_synchronized(records):
    current = batch('x')
    current['events']['sample_end'] = Event(False)
    timing.reserve(current, ['r'], set())
    with pytest.raises(ValueError, match='synchronization'):
        timing.generated(current, NS(req_ids=['r'], sampled_token_ids=[[1]]))
    assert not records


def test_generation_cannot_silently_lose_tokens(records):
    current = batch('x')
    timing.reserve(current, ['r'], set())
    with pytest.raises(ValueError, match='reserved ordinal'):
        timing.generated(current, NS(req_ids=['r'], sampled_token_ids=[[]]))


def test_toggle_does_not_change_frontend_ordinal_accounting(records, monkeypatch, tmp_path):
    mode = tmp_path / 'mode'
    monkeypatch.setenv('QH_SERVER_TIMING_MODE_FILE', str(mode))
    mode.write_text('off')
    timing.frontend('collector_put', 'off', [1])
    assert not records
    mode.write_text('on')
    timing.frontend('collector_put', 'r', [1, 2])
    timing.frontend('collector_put', 'r', [3])
    timing.frontend('collector_pop', 'r', [1, 2])
    assert [r['ordinal_start'] for r in records] == [0, 2, 0]
    mode.write_text('invalid')
    with pytest.raises(ValueError, match='invalid timing mode'):
        timing.enabled()


def test_buffered_log_has_contiguous_sequence_and_orderly_final_record(tmp_path, monkeypatch):
    read_text, readlink = Path.read_text, timing.os.readlink
    proc = {'/proc/sys/kernel/random/boot_id': 'test-boot\n', '/proc/self/timens_offsets': 'monotonic 0 0\n'}
    monkeypatch.setattr(Path, 'read_text', lambda path, *a, **k: proc[str(path)] if str(path) in proc else read_text(path, *a, **k))
    monkeypatch.setattr(timing.os, 'readlink', lambda path: 'time:[test]' if str(path) == '/proc/self/ns/time' else readlink(path))
    log = timing.EventLog(tmp_path)
    for i in range(257):
        log.write('sample', i=i)
    log.close()
    rows = [json.loads(line) for line in log.path.read_text().splitlines()]
    assert [row['sequence'] for row in rows] == list(range(1, 260))
    assert rows[-1]['dropped_records'] == 0
    assert rows[-1]['records_before_final'] == 258
    assert (rows[0]['boot_id'], rows[0]['time_namespace']) == ('test-boot', 'time:[test]')


def test_source_replacement_fails_on_ambiguity():
    with pytest.raises(ValueError, match='source mismatch'):
        timing.replace_once('same same', 'same', 'changed')


def test_preserved_fifo_delivers_distinct_outputs_and_errors_with_hooks(records):
    source = Path('outputs/a100-replay-completion-20260910T0116/stack/output_processor.patched.py').read_text()
    source = timing.replace_once(source, '        """Non-blocking put operation."""', '        """Non-blocking put operation."""\n        _qh_timing.collector("collector_put", self, output)')
    source = timing.replace_once(source, '            output = self._qh_outputs.popleft() if self._qh_outputs else None', '            output = self._qh_outputs.popleft() if self._qh_outputs else None\n            _qh_timing.collector("collector_pop", self, output)')
    node = next(node for node in ast.parse(source).body if isinstance(node, ast.ClassDef) and node.name == 'RequestOutputCollector')
    namespace = dict(asyncio=asyncio, RequestOutputKind=NS(DELTA=1), RequestOutput=NS, PoolingRequestOutput=NS, _qh_timing=timing)
    exec(compile('from __future__ import annotations\n' + ast.unparse(node), '<collector>', 'exec'), namespace)
    queue = namespace['RequestOutputCollector'](1, 'r')
    outputs = [NS(outputs=[NS(token_ids=[token], index=0)], finished=False) for token in (7, 8)]
    async def exercise():
        for output in outputs:
            queue.put(output)
        assert await queue.get() is outputs[0]
        assert await queue.get() is outputs[1]
        assert queue.get_nowait() is None
        queue.put(ValueError('retained failure'))
        with pytest.raises(ValueError, match='retained failure'):
            await queue.get()
    asyncio.run(exercise())
    assert [(r['kind'], r['ordinal_start']) for r in records[:4]] == [('collector_put', 0), ('collector_put', 1), ('collector_pop', 0), ('collector_pop', 1)]
    assert 'retained failure' in records[-1]['error']
