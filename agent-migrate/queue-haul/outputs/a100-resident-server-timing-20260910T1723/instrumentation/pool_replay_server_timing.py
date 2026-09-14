"""Exact vLLM 0.22.0 source hooks; CUDA durations are stream elapsed, not host time."""
import argparse
import atexit
import hashlib
import json
from multiprocessing.util import Finalize
import os
from pathlib import Path
import socket
import threading
import time

_LOG = None
_LOCK = threading.RLock()
_ORDINALS = {}


def enabled():
    path = os.environ.get('QH_SERVER_TIMING_MODE_FILE')
    if not path:
        return False
    mode = Path(path).read_text().strip()
    if mode not in ('off', 'on'):
        raise ValueError(f'invalid timing mode: {mode}')
    return mode == 'on'


class EventLog:
    def __init__(self, directory):
        Path(directory).mkdir(parents=True, exist_ok=True)
        self.path = Path(directory) / f'{socket.gethostname()}-{os.getpid()}.jsonl'
        self.file = self.path.open('x', buffering=1024 * 1024)
        self.count = 0
        self.write('process_start', host=socket.gethostname(), pid=os.getpid(),
                   boot_id=Path('/proc/sys/kernel/random/boot_id').read_text().strip(),
                   time_namespace=os.readlink('/proc/self/ns/time'),
                   time_namespace_offsets=Path('/proc/self/timens_offsets').read_text(),
                   cuda_visible_devices=os.environ.get('CUDA_VISIBLE_DEVICES'),
                   clock='CLOCK_MONOTONIC', wall_ns=time.time_ns(),
                   module_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest())
        atexit.register(self.close)
        self.finalizer = Finalize(self, self.close, exitpriority=10)

    def write(self, kind, **fields):
        with _LOCK:
            self.count += 1
            self.file.write(json.dumps(dict(sequence=self.count, kind=kind,
                                           mono_ns=time.monotonic_ns(), **fields)) + '\n')
            if self.count % 128 == 0 or kind in ('process_start', 'module', 'process_final'):
                self.file.flush()

    def close(self):
        with _LOCK:
            if not self.file.closed:
                self.write('process_final', dropped_records=0, records_before_final=self.count)
                self.file.close()


def emit(kind, **fields):
    global _LOG
    with _LOCK:
        if _LOG is None:
            _LOG = EventLog(os.environ['QH_SERVER_TIMING_DIR'])
        _LOG.write(kind, **fields)


def imported(path):
    if os.environ.get('QH_SERVER_TIMING_DIR'):
        emit('module', path=str(Path(path).resolve()), sha256=hashlib.sha256(Path(path).read_bytes()).hexdigest())


def schedule_start(scheduler):
    scheduler._qh_start = time.monotonic_ns() if enabled() else None


def scheduled(scheduler, output):
    if scheduler._qh_start is None:
        return
    scheduler._qh_iteration = getattr(scheduler, '_qh_iteration', 0) + 1
    output.qh_iteration = f'{socket.gethostname()}:{os.getpid()}:{scheduler._qh_iteration}'
    if sum(output.num_scheduled_tokens.values()) > 8192:
        raise ValueError('scheduled token budget exceeds frozen plan')
    rows = []
    for request in scheduler.requests.values():
        stats = request.prefill_stats
        rows.append(dict(request_id=request.request_id, prompt_tokens=request.num_prompt_tokens,
                         output_tokens=request.num_output_tokens, computed_before=request.num_computed_tokens,
                         scheduled_tokens=output.num_scheduled_tokens.get(request.request_id, 0),
                         status=request.status.name, preemptions=request.num_preemptions,
                         native_cached_tokens=None if stats is None else stats.num_local_cached_tokens,
                         external_cached_tokens=None if stats is None else stats.num_external_cached_tokens))
    emit('schedule', iteration=output.qh_iteration, start_ns=scheduler._qh_start,
         end_ns=time.monotonic_ns(), running=len(scheduler.running), waiting=len(scheduler.waiting), requests=rows)


def batch_start(runner, output):
    runner._qh_batch = None
    if output.qh_iteration is None:
        return
    if runner.speculative_config is not None or runner.is_pooling_model or runner.vllm_config.parallel_config.tensor_parallel_size != 1:
        raise ValueError('timing hooks require frozen TP1 generation without speculation')
    runner._qh_batch = dict(iteration=output.qh_iteration, events={}, ordinals={}, ready_ns=None)
    emit('worker_ingress', iteration=output.qh_iteration)


def mark(batch, phase):
    if batch is not None:
        import torch
        event = torch.cuda.Event(enable_timing=True)
        event.record()
        batch['events'][phase] = event


def ready(batch):
    if batch is not None:
        batch['ready_ns'] = time.monotonic_ns()


def reserve(batch, request_ids, invalid):
    if batch is not None:
        for i, req_id in enumerate(request_ids):
            key = ('worker', req_id)
            ordinal = _ORDINALS.get(key, 0)
            count = int(i not in invalid)
            batch['ordinals'][req_id] = (ordinal, count)
            _ORDINALS[key] = ordinal + count


def generated(batch, output):
    if batch is None:
        return
    events = batch['events']
    if batch['ready_ns'] is None or set(events) != {'forward_start', 'forward_end', 'logits_start', 'logits_end', 'sample_start', 'sample_end'}:
        raise ValueError('incomplete synchronized device timing boundaries')
    if not all(event.query() for event in events.values()):
        raise ValueError('existing output synchronization did not complete CUDA events')
    durations = {phase + '_stream_ms': events[phase + '_start'].elapsed_time(events[phase + '_end'])
                 for phase in ('forward', 'logits', 'sample')}
    requests = []
    for req_id, tokens in zip(output.req_ids, output.sampled_token_ids, strict=True):
        ordinal, count = batch['ordinals'][req_id]
        if len(tokens) != count:
            raise ValueError('generated token count differs from reserved ordinal')
        requests.append(dict(request_id=req_id, ordinal_start=ordinal, token_ids=list(tokens)))
    emit('worker_output', iteration=batch['iteration'], output_ready_ns=batch['ready_ns'],
         cuda_clock='relative GPU stream elapsed; includes waits', requests=requests, **durations)


def frontend(kind, request_id, tokens=(), **fields):
    if enabled():
        key = (kind, request_id)
        ordinal = _ORDINALS.get(key, 0)
        tokens = list(tokens)
        _ORDINALS[key] = ordinal + len(tokens)
        emit(kind, request_id=request_id, ordinal_start=ordinal, token_ids=tokens, **fields)


def collector(kind, collector, output):
    if output is None:
        return
    if isinstance(output, Exception):
        frontend(kind, collector.request_id, error=repr(output))
    else:
        for completion in output.outputs:
            frontend(kind, collector.request_id, completion.token_ids, finished=output.finished, index=completion.index)


def http_ingress(raw_request):
    if enabled():
        raw_request.state.qh_ingress = f'{os.getpid()}:{time.monotonic_ns()}'
        emit('http_ingress', ingress_id=raw_request.state.qh_ingress,
             boundary='completion route after body validation', path=raw_request.url.path)


def http_mapping(raw_request, request_id):
    if enabled() and raw_request is not None:
        emit('http_request_mapping', ingress_id=raw_request.state.qh_ingress, request_id=request_id)


def replace_once(source, old, new):
    if source.count(old) != 1:
        raise ValueError(f'vLLM source mismatch ({source.count(old)} matches): {old[:100]}')
    return source.replace(old, new, 1)


def patched_sources(root, fifo_source):
    """Prepare validated source strings without modifying the installed stack."""
    root = Path(root)
    metadata = root.parent / 'vllm-0.22.0.dist-info' / 'METADATA'
    if '\nVersion: 0.22.0\n' not in metadata.read_text():
        raise ValueError('vLLM 0.22.0 metadata required')
    result = {}
    def load(name):
        return (root / name).read_text()
    def save(name, source):
        source = 'import pool_replay_server_timing as _qh_timing\n' + source + '\n_qh_timing.imported(__file__)\n'
        compile(source, name, 'exec')
        result[name] = source
    name = 'v1/core/sched/output.py'
    source = replace_once(load(name), '    preempted_req_ids: set[str] | None = None', '    qh_iteration: str | None = None\n    preempted_req_ids: set[str] | None = None')
    save(name, source)
    name = 'v1/core/sched/scheduler.py'
    source = replace_once(load(name), '    def schedule(self) -> SchedulerOutput:\n', '    def schedule(self) -> SchedulerOutput:\n        _qh_timing.schedule_start(self)\n')
    source = replace_once(source, '            self._update_after_schedule(scheduler_output)', '            _qh_timing.scheduled(self, scheduler_output)\n            self._update_after_schedule(scheduler_output)')
    save(name, source)
    name = 'v1/worker/gpu_model_runner.py'
    source = replace_once(load(name), '        if self.execute_model_state is not None:\n', '        _qh_timing.batch_start(self, scheduler_output)\n        if self.execute_model_state is not None:\n')
    source = replace_once(source, '            model_output = self._model_forward(', '            _qh_timing.mark(self._qh_batch, "forward_start")\n            model_output = self._model_forward(')
    source = replace_once(source, '        with record_function_or_nullcontext("gpu_model_runner: postprocess"):', '            _qh_timing.mark(self._qh_batch, "forward_end")\n\n        with record_function_or_nullcontext("gpu_model_runner: postprocess"):')
    source = replace_once(source, '                logits = self.model.compute_logits(sample_hidden_states)\n            else:', '                _qh_timing.mark(self._qh_batch, "logits_start")\n                logits = self.model.compute_logits(sample_hidden_states)\n                _qh_timing.mark(self._qh_batch, "logits_end")\n            else:')
    source = replace_once(source, '            sampler_output = self._sample(logits, spec_decode_metadata)', '            _qh_timing.mark(self._qh_batch, "sample_start")\n            sampler_output = self._sample(logits, spec_decode_metadata)\n            _qh_timing.mark(self._qh_batch, "sample_end")')
    source = replace_once(source, '                valid_sampled_token_ids = self._to_list(sampled_token_ids)', '                valid_sampled_token_ids = self._to_list(sampled_token_ids)\n                _qh_timing.ready(self._qh_batch)')
    source = replace_once(source, '            return output\n\n        with record_function_or_nullcontext(\n            "gpu_model_runner: AsyncGPUModelRunnerOutput"', '            _qh_timing.reserve(self._qh_batch, output.req_ids, {i for i, tokens in enumerate(output.sampled_token_ids) if not tokens})\n            _qh_timing.generated(self._qh_batch, output)\n            return output\n\n        with record_function_or_nullcontext(\n            "gpu_model_runner: AsyncGPUModelRunnerOutput"')
    source = replace_once(source, '        return async_output\n', '        _qh_timing.reserve(self._qh_batch, output.req_ids, set(invalid_req_indices))\n        async_output._qh_batch = self._qh_batch\n        return async_output\n')
    source = replace_once(source, '        max_gen_len = self.sampled_token_ids_cpu.shape[-1]\n        self.async_copy_ready_event.synchronize()', '        max_gen_len = self.sampled_token_ids_cpu.shape[-1]\n        self.async_copy_ready_event.synchronize()\n        _qh_timing.ready(self._qh_batch)')
    source = replace_once(source, '        del self._routed_experts\n\n        return output', '        del self._routed_experts\n        _qh_timing.generated(self._qh_batch, output)\n\n        return output')
    save(name, source)
    name = 'v1/engine/output_processor.py'
    source = Path(fifo_source).read_text()
    if '_qh_outputs.popleft()' not in source or '_qh_outputs.append(output)' not in source:
        raise ValueError('existing FIFO collector patch required')
    source = replace_once(source, '        """Non-blocking put operation."""', '        """Non-blocking put operation."""\n        _qh_timing.collector("collector_put", self, output)')
    source = replace_once(source, '            output = self._qh_outputs.popleft() if self._qh_outputs else None', '            output = self._qh_outputs.popleft() if self._qh_outputs else None\n            _qh_timing.collector("collector_pop", self, output)')
    source = replace_once(source, '        self.request_states[request_id] = req_state', '        _qh_timing.frontend("frontend_registration", request_id, external_request_id=req_state.external_req_id, arrival_wall_s=request.arrival_time, arrival_clock="time.time at engine input processing")\n        self.request_states[request_id] = req_state')
    source = replace_once(source, '            req_id = engine_core_output.request_id\n', '            req_id = engine_core_output.request_id\n            _qh_timing.frontend("frontend_receipt", req_id, engine_core_output.new_token_ids, finish_reason=str(engine_core_output.finish_reason))\n')
    save(name, source)
    name = 'entrypoints/openai/completion/api_router.py'
    source = replace_once(load(name), 'async def create_completion(request: CompletionRequest, raw_request: Request):\n', 'async def create_completion(request: CompletionRequest, raw_request: Request):\n    _qh_timing.http_ingress(raw_request)\n')
    save(name, source)
    name = 'entrypoints/openai/completion/serving.py'
    source = replace_once(load(name), '        request_id = f"cmpl-{self._base_request_id(raw_request, request.request_id)}"', '        request_id = f"cmpl-{self._base_request_id(raw_request, request.request_id)}"\n        _qh_timing.http_mapping(raw_request, request_id)')
    save(name, source)
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--vllm-root', required=True, type=Path)
    parser.add_argument('--fifo-source', required=True, type=Path)
    parser.add_argument('--output', required=True, type=Path)
    args = parser.parse_args()
    sources = patched_sources(args.vllm_root, args.fifo_source)
    args.output.mkdir(parents=True, exist_ok=False)
    manifest = {}
    for name, source in sources.items():
        path = args.output / 'vllm' / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(source)
        manifest[name] = {'original_sha256': hashlib.sha256((args.vllm_root / name).read_bytes()).hexdigest(),
                          'patched_sha256': hashlib.sha256(path.read_bytes()).hexdigest()}
    (args.output / Path(__file__).name).write_bytes(Path(__file__).read_bytes())
    (args.output / 'manifest.json').write_text(json.dumps(manifest, indent=2) + '\n')


if __name__ == '__main__':
    main()
