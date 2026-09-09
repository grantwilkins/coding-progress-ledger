"""Preserve existing engine-output events; never split a multi-token engine event."""
import hashlib,json
from pathlib import Path

path=Path('/tmp/qh-replay-runtime/lib/python3.12/site-packages/vllm/v1/engine/output_processor.py')
out=Path(__file__).resolve().parent
source=path.read_text()
(out/'output_processor.original.py').write_text(source)
assert 'self._qh_outputs' not in source
patched=source.replace('        self.aggregate = output_kind == RequestOutputKind.DELTA',
    '        self._qh_outputs = __import__("collections").deque()\n        self.aggregate = output_kind == RequestOutputKind.DELTA')
patched=patched.replace('        """Non-blocking put operation."""\n', '''        """Non-blocking put operation."""
        if self.aggregate:
            self._qh_outputs.append(output)
            self.ready.set()
            return
''',1)
patched=patched.replace('        """Get operation blocks on put event."""\n', '''        """Get operation blocks on put event."""
        if self.aggregate:
            while not self._qh_outputs:
                await self.ready.wait()
            return self.get_nowait()
''',1)
patched=patched.replace('        """Non-blocking get operation."""\n', '''        """Non-blocking get operation."""
        if self.aggregate:
            output = self._qh_outputs.popleft() if self._qh_outputs else None
            if not self._qh_outputs:
                self.ready.clear()
            if isinstance(output, Exception):
                raise output
            return output
''',1)
assert patched!=source
compile(patched,str(path),'exec')
path.write_text(patched)
(out/'output_processor.patched.py').write_text(patched)
(out/'stream-patch.json').write_text(json.dumps({'runtime':'vllm0.22.0','path':str(path),
    'before_sha256':hashlib.sha256(source.encode()).hexdigest(),'after_sha256':hashlib.sha256(patched.encode()).hexdigest(),
    'change':'Preserve each engine RequestOutput in FIFO order instead of merging queued DELTA outputs; do not split engine events or alter sampling/scheduling.',
    'reason':'resident client token events failed the 99% exact-event prerequisite; original probes retained',
    'performance_equivalence':'not assumed; all subsequent paired arms use the same patch'},indent=2)+'\n')
