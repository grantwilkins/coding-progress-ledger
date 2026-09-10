import importlib.util
import tarfile
from pathlib import Path

import pytest

spec=importlib.util.spec_from_file_location('archive_raw',Path(__file__).with_name('archive-raw.py'))
module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)


def test_ignored_and_partial_raw_events_restore_losslessly(tmp_path):
    root=tmp_path/'scenario';root.mkdir();raw=b'{"event":1}\n{"partial":'
    (root/'events.jsonl').write_bytes(raw)
    result=module.archive(tmp_path)
    assert result['members'][0]['path']=='scenario/events.jsonl'
    with tarfile.open(tmp_path/result['archive']) as handle:assert handle.extractfile('scenario/events.jsonl').read()==raw
    with pytest.raises(FileExistsError):module.archive(tmp_path)
