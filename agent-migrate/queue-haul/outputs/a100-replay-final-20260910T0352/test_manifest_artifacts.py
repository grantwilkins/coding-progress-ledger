import importlib.util
import json
from pathlib import Path

import pytest


def test_manifest_rejects_raw_missing_from_compressed_archive(tmp_path):
    spec=importlib.util.spec_from_file_location('manifest_artifacts',Path(__file__).with_name('manifest-artifacts.py'))
    module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
    (tmp_path/'raw-telemetry-archive.json').write_text(json.dumps({'members':[],'archive':'raw.tar.gz','restore_command':'tar'}))
    (tmp_path/'events.jsonl').write_text('{}\n')
    with pytest.raises(ValueError,match='missing from archive'):module.manifest(tmp_path)
