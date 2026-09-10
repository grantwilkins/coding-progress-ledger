import csv
import json
import os
from pathlib import Path


def test_germany_shutdown_snapshot_requires_all_destination_and_no_source_closures(tmp_path):
    root=Path(__file__).parent
    plan=json.loads((root/'cleanup-plan.json').read_text())
    commands=plan['commands'];names=[x['operation'] for x in commands]
    name='snapshot Germany-only pool closures before stopping Sweden'
    assert names.index('stop Germany supervisor') < names.index(name) < names.index('stop Sweden supervisor')
    code=next(x['argv'][2] for x in commands if x['operation']==name).replace(str(root.resolve()),str(tmp_path))
    (tmp_path/'stack').mkdir();(tmp_path/'attribution-proof.json').write_text(json.dumps({'source_connection_ids':['s'],'destination_connection_ids':['d']}))
    (tmp_path/'runtime-launch.json').write_text(json.dumps({'pid':os.getpid()}))
    (tmp_path/'stack/proxy_connections.csv').write_text('connection_id,end_ns\nd,123\n')
    exec(compile(code,'cleanup-snapshot','exec'),{})
    assert json.loads((tmp_path/'germany-stop-origin-crosscheck.json').read_text())['verified']
    (tmp_path/'stack/proxy_connections.csv').write_text('connection_id,end_ns\nd,123\ns,124\n')
    import pytest
    with pytest.raises(AssertionError):exec(compile(code,'cleanup-snapshot','exec'),{})
    assert not json.loads((tmp_path/'germany-stop-origin-crosscheck.json').read_text())['verified']
