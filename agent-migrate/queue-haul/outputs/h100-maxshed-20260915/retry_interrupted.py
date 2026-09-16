"""Retry the authorized interrupted pack after the current campaign releases GPUs."""
import json
import subprocess
import sys
import time
from pathlib import Path

pid, root = int(sys.argv[1]), Path(sys.argv[2])
deadline = time.monotonic() + 28800
while Path(f'/proc/{pid}/cmdline').exists() and b'model_hardware_drain_campaign.py' in Path(f'/proc/{pid}/cmdline').read_bytes():
    if time.monotonic() >= deadline:
        raise TimeoutError('campaign did not release GPUs within eight hours')
    time.sleep(15)
case = root / 'arms/m0/scenarios/09a28a1d051a0511'
result = json.loads((case / 'attempt-0001/result.json').read_text())
if result['status'] != 'complete':
    assert result['failure_class'] == 'interrupted', result
    archive = root / 'authorized-interruption-retries/m0/09a28a1d051a0511'
    archive.parent.mkdir(parents=True, exist_ok=True)
    assert not archive.exists()
    case.rename(archive)
    (archive / 'retry_authorization.json').write_text(json.dumps({
        'reason': 'User explicitly authorized rerunning interrupted cases',
        'archived_failed_attempt': True, 'replacement_case': str(case),
        'same_frozen_workload_pack': True}, indent=2) + '\n')
subprocess.run(sys.argv[3:], check=True, timeout=28800)
