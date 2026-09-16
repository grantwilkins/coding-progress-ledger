"""Wait for both campaign hosts, then resume the frozen workload cases once."""
import subprocess
import sys
import time

end = time.monotonic() + 21600
hosts = ('azrsadmin@10.15.0.4', 'azureuser@10.13.0.4')
while True:
    missing = [host for host in hosts if subprocess.run([
        'timeout', '20', 'ssh', '-i', '/home/azureuser/.ssh/azrs',
        '-o', 'BatchMode=yes', '-o', 'ConnectTimeout=10', host,
        'test -x /datadrive/qh0912/.venv/bin/python'],
        stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL).returncode]
    if not missing:
        break
    print('Waiting for ' + ', '.join(missing), flush=True)
    if time.monotonic() >= end:
        raise TimeoutError('campaign hosts unavailable for six hours')
    time.sleep(30)
subprocess.run(sys.argv[1:], check=True, timeout=28800)
