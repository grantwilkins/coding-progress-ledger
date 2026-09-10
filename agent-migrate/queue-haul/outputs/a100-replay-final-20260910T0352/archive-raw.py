"""Archive every plain raw telemetry file and verify lossless restoration."""
import argparse
import subprocess
import hashlib
import json
import tarfile
from pathlib import Path


def archive(root):
    target=root/'raw-telemetry.tar.zst'
    if target.exists():raise FileExistsError(target)
    paths=sorted(p for p in root.rglob('*') if p.is_file() and p.suffix in ('.jsonl','.csv','.log','.raw') and '__pycache__' not in p.parts)
    members=[{'path':str(p.relative_to(root)),'bytes':p.stat().st_size,'sha256':hashlib.sha256(p.read_bytes()).hexdigest()} for p in paths]
    with target.open('wb') as sink,subprocess.Popen(['zstd','-q','--long=27','-6','-c'],stdin=subprocess.PIPE,stdout=sink) as compressed:
        with tarfile.open(fileobj=compressed.stdin,mode='w|') as handle:
            for path in paths:
                info=handle.gettarinfo(str(path),str(path.relative_to(root)));info.mtime=info.uid=info.gid=0;info.uname=info.gname='';info.mode=0o644
                with path.open('rb') as source:handle.addfile(info,source)
        compressed.stdin.close()
        if compressed.wait():raise RuntimeError('zstd compression failed')
    with subprocess.Popen(['zstd','-q','-d','--long=27','-c',str(target)],stdout=subprocess.PIPE) as restored:
        with tarfile.open(fileobj=restored.stdout,mode='r|') as handle:
            count=0
            for member in handle:
                if count>=len(members) or member.name!=members[count]['path']:raise ValueError('archive member list changed')
                row=members[count];data=handle.extractfile(member).read();count+=1
                if len(data)!=row['bytes'] or hashlib.sha256(data).hexdigest()!=row['sha256']:raise ValueError('archive differs from collecting-instance raw bytes')
            if count!=len(members):raise ValueError('archive member list changed')
        if restored.wait():raise RuntimeError('zstd decompression failed')
    result={'archive':target.name,'bytes':target.stat().st_size,'sha256':hashlib.sha256(target.read_bytes()).hexdigest(),
        'members':members,'verification':'Every member decompressed and checked against original collecting-instance byte length and SHA-256; originals retained.',
        'scope':'All plain JSONL/CSV/log/raw files; already-compressed raw Prometheus traces and structured JSON evidence remain separate tracked artifacts.',
        'compression':'zstd --long=27 -6;128MiB window; canonical tar metadata',
        'zstd_version':subprocess.check_output(['zstd','--version'],text=True).strip(),
        'restore_command':f'zstd -d --long=27 -c {target} | tar -xf - -C {root}'}
    (root/'raw-telemetry-archive.json').write_text(json.dumps(result,indent=2)+'\n')
    return result


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('root',type=Path);args=parser.parse_args()
    result=archive(args.root);print(json.dumps({'members':len(result['members']),'archive_bytes':result['bytes'],'sha256':result['sha256']}))
