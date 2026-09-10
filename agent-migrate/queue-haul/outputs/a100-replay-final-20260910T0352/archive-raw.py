"""Archive every plain raw telemetry file and verify lossless restoration."""
import argparse
import gzip
import hashlib
import json
import tarfile
from pathlib import Path


def archive(root):
    target=root/'raw-telemetry.tar.gz'
    if target.exists():raise FileExistsError(target)
    paths=sorted(p for p in root.rglob('*') if p.is_file() and p.suffix in ('.jsonl','.csv','.log','.raw') and '__pycache__' not in p.parts)
    members=[{'path':str(p.relative_to(root)),'bytes':p.stat().st_size,'sha256':hashlib.sha256(p.read_bytes()).hexdigest()} for p in paths]
    with target.open('wb') as sink,gzip.GzipFile(filename='',mode='wb',fileobj=sink,mtime=0) as compressed,tarfile.open(fileobj=compressed,mode='w') as handle:
        for path in paths:
            info=handle.gettarinfo(str(path),str(path.relative_to(root)));info.mtime=info.uid=info.gid=0;info.uname=info.gname='';info.mode=0o644
            with path.open('rb') as source:handle.addfile(info,source)
    with tarfile.open(target) as handle:
        if handle.getnames()!=[r['path'] for r in members]:raise ValueError('archive member list changed')
        for row in members:
            data=handle.extractfile(row['path']).read()
            if len(data)!=row['bytes'] or hashlib.sha256(data).hexdigest()!=row['sha256']:raise ValueError('archive differs from collecting-instance raw bytes')
    result={'archive':target.name,'bytes':target.stat().st_size,'sha256':hashlib.sha256(target.read_bytes()).hexdigest(),
        'members':members,'verification':'Every member decompressed and checked against original collecting-instance byte length and SHA-256; originals retained.',
        'scope':'All plain JSONL/CSV/log/raw files; already-compressed raw Prometheus traces and structured JSON evidence remain separate tracked artifacts.',
        'restore_command':f'tar -xzf {target} -C {root}'}
    (root/'raw-telemetry-archive.json').write_text(json.dumps(result,indent=2)+'\n')
    return result


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('root',type=Path);args=parser.parse_args()
    result=archive(args.root);print(json.dumps({'members':len(result['members']),'archive_bytes':result['bytes'],'sha256':result['sha256']}))
