"""Map every final artifact to standalone storage or a verified archive member."""
import argparse
import hashlib
import json
from pathlib import Path


def manifest(root):
    archive=json.loads((root/'raw-telemetry-archive.json').read_text())
    members={r['path']:r for r in archive['members']};rows=[]
    for path in sorted(root.rglob('*')):
        if not path.is_file() or '__pycache__' in path.parts or path.name=='artifact-sha256.json':continue
        name=str(path.relative_to(root));digest=hashlib.sha256(path.read_bytes()).hexdigest()
        if name in members:
            if digest!=members[name]['sha256'] or path.stat().st_size!=members[name]['bytes']:raise ValueError('raw artifact changed after archival: '+name)
            storage={'file':archive['archive'],'member':name}
        else:
            if path.suffix in ('.jsonl','.csv','.log','.raw'):raise ValueError('raw artifact missing from archive: '+name)
            storage={'file':name}
        rows.append({'path':name,'bytes':path.stat().st_size,'sha256':digest,'storage':storage})
    if not set(members)<={r['path'] for r in rows}:raise ValueError('archived original artifact is missing')
    value={'artifacts':rows,'archived_raw_files':len(members),'standalone_files':len(rows)-len(members),
        'scope':'Every final run file except this self-referential manifest and Python bytecode caches; raw paths resolve explicitly to losslessly verified compressed archive members.',
        'restore_command':archive['restore_command'],'coverage_gaps':[]}
    (root/'artifact-sha256.json').write_text(json.dumps(value,indent=2)+'\n');return value


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('root',type=Path);args=parser.parse_args()
    result=manifest(args.root);print(json.dumps({k:result[k] for k in ('archived_raw_files','standalone_files','coverage_gaps')}))
