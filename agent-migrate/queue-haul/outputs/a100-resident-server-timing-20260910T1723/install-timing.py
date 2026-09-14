import hashlib, json, os, shutil, sys
from pathlib import Path

source, target=map(Path,sys.argv[1:])
manifest=json.loads((source/'manifest.json').read_text())
assert len(manifest)==6
for name, hashes in manifest.items():
    assert hashlib.sha256((target/'vllm'/name).read_bytes()).hexdigest()==hashes['original_sha256'], name
for path in source.rglob('*.py'):
    dest=target/path.relative_to(source)
    shutil.copyfile(path,str(dest)+'.qh-new')
    os.replace(str(dest)+'.qh-new',dest)
for name,hashes in manifest.items():
    assert hashlib.sha256((target/'vllm'/name).read_bytes()).hexdigest()==hashes['patched_sha256'], name
print('All six pinned timing modules installed and verified.')
