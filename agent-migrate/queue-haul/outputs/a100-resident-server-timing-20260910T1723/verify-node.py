import hashlib, json, os, platform, subprocess, sys, time
from importlib.metadata import version
from pathlib import Path

out, reference = map(Path, sys.argv[1:])
out.mkdir(parents=True, exist_ok=True)
model = Path('/datadrive/hub/models--openai--gpt-oss-20b/snapshots/6cee5e81ee83917806bbde320786a8fb61efebee')
expected = json.loads(reference.read_text())
required=set(json.loads((model/'model.safetensors.index.json').read_text())['weight_map'].values()) | {'model.safetensors.index.json','config.json','tokenizer.json','tokenizer_config.json','chat_template.jinja','generation_config.json'}
assert required <= expected.keys() and all((model/name).is_file() for name in required), 'missing frozen serving weights or tokenizer'
missing=[name for name in expected if not (model/name).is_file()]
assert all(name.startswith('original/') for name in missing), 'missing non-original reference file'
actual = {name: {'sha256': hashlib.file_digest((model/name).open('rb'), 'sha256').hexdigest(), 'bytes': (model/name).stat().st_size} for name in expected if name not in missing}
(out/'model-sha256.json').write_text(json.dumps(actual, indent=2)+'\n')
(out/'model-coverage.json').write_text(json.dumps({'required_serving_files':sorted(required),'missing_unused_original_files':missing},indent=2)+'\n')
assert all(value == expected[name] for name,value in actual.items()), 'model/tokenizer differs from frozen reference'
import torch
identity = {'hostname': platform.node(), 'pid': os.getpid(), 'boot_id': Path('/proc/sys/kernel/random/boot_id').read_text().strip(),
    'time_namespace': os.readlink('/proc/self/ns/time'), 'wall_ns': time.time_ns(), 'monotonic_ns': time.monotonic_ns(),
    'gpu': subprocess.check_output(['nvidia-smi', '--query-gpu=name,uuid,memory.total,power.limit,mig.mode.current,driver_version', '--format=csv'], text=True),
    'active_compute': subprocess.check_output(['nvidia-smi', '--query-compute-apps=pid,process_name,used_memory', '--format=csv'], text=True)}
(out/'identity.json').write_text(json.dumps(identity, indent=2)+'\n')
gpu = identity['gpu'].splitlines()[1].split(', ')
assert len(identity['gpu'].splitlines()) == 2 and gpu[0]=='NVIDIA A100 80GB PCIe'
assert gpu[2]=='81920 MiB' and gpu[3]=='300.00 W' and gpu[4]=='Disabled', 'full idle reference GPU at 300W required'
assert len(identity['active_compute'].splitlines()) == 1, 'GPU already owned by another process'
runtime = {key: version(key) for key in ('vllm','lmcache','torch','transformers')}
runtime.update(python=sys.version, torch_build=torch.__version__, cuda_build=torch.version.cuda)
(out/'runtime.json').write_text(json.dumps(runtime, indent=2)+'\n')
import vllm._C, lmcache.c_ops
assert runtime['vllm']=='0.22.0' and runtime['lmcache']=='0.5.1'
assert torch.cuda.get_device_name()=='NVIDIA A100 80GB PCIe'
(out/'packages.txt').write_text(subprocess.check_output(['/datadrive/queue-haul-tools/bin/uv','pip','freeze','--python',sys.executable], text=True))
print(json.dumps({'model_verified': True, 'identity': identity, 'runtime': runtime}))
