"""
Claim:
setup.sh installs the pinned native Sherlock-equivalent stack and model in /datadrive.

Plausible wrong implementations:
- Install a different vLLM, LMCache, CUDA, Python, or Rust version.
- Download a moving model revision or duplicate unused weight formats.
- Configure a home-disk Hugging Face cache or append duplicate shell settings.
"""

import subprocess
from pathlib import Path


SCRIPT = Path(__file__).parents[1] / "setup.sh"


def test_setup_script_has_valid_bash_and_pinned_runtime_contract():
    subprocess.run(["bash", "-n", SCRIPT], check=True)
    text = SCRIPT.read_text()

    for value in ("0.11.32", "3.12", "1.96.0", "vllm==0.22.0", "lmcache==0.5.1", "cu129"):
        assert value in text
    assert "6cee5e81ee83917806bbde320786a8fb61efebee" in text
    assert "--exclude 'original/*' 'metal/*'" in text
    assert "HF_HOME=/datadrive" in text
    assert "HF_HUB_CACHE=/datadrive/hub" in text
    assert "QH_CACHE_ROOT=/datadrive/queue-haul-cache" in text
    assert "QH_RUNTIME=native" in text


def test_setup_replaces_one_managed_bashrc_block():
    text = SCRIPT.read_text()

    assert 'profile="$HOME/.bashrc"' in text
    assert "sed '/# BEGIN QUEUE-HAUL/,/# END QUEUE-HAUL/d'" in text
    assert text.count("# BEGIN QUEUE-HAUL") == 2
    assert text.count("# END QUEUE-HAUL") == 2
