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

    for value in ("0.11.32", "3.12", "1.96.0", "QH_VLLM_VERSION",
                  "0.22.0", "0.24.0", "lmcache==0.5.1", "cu129"):
        assert value in text
    for model, revision in (
        ("openai/gpt-oss-20b", "6cee5e81ee83917806bbde320786a8fb61efebee"),
        ("Qwen/Qwen3.8-27B", "1d4bf0f2ff6012fd82039f2fa52739d0dd7c60c0"),
        ("google/gemma-4-26B-A4B-it", "4d7ae4984b7db7de8f8457170b3f1a419ee76d52"),
    ):
        assert model in text and revision in text
    assert "--exclude 'original/*' --exclude 'metal/*'" in text
    assert "model-*.safetensors" not in text
    assert "HF_HOME=/datadrive" in text
    assert "HF_HUB_CACHE=/datadrive/hub" in text
    assert "QH_CACHE_ROOT=/datadrive/queue-haul-cache" in text
    assert "QH_RUNTIME=native" in text
    assert "QH_NATIVE_RUNTIME_VERSIONS" in text
    assert 'VIRTUAL_ENV="$repo_dir/.venv" .venv/bin/maturin develop --release' in text
    assert "greedy_compact" in text and "greedy_csc" in text
    assert "command -v dnf" in text
    assert "valkey chrony iperf3" in text
    assert "refclock PHC /dev/ptp_hyperv poll 3 dpoll -2 offset 0 stratum 2" in text
    assert "systemctl enable --now chronyd" in text
    assert "chronyc waitsync 60 0.002" in text
    assert "apt-get" not in text


def test_setup_replaces_one_managed_bashrc_block():
    text = SCRIPT.read_text()

    assert 'profile="$HOME/.bashrc"' in text
    assert "sed '/# BEGIN QUEUE-HAUL/,/# END QUEUE-HAUL/d'" in text
    assert text.count("# BEGIN QUEUE-HAUL") == 2
    assert text.count("# END QUEUE-HAUL") == 2
