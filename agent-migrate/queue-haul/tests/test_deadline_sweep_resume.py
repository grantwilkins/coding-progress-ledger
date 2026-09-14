import importlib.util
from pathlib import Path
from types import SimpleNamespace

import pytest
import network_campaign as network


@pytest.fixture
def resume(monkeypatch):
    path = Path(__file__).resolve().parents[1] / 'outputs/h100-controls-20260914/resume_deadline_sweep.py'
    spec = importlib.util.spec_from_file_location('deadline_resume', path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    node = SimpleNamespace(id='germany', repo_root='/datadrive/qh0912')
    monkeypatch.setattr(network.Cluster, 'load', lambda _: SimpleNamespace(destinations=[node]))
    monkeypatch.setattr(network, 'ssh_command', lambda node, key, command: command)
    monkeypatch.setattr(module, 'save', lambda state: None)
    return module


def test_wait_retains_failed_job_until_destination_returns(resume, monkeypatch):
    responses = iter((255, 0))
    commands, sleeps = [], []
    def run(command, **kwargs):
        commands.append(command)
        return SimpleNamespace(returncode=next(responses))
    monkeypatch.setattr(resume.subprocess, 'run', run)
    monkeypatch.setattr(resume.time, 'sleep', sleeps.append)
    state = {'jobs': {'gemma_repeat_zero': {'returncode': 1}}}
    resume.wait_for_hosts(state)
    assert sleeps == [30]
    assert commands == [['timeout', '20', 'test', '-x', '/datadrive/qh0912/.venv/bin/python']] * 2
    assert 'waiting_for_hosts' not in state
    assert state['jobs']['gemma_repeat_zero']['returncode'] == 1


def test_unavailable_destination_has_bounded_wait(resume, monkeypatch):
    times = iter((0, 21601))
    monkeypatch.setattr(resume.time, 'monotonic', lambda: next(times))
    monkeypatch.setattr(resume.subprocess, 'run', lambda *args, **kwargs: SimpleNamespace(returncode=255))
    state = {'jobs': {}}
    with pytest.raises(TimeoutError, match='six hours'):
        resume.wait_for_hosts(state)
    assert state['status'] == 'waiting_for_hosts'
    assert state['waiting_for_hosts'] == ['germany']
