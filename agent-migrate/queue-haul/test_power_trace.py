from types import SimpleNamespace

import pytest

from power_trace import NvmlPowerSampler


class Unsupported(Exception):
    pass


class Nvml:
    NVMLError_NotSupported = Unsupported
    NVMLError = RuntimeError
    NVML_SUCCESS = 0
    NVML_ERROR_NOT_SUPPORTED = 3
    NVML_VALUE_TYPE_UNSIGNED_INT = 1
    NVML_FI_DEV_POWER_AVERAGE = 185
    NVML_FI_DEV_POWER_INSTANT = 186
    NVML_CLOCK_SM = 0
    NVML_CLOCK_MEM = 1
    status = 0

    def nvmlDeviceGetUUID(self, handle): return b"GPU-test"
    def nvmlDeviceGetFieldValues(self, handle, fields):
        return [SimpleNamespace(nvmlReturn=self.status, timestamp=100, latencyUsec=2,
                                valueType=1, value=SimpleNamespace(uiVal=61000))]
    def nvmlDeviceGetTotalEnergyConsumption(self, handle): return 700000
    def nvmlDeviceGetEnforcedPowerLimit(self, handle): return 400000
    def nvmlDeviceGetClockInfo(self, handle, clock): return 1500
    def nvmlDeviceGetUtilizationRates(self, handle): return SimpleNamespace(gpu=20)
    def nvmlDeviceGetMemoryInfo(self, handle): return SimpleNamespace(used=1024)


def test_sensor_units_and_timestamps():
    row = NvmlPowerSampler.sample(Nvml(), 0)
    assert row["gpu_uuid"] == "GPU-test"
    assert row["average_power_mw"]["value"] == 61000
    assert row["instantaneous_power_mw"]["timestamp_us"] == 100
    assert row["total_energy_mj"]["value"] == 700000
    assert row["framebuffer_used_bytes"]["value"] == 1024
    assert row["query_start_monotonic_ns"] <= row["query_end_monotonic_ns"]
    assert row["query_start_wall_ns"] <= row["query_end_wall_ns"]


def test_unsupported_is_explicit_and_other_errors_fail():
    nvml = Nvml()
    nvml.status = 3
    def unsupported(handle): raise Unsupported()
    nvml.nvmlDeviceGetTotalEnergyConsumption = unsupported
    row = NvmlPowerSampler.sample(nvml, 0)
    assert row["average_power_mw"]["value"] is None
    assert row["total_energy_mj"] == {"status": "not_supported", "value": None}
    nvml.status = 4
    with pytest.raises(RuntimeError):
        NvmlPowerSampler.sample(nvml, 0)


def test_worker_error_is_not_suppressed(tmp_path):
    sampler = NvmlPowerSampler(tmp_path / "power.jsonl")
    sampler.thread = __import__('threading').Thread(target=lambda: None)
    sampler.thread.start()
    sampler.error = OSError("sensor lost")
    with pytest.raises(RuntimeError, match="sampler failed"):
        sampler.close()


@pytest.mark.parametrize("interval,device", [(0, 0), (-1, 0), (.1, -1)])
def test_invalid_configuration(tmp_path, interval, device):
    with pytest.raises(ValueError):
        NvmlPowerSampler(tmp_path / "power.jsonl", interval, device)
