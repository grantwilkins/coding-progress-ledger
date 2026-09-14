"""Read-only NVML power telemetry with explicit sensor and query timestamps."""
from __future__ import annotations

import argparse
import json
import threading
import time
from pathlib import Path


class NvmlPowerSampler:
    def __init__(self, path: Path, interval_s: float = .1, device: int = 0):
        if interval_s <= 0 or device < 0:
            raise ValueError("invalid power sampling interval or device")
        self.path, self.interval_s, self.device = Path(path), interval_s, device
        self.stop = threading.Event()
        self.ready = threading.Event()
        self.error = None
        self.thread = threading.Thread(target=self._run, daemon=True)

    def start(self):
        self.thread.start()
        if not self.ready.wait(10):
            self.close()
            raise TimeoutError("NVML sampler produced no initial sample")
        if self.error:
            self.close()

    @staticmethod
    def sample(nvml, handle):
        row = {"query_start_monotonic_ns": time.monotonic_ns(),
               "query_start_wall_ns": time.time_ns()}

        def read(call):
            try:
                return {"status": "ok", "value": call()}
            except nvml.NVMLError_NotSupported:
                return {"status": "not_supported", "value": None}

        uuid = nvml.nvmlDeviceGetUUID(handle)
        row["gpu_uuid"] = uuid.decode() if isinstance(uuid, bytes) else uuid
        for name, constant in (("average_power_mw", "NVML_FI_DEV_POWER_AVERAGE"),
                               ("instantaneous_power_mw", "NVML_FI_DEV_POWER_INSTANT")):
            field_id = getattr(nvml, constant, None)
            if field_id is None:
                row[name] = {"status": "binding_missing", "value": None}
                continue
            field = nvml.nvmlDeviceGetFieldValues(handle, [field_id])[0]
            value = {"field_id": field_id, "status_code": field.nvmlReturn,
                     "timestamp_us": field.timestamp, "latency_us": field.latencyUsec,
                     "value": None}
            if field.nvmlReturn == nvml.NVML_ERROR_NOT_SUPPORTED:
                value["status"] = "not_supported"
            elif field.nvmlReturn != nvml.NVML_SUCCESS:
                raise nvml.NVMLError(field.nvmlReturn)
            else:
                if field.valueType != nvml.NVML_VALUE_TYPE_UNSIGNED_INT:
                    raise RuntimeError("unexpected NVML power field type")
                value.update(status="ok", value=field.value.uiVal)
            row[name] = value
        for name, call in {
            "total_energy_mj": lambda: nvml.nvmlDeviceGetTotalEnergyConsumption(handle),
            "enforced_power_limit_mw": lambda: nvml.nvmlDeviceGetEnforcedPowerLimit(handle),
            "sm_clock_mhz": lambda: nvml.nvmlDeviceGetClockInfo(handle, nvml.NVML_CLOCK_SM),
            "memory_clock_mhz": lambda: nvml.nvmlDeviceGetClockInfo(handle, nvml.NVML_CLOCK_MEM),
            "utilization_gpu_pct": lambda: nvml.nvmlDeviceGetUtilizationRates(handle).gpu,
            "framebuffer_used_bytes": lambda: nvml.nvmlDeviceGetMemoryInfo(handle).used,
        }.items():
            row[name] = read(call)
        row.update(query_end_monotonic_ns=time.monotonic_ns(),
                   query_end_wall_ns=time.time_ns())
        return row

    def _run(self):
        try:
            import pynvml
            pynvml.nvmlInit()
            try:
                handle = pynvml.nvmlDeviceGetHandleByIndex(self.device)
                with self.path.open("x") as output:
                    while not self.stop.is_set():
                        output.write(json.dumps(self.sample(pynvml, handle)) + "\n")
                        output.flush()
                        self.ready.set()
                        self.stop.wait(self.interval_s)
            finally:
                pynvml.nvmlShutdown()
        except BaseException as exc:
            self.error = exc
            self.ready.set()
            self.stop.set()

    def close(self):
        self.stop.set()
        self.thread.join(10)
        if self.thread.is_alive():
            raise TimeoutError("NVML power sampler did not stop")
        if self.error:
            raise RuntimeError("NVML power sampler failed") from self.error


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--interval-s", type=float, default=.1)
    parser.add_argument("--seconds", type=float, required=True)
    parser.add_argument("--device", type=int, default=0)
    args = parser.parse_args()
    if args.seconds <= 0:
        parser.error("seconds must be positive")
    sampler = NvmlPowerSampler(args.out, args.interval_s, args.device)
    sampler.start()
    try:
        if sampler.stop.wait(args.seconds):
            raise RuntimeError("power sampling interrupted")
    finally:
        sampler.close()


if __name__ == "__main__":
    main()
