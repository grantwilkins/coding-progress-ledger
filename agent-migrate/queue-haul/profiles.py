"""Versioned inputs for measured machine behavior and sampled workloads."""

from __future__ import annotations

import json
import math
from dataclasses import dataclass
from pathlib import Path

import numpy as np


PROFILE_SCHEMA = "queue-haul-model-profile-v1"
WORKLOAD_SCHEMA = "queue-haul-workload-profile-v1"
SOURCE_SECTIONS = ("power", "service", "replay", "kv_transfer", "transitions")
ACTION_POWER = {"replay", "kv_transfer", "replay_on_request", "catch_up", "sleep", "off"}


@dataclass(frozen=True)
class Source:
    kind: str
    reference: str
    valid_range: tuple[float, float]
    relative_error: float

    @classmethod
    def parse(cls, raw: dict) -> "Source":
        required = {"kind", "reference", "valid_range", "relative_error"}
        if set(raw) != required:
            raise ValueError(f"source fields must be {sorted(required)}")
        lo, hi = map(float, raw["valid_range"])
        error = float(raw["relative_error"])
        if raw["kind"] not in {"calculated", "measured", "published", "assumed"}:
            raise ValueError(f"unknown source kind {raw['kind']!r}")
        if not raw["reference"] or lo < 0 or hi <= lo or error < 0:
            raise ValueError("invalid source reference, range, or error")
        return cls(raw["kind"], raw["reference"], (lo, hi), error)


@dataclass(frozen=True)
class RateCurve:
    by_concurrency: dict[int, tuple[np.ndarray, np.ndarray]]

    @classmethod
    def parse(cls, raw: dict) -> "RateCurve":
        curves = {}
        for key, points in raw.items():
            concurrency = int(key)
            a = np.asarray(points, float)
            if concurrency < 1 or a.ndim != 2 or a.shape[0] < 2 or a.shape[1] != 2:
                raise ValueError("rate curves require two or more [context_tokens, tokens_per_s] points")
            x, y = a.T
            if np.any(np.diff(x) <= 0) or np.any(y <= 0):
                raise ValueError("rate curve contexts and rates must be positive and ordered")
            curves[concurrency] = x, y
        if not curves:
            raise ValueError("rate curve must not be empty")
        return cls(curves)

    def rate(self, context_tokens: float, concurrency: int) -> float:
        if concurrency not in self.by_concurrency:
            raise ValueError(f"unsupported concurrency {concurrency}")
        x, y = self.by_concurrency[concurrency]
        if not x[0] <= context_tokens <= x[-1]:
            raise ValueError(f"context {context_tokens} outside [{x[0]}, {x[-1]}]")
        return float(np.interp(context_tokens, x, y))


@dataclass(frozen=True)
class PowerCurve:
    ell: np.ndarray
    watts: np.ndarray

    @classmethod
    def parse(cls, points) -> "PowerCurve":
        a = np.asarray(points, float)
        if a.ndim != 2 or a.shape[0] < 2 or a.shape[1] != 2:
            raise ValueError("power curve requires two or more [ell, watts] points")
        ell, watts = a.T
        slopes = np.diff(watts) / np.diff(ell)
        if ell[0] != 0 or np.any(np.diff(ell) <= 0) or np.any(np.diff(watts) < 0):
            raise ValueError("power curve must start at ell=0 and be nondecreasing")
        if np.any(np.diff(slopes) > 1e-9):
            raise ValueError("power curve must be concave")
        return cls(ell, watts)

    def power(self, ell: float) -> float:
        if not self.ell[0] <= ell <= self.ell[-1]:
            raise ValueError(f"ell {ell} outside [{self.ell[0]}, {self.ell[-1]}]")
        return float(np.interp(ell, self.ell, self.watts))


@dataclass(frozen=True)
class KVTransfer:
    block_tokens: int
    block_bytes: int
    setup_s: float
    block_processing_s: float
    sync_s: float

    @classmethod
    def parse(cls, raw: dict) -> "KVTransfer":
        value = cls(int(raw["block_tokens"]), int(raw["block_bytes"]),
                    float(raw["setup_s"]), float(raw["block_processing_s"]),
                    float(raw["sync_s"]))
        if value.block_tokens < 1 or value.block_bytes < 1 or min(
            value.setup_s, value.block_processing_s, value.sync_s
        ) < 0:
            raise ValueError("invalid KV transfer parameters")
        return value

    def blocks(self, tokens: int) -> int:
        return max(0, math.ceil(int(tokens) / self.block_tokens))

    def bytes(self, tokens: int) -> int:
        return self.blocks(tokens) * self.block_bytes


@dataclass(frozen=True)
class ProfileCase:
    case_id: str
    F: float
    G: float
    power_curve: PowerCurve
    prefill: RateCurve
    decode: RateCurve
    replay: RateCurve
    kv_transfer: KVTransfer
    switch_s: float
    sleep_power_w: float
    sleep_s: float
    shutdown_s: float
    action_power_w: dict[str, tuple[float, float]]

    @classmethod
    def parse(cls, case_id: str, raw: dict) -> "ProfileCase":
        value = cls(
            case_id, float(raw["F"]), float(raw["G"]),
            PowerCurve.parse(raw["power_curve"]),
            RateCurve.parse(raw["prefill_tps"]), RateCurve.parse(raw["decode_tps"]),
            RateCurve.parse(raw["replay_tps"]), KVTransfer.parse(raw["kv_transfer"]),
            float(raw["switch_s"]), float(raw["sleep_power_w"]),
            float(raw["sleep_s"]), float(raw["shutdown_s"]),
            {str(k): tuple(map(float, v)) for k, v in raw["action_power_w"].items()},
        )
        if set(value.action_power_w) != ACTION_POWER:
            raise ValueError(f"action_power_w fields must be {sorted(ACTION_POWER)}")
        if any(len(v) != 2 for v in value.action_power_w.values()):
            raise ValueError("action power requires [source_w, destination_w]")
        if value.F <= 0 or value.G <= 0 or min(
            value.switch_s, value.sleep_power_w, value.sleep_s, value.shutdown_s,
            *(v for pair in value.action_power_w.values() for v in pair),
        ) < 0:
            raise ValueError("rates, times, and power must be nonnegative; F and G must be positive")
        return value


@dataclass(frozen=True)
class ModelProfile:
    profile_id: str
    status: str
    model: str
    hardware: str
    precision: str
    tensor_parallel: int
    gpus_per_node: int
    power_scope: str
    power_window_s: float
    max_ell: float
    max_parallel_moves: int
    sources: dict[str, Source]
    cases: dict[str, ProfileCase]

    @classmethod
    def load(cls, path: str | Path) -> "ModelProfile":
        raw = json.loads(Path(path).read_text())
        if raw.get("schema") != PROFILE_SCHEMA:
            raise ValueError(f"expected schema {PROFILE_SCHEMA!r}")
        sources = {k: Source.parse(v) for k, v in raw["sources"].items()}
        missing = set(SOURCE_SECTIONS) - set(sources)
        if missing:
            raise ValueError(f"missing sources: {sorted(missing)}")
        cases = {k: ProfileCase.parse(k, v) for k, v in raw["cases"].items()}
        required_cases = {"central"} if raw["status"] != "estimated" else {"central", "faster", "slower"}
        if set(cases) != required_cases:
            raise ValueError(f"profile cases must be {sorted(required_cases)}")
        value = cls(
            raw["profile_id"], raw["status"], raw["model"], raw["hardware"],
            raw["precision"], int(raw["tensor_parallel"]), int(raw["gpus_per_node"]),
            raw["power_scope"], float(raw["power_window_s"]), float(raw["max_ell"]),
            int(raw["max_parallel_moves"]), sources, cases,
        )
        if value.status not in {"fitted", "validated", "estimated"}:
            raise ValueError(f"unknown profile status {value.status!r}")
        if value.power_scope not in {"gpu", "server"}:
            raise ValueError(f"unknown power scope {value.power_scope!r}")
        if not value.profile_id or value.tensor_parallel < 1 or value.gpus_per_node < 1 \
                or value.power_window_s <= 0 or value.max_ell <= 0 or value.max_parallel_moves < 1:
            raise ValueError("invalid profile identity or limits")
        for case in cases.values():
            if value.max_ell > case.power_curve.ell[-1]:
                raise ValueError("max_ell exceeds the calibrated power curve")
        return value

    def case(self, case_id: str = "central") -> ProfileCase:
        try:
            return self.cases[case_id]
        except KeyError as exc:
            raise ValueError(f"unknown profile case {case_id!r}") from exc


@dataclass(frozen=True)
class WorkloadRecord:
    job_type: str
    state: str
    context_tokens: int
    output_tokens: int
    request_gap_s: float
    tool_delay_s: float
    log_bytes: int
    log_external: bool

    @classmethod
    def parse(cls, raw: dict) -> "WorkloadRecord":
        value = cls(
            raw["job_type"], raw["state"], int(raw["context_tokens"]),
            int(raw["output_tokens"]), float(raw["request_gap_s"]),
            float(raw["tool_delay_s"]), int(raw["log_bytes"]), bool(raw["log_external"]),
        )
        if value.state not in {"active", "idle", "cold"} or value.context_tokens < 1 \
                or value.output_tokens < 0 or min(value.request_gap_s, value.tool_delay_s) < 0 \
                or value.log_bytes < 1:
            raise ValueError("invalid workload record")
        return value


@dataclass(frozen=True)
class WorkloadProfile:
    profile_id: str
    source: Source
    records: tuple[WorkloadRecord, ...]

    @classmethod
    def load(cls, path: str | Path) -> "WorkloadProfile":
        raw = json.loads(Path(path).read_text())
        if raw.get("schema") != WORKLOAD_SCHEMA:
            raise ValueError(f"expected schema {WORKLOAD_SCHEMA!r}")
        records = tuple(WorkloadRecord.parse(r) for r in raw["records"])
        if not raw["profile_id"] or not records:
            raise ValueError("workload profile and records must not be empty")
        return cls(raw["profile_id"], Source.parse(raw["source"]), records)

    def sample(self, n: int, seed: int) -> tuple[WorkloadRecord, ...]:
        if n < 1:
            raise ValueError("sample size must be positive")
        indices = np.random.default_rng(seed).integers(0, len(self.records), n)
        return tuple(self.records[i] for i in indices)
