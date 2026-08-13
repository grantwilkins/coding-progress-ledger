"""Versioned inputs for measured machine behavior and sampled workloads."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np


PROFILE_SCHEMA = "queue-haul-model-profile-v4"
PHASE_PROFILE_SCHEMA = "queue-haul-model-profile-v5"
WORKLOAD_SCHEMA = "queue-haul-workload-profile-v2"
SOURCE_SECTIONS = ("power", "service", "capacity", "replay", "kv_transfer", "transitions")
ACTION_POWER = {"replay", "kv_transfer", "replay_on_request", "catch_up", "sleep", "off"}
WORKLOAD_STATES = {"active", "cold"}


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
        if ell < self.ell[0] - 1e-12 or ell > self.ell[-1] + 1e-12:
            raise ValueError(f"ell {ell} outside [{self.ell[0]}, {self.ell[-1]}]")
        return float(np.interp(np.clip(ell, self.ell[0], self.ell[-1]), self.ell, self.watts))


@dataclass(frozen=True)
class PhasePower:
    """Identifiable phase-aware source power calibration."""

    p0_w: float
    delta_w: float
    a_s_per_prefill_token: float
    b_s_per_decode_token: float
    valid_hull: tuple[tuple[float, float], ...]
    grouped_cv_rmse_w: float
    within_5w_fraction: float
    bootstrap: tuple[tuple[float, float, float, float], ...]
    provenance_sha256: str
    measured_power_curve: tuple[tuple[float, float], ...] = ()

    @classmethod
    def parse(cls, raw: dict) -> "PhasePower":
        required = {
            "p0_w", "delta_w", "a_s_per_prefill_token",
            "b_s_per_decode_token", "valid_hull", "grouped_cv_rmse_w",
            "within_5w_fraction", "bootstrap", "provenance_sha256",
        }
        if set(raw) not in (required, required | {"measured_power_curve"}):
            raise ValueError(f"phase_power fields must be {sorted(required)} plus optional measured_power_curve")
        hull = tuple(tuple(map(float, point)) for point in raw["valid_hull"])
        bootstrap = tuple(tuple(map(float, row)) for row in raw["bootstrap"])
        value = cls(
            float(raw["p0_w"]), float(raw["delta_w"]),
            float(raw["a_s_per_prefill_token"]),
            float(raw["b_s_per_decode_token"]), hull,
            float(raw["grouped_cv_rmse_w"]),
            float(raw["within_5w_fraction"]), bootstrap,
            str(raw["provenance_sha256"]),
            tuple(tuple(map(float, point)) for point in raw.get("measured_power_curve", ())),
        )
        curve = np.asarray(value.measured_power_curve, float)
        if value.p0_w < 0 or value.delta_w <= 0 \
                or min(value.a_s_per_prefill_token,
                       value.b_s_per_decode_token) <= 0 \
                or len(hull) < 3 or any(len(point) != 2 or min(point) < 0
                                       for point in hull) \
                or value.grouped_cv_rmse_w < 0 \
                or not 0 <= value.within_5w_fraction <= 1 \
                or any(len(row) != 4 or min(row) <= 0 for row in bootstrap) \
                or len(value.provenance_sha256) != 64 \
                or value.measured_power_curve and (curve.ndim != 2 or curve.shape[1] != 2
                    or curve[0, 0] != 0 or np.any(np.diff(curve[:, 0]) <= 0)
                    or np.any(np.diff(curve[:, 1]) < 0)):
            raise ValueError("invalid phase-aware power calibration")
        return value

    def load(self, prefill_tps: float, decode_tps: float) -> float:
        if min(prefill_tps, decode_tps) < 0:
            raise ValueError("phase rates must be nonnegative")
        return (self.a_s_per_prefill_token * prefill_tps
                + self.b_s_per_decode_token * decode_tps)

    def power(self, load: float) -> float:
        if load < -1e-12:
            raise ValueError("power load must be nonnegative")
        z = max(0.0, load)
        if self.measured_power_curve:
            curve = np.asarray(self.measured_power_curve)
            if z > curve[-1, 0] + 1e-12:
                raise ValueError(f"power load {z} outside measured curve")
            return float(np.interp(z, curve[:, 0], curve[:, 1]))
        return self.p0_w + self.delta_w * z / (1 + z)

    def contains(self, prefill_tps: float, decode_tps: float) -> bool:
        point = np.asarray((prefill_tps, decode_tps), float)
        hull = np.asarray(self.valid_hull, float)
        edges, offsets = np.roll(hull, -1, axis=0) - hull, point - hull
        crosses = edges[:, 0] * offsets[:, 1] - edges[:, 1] * offsets[:, 0]
        return bool(np.all(crosses >= -1e-8) or np.all(crosses <= 1e-8))


@dataclass(frozen=True)
class ActionPower:
    concurrency: np.ndarray
    source_w: np.ndarray
    destination_w: np.ndarray

    @classmethod
    def parse(cls, raw: dict) -> "ActionPower":
        points = sorted((int(key), *map(float, value)) for key, value in raw.items())
        if not points or any(len(value) != 2 for value in raw.values()):
            raise ValueError("action power requires concurrency: [source_w, destination_w]")
        concurrency, source, destination = map(np.asarray, zip(*points))
        if concurrency[0] != 1 or np.any(np.diff(concurrency) <= 0) \
                or min(*source, *destination) < 0 \
                or np.any(np.diff(source) < 0) or np.any(np.diff(destination) < 0):
            raise ValueError("action power must start at concurrency one and be nondecreasing")
        return cls(concurrency, source, destination)

    def power(self, concurrency: int, local: bool) -> float:
        if not 1 <= concurrency <= self.concurrency[-1]:
            raise ValueError(f"unsupported concurrency {concurrency}")
        return float(np.interp(concurrency, self.concurrency,
                               self.source_w if local else self.destination_w))


@dataclass(frozen=True)
class KVTransfer:
    """Transport-agnostic sealed-block movement and endpoint timing."""

    block_tokens: int
    block_bytes: int
    setup_s: float
    destination_bytes_per_s: float
    initial_completion_s: float
    catch_up_fixed_s: float
    tail_replay_tps: float

    @classmethod
    def parse(cls, raw: dict) -> "KVTransfer":
        value = cls(
            int(raw["block_tokens"]), int(raw["block_bytes"]),
            float(raw["setup_s"]), float(raw["destination_bytes_per_s"]),
            float(raw["initial_completion_s"]), float(raw["catch_up_fixed_s"]),
            float(raw["tail_replay_tps"]),
        )
        if value.block_tokens < 1 or value.block_bytes < 1 \
                or value.block_bytes % value.block_tokens \
                or min(value.setup_s, value.initial_completion_s,
                       value.catch_up_fixed_s) < 0 \
                or min(value.destination_bytes_per_s, value.tail_replay_tps) <= 0:
            raise ValueError("invalid KV transfer parameters")
        return value

    def sealed_blocks(self, tokens: int) -> int:
        return max(0, int(tokens)) // self.block_tokens

    def sealed_bytes(self, tokens: int) -> int:
        return self.sealed_blocks(tokens) * self.block_bytes

    def tail_tokens(self, tokens: int) -> int:
        return max(0, int(tokens)) % self.block_tokens


@dataclass(frozen=True)
class ProfileCase:
    case_id: str
    F: float
    G: float
    power_curve: PowerCurve
    prefill: RateCurve
    decode: RateCurve
    replay: RateCurve
    replay_completion_s: float
    kv_transfer: KVTransfer
    switch_s: float
    sleep_power_delta_w: float
    sleep_s: float
    shutdown_s: float | None
    action_power_w: dict[str, ActionPower]
    phase_power: PhasePower | None = None

    @classmethod
    def parse(cls, case_id: str, raw: dict) -> "ProfileCase":
        value = cls(
            case_id, float(raw["F"]), float(raw["G"]),
            PowerCurve.parse(raw["power_curve"]),
            RateCurve.parse(raw["prefill_tps"]), RateCurve.parse(raw["decode_tps"]),
            RateCurve.parse(raw["replay_tps"]), float(raw["replay_completion_s"]),
            KVTransfer.parse(raw["kv_transfer"]),
            float(raw["switch_s"]), float(raw["sleep_power_delta_w"]),
            float(raw["sleep_s"]),
            None if raw["shutdown_s"] is None else float(raw["shutdown_s"]),
            {str(k): ActionPower.parse(v) for k, v in raw["action_power_w"].items()},
            PhasePower.parse(raw["phase_power"]) if "phase_power" in raw else None,
        )
        if set(value.action_power_w) != ACTION_POWER:
            raise ValueError(f"action_power_w fields must be {sorted(ACTION_POWER)}")
        if value.F <= 0 or value.G <= 0 or min(
            value.replay_completion_s, value.switch_s, value.sleep_s,
        ) < 0:
            raise ValueError("rates, times, and power must be nonnegative; F and G must be positive")
        if value.shutdown_s is not None and value.shutdown_s < 0:
            raise ValueError("shutdown time must be nonnegative")
        if value.power(0) + value.sleep_power_delta_w < 0:
            raise ValueError("sleep power must be nonnegative")
        return value

    def service_load(self, prefill_tps: float, decode_tps: float) -> float:
        return prefill_tps / self.F + decode_tps / self.G

    def power_load(self, prefill_tps: float, decode_tps: float) -> float:
        return (self.phase_power.load(prefill_tps, decode_tps)
                if self.phase_power else self.service_load(prefill_tps, decode_tps))

    def power(self, load: float) -> float:
        return (self.phase_power.power(load)
                if self.phase_power else self.power_curve.power(load))


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
    max_power_load: float
    kv_capacity_tokens: int
    max_destination_replays: int
    max_destination_kv_streams: int
    sources: dict[str, Source]
    cases: dict[str, ProfileCase]

    @classmethod
    def load(cls, path: str | Path) -> "ModelProfile":
        raw = json.loads(Path(path).read_text())
        if raw.get("schema") not in {PROFILE_SCHEMA, PHASE_PROFILE_SCHEMA}:
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
            float(raw.get("max_power_load", raw["max_ell"])),
            int(raw["kv_capacity_tokens"]), int(raw["max_destination_replays"]),
            int(raw["max_destination_kv_streams"]), sources, cases,
        )
        if value.status not in {"fitted", "validated", "estimated"}:
            raise ValueError(f"unknown profile status {value.status!r}")
        if value.power_scope not in {"gpu", "server"}:
            raise ValueError(f"unknown power scope {value.power_scope!r}")
        if not value.profile_id or value.tensor_parallel < 1 or value.gpus_per_node < 1 \
                or min(value.power_window_s, value.max_ell, value.max_power_load,
                       value.kv_capacity_tokens) <= 0:
            raise ValueError("invalid profile identity or limits")
        if min(value.max_destination_replays, value.max_destination_kv_streams) < 1:
            raise ValueError("destination concurrency limits must be positive")
        for case in cases.values():
            if case.phase_power is None and value.max_power_load > case.power_curve.ell[-1]:
                raise ValueError("max_power_load exceeds the calibrated power curve")
            if raw["schema"] == PHASE_PROFILE_SCHEMA and case.phase_power is None:
                raise ValueError("v5 profiles require phase-aware power")
            if raw["status"] == "validated" and case.phase_power is None:
                raise ValueError("validated profiles require phase-aware power")
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
    prompt_tokens: int
    output_tokens: int
    request_gap_s: float
    tool_delay_s: float
    log_bytes: int
    log_location: str

    @classmethod
    def parse(cls, raw: dict) -> "WorkloadRecord":
        value = cls(
            raw["job_type"], "cold" if raw["state"] == "idle" else raw["state"],
            int(raw["context_tokens"]),
            int(raw["prompt_tokens"]), int(raw["output_tokens"]), float(raw["request_gap_s"]),
            float(raw["tool_delay_s"]), int(raw["log_bytes"]), raw["log_location"],
        )
        if value.state not in WORKLOAD_STATES or value.context_tokens < 1 \
                or value.prompt_tokens < 1 or value.output_tokens < 0 \
                or min(value.request_gap_s, value.tool_delay_s) < 0 \
                or value.log_bytes < 1 or value.log_location != "source_dc":
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
