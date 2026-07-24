"""Measured destination admission inputs."""

from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path

import numpy as np


DESTINATION_SCHEMA = "queue-haul-destination-v1"
MODES = ("normal", "emergency", "stable")


@dataclass(frozen=True)
class CompatibilityFingerprint:
    model: str
    tokenizer: str
    durable_log: str
    kv_abi: str

    def __post_init__(self):
        if not all((self.model, self.tokenizer, self.durable_log, self.kv_abi)):
            raise ValueError("compatibility fingerprints must be complete")

    def supports(self, other: "CompatibilityFingerprint", method: str) -> bool:
        replay = (self.model, self.tokenizer, self.durable_log) == (
            other.model, other.tokenizer, other.durable_log,
        )
        return replay and (method == "replay" or self.kv_abi == other.kv_abi)


@dataclass(frozen=True)
class ContextRate:
    contexts: tuple[float, ...]
    rates: tuple[float, ...]

    def __post_init__(self):
        if len(self.contexts) < 2 or len(self.contexts) != len(self.rates) \
                or any(x <= 0 for x in self.contexts + self.rates) \
                or any(b <= a for a, b in zip(self.contexts, self.contexts[1:])):
            raise ValueError("context rates require ordered positive points")

    def at(self, context: float) -> float:
        if not self.contexts[0] <= context <= self.contexts[-1]:
            raise ValueError(f"context {context:g} outside measured range")
        return float(np.interp(context, self.contexts, self.rates))


@dataclass(frozen=True)
class LoadedCoefficients:
    rho: tuple[float, ...]
    slowdown: tuple[float, ...]
    context_range: tuple[float, float]
    bandwidth_range_bytes_per_s: tuple[float, float]
    provenance: str
    baseline_factor: float = 1.0

    def __post_init__(self):
        if len(self.rho) < 2 or len(self.rho) != len(self.slowdown) \
                or self.rho[0] < 0 or any(b <= a for a, b in zip(self.rho, self.rho[1:])) \
                or any(x < 1 for x in self.slowdown) \
                or not 0 < self.context_range[0] < self.context_range[1] \
                or not 0 < self.bandwidth_range_bytes_per_s[0] \
                < self.bandwidth_range_bytes_per_s[1] or not self.provenance \
                or self.baseline_factor <= 0:
            raise ValueError("invalid loaded migration coefficients")

    def worst(self, initial_rho: float, boundary_rho: float, context: float,
              bandwidth_bytes_per_s: float) -> float:
        if not self.context_range[0] <= context <= self.context_range[1] \
                or not self.bandwidth_range_bytes_per_s[0] <= bandwidth_bytes_per_s \
                <= self.bandwidth_range_bytes_per_s[1] \
                or initial_rho < self.rho[0] or initial_rho > boundary_rho \
                or boundary_rho > self.rho[-1]:
            raise ValueError("migration candidate outside loaded-profile range")
        values = [np.interp(initial_rho, self.rho, self.slowdown),
                  np.interp(boundary_rho, self.rho, self.slowdown)]
        values += [v for r, v in zip(self.rho, self.slowdown)
                   if initial_rho <= r <= boundary_rho]
        return self.baseline_factor * float(max(values))


@dataclass(frozen=True)
class DestinationType:
    type_id: str
    compatibility: CompatibilityFingerprint
    prefill: ContextRate
    decode: ContextRate
    normals: tuple[tuple[float, float], ...]
    bounds: dict[str, tuple[float, ...]]
    kv_capacity_tokens: int
    loaded: dict[str, LoadedCoefficients]
    workload_prefill_fraction_range: tuple[float, float]
    provenance: str
    synthetic: bool = False

    def __post_init__(self):
        normals = np.asarray(self.normals, float)
        bounds = {mode: np.asarray(self.bounds.get(mode, ()), float) for mode in MODES}
        if not self.type_id or normals.ndim != 2 or normals.shape[1] != 2 \
                or not len(normals) or np.any(normals < 0) \
                or np.any(normals.sum(1) <= 0) \
                or any(v.shape != (len(normals),) or np.any(v <= 0) for v in bounds.values()) \
                or np.any(bounds["normal"] > bounds["emergency"]) \
                or np.any(bounds["emergency"] > bounds["stable"]) \
                or self.kv_capacity_tokens < 1 or set(self.loaded) != {"replay", "kv_transfer"} \
                or not 0 <= self.workload_prefill_fraction_range[0] \
                <= self.workload_prefill_fraction_range[1] <= 1 or not self.provenance:
            raise ValueError("invalid or nonnested destination envelope")

    def work(self, expected_f: float, expected_g: float, context: float) -> np.ndarray:
        work = np.array((expected_f / self.prefill.at(context),
                         expected_g / self.decode.at(context)))
        fraction = work[0] / work.sum() if work.sum() else 0.5
        lo, hi = self.workload_prefill_fraction_range
        if not lo <= fraction <= hi:
            raise ValueError("workload direction outside measured range")
        return work


@dataclass(frozen=True)
class DestinationReplica:
    replica_id: str
    baseline_work: tuple[float, float] = (0.0, 0.0)
    baseline_kv_tokens: int = 0

    def __post_init__(self):
        if not self.replica_id or min(*self.baseline_work, self.baseline_kv_tokens) < 0:
            raise ValueError("invalid destination replica baseline")


@dataclass(frozen=True)
class DestinationPool:
    pool_id: str
    type_id: str
    replicas: tuple[DestinationReplica, ...]
    route_id: str
    route: tuple[str, ...]
    methods: tuple[str, ...] = ("replay", "kv_transfer")

    def __post_init__(self):
        if not self.pool_id or not self.type_id or not self.replicas or not self.route_id \
                or not self.route or len({r.replica_id for r in self.replicas}) != len(self.replicas) \
                or not set(self.methods) <= {"replay", "kv_transfer"}:
            raise ValueError("invalid destination pool")


@dataclass(frozen=True)
class DestinationArchitecture:
    schema: str
    source_compatibility: CompatibilityFingerprint
    types: tuple[DestinationType, ...]
    pools: tuple[DestinationPool, ...]
    residency_horizon_s: float | None = None

    def __post_init__(self):
        type_ids = [q.type_id for q in self.types]
        pool_ids = [p.pool_id for p in self.pools]
        replicas = [r.replica_id for p in self.pools for r in p.replicas]
        if self.schema != DESTINATION_SCHEMA or not self.types or not self.pools \
                or len(set(type_ids)) != len(type_ids) or len(set(pool_ids)) != len(pool_ids) \
                or len(set(replicas)) != len(replicas) \
                or not set(p.type_id for p in self.pools) <= set(type_ids) \
                or self.residency_horizon_s is not None and self.residency_horizon_s < 0:
            raise ValueError("invalid destination architecture")

    @classmethod
    def load(cls, path: str | Path) -> "DestinationArchitecture":
        raw = json.loads(Path(path).read_text())
        def fingerprint(value): return CompatibilityFingerprint(**value)
        def rate(value): return ContextRate(tuple(value[0]), tuple(value[1]))
        types = []
        for item in raw["types"]:
            loaded = {
                method: LoadedCoefficients(
                    tuple(value["rho"]), tuple(value["slowdown"]),
                    tuple(value["context_range"]),
                    tuple(value["bandwidth_range_bytes_per_s"]), value["provenance"],
                    value.get("baseline_factor", 1),
                ) for method, value in item["loaded"].items()
            }
            types.append(DestinationType(
                item["type_id"], fingerprint(item["compatibility"]),
                rate(item["prefill"]), rate(item["decode"]),
                tuple(map(tuple, item["normals"])),
                {key: tuple(value) for key, value in item["bounds"].items()},
                item["kv_capacity_tokens"], loaded,
                tuple(item["workload_prefill_fraction_range"]), item["provenance"],
                item.get("synthetic", False),
            ))
        pools = tuple(DestinationPool(
            item["pool_id"], item["type_id"],
            tuple(DestinationReplica(
                r["replica_id"], tuple(r.get("baseline_work", (0, 0))),
                r.get("baseline_kv_tokens", 0),
            ) for r in item["replicas"]), item["route_id"], tuple(item["route"]),
            tuple(item.get("methods", ("replay", "kv_transfer"))),
        ) for item in raw["pools"])
        return cls(raw["schema"], fingerprint(raw["source_compatibility"]),
                   tuple(types), pools, raw.get("residency_horizon_s"))

    @property
    def type_by_id(self) -> dict[str, DestinationType]:
        return {q.type_id: q for q in self.types}
