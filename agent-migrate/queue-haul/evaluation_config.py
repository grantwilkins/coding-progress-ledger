"""Canonical Queue-Haul evaluation operating points."""

from dataclasses import dataclass


PROVENANCE = {"measured", "fitted", "assumed", "simulated"}


@dataclass(frozen=True)
class EvaluationValue:
    value: object
    unit: str
    provenance: str
    valid_range: str
    replacement_evidence: str

    def __post_init__(self):
        if not self.unit or self.provenance not in PROVENANCE \
                or not self.valid_range or not self.replacement_evidence:
            raise ValueError("evaluation values require units, provenance, range, and evidence")

    def record(self, evidence_status: str = "sensitivity") -> dict:
        if evidence_status == "accepted" and self.provenance == "assumed":
            raise ValueError("assumed values cannot support accepted evidence")
        if evidence_status not in {"accepted", "sensitivity"}:
            raise ValueError("invalid evidence status")
        return {
            "value": self.value,
            "unit": self.unit,
            "provenance": self.provenance,
            "valid_range": self.valid_range,
            "replacement_evidence": self.replacement_evidence,
            "evidence_status": evidence_status,
        }


# TODO: ASSUMED Replace with the declared curtailment notice distribution.
DEADLINES_S = EvaluationValue(
    (30, 60, 120, 300), "s", "assumed", "30-300",
    "operator event notice records",
)

# TODO: ASSUMED Replace with reserved source-to-pool route telemetry.
ROUTE_GBPS = EvaluationValue(
    (1, 5, 10), "Gb/s", "assumed", "1-10",
    "leased route throughput and queue telemetry",
)

# TODO: ASSUMED Replace with destination event-admission policy.
SERVICE_FLEX = EvaluationValue(
    (0, .05, .10, .20), "fraction of stable pool capacity", "assumed", "0-0.20",
    "normal/stable boundary and operator event policy",
)

# TODO: ASSUMED Replace with destination transition-debt policy.
SERVICE_DEBT = EvaluationValue(
    (0, .05, .10, .20), "stable pool capacity x migration horizon",
    "assumed", "0-0.20", "measured queued work and operator debt policy",
)

# TODO: ASSUMED Pools are a controlled diversity sweep, not a fleet claim.
POOL_COUNTS = EvaluationValue(
    (1, 2, 4, 8), "compatible pools", "assumed", "1-8",
    "operator pool inventory",
)

# TODO: ASSUMED Scales test frontier and runtime stability.
SESSION_COUNTS = EvaluationValue(
    (10_000, 100_000, 1_000_000), "sessions", "assumed", "10000-1000000",
    "source session inventory",
)

WORKLOADS = EvaluationValue(
    ("coding", "interactive_coding", "agentic_tool_loop", "conversation"),
    "workload class", "simulated", "listed classes",
    "checksum-pinned content-free trace shapes",
)

# TODO: ASSUMED Replace with observed source placement imbalance.
SOURCE_SKEWS = EvaluationValue(
    ("measured_normal", "balanced", "moderate", "high"), "placement case",
    "assumed", "listed cases", "source scheduler placement telemetry",
)


EVALUATION_GRID = {
    "deadlines": DEADLINES_S,
    "routes": ROUTE_GBPS,
    "service_flex": SERVICE_FLEX,
    "service_debt": SERVICE_DEBT,
    "pools": POOL_COUNTS,
    "sessions": SESSION_COUNTS,
    "workloads": WORKLOADS,
    "source_skews": SOURCE_SKEWS,
}


def evaluation_manifest() -> dict:
    return {name: value.record() for name, value in EVALUATION_GRID.items()}
