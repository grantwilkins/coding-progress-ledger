from __future__ import annotations

import re
from dataclasses import asdict, dataclass
from typing import Any


@dataclass(frozen=True)
class VerifierSignature:
    exit_code: int
    collected: int | None
    passed: int
    failed: int
    errors: int
    warnings: int
    failed_tests: tuple[str, ...]
    error_tests: tuple[str, ...]

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def verifier_signature(*, stdout: str, stderr: str, exit_code: int) -> VerifierSignature:
    """Build a stable semantic verifier outcome from noisy verifier logs."""
    text = stdout + "\n" + stderr
    return VerifierSignature(
        exit_code=exit_code,
        collected=_last_int(r"\bcollected\s+(\d+)\s+items?\b", text),
        passed=_last_int(r"\b(\d+)\s+passed\b", text) or 0,
        failed=_last_int(r"\b(\d+)\s+failed\b", text) or 0,
        errors=_last_int(r"\b(\d+)\s+errors?\b", text) or 0,
        warnings=_last_int(r"\b(\d+)\s+warnings?\b", text) or 0,
        failed_tests=tuple(_failed_test_ids(text)),
        error_tests=tuple(_error_test_ids(text)),
    )


def signatures_match(expected: VerifierSignature, observed: VerifierSignature) -> bool:
    return expected == observed


def determinism_report(
    *,
    expected: VerifierSignature,
    observed: list[VerifierSignature],
) -> dict[str, Any]:
    mismatches = [
        {"trial": index, "signature": signature.to_dict()}
        for index, signature in enumerate(observed, 1)
        if not signatures_match(expected, signature)
    ]
    return {
        "expected_signature": expected.to_dict(),
        "observed_signatures": [signature.to_dict() for signature in observed],
        "trials": len(observed),
        "deterministic": not mismatches,
        "mismatches": mismatches,
    }


def _last_int(pattern: str, text: str) -> int | None:
    matches = re.findall(pattern, text)
    return int(matches[-1]) if matches else None


def _failed_test_ids(text: str) -> list[str]:
    failed: list[str] = []
    for line in text.splitlines():
        if line.startswith("FAILED "):
            parts = line.split()
            if len(parts) >= 2:
                failed.append(parts[1])
    return sorted(set(failed))


def _error_test_ids(text: str) -> list[str]:
    errors: list[str] = []
    for line in text.splitlines():
        if line.startswith("ERROR "):
            parts = line.split()
            if len(parts) >= 2:
                errors.append(parts[1])
    return sorted(set(errors))
