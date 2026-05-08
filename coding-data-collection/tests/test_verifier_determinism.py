"""
Claim:
Verifier determinism is a semantic outcome check: reruns must reproduce the
recorded verifier exit code, pytest outcome counts, and failed test identities
without requiring byte-identical Docker or package-install logs.

Plausible wrong implementations:
- Compare raw stdout/stderr and fail deterministic runs because install logs differ.
- Compare only exit code and miss a different set of failing tests.
- Count PASSED/FAILED detail lines instead of the final pytest summary.
- Treat warnings as irrelevant even though warning count is part of the verifier outcome.
- Ignore pytest ERROR identities and false-pass different setup/import failures.
"""

from coding_data_collection.verifier_determinism import determinism_report, verifier_signature


def test_signature_ignores_noisy_install_logs_but_keeps_outcome_counts() -> None:
    expected = verifier_signature(
        stdout="""
Downloading package
collected 4 items
tests/test_outputs.py .... [100%]
========================= 4 passed, 1 warning in 0.03s =========================
""",
        stderr="warning: cache path /tmp/random-a",
        exit_code=0,
    )
    observed = verifier_signature(
        stdout="""
Resolved packages in 92ms
collected 4 items
tests/test_outputs.py .... [100%]
========================= 4 passed, 1 warning in 0.05s =========================
""",
        stderr="warning: cache path /tmp/random-b",
        exit_code=0,
    )

    assert determinism_report(expected=expected, observed=[observed])["deterministic"] is True


def test_signature_distinguishes_same_exit_code_with_different_failed_tests() -> None:
    expected = verifier_signature(
        stdout="""
collected 2 items
FAILED tests/test_outputs.py::test_answer - AssertionError
========================= 1 failed, 1 passed in 0.01s =========================
""",
        stderr="",
        exit_code=1,
    )
    observed = verifier_signature(
        stdout="""
collected 2 items
FAILED tests/test_outputs.py::test_required_file - AssertionError
========================= 1 failed, 1 passed in 0.01s =========================
""",
        stderr="",
        exit_code=1,
    )

    report = determinism_report(expected=expected, observed=[observed])

    assert report["deterministic"] is False
    assert report["mismatches"][0]["signature"]["failed_tests"] == ("tests/test_outputs.py::test_required_file",)


def test_signature_uses_final_summary_not_verbose_pass_lines() -> None:
    signature = verifier_signature(
        stdout="""
PASSED tests/test_outputs.py::test_a
PASSED tests/test_outputs.py::test_b
========================= 2 passed, 3 warnings in 0.01s =========================
""",
        stderr="",
        exit_code=0,
    )

    assert signature.passed == 2
    assert signature.warnings == 3


def test_signature_distinguishes_same_error_count_with_different_error_tests() -> None:
    expected = verifier_signature(
        stdout="""
collected 2 items
ERROR tests/test_outputs.py::test_import - ModuleNotFoundError
========================= 1 error, 1 passed in 0.01s =========================
""",
        stderr="",
        exit_code=1,
    )
    observed = verifier_signature(
        stdout="""
collected 2 items
ERROR tests/test_outputs.py::test_setup - RuntimeError
========================= 1 error, 1 passed in 0.01s =========================
""",
        stderr="",
        exit_code=1,
    )

    report = determinism_report(expected=expected, observed=[observed])

    assert report["deterministic"] is False
    assert report["mismatches"][0]["signature"]["error_tests"] == ("tests/test_outputs.py::test_setup",)
