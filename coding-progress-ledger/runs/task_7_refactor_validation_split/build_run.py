from __future__ import annotations

import csv
import json
import shutil
import subprocess
import sys
from pathlib import Path


RUN_ROOT = Path(__file__).resolve().parent
PROJECT_ROOT = RUN_ROOT.parents[1]
REPO = RUN_ROOT / "repo"
TEST_PYTHON = PROJECT_ROOT / ".venv" / "bin" / "python"
TEST_CMD = [str(TEST_PYTHON), "-m", "pytest"]
TEST_CMD_DISPLAY = "../../../.venv/bin/python -m pytest"

sys.path.insert(0, str(PROJECT_ROOT))
from ledger_progress import LedgerSession  # noqa: E402


def write(path: Path, content: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(content, encoding="utf-8")


def run(cmd: list[str], cwd: Path) -> subprocess.CompletedProcess[str]:
    return subprocess.run(cmd, cwd=cwd, text=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, check=False)


def must(cmd: list[str], cwd: Path) -> subprocess.CompletedProcess[str]:
    result = run(cmd, cwd)
    if result.returncode != 0:
        raise RuntimeError(f"{cmd} failed in {cwd}:\n{result.stdout}")
    return result


def initial_module() -> str:
    return '''"""Invoice summarization utilities for a small checkout flow."""

from __future__ import annotations


def summarize_invoice(items, *, tax_rate=0.0, discount_codes=()):
    """Return subtotal, discount, tax, and total for invoice line items."""
    if tax_rate < 0:
        raise ValueError("tax_rate must be non-negative")

    normalized_codes = {str(code).upper() for code in discount_codes}
    subtotal_cents = 0
    discount_cents = 0
    categories = {}
    line_count = 0

    for item in items:
        if "name" not in item:
            raise ValueError("item missing name")
        if "unit_price_cents" not in item:
            raise ValueError("item missing unit_price_cents")

        name = str(item["name"]).strip()
        if not name:
            raise ValueError("item name must not be blank")

        quantity = int(item.get("quantity", 1))
        unit_price_cents = int(item["unit_price_cents"])
        if quantity <= 0:
            raise ValueError("quantity must be positive")
        if unit_price_cents < 0:
            raise ValueError("unit_price_cents must be non-negative")

        category = str(item.get("category", "uncategorized")).strip().lower() or "uncategorized"
        line_subtotal = quantity * unit_price_cents
        line_discount = 0

        if item.get("clearance"):
            line_discount += line_subtotal // 10
        if "BULK5" in normalized_codes and quantity >= 5:
            line_discount += line_subtotal * 5 // 100
        if "SAVE10" in normalized_codes:
            line_discount += 1000

        if line_discount > line_subtotal:
            line_discount = line_subtotal

        subtotal_cents += line_subtotal
        discount_cents += line_discount
        categories[category] = categories.get(category, 0) + line_subtotal - line_discount
        line_count += 1

    taxable_cents = subtotal_cents - discount_cents
    tax_cents = round(taxable_cents * tax_rate)
    total_cents = taxable_cents + tax_cents

    return {
        "subtotal_cents": subtotal_cents,
        "discount_cents": discount_cents,
        "tax_cents": tax_cents,
        "total_cents": total_cents,
        "line_count": line_count,
        "category_totals": dict(sorted(categories.items())),
    }
'''


def final_module() -> str:
    return '''"""Invoice summarization utilities for a small checkout flow."""

from __future__ import annotations


def _normalize_item(item):
    if "name" not in item:
        raise ValueError("item missing name")
    if "unit_price_cents" not in item:
        raise ValueError("item missing unit_price_cents")

    name = str(item["name"]).strip()
    if not name:
        raise ValueError("item name must not be blank")

    quantity = int(item.get("quantity", 1))
    unit_price_cents = int(item["unit_price_cents"])
    if quantity <= 0:
        raise ValueError("quantity must be positive")
    if unit_price_cents < 0:
        raise ValueError("unit_price_cents must be non-negative")

    category = str(item.get("category", "uncategorized")).strip().lower() or "uncategorized"
    return {
        "name": name,
        "quantity": quantity,
        "unit_price_cents": unit_price_cents,
        "category": category,
        "clearance": bool(item.get("clearance")),
    }


def _line_totals(item, discount_codes):
    line_subtotal = item["quantity"] * item["unit_price_cents"]
    line_discount = 0

    if item["clearance"]:
        line_discount += line_subtotal // 10
    if "BULK5" in discount_codes and item["quantity"] >= 5:
        line_discount += line_subtotal * 5 // 100
    if "SAVE10" in discount_codes:
        line_discount += 1000

    return line_subtotal, min(line_discount, line_subtotal)


def summarize_invoice(items, *, tax_rate=0.0, discount_codes=()):
    """Return subtotal, discount, tax, and total for invoice line items."""
    if tax_rate < 0:
        raise ValueError("tax_rate must be non-negative")

    normalized_codes = {str(code).upper() for code in discount_codes}
    subtotal_cents = 0
    discount_cents = 0
    categories = {}
    line_count = 0

    for raw_item in items:
        item = _normalize_item(raw_item)
        line_subtotal, line_discount = _line_totals(item, normalized_codes)

        subtotal_cents += line_subtotal
        discount_cents += line_discount
        categories[item["category"]] = categories.get(item["category"], 0) + line_subtotal - line_discount
        line_count += 1

    taxable_cents = subtotal_cents - discount_cents
    tax_cents = round(taxable_cents * tax_rate)
    total_cents = taxable_cents + tax_cents

    return {
        "subtotal_cents": subtotal_cents,
        "discount_cents": discount_cents,
        "tax_cents": tax_cents,
        "total_cents": total_cents,
        "line_count": line_count,
        "category_totals": dict(sorted(categories.items())),
    }
'''


def tests_targeted() -> str:
    return '''"""
Claim:
summarize_invoice preserves invoice accounting while applying per-line
discounts before tax and before category aggregation.

Plausible wrong implementations:
- Apply SAVE10 once per invoice instead of once per eligible line.
- Apply tax to the undiscounted subtotal rather than the taxable amount.
- Aggregate category totals before subtracting the line's discount.
"""

import pytest

from invoice_summary import summarize_invoice


def test_hand_checked_discounts_tax_and_categories():
    items = [
        {"name": "Widget", "unit_price_cents": 2000, "quantity": 5, "category": "Hardware"},
        {"name": "Cable", "unit_price_cents": 500, "quantity": 2, "category": "Hardware", "clearance": True},
        {"name": "Plan", "unit_price_cents": 2500, "category": "Service"},
    ]

    result = summarize_invoice(items, tax_rate=0.08, discount_codes=["bulk5", "save10"])

    assert result == {
        "subtotal_cents": 13500,
        "discount_cents": 3500,
        "tax_cents": 800,
        "total_cents": 10800,
        "line_count": 3,
        "category_totals": {"hardware": 8500, "service": 1500},
    }


@pytest.mark.parametrize(
    ("bad_item", "message"),
    [
        ({"unit_price_cents": 100}, "item missing name"),
        ({"name": "   ", "unit_price_cents": 100}, "item name must not be blank"),
        ({"name": "x", "unit_price_cents": 100, "quantity": 0}, "quantity must be positive"),
        ({"name": "x", "unit_price_cents": -1}, "unit_price_cents must be non-negative"),
    ],
)
def test_validation_boundaries_keep_public_errors(bad_item, message):
    with pytest.raises(ValueError, match=message):
        summarize_invoice([bad_item])
'''


def tests_regression() -> str:
    return '''"""
Claim:
Invoice totals are compositional: recombining independent groups should preserve
the sum of subtotal, discount, and post-tax total at zero tax.

Plausible wrong implementations:
- Let category state or discount caps leak between adjacent line items.
- Count lines from categories instead of the original items.
- Drop uncategorized or blank-category items during normalization.
"""

from invoice_summary import summarize_invoice


def test_zero_tax_totals_recompose_across_item_groups():
    group_a = [
        {"name": "A", "unit_price_cents": 300, "quantity": 5, "category": "Parts"},
        {"name": "B", "unit_price_cents": 250, "quantity": 1, "category": ""},
    ]
    group_b = [
        {"name": "C", "unit_price_cents": 900, "quantity": 2, "clearance": True},
        {"name": "D", "unit_price_cents": 1200, "quantity": 1, "category": "Service"},
    ]

    combined = summarize_invoice(group_a + group_b, discount_codes=["BULK5"])
    left = summarize_invoice(group_a, discount_codes=["BULK5"])
    right = summarize_invoice(group_b, discount_codes=["BULK5"])

    assert combined["subtotal_cents"] == left["subtotal_cents"] + right["subtotal_cents"]
    assert combined["discount_cents"] == left["discount_cents"] + right["discount_cents"]
    assert combined["total_cents"] == left["total_cents"] + right["total_cents"]
    assert combined["line_count"] == 4
    assert combined["category_totals"] == {
        "parts": 1425,
        "service": 1200,
        "uncategorized": 1870,
    }


def test_discount_never_makes_line_or_total_negative():
    result = summarize_invoice(
        [{"name": "Sticker", "unit_price_cents": 150, "quantity": 1, "category": "Promo"}],
        discount_codes=["SAVE10"],
    )

    assert result["discount_cents"] == 150
    assert result["total_cents"] == 0
    assert result["category_totals"] == {"promo": 0}
'''


def tests_api() -> str:
    return '''"""
Claim:
The refactor preserves the public API: callers still import summarize_invoice
and call it with items plus keyword-only tax and discount options.

Plausible wrong implementations:
- Rename or remove summarize_invoice while extracting helpers.
- Make tax_rate or discount_codes positional-only/required by accident.
- Return a tuple or helper-oriented structure instead of the documented dict.
"""

import inspect

import invoice_summary


def test_public_api_signature_is_compatible():
    signature = inspect.signature(invoice_summary.summarize_invoice)
    assert list(signature.parameters) == ["items", "tax_rate", "discount_codes"]
    assert signature.parameters["tax_rate"].kind is inspect.Parameter.KEYWORD_ONLY
    assert signature.parameters["discount_codes"].kind is inspect.Parameter.KEYWORD_ONLY
    assert signature.parameters["tax_rate"].default == 0.0
    assert signature.parameters["discount_codes"].default == ()


def test_public_result_keys_are_stable():
    result = invoice_summary.summarize_invoice([{"name": "A", "unit_price_cents": 100}])

    assert set(result) == {
        "subtotal_cents",
        "discount_cents",
        "tax_cents",
        "total_cents",
        "line_count",
        "category_totals",
    }
'''


def read_progress(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def main() -> None:
    if REPO.exists():
        shutil.rmtree(REPO)

    session = LedgerSession("TASK 7: Refactor with validation subtree split")
    transcript: list[str] = []

    step = 1
    s_repo = session.add("Create tiny self-contained invoice repo", step=step, weight=1.0)
    s_initial = session.add("Implement initial moderately long public function", step=step, weight=1.0)
    s_refactor = session.add("Refactor public function into two helpers without behavior changes", step=step, weight=1.0)
    s_validation = session.add("Validate behavior and API stayed unchanged", step=step, weight=1.0)
    s_artifacts = session.add("Export empirical run artifacts", step=step, weight=1.0)
    transcript.append("Step 1: Started with five concrete subtasks, including one intentionally broad validation task.")

    step += 1
    session.start(s_repo, step=step, evidence="Creating repo files and pytest configuration")
    write(
        REPO / "pyproject.toml",
        """[project]
name = "task-7-refactor-validation-split"
version = "0.1.0"

[tool.pytest.ini_options]
pythonpath = ["."]
testpaths = ["tests"]
""",
    )
    write(REPO / ".gitignore", "__pycache__/\n.pytest_cache/\n*.pyc\n")
    write(REPO / "README.md", "# Invoice Summary Toy Repo\n\nRun tests with `../../../.venv/bin/python -m pytest`.\n")
    session.complete(s_repo, "Created toy repo with pyproject and README", step=step)
    transcript.append("Step 2: Created a tiny standalone Python repo under the run directory.")

    step += 1
    session.start(s_initial, step=step, evidence="Writing long summarize_invoice implementation")
    write(REPO / "invoice_summary.py", initial_module())
    write(REPO / "tests" / "test_targeted_invoice.py", tests_targeted())
    initial_tests = run(TEST_CMD, REPO)
    if initial_tests.returncode != 0:
        raise RuntimeError(f"Initial tests unexpectedly failed:\n{initial_tests.stdout}")
    shutil.rmtree(REPO / "__pycache__", ignore_errors=True)
    shutil.rmtree(REPO / "tests" / "__pycache__", ignore_errors=True)
    shutil.rmtree(REPO / ".pytest_cache", ignore_errors=True)
    must(["git", "init"], REPO)
    must(["git", "add", "."], REPO)
    must(
        ["git", "-c", "user.name=Ledger Worker", "-c", "user.email=ledger@example.invalid", "commit", "-m", "Initial long invoice summary implementation"],
        REPO,
    )
    session.complete(s_initial, "Initial long public function passed targeted behavior tests and was committed", step=step)
    transcript.append("Step 3: Baseline function and targeted tests passed, then the initial repo state was committed.")

    step += 1
    session.start(s_refactor, step=step, evidence="Extracting normalization and per-line total helpers")
    write(REPO / "invoice_summary.py", final_module())
    after_refactor_targeted = run(TEST_CMD, REPO)
    if after_refactor_targeted.returncode != 0:
        session.block(s_refactor, step=step, reason="Refactor broke targeted tests", evidence=after_refactor_targeted.stdout)
        raise RuntimeError(after_refactor_targeted.stdout)
    session.complete(s_refactor, "Refactored into _normalize_item and _line_totals; targeted tests still passed", step=step)
    transcript.append("Step 4: Refactored the long function into two private helpers while preserving targeted behavior.")

    step += 1
    session.start(s_validation, step=step, evidence="Running targeted tests as first validation pass")
    session.complete(s_validation, "Targeted tests passed after refactor, but validation scope was still vague", step=step)
    transcript.append("Step 5: The broad validation task was initially marked complete from targeted tests alone.")

    step += 1
    targeted_id, regression_id, api_id = session.split(
        s_validation,
        [
            "Targeted unit tests for hand-checked invoice math and validation boundaries",
            "Broader regression tests for recomposition, category aggregation, and discount caps",
            "API compatibility checks for signature and result shape",
        ],
        step=step,
        reason="Validation needed to be auditable as targeted unit, broader regression, and API compatibility checks",
    )
    session.complete(targeted_id, "Targeted tests passed before and after refactor", step=step)
    transcript.append("Step 6: Split vague validation into three leaves; progress dropped because two new validation leaves remained open.")

    step += 1
    session.start(regression_id, step=step, evidence="Adding broader regression tests")
    write(REPO / "tests" / "test_regression_invoice.py", tests_regression())
    regression_tests = run(TEST_CMD, REPO)
    if regression_tests.returncode != 0:
        session.block(regression_id, step=step, reason="Regression tests failed", evidence=regression_tests.stdout)
        raise RuntimeError(regression_tests.stdout)
    session.complete(regression_id, "Broader regression tests passed on refactored implementation", step=step)
    transcript.append("Step 7: Added and passed recomposition, aggregation, and discount-cap regression tests.")

    step += 1
    session.start(api_id, step=step, evidence="Adding API compatibility tests")
    write(REPO / "tests" / "test_api_compatibility.py", tests_api())
    final_tests = run(TEST_CMD, REPO)
    if final_tests.returncode != 0:
        session.block(api_id, step=step, reason="API compatibility tests failed", evidence=final_tests.stdout)
        raise RuntimeError(final_tests.stdout)
    session.complete(api_id, "API compatibility tests passed with stable signature and result keys", step=step)
    transcript.append("Step 8: Added API compatibility checks and confirmed the full pytest suite passed.")

    step += 1
    session.start(s_artifacts, step=step, evidence="Writing transcript, notes, patch, ledger, progress, and summary")
    must(["git", "add", "-N", "tests/test_regression_invoice.py", "tests/test_api_compatibility.py"], REPO)
    patch = must(["git", "diff", "HEAD"], REPO)
    write(RUN_ROOT / "final_diff.patch", patch.stdout)
    write(
        RUN_ROOT / "test_output.txt",
        f"$ {TEST_CMD_DISPLAY}\n\n"
        "Initial baseline output:\n"
        f"{initial_tests.stdout}\n"
        "After helper refactor with targeted tests only:\n"
        f"{after_refactor_targeted.stdout}\n"
        "After broader regression tests were added:\n"
        f"{regression_tests.stdout}\n"
        "Final output with API compatibility tests:\n"
        f"{final_tests.stdout}",
    )
    write(
        RUN_ROOT / "task.md",
        """# TASK 7: Refactor with validation subtree split

Create a tiny self-contained Python repo with a moderately long public function.
The user-facing task is to refactor that function into two helpers without
changing behavior, while preserving the public API and tests.
""",
    )
    write(
        RUN_ROOT / "README.md",
        """# Task 7 Refactor Validation Split

Toy repo: `repo/`

From the repo directory, run:

```bash
../../../.venv/bin/python -m pytest
```

The final implementation refactors `summarize_invoice` into two private helpers
while preserving behavior and public API compatibility.
""",
    )
    write(RUN_ROOT / "agent_transcript.md", "# Agent Transcript\n\n" + "\n".join(f"- {line}" for line in transcript) + "\n")
    session.complete(s_artifacts, "Required artifact bundle written except ledger/progress/summary finalization", step=step)

    session.export_jsonl(str(RUN_ROOT / "ledger.jsonl"))
    session.export_curve_csv(str(RUN_ROOT / "progress.csv"))

    rows = read_progress(RUN_ROOT / "progress.csv")
    progress_values = [float(row["progress"]) for row in rows]
    drops = [progress_values[i] - progress_values[i + 1] for i in range(len(progress_values) - 1)]
    largest_drop = max([drop for drop in drops if drop > 0], default=0.0)
    events = session.ledger.events
    summary = {
        "task_id": "task_7_refactor_validation_split",
        "final_progress": session.score().progress,
        "subtasks_created": len(session.ledger.subtasks),
        "completed_subtasks": sum(1 for subtask in session.ledger.subtasks.values() if subtask.status.value == "complete"),
        "splits": sum(1 for event in events if event.event_type.value == "split_subtask"),
        "reopens": sum(1 for event in events if event.event_type.value == "reopen_subtask"),
        "invalidations": sum(1 for event in events if event.event_type.value == "invalidate_subtask"),
        "largest_progress_drop": largest_drop,
        "non_monotonic": any(drop > 0 for drop in drops),
        "test_command": TEST_CMD_DISPLAY,
        "test_status": "passed" if final_tests.returncode == 0 else "failed",
        "artifact_paths": [
            "task.md",
            "README.md",
            "agent_transcript.md",
            "ledger.jsonl",
            "progress.csv",
            "final_diff.patch",
            "test_output.txt",
            "run_notes.md",
            "summary.json",
            "repo/",
        ],
    }
    write(RUN_ROOT / "summary.json", json.dumps(summary, indent=2) + "\n")
    write(
        RUN_ROOT / "run_notes.md",
        f"""# Run Notes

## Progress Changes

The ledger started with five active subtasks. Progress rose as the repo,
baseline implementation, refactor, and first validation pass completed. It then
dropped when the broad validation task was split into three concrete leaves:
targeted unit tests, broader regression tests, and API compatibility checks.
The largest observed progress drop was {largest_drop:.6f}.

## Validation Split

The validation task was deliberately vague at first. After the helper extraction
passed the targeted tests, the work model changed to expose the missing evidence:
broader regression tests for accounting invariants and API checks for callers
that import and invoke `summarize_invoice`.

## Evidence-Backed Completions

The initial implementation completion is backed by passing baseline pytest
output and the initial git commit. The refactor completion is backed by targeted
tests passing after extracting `_normalize_item` and `_line_totals`. The final
validation leaves are backed by full pytest output captured in `test_output.txt`.

## Ledger Notes

The ledger was useful because the split made incomplete validation visible as a
non-monotonic progress event instead of allowing one broad checkbox to hide the
remaining test work. The awkward part is that final artifact export is itself a
task, but `ledger.jsonl` and `progress.csv` can only be exported after the last
ledger event, so the artifact completion evidence describes that pending final
side effect.
""",
    )

    if final_tests.returncode != 0:
        raise SystemExit(final_tests.returncode)


if __name__ == "__main__":
    main()
