#!/usr/bin/env python3
"""SWE-agent pilot sampler (Workstream B, task B2).

Reads the deterministic manifest produced by A3
(``external_data/swe_agent/manifests/swe_agent_inventory.csv``) and
emits a balanced pilot sample CSV at the path passed via ``--output``,
implementing the policy specified in
``external_data/swe_agent/PILOT_SAMPLING_POLICY.md`` (B1).

Determinism contract
--------------------
TASKS.md (B2) requires "re-running with same seed produces byte-identical
CSV". This script:

  1. Reads the manifest with ``csv.DictReader`` and explicitly converts
     boolean columns from the literal strings ``"True"``/``"False"``
     (never via ``bool()`` on the raw string).
  2. Sorts each side's eligible pool by ``instance_id`` lexicographically
     before sampling, so that even if ``DictReader`` returned rows in
     a different order, ``random.Random(seed).sample`` sees a stable
     input.
  3. Uses an explicit ``random.Random(seed)`` instance (never the
     module-global ``random.sample``). The same RNG instance is used for
     both sides; the **success side is sampled first**, then the failure
     side, fixing the order of consumption.
  4. After sampling, re-sorts each status group by ``instance_id`` and
     assigns ``pilot_id`` deterministically (``s_NN`` for successes,
     ``f_NN`` for failures), zero-padded to 2 digits unless n > 99.
  5. Writes the final CSV sorted by ``pilot_id``. Because ASCII has
     ``f`` < ``s``, the output begins with ``f_01..f_NN`` and ends with
     ``s_01..s_NN`` — that is the documented, stable order.
  6. CSV writer uses ``csv.QUOTE_MINIMAL``, ``newline=""``,
     ``lineterminator="\n"``, UTF-8, header row first. Booleans are
     written as the literals ``True``/``False``.

Source mutation contract
------------------------
The script never modifies the inventory CSV or anything under
``external_data/swe_agent/raw/``. The policy doc path passed via
``--policy-doc`` is logged for diagnostics only; this script does not
parse the policy.

CLI
---
    python scripts/sample_swe_agent_pilot.py \\
      --inventory-csv external_data/swe_agent/manifests/swe_agent_inventory.csv \\
      --output        external_data/swe_agent/manifests/swe_agent_pilot_sample.csv \\
      --n-success 10 \\
      --n-failure 10 \\
      --seed 0
"""

from __future__ import annotations

import argparse
import csv
import random
import sys
from collections import Counter
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple


CSV_COLUMNS: Tuple[str, ...] = (
    "pilot_id",
    "source_id",
    "instance_id",
    "model_name",
    "final_success",
    "trajectory_length",
    "patch_available",
    "eval_log_available",
    "raw_path_or_dataset_index",
    "selection_reason",
)

DEFAULT_POLICY_DOC = Path("external_data/swe_agent/PILOT_SAMPLING_POLICY.md")
PILOT_MODEL_NAME = "swe-agent-llama-70b"


def _parse_args(argv: Optional[List[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Deterministically sample a SWE-agent pilot from the A3 "
            "inventory CSV per the B1 policy."
        ),
    )
    parser.add_argument(
        "--inventory-csv",
        required=True,
        type=Path,
        help="Path to the A3 manifest CSV.",
    )
    parser.add_argument(
        "--output",
        required=True,
        type=Path,
        help="Destination pilot sample CSV path.",
    )
    parser.add_argument(
        "--n-success",
        type=int,
        default=10,
        help="Target number of success rows (default 10).",
    )
    parser.add_argument(
        "--n-failure",
        type=int,
        default=10,
        help="Target number of failure rows (default 10).",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=0,
        help="Seed for random.Random (default 0).",
    )
    parser.add_argument(
        "--policy-doc",
        type=Path,
        default=DEFAULT_POLICY_DOC,
        help=(
            "Path to the B1 policy doc (diagnostic only; not parsed). "
            "Default: external_data/swe_agent/PILOT_SAMPLING_POLICY.md."
        ),
    )
    return parser.parse_args(argv)


def _parse_bool(value: str) -> Optional[bool]:
    """Strict converter: ``'True'`` -> True, ``'False'`` -> False, else None.

    Note: `bool('False')` is True in Python, so we never trust ``bool()``
    on the raw string. Empty string maps to None ("missing").
    """
    if value == "True":
        return True
    if value == "False":
        return False
    return None


def _parse_int(value: str) -> Optional[int]:
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def _parse_dataset_index(value: str) -> Optional[int]:
    """Extract the trailing int from ``nebius/SWE-agent-trajectories:train:<N>``.

    Compare numerically to avoid lexical ordering bugs (``'10' < '2'`` in
    string sort).
    """
    if not isinstance(value, str) or not value:
        return None
    tail = value.rsplit(":", 1)[-1]
    try:
        return int(tail)
    except (TypeError, ValueError):
        return None


def _load_inventory(path: Path) -> List[Dict[str, Any]]:
    """Load the manifest CSV. Returns a list of raw dict rows."""
    with path.open("r", encoding="utf-8", newline="") as fh:
        reader = csv.DictReader(fh)
        return list(reader)


def _apply_filters(
    rows: List[Dict[str, Any]],
    *,
    require_model: bool,
    min_trajectory_length: int,
) -> Tuple[List[Dict[str, Any]], Dict[str, int]]:
    """Apply inclusion filters in the policy-specified order.

    Returns (filtered rows, funnel counts dict). Funnel keys are the
    step labels used in the end-of-run report.
    """
    funnel: Dict[str, int] = {}
    funnel["total"] = len(rows)

    # I1: parse_status == "ok"
    step1 = [r for r in rows if r.get("parse_status") == "ok"]
    funnel["after_parse_status_ok"] = len(step1)

    # I2: trajectory_available == True
    step2 = [r for r in step1 if _parse_bool(r.get("trajectory_available", "")) is True]
    funnel["after_trajectory_available"] = len(step2)

    # I3: final_success_available == True
    step3 = [
        r for r in step2 if _parse_bool(r.get("final_success_available", "")) is True
    ]
    funnel["after_final_success_available"] = len(step3)

    # I4: patch_available == True
    step4 = [r for r in step3 if _parse_bool(r.get("patch_available", "")) is True]
    funnel["after_patch_available"] = len(step4)

    # I5: eval_log_available == True
    step5 = [r for r in step4 if _parse_bool(r.get("eval_log_available", "")) is True]
    funnel["after_eval_log_available"] = len(step5)

    # I6: trajectory_length >= min_trajectory_length (parse to int; drop bad)
    step6: List[Dict[str, Any]] = []
    bad_traj_len = 0
    for r in step5:
        n = _parse_int(r.get("trajectory_length", ""))
        if n is None:
            bad_traj_len += 1
            print(
                f"[sample_swe_agent_pilot] WARNING: dropping row with "
                f"non-int trajectory_length="
                f"{r.get('trajectory_length', '')!r} "
                f"source_id={r.get('source_id', '')!r}",
                file=sys.stderr,
            )
            continue
        if n >= min_trajectory_length:
            step6.append(r)
    funnel[f"after_trajectory_length_ge_{min_trajectory_length}"] = len(step6)
    if bad_traj_len:
        funnel["dropped_bad_trajectory_length"] = bad_traj_len

    # I7: model_name == swe-agent-llama-70b (optional)
    if require_model:
        step7 = [r for r in step6 if r.get("model_name") == PILOT_MODEL_NAME]
        funnel[f"after_model_name_eq_{PILOT_MODEL_NAME}"] = len(step7)
    else:
        step7 = step6
        funnel["after_model_name_any"] = len(step7)

    return step7, funnel


def _dedupe_by_instance(
    rows: List[Dict[str, Any]],
) -> Tuple[List[Dict[str, Any]], int]:
    """For each instance_id, keep the row with the lowest dataset index.

    Returns (deduped rows, number of unique instance_ids). Rows without
    a parseable dataset index are sorted last (None treated as +inf).
    """
    by_instance: Dict[str, Dict[str, Any]] = {}
    by_instance_idx: Dict[str, int] = {}
    for r in rows:
        inst = r.get("instance_id", "")
        if not inst:
            continue
        idx = _parse_dataset_index(r.get("raw_path_or_dataset_index", ""))
        # Treat un-parseable indices as worst (largest) so any parseable
        # row beats them.
        idx_cmp = idx if idx is not None else float("inf")
        existing_idx = by_instance_idx.get(inst)
        if existing_idx is None or idx_cmp < existing_idx:
            by_instance[inst] = r
            by_instance_idx[inst] = idx_cmp  # type: ignore[assignment]
    return list(by_instance.values()), len(by_instance)


def _split_by_success(
    rows: List[Dict[str, Any]],
) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
    """Split into (success, failure) lists by final_success column."""
    success: List[Dict[str, Any]] = []
    failure: List[Dict[str, Any]] = []
    for r in rows:
        b = _parse_bool(r.get("final_success", ""))
        if b is True:
            success.append(r)
        elif b is False:
            failure.append(r)
        # rows without a parseable final_success are dropped here; the
        # I3 filter (final_success_available) should already have removed
        # them.
    return success, failure


def _sample_side(
    rows: List[Dict[str, Any]], n: int, rng: random.Random
) -> List[Dict[str, Any]]:
    """Sort by instance_id, then sample n without replacement.

    If len(rows) < n, return all rows (caller decides fallback).
    """
    sorted_rows = sorted(rows, key=lambda r: r.get("instance_id", ""))
    if len(sorted_rows) <= n:
        return sorted_rows
    return rng.sample(sorted_rows, n)


def _format_pilot_id(status_letter: str, ordinal_1based: int, total: int) -> str:
    """Build a pilot_id like ``swe_agent_pilot_s_01``.

    Zero-pads to 2 digits unless total > 99 (then 3, with a stderr warning).
    """
    width = 2
    if total > 99:
        width = 3
    return f"swe_agent_pilot_{status_letter}_{ordinal_1based:0{width}d}"


def _format_cell(value: Any) -> str:
    """Render a sample CSV cell to a CSV-safe string. Bools as True/False."""
    if isinstance(value, bool):
        return "True" if value else "False"
    if value is None:
        return ""
    return str(value)


def _build_output_rows(
    success_picks: List[Dict[str, Any]],
    failure_picks: List[Dict[str, Any]],
    selection_reason: str,
) -> List[Dict[str, Any]]:
    """Assign pilot_ids, attach selection_reason, and project to CSV columns."""
    out: List[Dict[str, Any]] = []

    success_sorted = sorted(success_picks, key=lambda r: r.get("instance_id", ""))
    failure_sorted = sorted(failure_picks, key=lambda r: r.get("instance_id", ""))

    if len(success_sorted) > 99 or len(failure_sorted) > 99:
        print(
            "[sample_swe_agent_pilot] WARNING: pilot scope exceeded "
            "(n>99); padding pilot_id ordinal to 3 digits.",
            file=sys.stderr,
        )

    for i, row in enumerate(success_sorted, start=1):
        pid = _format_pilot_id("s", i, len(success_sorted))
        out.append(_project_row(row, pid, selection_reason))

    for i, row in enumerate(failure_sorted, start=1):
        pid = _format_pilot_id("f", i, len(failure_sorted))
        out.append(_project_row(row, pid, selection_reason))

    # Final sort by pilot_id. ASCII: 'f' < 's', so failures come first.
    out.sort(key=lambda r: r["pilot_id"])
    return out


def _project_row(
    row: Dict[str, Any], pilot_id: str, selection_reason: str
) -> Dict[str, Any]:
    """Project a manifest row to the B2 output columns."""
    final_success = _parse_bool(row.get("final_success", ""))
    patch_available = _parse_bool(row.get("patch_available", ""))
    eval_log_available = _parse_bool(row.get("eval_log_available", ""))
    traj_len = _parse_int(row.get("trajectory_length", ""))
    return {
        "pilot_id": pilot_id,
        "source_id": row.get("source_id", ""),
        "instance_id": row.get("instance_id", ""),
        "model_name": row.get("model_name", ""),
        "final_success": final_success if final_success is not None else "",
        "trajectory_length": traj_len if traj_len is not None else "",
        "patch_available": patch_available if patch_available is not None else "",
        "eval_log_available": (
            eval_log_available if eval_log_available is not None else ""
        ),
        "raw_path_or_dataset_index": row.get("raw_path_or_dataset_index", ""),
        "selection_reason": selection_reason,
    }


def _write_output(rows: List[Dict[str, Any]], output: Path) -> None:
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", encoding="utf-8", newline="") as fh:
        writer = csv.writer(fh, quoting=csv.QUOTE_MINIMAL, lineterminator="\n")
        writer.writerow(list(CSV_COLUMNS))
        for r in rows:
            writer.writerow([_format_cell(r[c]) for c in CSV_COLUMNS])


def _log_funnel(funnel: Dict[str, int], unique_instances: int) -> None:
    """Print the canonical end-of-run funnel block to stderr."""
    print("[sample_swe_agent_pilot] funnel:", file=sys.stderr)
    print(f"  total rows: {funnel.get('total', 0)}", file=sys.stderr)
    print(
        f"  after parse_status==ok: {funnel.get('after_parse_status_ok', 0)}",
        file=sys.stderr,
    )
    print(
        f"  after trajectory_available: "
        f"{funnel.get('after_trajectory_available', 0)}",
        file=sys.stderr,
    )
    print(
        f"  after final_success_available: "
        f"{funnel.get('after_final_success_available', 0)}",
        file=sys.stderr,
    )
    print(
        f"  after patch_available: {funnel.get('after_patch_available', 0)}",
        file=sys.stderr,
    )
    print(
        f"  after eval_log_available: {funnel.get('after_eval_log_available', 0)}",
        file=sys.stderr,
    )
    # Trajectory-length step name varies (10 or 5 depending on fallback).
    traj_key = next(
        (k for k in funnel if k.startswith("after_trajectory_length_ge_")),
        None,
    )
    if traj_key is not None:
        threshold = traj_key.rsplit("_", 1)[-1]
        print(
            f"  after trajectory_length >= {threshold}: {funnel[traj_key]}",
            file=sys.stderr,
        )
    # Model step varies between primary and fallback1.
    model_key = next(
        (k for k in funnel if k.startswith("after_model_name_")),
        None,
    )
    if model_key is not None:
        if model_key == "after_model_name_any":
            print(
                f"  after model_name == any (fallback): {funnel[model_key]}",
                file=sys.stderr,
            )
        else:
            mn = model_key[len("after_model_name_eq_"):]
            print(
                f"  after model_name == {mn}: {funnel[model_key]}",
                file=sys.stderr,
            )
    print(
        f"  after dedupe on instance_id: {funnel.get('after_dedupe', 0)} "
        f"({unique_instances})",
        file=sys.stderr,
    )
    print(
        f"  eligible_success: {funnel.get('eligible_success', 0)}",
        file=sys.stderr,
    )
    print(
        f"  eligible_failure: {funnel.get('eligible_failure', 0)}",
        file=sys.stderr,
    )


def _run_funnel(
    rows: List[Dict[str, Any]],
    *,
    require_model: bool,
    min_trajectory_length: int,
) -> Tuple[
    List[Dict[str, Any]],
    List[Dict[str, Any]],
    Dict[str, int],
    int,
]:
    """Apply filters + dedupe + split. Returns (success_pool, failure_pool, funnel, unique_instances)."""
    filtered, funnel = _apply_filters(
        rows,
        require_model=require_model,
        min_trajectory_length=min_trajectory_length,
    )
    deduped, unique_instances = _dedupe_by_instance(filtered)
    funnel["after_dedupe"] = len(deduped)
    success_pool, failure_pool = _split_by_success(deduped)
    funnel["eligible_success"] = len(success_pool)
    funnel["eligible_failure"] = len(failure_pool)
    return success_pool, failure_pool, funnel, unique_instances


def _try_sample(
    rows: List[Dict[str, Any]],
    *,
    n_success: int,
    n_failure: int,
    seed: int,
    require_model: bool,
    min_trajectory_length: int,
) -> Tuple[
    List[Dict[str, Any]],
    List[Dict[str, Any]],
    Dict[str, int],
    int,
]:
    """Run one funnel + sampling pass. Returns picks + funnel state.

    Sampling order: success first, then failure, on the same Random
    instance. Sorting by instance_id happens inside ``_sample_side``.
    """
    success_pool, failure_pool, funnel, unique_instances = _run_funnel(
        rows,
        require_model=require_model,
        min_trajectory_length=min_trajectory_length,
    )
    rng = random.Random(seed)
    success_picks = _sample_side(success_pool, n_success, rng)
    failure_picks = _sample_side(failure_pool, n_failure, rng)
    return success_picks, failure_picks, funnel, unique_instances


def _select_with_fallbacks(
    rows: List[Dict[str, Any]],
    *,
    n_success: int,
    n_failure: int,
    seed: int,
) -> Tuple[
    List[Dict[str, Any]],
    List[Dict[str, Any]],
    str,
    str,
    Dict[str, int],
    int,
]:
    """Run primary; descend the fallback ladder until both sides hit target.

    Returns:
      (success_picks, failure_picks, selection_reason, fallback_level,
       final_funnel, final_unique_instances)
    """
    # Level 0: primary
    success_picks, failure_picks, funnel, unique_instances = _try_sample(
        rows,
        n_success=n_success,
        n_failure=n_failure,
        seed=seed,
        require_model=True,
        min_trajectory_length=10,
    )
    if len(success_picks) >= n_success and len(failure_picks) >= n_failure:
        return (
            success_picks,
            failure_picks,
            f"primary_balanced_{n_success}_{n_failure}",
            "primary",
            funnel,
            unique_instances,
        )
    print(
        f"[sample_swe_agent_pilot] FALLBACK fallback1: "
        f"success={len(success_picks)}/{n_success} "
        f"failure={len(failure_picks)}/{n_failure}",
        file=sys.stderr,
    )

    # Level 1: drop model restriction
    success_picks, failure_picks, funnel, unique_instances = _try_sample(
        rows,
        n_success=n_success,
        n_failure=n_failure,
        seed=seed,
        require_model=False,
        min_trajectory_length=10,
    )
    if len(success_picks) >= n_success and len(failure_picks) >= n_failure:
        return (
            success_picks,
            failure_picks,
            f"fallback1_all_models_{n_success}_{n_failure}",
            "fallback1",
            funnel,
            unique_instances,
        )
    print(
        f"[sample_swe_agent_pilot] FALLBACK fallback2: "
        f"success={len(success_picks)}/{n_success} "
        f"failure={len(failure_picks)}/{n_failure}",
        file=sys.stderr,
    )

    # Level 2: relax trajectory_length to >= 5
    success_picks, failure_picks, funnel, unique_instances = _try_sample(
        rows,
        n_success=n_success,
        n_failure=n_failure,
        seed=seed,
        require_model=False,
        min_trajectory_length=5,
    )
    if len(success_picks) >= n_success and len(failure_picks) >= n_failure:
        return (
            success_picks,
            failure_picks,
            f"fallback2_short_traj_{n_success}_{n_failure}",
            "fallback2",
            funnel,
            unique_instances,
        )
    print(
        f"[sample_swe_agent_pilot] FALLBACK fallback3: "
        f"success={len(success_picks)}/{n_success} "
        f"failure={len(failure_picks)}/{n_failure}",
        file=sys.stderr,
    )

    # Level 3: halve targets
    halved_success = max(1, n_success // 2)
    halved_failure = max(1, n_failure // 2)
    success_picks, failure_picks, funnel, unique_instances = _try_sample(
        rows,
        n_success=halved_success,
        n_failure=halved_failure,
        seed=seed,
        require_model=False,
        min_trajectory_length=5,
    )
    if (
        len(success_picks) >= halved_success
        and len(failure_picks) >= halved_failure
    ):
        return (
            success_picks,
            failure_picks,
            f"fallback3_half_targets_{halved_success}_{halved_failure}",
            "fallback3",
            funnel,
            unique_instances,
        )
    print(
        f"[sample_swe_agent_pilot] FALLBACK fallback4: "
        f"success={len(success_picks)}/{halved_success} "
        f"failure={len(failure_picks)}/{halved_failure}",
        file=sys.stderr,
    )

    # Level 4: take what we can; do NOT silently rebalance.
    print(
        "[sample_swe_agent_pilot] WARNING: fallback4 reached — emitting "
        f"imbalanced sample success={len(success_picks)} "
        f"failure={len(failure_picks)}.",
        file=sys.stderr,
    )
    return (
        success_picks,
        failure_picks,
        f"fallback_imbalance_{len(success_picks)}_{len(failure_picks)}",
        "fallback4",
        funnel,
        unique_instances,
    )


def main(argv: Optional[List[str]] = None) -> int:
    args = _parse_args(argv)

    print(
        f"[sample_swe_agent_pilot] implementing policy at {args.policy_doc}",
        file=sys.stderr,
    )

    rows = _load_inventory(args.inventory_csv)

    (
        success_picks,
        failure_picks,
        selection_reason,
        fallback_level,
        funnel,
        unique_instances,
    ) = _select_with_fallbacks(
        rows,
        n_success=args.n_success,
        n_failure=args.n_failure,
        seed=args.seed,
    )

    output_rows = _build_output_rows(success_picks, failure_picks, selection_reason)

    if not output_rows:
        print(
            "[sample_swe_agent_pilot] FATAL: no rows selected after all "
            "fallbacks; refusing to write a header-only file.",
            file=sys.stderr,
        )
        return 2

    _write_output(output_rows, args.output)

    _log_funnel(funnel, unique_instances)
    print(
        f"[sample_swe_agent_pilot] selected: {len(success_picks)} success "
        f"/ {len(failure_picks)} failure "
        f"(selection_reason={selection_reason})",
        file=sys.stderr,
    )
    print(
        f"[sample_swe_agent_pilot] fallback level: {fallback_level}",
        file=sys.stderr,
    )
    print(
        f"[sample_swe_agent_pilot] wrote sample -> {args.output}",
        file=sys.stderr,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
