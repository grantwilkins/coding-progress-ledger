"""N1/N2 — model card builders and JSON sidecar emitters.

A model card has two faces:
- `model_card.md` — human-readable, follows `docs/MODEL_CARD_TEMPLATE.md`.
- `model_card.json` — machine-readable, validates against
  `schemas/model_card_schema.json`. The JSON is the source of truth.

`build_card_record` aggregates everything required for the JSON; the
markdown is rendered from the same record so they cannot drift.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import jsonschema

from coding_estimator.checkpoints.features.registry import GROUPS
from coding_estimator.eval.harness import EvalCell
from coding_estimator.labels.registry import V0_TARGETS

REPO_ROOT = Path(__file__).resolve().parents[2]
SCHEMA_PATH = REPO_ROOT / "schemas" / "model_card_schema.json"


def load_schema() -> dict[str, Any]:
    return json.loads(SCHEMA_PATH.read_text(encoding="utf-8"))


def validate_card(record: dict[str, Any]) -> None:
    """Validate `record` against the JSON schema. Raises
    jsonschema.ValidationError on failure."""
    jsonschema.validate(record, load_schema())


@dataclass(frozen=True)
class TrainingDataMeta:
    checkpoints_path: str
    labels_path: str
    n_runs: int
    n_checkpoints: int


def _calibration_status_from_cells(
    cells: list[EvalCell],
    *,
    scheme: str,
    calibration_method: str,
) -> dict[str, dict[str, Any]]:
    """For each target with a feasible cell on the requested scheme,
    emit one calibration_status entry."""
    out: dict[str, dict[str, Any]] = {}
    by_target: dict[str, EvalCell] = {}
    for cell in cells:
        if cell.scheme != scheme or not cell.feasible:
            continue
        by_target[cell.target] = cell
    for target, cell in by_target.items():
        out[target] = {
            "brier": cell.brier,
            "ece": cell.ece,
            "ece_3bin": None,
            "auroc": cell.auroc,
            "n_checkpoints": int(cell.n_checkpoints_test or 0),
            "calibration_method": calibration_method,
        }
    return out


def _target_definitions(targets: list[str]) -> dict[str, dict[str, Any]]:
    out: dict[str, dict[str, Any]] = {}
    for name in targets:
        meta = V0_TARGETS.get(name)
        if meta is None:
            continue
        out[name] = {
            "family": meta.family,
            "horizon_units": meta.horizon_units,
            "horizon_value": meta.horizon_value,
            "run_constant_flag": bool(meta.run_constant_flag),
        }
    return out


def _source_versions(checkpoints_df) -> dict[str, str]:
    if (
        "source" not in checkpoints_df.columns
        or "source_protocol_version" not in checkpoints_df.columns
    ):
        return {}
    out: dict[str, str] = {}
    for source, sub in checkpoints_df.groupby("source", sort=True):
        versions = sorted(set(str(v) for v in sub["source_protocol_version"].unique()))
        out[str(source)] = ",".join(versions) if versions else "unknown"
    return out


def build_card_record(
    *,
    estimator_id: str,
    estimator_version: str,
    model_family: str,
    checkpoints_df,
    labels_df,
    feature_groups: tuple[str, ...],
    targets: list[str],
    eval_cells: list[EvalCell],
    headline_scheme: str,
    diagnostic_schemes: tuple[str, ...],
    headline_seed: int,
    calibration_method: str,
    intended_use: list[str],
    non_use_cases: list[str],
    known_limits: list[str],
    not_safe_for_control: bool,
    commit_sha: str,
    failure_mode_results: dict[str, Any],
    go_no_go_gate: dict[str, str] | None = None,
    checkpoints_path: str = "datasets/checkpoints_all.parquet",
    labels_path: str = "datasets/labels_all.parquet",
) -> dict[str, Any]:
    """Aggregate all required fields into a record that validates
    against `model_card_schema.json`."""
    # Validate referenced groups
    for g in feature_groups:
        if g not in GROUPS:
            raise KeyError(f"unknown feature group: {g}")
    n_runs = int(checkpoints_df["run_id"].nunique())
    n_checkpoints = int(len(checkpoints_df))
    record: dict[str, Any] = {
        "estimator_id": estimator_id,
        "estimator_version": estimator_version,
        "model_family": model_family,
        "training_data": {
            "checkpoints_path": checkpoints_path,
            "labels_path": labels_path,
            "n_runs": n_runs,
            "n_checkpoints": n_checkpoints,
        },
        "source_versions": _source_versions(checkpoints_df),
        "feature_groups": list(feature_groups),
        "target_definitions": _target_definitions(targets),
        "split_protocol": {
            "headline_scheme": headline_scheme,
            "diagnostic_schemes": list(diagnostic_schemes),
            "headline_seed": headline_seed,
        },
        "known_limits": list(known_limits),
        "not_safe_for_control": bool(not_safe_for_control),
        "calibration_status": _calibration_status_from_cells(
            eval_cells,
            scheme=headline_scheme,
            calibration_method=calibration_method,
        ),
        "intended_use": list(intended_use),
        "non_use_cases": list(non_use_cases),
        "commit_sha": commit_sha,
        "failure_mode_results": failure_mode_results,
    }
    if go_no_go_gate is not None:
        record["go_no_go_gate"] = go_no_go_gate
    validate_card(record)
    return record


def render_card_markdown(record: dict[str, Any]) -> str:
    """Render the human-facing markdown card from a validated record."""
    lines = [f"# {record['estimator_id']}", ""]
    lines.append("## Intended use")
    lines.append("")
    for item in record["intended_use"]:
        lines.append(f"- {item}")
    lines.append("")

    lines.append("## Non-use cases")
    lines.append("")
    for item in record["non_use_cases"]:
        lines.append(f"- {item}")
    lines.append("")

    lines.append("## Not safe for control")
    lines.append("")
    lines.append(f"- `{str(record['not_safe_for_control']).lower()}`")
    lines.append("")

    td = record["training_data"]
    lines.append("## Training data")
    lines.append("")
    lines.append(f"- canonical sources: `{', '.join(record['source_versions'].keys())}`")
    lines.append(f"- inputs: `{td['checkpoints_path']}`, `{td['labels_path']}`")
    lines.append(f"- n_runs: {td['n_runs']}")
    lines.append(f"- n_checkpoints: {td['n_checkpoints']}")
    lines.append(f"- commit_sha: `{record['commit_sha']}`")
    lines.append("")

    lines.append("## Source versions")
    lines.append("")
    for source, ver in record["source_versions"].items():
        lines.append(f"- `{source}`: `{ver}`")
    lines.append("")

    lines.append("## Features")
    lines.append("")
    lines.append(f"- groups: {', '.join(record['feature_groups']) or '(none)'}")
    lines.append("")

    lines.append("## Target definitions")
    lines.append("")
    for target, meta in sorted(record["target_definitions"].items()):
        h_value = meta["horizon_value"] if meta["horizon_value"] is not None else "n/a"
        lines.append(
            f"- `{target}` — family={meta['family']}, "
            f"horizon={meta['horizon_units']}/{h_value}, "
            f"run_constant={meta['run_constant_flag']}"
        )
    lines.append("")

    sp = record["split_protocol"]
    lines.append("## Split protocol")
    lines.append("")
    lines.append(
        f"- headline metrics: `{sp['headline_scheme']}`, seed={sp.get('headline_seed', 0)}"
    )
    lines.append(f"- diagnostics: {', '.join(sp['diagnostic_schemes'])}")
    lines.append("")

    lines.append("## Calibration status")
    lines.append("")
    if record["calibration_status"]:
        for target, st in sorted(record["calibration_status"].items()):
            brier = "n/a" if st["brier"] is None else f"{st['brier']:.3f}"
            ece = "n/a" if st["ece"] is None else f"{st['ece']:.3f}"
            auroc = "n/a" if st["auroc"] is None else f"{st['auroc']:.3f}"
            lines.append(
                f"- `{target}` — Brier={brier}, ECE={ece}, AUROC={auroc}, "
                f"n={st['n_checkpoints']}, method=`{st['calibration_method']}`"
            )
    else:
        lines.append("- no feasible cells on the headline scheme")
    lines.append("")

    lines.append("## Failure-mode results (Workstream O)")
    lines.append("")
    for tid in ("O1", "O5", "O7"):
        rec = record["failure_mode_results"].get(tid)
        if rec is None:
            continue
        if isinstance(rec, list):
            lines.append(f"- `{tid}`:")
            for sub in rec:
                lines.append(
                    f"  - {sub.get('detail', {}).get('source', '?')}: "
                    f"**{sub['outcome']}** ({sub['metric_name']}={_fmt(sub.get('metric_value'))}; "
                    f"threshold={_fmt(sub['threshold'])})"
                )
        else:
            lines.append(
                f"- `{tid}`: **{rec['outcome']}** "
                f"({rec['metric_name']}={_fmt(rec.get('metric_value'))}; "
                f"threshold={_fmt(rec['threshold'])})"
            )
    lines.append("")

    if "go_no_go_gate" in record:
        gate = record["go_no_go_gate"]
        lines.append("## Go/no-go gate (Workstream P)")
        lines.append("")
        lines.append(f"- verdict: **{gate.get('verdict', '?')}**")
        if gate.get("report_path"):
            lines.append(f"- report_path: `{gate['report_path']}`")
        lines.append("")

    lines.append("## Known limits")
    lines.append("")
    for item in record["known_limits"]:
        lines.append(f"- {item}")
    return "\n".join(lines) + "\n"


def _fmt(v: Any) -> str:
    if v is None:
        return "n/a"
    if isinstance(v, float):
        return f"{v:.3f}"
    return str(v)


def write_card(out_dir: Path, record: dict[str, Any]) -> tuple[Path, Path]:
    """Write `model_card.json` and `model_card.md` into `out_dir`.
    Validates the record before writing — invalid records raise."""
    validate_card(record)
    out_dir.mkdir(parents=True, exist_ok=True)
    json_path = out_dir / "model_card.json"
    md_path = out_dir / "model_card.md"
    with json_path.open("w", encoding="utf-8", newline="\n") as fh:
        json.dump(record, fh, sort_keys=True, indent=2, ensure_ascii=False)
        fh.write("\n")
    md_path.write_text(render_card_markdown(record), encoding="utf-8", newline="\n")
    return json_path, md_path


__all__ = [
    "SCHEMA_PATH",
    "TrainingDataMeta",
    "load_schema",
    "validate_card",
    "build_card_record",
    "render_card_markdown",
    "write_card",
]
