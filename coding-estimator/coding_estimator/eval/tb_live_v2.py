"""U6 — tb_live_v2 evaluation under exact-task holdout.

The headline claim for tb_live_v2 is intentionally narrow:

- Process-dynamics performance is reported under exact-task holdout,
  not under overlap-heavy run-level splits that let the model see task
  X / arm A while testing on task X / arm B.
- Easier overlap-heavy splits (LORO, random holdout) are auxiliary
  diagnostics only.
- Terminal-success results are secondary and caveated because the
  corpus is arm-concentrated, ceiling-limited, and optimistic due to
  the seeded `solution.sh` leak recorded in the source registry.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import UTC, datetime
import re
from pathlib import Path

import pandas as pd

from coding_estimator.baselines import LEDGER_BASIC, TIME_ONLY, BaselineSpec
from coding_estimator.eval.harness import EvalCell, evaluate_cell
from coding_estimator.profile.budget import BudgetCell, compute_budget
from coding_estimator.splits.protocol import Split, holdout, loro, ltfo

TB_LIVE_V2 = "tb_live_v2"
HEADLINE_TARGETS: tuple[str, ...] = (
    "y_future_progress_drop_h5",
    "y_validation_new_work_h5",
)
SECONDARY_TARGETS: tuple[str, ...] = ("y_success_eventual",)
EVAL_TARGETS: tuple[str, ...] = HEADLINE_TARGETS + SECONDARY_TARGETS
EVAL_MODELS: tuple[BaselineSpec, ...] = (TIME_ONLY, LEDGER_BASIC)
SCHEME_LABELS: dict[str, str] = {
    "ltfo": "exact-task-holdout",
    "loro": "loro-overlap",
    "holdout": "holdout-overlap",
}

_RUN_ID_ARM_RE = re.compile(r"__arm(?P<arm>[A-Z])(?:__|$)")


@dataclass(frozen=True)
class ProfileOutcomeRow:
    name: str
    n_success: int
    n_runs: int
    model_name: str | None = None

    @property
    def n_fail(self) -> int:
        return self.n_runs - self.n_success

    @property
    def success_rate(self) -> float:
        return 0.0 if self.n_runs == 0 else self.n_success / self.n_runs


@dataclass(frozen=True)
class TBLiveV2Profile:
    n_runs: int
    n_success: int
    n_fail: int
    n_unresolved: int
    n_exact_tasks: int
    n_coarse_families: int
    exact_task_group_sizes: tuple[tuple[int, int], ...]
    arm_rows: tuple[ProfileOutcomeRow, ...]
    family_rows: tuple[ProfileOutcomeRow, ...]
    failure_task_rows: tuple[ProfileOutcomeRow, ...]


def _na_cell(
    *,
    target: str,
    spec: BaselineSpec,
    split: Split,
    source_slice: str,
    note: str,
) -> EvalCell:
    return EvalCell(
        target=target,
        model=spec.name,
        scheme=split.scheme,
        source_slice=source_slice,
        feasible=False,
        n_runs_train=None,
        n_runs_test=None,
        n_checkpoints_test=None,
        positive_rate_data=None,
        predicted_positive_rate=None,
        auroc=None,
        brier=None,
        log_loss=None,
        ece=None,
        brier_ci_low=None,
        brier_ci_high=None,
        note=note,
    )


def _infer_arm(run_id: str) -> str | None:
    m = _RUN_ID_ARM_RE.search(str(run_id))
    return None if m is None else str(m.group("arm"))


def _infer_task_id(run_id: str) -> str:
    text = str(run_id)
    if "__arm" in text:
        return text.split("__arm", 1)[0]
    return text


def _run_level_manifest(
    *, checkpoints_df: pd.DataFrame, labels_df: pd.DataFrame, manifest_df: pd.DataFrame | None
) -> pd.DataFrame:
    if manifest_df is not None and not manifest_df.empty:
        sub = manifest_df[manifest_df["source"] == TB_LIVE_V2].copy()
        if not sub.empty:
            if "arm" not in sub.columns:
                sub["arm"] = sub["run_id"].map(_infer_arm)
            if "task_id" not in sub.columns:
                sub["task_id"] = sub["run_id"].map(_infer_task_id)
            return sub.drop_duplicates("run_id")

    def _frame_meta(df: pd.DataFrame) -> pd.DataFrame | None:
        if df.empty or "run_id" not in df.columns:
            return None
        cols = ["run_id"]
        for name in ("task_id", "task_family", "arm", "model_name", "final_success"):
            if name in df.columns:
                cols.append(name)
        if len(cols) == 1:
            return None
        out = df[cols].drop_duplicates("run_id").copy()
        if "arm" not in out.columns:
            out["arm"] = out["run_id"].map(_infer_arm)
        if "task_id" not in out.columns:
            out["task_id"] = out["run_id"].map(_infer_task_id)
        return out

    ck_meta = _frame_meta(checkpoints_df[checkpoints_df["source"] == TB_LIVE_V2])
    if ck_meta is not None:
        return ck_meta
    lb_meta = _frame_meta(labels_df[labels_df["source"] == TB_LIVE_V2])
    if lb_meta is not None:
        return lb_meta
    raise ValueError("tb_live_v2 run metadata unavailable")


def exact_task_group_map(
    *,
    checkpoints_df: pd.DataFrame,
    labels_df: pd.DataFrame,
    manifest_df: pd.DataFrame | None = None,
) -> dict[str, str]:
    meta = _run_level_manifest(
        checkpoints_df=checkpoints_df,
        labels_df=labels_df,
        manifest_df=manifest_df,
    )
    task_ids = meta["task_id"].fillna(meta["run_id"].map(_infer_task_id)).astype(str)
    return dict(zip(meta["run_id"].astype(str), task_ids, strict=True))


def build_tb_live_v2_splits(
    *,
    checkpoints_df: pd.DataFrame,
    labels_df: pd.DataFrame,
    manifest_df: pd.DataFrame | None = None,
) -> dict[str, Split]:
    sub = checkpoints_df[checkpoints_df["source"] == TB_LIVE_V2].copy()
    if sub.empty:
        raise ValueError("no tb_live_v2 checkpoints present")
    groups = exact_task_group_map(
        checkpoints_df=sub,
        labels_df=labels_df,
        manifest_df=manifest_df,
    )
    sub["task_family"] = sub["run_id"].map(groups)
    if sub["task_family"].nunique() < 2:
        raise ValueError("tb_live_v2 exact-task LTFO requires at least 2 unique task_ids")
    return {
        "ltfo": ltfo(sub),
        "loro": loro(sub),
        "holdout": holdout(sub),
    }


def _wide_targets(labels_df: pd.DataFrame) -> pd.DataFrame:
    sub = labels_df[
        (labels_df["source"] == TB_LIVE_V2)
        & (labels_df["target_name"].isin(EVAL_TARGETS))
        & (~labels_df["is_masked"].astype(bool))
    ].copy()
    return sub.pivot(
        index=["run_id", "source", "checkpoint_id"],
        columns="target_name",
        values="label_value",
    ).reset_index()


def _budget_lookup(
    *,
    labels_df: pd.DataFrame,
    exact_task_groups: dict[str, str],
) -> dict[tuple[str, str], BudgetCell]:
    wide = _wide_targets(labels_df)
    if wide.empty:
        return {}
    wide = wide.assign(task_family=wide["run_id"].map(exact_task_groups))
    cells = compute_budget(
        wide,
        targets=EVAL_TARGETS,
        sources=(TB_LIVE_V2,),
        schemes=("ltfo", "loro", "holdout"),
    )
    return {(c.target, c.split_scheme): c for c in cells}


def evaluate_tb_live_v2(
    *,
    checkpoints_df: pd.DataFrame,
    labels_df: pd.DataFrame,
    manifest_df: pd.DataFrame | None = None,
    bootstrap_b: int = 1000,
    bootstrap_seed: int = 0,
) -> list[EvalCell]:
    sub = checkpoints_df[checkpoints_df["source"] == TB_LIVE_V2].copy()
    if sub.empty:
        raise ValueError("no tb_live_v2 checkpoints present")

    exact_groups = exact_task_group_map(
        checkpoints_df=sub,
        labels_df=labels_df,
        manifest_df=manifest_df,
    )
    budget = _budget_lookup(labels_df=labels_df, exact_task_groups=exact_groups)
    splits = build_tb_live_v2_splits(
        checkpoints_df=sub,
        labels_df=labels_df,
        manifest_df=manifest_df,
    )

    cells: list[EvalCell] = []
    for scheme in ("ltfo", "loro", "holdout"):
        split = splits[scheme]
        source_slice = SCHEME_LABELS[scheme]
        for target in EVAL_TARGETS:
            budget_cell = budget.get((target, scheme))
            feasible = budget_cell is not None and budget_cell.feasible
            note = None if feasible else (
                "no budget row" if budget_cell is None else budget_cell.reason
            )
            for spec in EVAL_MODELS:
                if feasible:
                    cells.append(
                        evaluate_cell(
                            checkpoints_df=sub,
                            labels_df=labels_df,
                            target=target,
                            spec=spec,
                            split=split,
                            source_slice=source_slice,
                            sources_in_train=(TB_LIVE_V2,),
                            feasible=True,
                            bootstrap_b=bootstrap_b,
                            bootstrap_seed=bootstrap_seed,
                        )
                    )
                else:
                    cells.append(
                        _na_cell(
                            target=target,
                            spec=spec,
                            split=split,
                            source_slice=source_slice,
                            note=note or "insufficient data",
                        )
                    )
    return cells


def build_tb_live_v2_profile(
    *,
    manifest_df: pd.DataFrame,
) -> TBLiveV2Profile:
    sub = manifest_df[manifest_df["source"] == TB_LIVE_V2].copy()
    if sub.empty:
        raise ValueError("no tb_live_v2 manifest rows present")

    if "arm" not in sub.columns:
        sub["arm"] = sub["run_id"].map(_infer_arm)
    if "task_id" not in sub.columns:
        sub["task_id"] = sub["run_id"].map(_infer_task_id)

    if "final_success" not in sub.columns:
        raise ValueError("tb_live_v2 manifest missing final_success")

    resolved = sub["final_success"].notna()
    success = (sub["final_success"] == True)  # noqa: E712
    n_runs = int(len(sub))
    n_success = int(success.sum())
    n_unresolved = int((~resolved).sum())
    n_fail = n_runs - n_success - n_unresolved

    exact_hist = (
        sub.groupby("task_id")["run_id"]
        .size()
        .value_counts()
        .sort_index()
    )
    exact_task_group_sizes = tuple((int(size), int(count)) for size, count in exact_hist.items())

    arm_rows: list[ProfileOutcomeRow] = []
    for arm, grp in sub.groupby("arm", dropna=False, sort=True):
        models = grp["model_name"].dropna().unique().tolist() if "model_name" in grp.columns else []
        arm_rows.append(
            ProfileOutcomeRow(
                name="unknown" if pd.isna(arm) else str(arm),
                model_name=models[0] if len(models) == 1 else None,
                n_success=int((grp["final_success"] == True).sum()),  # noqa: E712
                n_runs=int(len(grp)),
            )
        )

    family_rows = tuple(
        ProfileOutcomeRow(
            name=str(name),
            n_success=int((grp["final_success"] == True).sum()),  # noqa: E712
            n_runs=int(len(grp)),
        )
        for name, grp in sorted(
            sub.groupby("task_family", dropna=False),
            key=lambda item: (
                -int((item[1]["final_success"] == False).sum()),  # noqa: E712
                str(item[0]),
            ),
        )
    )

    failure_task_rows = tuple(
        ProfileOutcomeRow(
            name=str(name),
            n_success=int((grp["final_success"] == True).sum()),  # noqa: E712
            n_runs=int(len(grp)),
        )
        for name, grp in sorted(
            (
                (name, grp)
                for name, grp in sub.groupby("task_id", dropna=False)
                if int((grp["final_success"] == False).sum()) > 0  # noqa: E712
            ),
            key=lambda item: (
                -int((item[1]["final_success"] == False).sum()),  # noqa: E712
                -int(len(item[1])),
                str(item[0]),
            ),
        )
    )

    return TBLiveV2Profile(
        n_runs=n_runs,
        n_success=n_success,
        n_fail=n_fail,
        n_unresolved=n_unresolved,
        n_exact_tasks=int(sub["task_id"].nunique(dropna=True)),
        n_coarse_families=int(sub["task_family"].nunique(dropna=True)),
        exact_task_group_sizes=exact_task_group_sizes,
        arm_rows=tuple(arm_rows),
        family_rows=family_rows,
        failure_task_rows=failure_task_rows,
    )


def _fmt(v: float | int | None) -> str:
    if v is None or (isinstance(v, float) and pd.isna(v)):
        return "n/a"
    if isinstance(v, float):
        return f"{v:.3f}"
    return str(v)


def _render_eval_table(cells: list[EvalCell]) -> list[str]:
    lines = [
        "| split | target | model | n_train | n_test | n_ckpts | pos_rate | AUROC | Brier | Brier 95% CI | note |",
        "|---|---|---|---:|---:|---:|---:|---:|---:|---|---|",
    ]
    for c in cells:
        if c.feasible:
            ci = f"[{_fmt(c.brier_ci_low)}, {_fmt(c.brier_ci_high)}]"
            lines.append(
                f"| {c.source_slice} | {c.target} | {c.model} | {_fmt(c.n_runs_train)} | "
                f"{_fmt(c.n_runs_test)} | {_fmt(c.n_checkpoints_test)} | {_fmt(c.positive_rate_data)} | "
                f"{_fmt(c.auroc)} | {_fmt(c.brier)} | {ci} | {c.note or ''} |"
            )
        else:
            lines.append(
                f"| {c.source_slice} | {c.target} | {c.model} | n/a | n/a | n/a | "
                f"n/a | n/a | n/a | n/a | {c.note or 'insufficient data'} |"
            )
    return lines


def render_tb_live_v2_report(
    *,
    cells: list[EvalCell],
    profile: TBLiveV2Profile,
) -> str:
    now = datetime.now(UTC).isoformat(timespec="seconds")
    headline = [
        c
        for c in cells
        if c.scheme == "ltfo" and c.target in HEADLINE_TARGETS
    ]
    overlap = [
        c
        for c in cells
        if c.scheme in {"loro", "holdout"} and c.target in HEADLINE_TARGETS
    ]
    success = [c for c in cells if c.target == "y_success_eventual"]

    group_hist = ", ".join(
        f"{n_tasks} tasks x {group_size} runs"
        for group_size, n_tasks in profile.exact_task_group_sizes
    )
    lines = [
        "# tb_live_v2 estimator evaluation — exact-task holdout",
        "",
        f"_Generated {now}._",
        "",
        "Headline metrics are exact-task holdout (`task_id` held out across all arms). "
        "Overlap-heavy run-level splits are reported separately as easier auxiliary diagnostics.",
        "",
        "## Headline: exact-task holdout process dynamics",
        "",
        f"`tb_live_v2` contains {profile.n_runs} runs across {profile.n_exact_tasks} exact tasks "
        f"and {profile.n_coarse_families} coarse shape families. Exact-task group histogram: {group_hist}.",
        "At the current base rates, `y_future_progress_drop_h5` is evaluable under exact-task holdout; "
        "`y_validation_new_work_h5` remains below the per-fold positive-count budget and is reported as `n/a`.",
        "",
    ]
    lines.extend(_render_eval_table(sorted(headline, key=lambda c: (c.target, c.model))))
    lines.extend(
        [
            "",
            "## Auxiliary only: overlap-heavy splits",
            "",
            "These splits can place task X / arm A in train and task X / arm B in test, so "
            "they are easier than the exact-task claim and are not the headline result.",
            "",
        ]
    )
    lines.extend(_render_eval_table(sorted(overlap, key=lambda c: (c.scheme, c.target, c.model))))
    lines.extend(
        [
            "",
            "## Secondary only: terminal success",
            "",
            "Read this section conservatively:",
            f"- Arm concentration is strong: {', '.join(f'{r.name} {r.n_success}/{r.n_runs}' for r in profile.arm_rows)}.",
            f"- The corpus is ceiling-limited in several shape families, so terminal success is not a balanced substrate.",
            "- `solution.sh` was present in the seeded workspace during collection, so observed success is optimistic.",
            "",
        ]
    )
    lines.extend(_render_eval_table(sorted(success, key=lambda c: (c.scheme, c.model))))
    lines.append("")
    return "\n".join(lines) + "\n"


def render_tb_live_v2_shape_profile(profile: TBLiveV2Profile) -> str:
    now = datetime.now(UTC).isoformat(timespec="seconds")
    group_hist = ", ".join(
        f"{n_tasks} tasks x {group_size} runs"
        for group_size, n_tasks in profile.exact_task_group_sizes
    )
    lines = [
        "# tb_live_v2 shape profile",
        "",
        f"_Generated {now}._",
        "",
        f"`tb_live_v2` has {profile.n_runs} runs: {profile.n_success} successes, {profile.n_fail} failures, "
        f"{profile.n_unresolved} unresolved. The exact-task unit is {profile.n_exact_tasks} unique `task_id`s "
        f"spread across {profile.n_coarse_families} coarse shape families.",
        "",
        f"Exact-task replication histogram: {group_hist}. This is why exact-task holdout is stricter than "
        "coarse shape holdout on this corpus.",
        "",
        "## By arm",
        "",
        "| arm | model | pass | total | pass_rate | fail |",
        "|---|---|---:|---:|---:|---:|",
    ]
    for r in profile.arm_rows:
        lines.append(
            f"| {r.name} | {r.model_name or ''} | {r.n_success} | {r.n_runs} | "
            f"{r.success_rate:.3f} | {r.n_fail} |"
        )
    lines.extend(
        [
            "",
            "## By coarse family",
            "",
            "| family | pass | total | pass_rate | fail |",
            "|---|---:|---:|---:|---:|",
        ]
    )
    for r in profile.family_rows:
        lines.append(
            f"| {r.name} | {r.n_success} | {r.n_runs} | {r.success_rate:.3f} | {r.n_fail} |"
        )
    lines.extend(
        [
            "",
            "## Failure-concentrated exact tasks",
            "",
            "| task_id | pass | total | pass_rate | fail |",
            "|---|---:|---:|---:|---:|",
        ]
    )
    if profile.failure_task_rows:
        for r in profile.failure_task_rows:
            lines.append(
                f"| {r.name} | {r.n_success} | {r.n_runs} | {r.success_rate:.3f} | {r.n_fail} |"
            )
    else:
        lines.append("| none | 0 | 0 | n/a | 0 |")
    lines.extend(
        [
            "",
            "## Caveat",
            "",
            "The seeded workspace included `solution.sh` during collection. Any terminal-success analysis on this "
            "corpus should therefore be treated as optimistic rather than deployment-grade.",
            "",
        ]
    )
    return "\n".join(lines) + "\n"


def write_tb_live_v2_report(
    path: Path,
    *,
    cells: list[EvalCell],
    profile: TBLiveV2Profile,
) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        render_tb_live_v2_report(cells=cells, profile=profile),
        encoding="utf-8",
        newline="\n",
    )
    return path


def write_tb_live_v2_shape_profile(path: Path, *, profile: TBLiveV2Profile) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        render_tb_live_v2_shape_profile(profile),
        encoding="utf-8",
        newline="\n",
    )
    return path


__all__ = [
    "TB_LIVE_V2",
    "HEADLINE_TARGETS",
    "SECONDARY_TARGETS",
    "TBLiveV2Profile",
    "build_tb_live_v2_profile",
    "build_tb_live_v2_splits",
    "evaluate_tb_live_v2",
    "exact_task_group_map",
    "render_tb_live_v2_report",
    "render_tb_live_v2_shape_profile",
    "write_tb_live_v2_report",
    "write_tb_live_v2_shape_profile",
]
