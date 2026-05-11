from __future__ import annotations

import json
import re
from dataclasses import dataclass

from .classify import classify_turn
from .models import Category, Row, Summary, Turn
from .path_tracker import first_write_target, is_source_path


STUCK_RESPONSE_MIN_CHARS = 80
STUCK_ERROR_MARKERS = (
    "error",
    "exception",
    "failed",
    "failure",
    "permission denied",
    "returncode=1",
    "traceback",
)
RETURNCODE_RE = re.compile(r"<returncode>\s*(-?\d+)\s*</returncode>", re.IGNORECASE)
RETURN_ASSIGN_RE = re.compile(r"\breturncode\s*=\s*(-?\d+)\b", re.IGNORECASE)
RECENT_ERROR_WINDOW_STEPS = 10


@dataclass
class _Unit:
    category: Category
    opened_step: int
    closed_step: int | None = None
    status: str = "open"
    target: str | None = None


class Annotator:
    def __init__(self, *, instance_id: str = "", exit_status: str = "unknown") -> None:
        self.instance_id = instance_id
        self.exit_status = exit_status
        self.units: list[_Unit] = []
        self.recent_responses: list[str] = []
        self.recent_observations: list[tuple[int, bool]] = []
        self.action_count = 0
        self.investigation_action_count = 0
        self.touched_source = False
        self.had_stuck_episode = False
        self.last_seen_step = 0

    def feed(self, turn: Turn) -> Row:
        self.last_seen_step = turn.step
        if not self.instance_id and turn.instance_id:
            self.instance_id = turn.instance_id
        if turn.metadata.get("exit_status"):
            self.exit_status = str(turn.metadata["exit_status"])

        if turn.kind == "observation":
            self._observe(turn.step, turn.response or "")
        elif turn.kind == "action":
            self.action_count += 1
            category = classify_turn(turn)
            if category is not None:
                self.investigation_action_count += int(category is Category.INVESTIGATION)
                self._handle_action(turn, category)

        current = self._current_unit()
        current_category = current.category.value if current else ""
        age = turn.step - current.opened_step + 1 if current else 0
        recent_error_rate = self._recent_error_rate(turn.step)
        investigation_ratio = self._investigation_ratio()
        return Row(
            step=turn.step,
            total=len(self.units),
            done=self._done_count(),
            current_category=current_category,
            current_unit_age=age,
            had_stuck_episode=self.had_stuck_episode,
            recent_error_bucket=_error_bucket(recent_error_rate),
            recent_error_rate=recent_error_rate,
            touched_source=self.touched_source,
            investigation_ratio_bucket=_investigation_bucket(investigation_ratio),
            investigation_ratio=investigation_ratio,
            kind=turn.kind,
            tool=turn.tool or "",
        )

    def finalize(self) -> Summary:
        current = self._current_unit()
        if current and current.status in {"open", "stuck"}:
            current.closed_step = self.last_seen_step
            current.status = "done"
        return Summary(
            instance_id=self.instance_id,
            final_total=len(self.units),
            final_done=self._done_count(),
            had_stuck_episode=self.had_stuck_episode,
            exit_status=self.exit_status,
        )

    def _handle_action(self, turn: Turn, category: Category) -> None:
        target = first_write_target(turn) if category is Category.PRODUCT else None
        if target and is_source_path(target):
            self.touched_source = True
        current = self._current_unit()
        if current is None:
            self._open(category, turn.step, target)
            return

        same_category = current.category is category
        target_changed = (
            category is Category.PRODUCT
            and target is not None
            and current.target is not None
            and target != current.target
        )
        if same_category and not target_changed:
            if category is Category.PRODUCT and target is not None:
                current.target = target
            return

        self._close_current(turn.step - 1)
        self._open(category, turn.step, target)

    def _observe(self, step: int, response: str) -> None:
        body = response.strip()
        self.recent_responses.append(body)
        self.recent_responses = self.recent_responses[-3:]
        self.recent_observations.append((step, _is_failure_observation(body)))
        self._prune_recent_observations(step)
        current = self._current_unit()
        if (
            current
            and current.status == "open"
            and len(self.recent_responses) == 3
            and self.recent_responses[0] == self.recent_responses[1] == self.recent_responses[2]
            and _is_stuck_evidence(body)
        ):
            current.status = "stuck"
            self.had_stuck_episode = True

    def _open(self, category: Category, step: int, target: str | None) -> None:
        inherited_target = None
        if category is Category.PRODUCT and target is None:
            previous = self._current_unit()
            if previous and previous.category is Category.PRODUCT:
                inherited_target = previous.target
        self.units.append(_Unit(category=category, opened_step=step, target=target or inherited_target))
        self.recent_responses = []

    def _close_current(self, step: int) -> None:
        current = self._current_unit()
        if current is None:
            return
        current.closed_step = step
        current.status = "done"
        self.recent_responses = []

    def _current_unit(self) -> _Unit | None:
        return self.units[-1] if self.units else None

    def _done_count(self) -> int:
        return sum(1 for unit in self.units if unit.status == "done")

    def _recent_error_bucket(self, step: int) -> str:
        return _error_bucket(self._recent_error_rate(step))

    def _recent_error_rate(self, step: int) -> float:
        self._prune_recent_observations(step)
        if not self.recent_observations:
            return 0.0
        failures = sum(int(failed) for _, failed in self.recent_observations)
        return failures / len(self.recent_observations)

    def _prune_recent_observations(self, step: int) -> None:
        lower = step - RECENT_ERROR_WINDOW_STEPS
        self.recent_observations = [(obs_step, failed) for obs_step, failed in self.recent_observations if obs_step >= lower]

    def _investigation_ratio_bucket(self) -> str:
        return _investigation_bucket(self._investigation_ratio())

    def _investigation_ratio(self) -> float:
        if not self.action_count:
            return 0.0
        return self.investigation_action_count / self.action_count


def _is_stuck_evidence(body: str) -> bool:
    normalized = body.strip()
    if len(normalized) >= STUCK_RESPONSE_MIN_CHARS:
        return True
    lowered = normalized.lower()
    return any(marker in lowered for marker in STUCK_ERROR_MARKERS)


def _is_failure_observation(body: str) -> bool:
    stripped = body.strip()
    if stripped.startswith("Traceback") or "Exception" in stripped:
        return True
    if "error" in stripped or "Error" in stripped:
        return True
    for match in (*RETURNCODE_RE.finditer(stripped), *RETURN_ASSIGN_RE.finditer(stripped)):
        if int(match.group(1)) != 0:
            return True
    try:
        parsed = json.loads(stripped)
    except json.JSONDecodeError:
        return False
    return isinstance(parsed, dict) and parsed.get("success") is False


def _error_bucket(rate: float) -> str:
    if rate <= 0.1:
        return "clean"
    if rate <= 0.3:
        return "mild"
    if rate <= 0.6:
        return "moderate"
    return "heavy"


def _investigation_bucket(rate: float) -> str:
    if rate <= 0.25:
        return "low"
    if rate <= 0.5:
        return "moderate"
    if rate <= 0.75:
        return "high"
    return "dominant"
