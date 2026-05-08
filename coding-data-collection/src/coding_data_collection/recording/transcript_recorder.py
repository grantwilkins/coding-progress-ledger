from __future__ import annotations

from pathlib import Path
from typing import Any

from coding_data_collection.artifacts import utc_now
from coding_data_collection.ledger import transcript_to_wire_events, write_wire_events
from coding_data_collection.observation import build_observation_events, read_jsonl, write_jsonl


class RunRecorder:
    def __init__(self, *, run_dir: Path, run_id: str) -> None:
        self.run_dir = run_dir
        self.run_id = run_id
        self.run_dir.mkdir(parents=True, exist_ok=True)
        self._transcript: list[dict[str, Any]] = read_jsonl(run_dir / "transcript.jsonl")
        self.step = max((int(row.get("step", 0)) for row in self._transcript), default=0)

    def record(self, kind: str, **fields: Any) -> dict[str, Any]:
        self.step += 1
        row = {
            "step": self.step,
            "ts": utc_now(),
            "kind": kind,
            **{key: value for key, value in fields.items() if value is not None},
        }
        self._transcript.append(row)
        write_jsonl(self.run_dir / "transcript.jsonl", self._transcript)
        return row

    def read_transcript(self) -> list[dict[str, Any]]:
        return list(self._transcript)

    def write_derived_artifacts(
        self,
        *,
        verifier_exit_code: int | None = None,
        expected_paths: set[str] | None = None,
    ) -> None:
        write_wire_events(
            self.run_dir / "events.jsonl",
            transcript_to_wire_events(
                self._transcript,
                run_id=self.run_id,
                verifier_exit_code=verifier_exit_code,
            ),
        )
        write_jsonl(
            self.run_dir / "observation_events.jsonl",
            build_observation_events(
                self._transcript,
                run_id=self.run_id,
                verifier_exit_code=verifier_exit_code,
                expected_paths=expected_paths or set(),
            ),
        )
