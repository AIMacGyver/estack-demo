"""Versioned thinker decision records and strict offline replay."""

from __future__ import annotations

import json
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from types import SimpleNamespace

DECISION_RECORD_VERSION = 1


class ReplayError(ValueError):
    """A malformed, mismatched, exhausted, or partly consumed replay."""


def _jsonable(value: object) -> object:
    """Convert known question metadata to stable JSON-compatible values."""
    if value is None or isinstance(value, (bool, int, float, str)):
        return value
    if isinstance(value, Mapping):
        return {str(key): _jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(item) for item in value]
    model_dump = getattr(value, "model_dump", None)
    if callable(model_dump):
        return _jsonable(model_dump())
    metadata = {"type": type(value).__name__}
    for name in ("instructions", "criteria"):
        if hasattr(value, name):
            metadata[name] = _jsonable(getattr(value, name))
    return metadata


def question_snapshot(questions: Mapping[str, object]) -> dict[str, object]:
    """Capture question IDs and available instructions/criteria for audit."""
    return {key: _jsonable(value) for key, value in sorted(questions.items())}


def _answer_snapshot(answer: object, fields: Sequence[str]) -> dict[str, object]:
    """Capture named public fields from one duck-typed backend answer."""
    snapshot = {}
    for field in fields:
        if hasattr(answer, field):
            snapshot[field] = _jsonable(getattr(answer, field))
    return snapshot


def response_snapshot(response: object) -> dict[str, object]:
    """Capture the normalized answer maps and available raw backend output."""
    snapshot: dict[str, object] = {
        "nouls": {
            key: _answer_snapshot(answer, ("noul", "confidence")) for key, answer in sorted(response.nouls.items())
        },
        "scores": {
            key: _answer_snapshot(answer, ("score", "confidence", "probabilities"))
            for key, answer in sorted(response.scores.items())
        },
        "choices": {
            key: _answer_snapshot(answer, ("choice", "confidence", "probabilities"))
            for key, answer in sorted(response.choices.items())
        },
        "confidence_kind": str(getattr(response, "confidence_kind", "jev_noul")),
    }
    raw_output = getattr(response, "raw_output", None)
    if isinstance(raw_output, str):
        snapshot["raw_output"] = raw_output
    return snapshot


@dataclass(frozen=True)
class DecisionRecord:
    """One backend call with enough input and output to replay it strictly."""

    state: dict[str, object]
    questions: dict[str, object]
    response: dict[str, object]
    schema_version: int = DECISION_RECORD_VERSION

    def as_dict(self) -> dict[str, object]:
        """Return the stable JSON object written as one JSONL line."""
        return {
            "schema_version": self.schema_version,
            "state": self.state,
            "questions": self.questions,
            "response": self.response,
        }


def capture_decision_record(
    state: Mapping[str, object],
    questions: Mapping[str, object],
    response: object,
) -> DecisionRecord:
    """Capture one successful backend call after its response is normalized."""
    return DecisionRecord(
        state=dict(_jsonable(state)),
        questions=question_snapshot(questions),
        response=response_snapshot(response),
    )


def write_decision_records(path: str | Path, records: Sequence[DecisionRecord]) -> Path:
    """Write versioned decision records as deterministic JSON Lines."""
    destination = Path(path)
    lines = [json.dumps(record.as_dict(), sort_keys=True, separators=(",", ":")) for record in records]
    destination.write_text("".join(f"{line}\n" for line in lines), encoding="utf-8")
    return destination


def _record_from_dict(value: object, *, line_number: int) -> DecisionRecord:
    if not isinstance(value, dict):
        raise ReplayError(f"Decision replay line {line_number} must be a JSON object")
    version = value.get("schema_version")
    if version != DECISION_RECORD_VERSION:
        raise ReplayError(
            f"Decision replay line {line_number} uses schema version {version!r}; expected {DECISION_RECORD_VERSION}"
        )
    state = value.get("state")
    questions = value.get("questions")
    response = value.get("response")
    if not isinstance(state, dict) or not isinstance(questions, dict) or not isinstance(response, dict):
        raise ReplayError(f"Decision replay line {line_number} is missing state, questions, or response objects")
    return DecisionRecord(state=state, questions=questions, response=response)


def read_decision_records(path: str | Path) -> list[DecisionRecord]:
    """Load and validate all decision records from a JSONL file."""
    source = Path(path)
    records = []
    for line_number, raw in enumerate(source.read_text(encoding="utf-8").splitlines(), start=1):
        if not raw.strip():
            continue
        try:
            value = json.loads(raw)
        except json.JSONDecodeError as exc:
            raise ReplayError(f"Decision replay line {line_number} is invalid JSON") from exc
        records.append(_record_from_dict(value, line_number=line_number))
    return records


def _response_from_snapshot(snapshot: Mapping[str, object]) -> SimpleNamespace:
    """Recreate the duck-typed response consumed by thinker policy code."""

    def answers(name: str) -> dict[str, SimpleNamespace]:
        values = snapshot.get(name)
        if not isinstance(values, Mapping):
            raise ReplayError(f"Decision replay response is missing {name}")
        restored = {}
        for key, fields in values.items():
            if not isinstance(key, str) or not isinstance(fields, Mapping):
                raise ReplayError(f"Decision replay response {name} must map IDs to objects")
            restored[key] = SimpleNamespace(**fields)
        return restored

    extra = {}
    raw_output = snapshot.get("raw_output")
    if isinstance(raw_output, str):
        extra["raw_output"] = raw_output
    return SimpleNamespace(
        nouls=answers("nouls"),
        scores=answers("scores"),
        choices=answers("choices"),
        confidence_kind=str(snapshot.get("confidence_kind", "jev_noul")),
        **extra,
    )


class ReplayThinkerClient:
    """Return recorded backend responses while validating call order and state."""

    uses_sdk_questions = False

    def __init__(self, records: Sequence[DecisionRecord]):
        """Bind an ordered record stream for one simulation replay."""
        self._records = tuple(records)
        self._index = 0

    @classmethod
    def from_path(cls, path: str | Path) -> ReplayThinkerClient:
        """Load a replay client from a decision JSONL path."""
        return cls(read_decision_records(path))

    @property
    def consumed(self) -> int:
        """Number of records consumed so far."""
        return self._index

    def system_one(self, state: object, questions: Mapping[str, object]) -> SimpleNamespace:
        """Validate the next call and return its recorded response."""
        if self._index >= len(self._records):
            raise ReplayError(f"Decision replay exhausted after {self._index} calls")
        record = self._records[self._index]
        current_state = _jsonable(state)
        if current_state != record.state:
            raise ReplayError(f"Decision replay state mismatch at call {self._index + 1}")
        current_ids = sorted(questions)
        recorded_ids = sorted(record.questions)
        if current_ids != recorded_ids:
            raise ReplayError(
                f"Decision replay question IDs mismatch at call {self._index + 1}: "
                f"got {current_ids}, expected {recorded_ids}"
            )
        self._index += 1
        return _response_from_snapshot(record.response)

    def assert_exhausted(self) -> None:
        """Fail if the replay file contains calls the simulation did not consume."""
        remaining = len(self._records) - self._index
        if remaining:
            raise ReplayError(f"Decision replay has {remaining} unused record(s)")
