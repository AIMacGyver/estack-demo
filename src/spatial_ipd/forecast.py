"""Score backend probabilities against committed arena outcomes."""

from __future__ import annotations

import argparse
import csv
import json
from collections.abc import Mapping, Sequence
from pathlib import Path
from random import Random
from types import SimpleNamespace

from spatial_ipd.calibrate import binary_log_loss, brier_score
from spatial_ipd.label import load_dotenv
from spatial_ipd.local_llm import (
    DEFAULT_LOCAL_LLM_ENDPOINT,
    DEFAULT_LOCAL_LLM_TIMEOUT,
    LOCAL_LLM_REASONING_EFFORTS,
    LocalLLMError,
    _boolean_decision,
    _bounded_number,
    _content_json,
    _message_content,
    _post_json,
)
from spatial_ipd.think import BACKEND_JEV, BACKEND_LOCAL, BACKEND_RANDOM, BACKENDS

REPO_ROOT = Path(__file__).resolve().parents[2]
MINIMUM_EVENTS = 50
FORECAST_INSTRUCTIONS = (
    "Is the statement in `question` true for the described experiment? "
    "Use only `mechanism`, `condition`, `seed`, `roster`, `generations`, "
    "`encounters_per_generation`, and `rounds_per_match`. "
    "The outcome count is not in the state."
)
FORECAST_CRITERIA = {
    "true": "The statement in `question` holds for that replicate.",
    "false": "The statement in `question` does not hold for that replicate.",
}


def _final_defect_counts(path: Path) -> dict[int, int]:
    with path.open(encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle))
    final_generation = max(int(row["generation"]) for row in rows)
    return {
        int(row["replicate_seed"]): int(row["next_count"])
        for row in rows
        if int(row["generation"]) == final_generation and row["policy"] == "always_defect"
    }


def _items_from_counts(
    *,
    mechanism: str,
    condition: str,
    roster: str,
    counts: Mapping[int, int],
    question: str,
    label_for: object,
) -> list[dict[str, object]]:
    items = []
    for seed, count in sorted(counts.items()):
        label = bool(label_for(count))
        items.append(
            {
                "id": f"{mechanism}-{condition}-{seed}",
                "state": {
                    "mechanism": mechanism,
                    "condition": condition,
                    "seed": seed,
                    "roster": roster,
                    "generations": 10,
                    "encounters_per_generation": 50,
                    "rounds_per_match": 10,
                    "mutation_rate": 0,
                    "question": question,
                },
                "label": label,
            }
        )
    return items


def committed_propositions(root: Path | None = None) -> list[dict[str, object]]:
    """Build yes/no items from committed selection tables. Outcomes stay out of the state."""
    base = root or REPO_ROOT
    reputation = "25 each of Always Cooperate, Always Defect, Reputation Guard, and Tit-for-Tat"
    communication = "25 each of Always Cooperate, Always Defect, Communication Guard, and Tit-for-Tat"
    memory = "20 each of Always Cooperate, Always Defect, Forgiving Tit-for-Tat, Pavlov, and Tit-for-Tat"
    below_ten = "Is the final Always Defect count below 10?"
    above_forty = "Is the final Always Defect count above 40?"
    items = []
    items.extend(
        _items_from_counts(
            mechanism="reputation_selection",
            condition="enabled",
            roster=reputation,
            counts=_final_defect_counts(base / "evidence/2026-09-21-reputation-selection/enabled.csv"),
            question=below_ten,
            label_for=lambda count: count < 10,
        )
    )
    items.extend(
        _items_from_counts(
            mechanism="reputation_selection",
            condition="disabled",
            roster=reputation,
            counts=_final_defect_counts(base / "evidence/2026-09-21-reputation-selection/disabled.csv"),
            question=below_ten,
            label_for=lambda count: count < 10,
        )
    )
    items.extend(
        _items_from_counts(
            mechanism="communication_selection",
            condition="enabled",
            roster=communication,
            counts=_final_defect_counts(base / "evidence/2026-09-21-communication-selection/enabled.csv"),
            question=below_ten,
            label_for=lambda count: count < 10,
        )
    )
    items.extend(
        _items_from_counts(
            mechanism="communication_selection",
            condition="disabled",
            roster=communication,
            counts=_final_defect_counts(base / "evidence/2026-09-21-communication-selection/disabled.csv"),
            question=below_ten,
            label_for=lambda count: count < 10,
        )
    )
    items.extend(
        _items_from_counts(
            mechanism="memory_selection",
            condition="window_1",
            roster=memory,
            counts=_final_defect_counts(base / "evidence/2026-09-21-memory-selection/memory-1.csv"),
            question=above_forty,
            label_for=lambda count: count > 40,
        )
    )
    items.extend(
        _items_from_counts(
            mechanism="memory_selection",
            condition="full",
            roster=memory,
            counts=_final_defect_counts(base / "evidence/2026-09-21-memory-selection/memory-full.csv"),
            question=above_forty,
            label_for=lambda count: count > 40,
        )
    )
    return items


class RandomForecastClient:
    """Seeded Uniform[0, 1] probability that the statement is true."""

    def __init__(self, seed: int):
        """Bind one random stream."""
        self._rng = Random(seed)

    def system_one(self, state: object, questions: Mapping[str, object]) -> SimpleNamespace:
        """Draw P(true). The draw is not a calibrated forecast."""
        del state, questions
        probability = self._rng.random()
        return SimpleNamespace(
            nouls={"statement_true": SimpleNamespace(noul=probability, confidence=probability)},
            confidence_kind="random_draw",
        )


class JevForecastClient:
    """Ask TypeSafe whether a committed statement is true."""

    def __init__(self):
        """Open one TypeSafe client."""
        try:
            from typesafe_sdk import Noul, TypeSafeClient
        except ImportError as exc:
            raise ImportError('TypeSafe is optional. Install with: python3 -m pip install -e ".[typesafe]"') from exc
        self._questions = {
            "statement_true": Noul(instructions=FORECAST_INSTRUCTIONS, criteria=FORECAST_CRITERIA),
        }
        self._context = TypeSafeClient()
        self._client = self._context.__enter__()

    def close(self) -> None:
        """Close the TypeSafe client."""
        self._context.__exit__(None, None, None)

    def system_one(self, state: object, questions: Mapping[str, object]) -> SimpleNamespace:
        """Return Jev's Noul as P(statement is true)."""
        del questions
        response = self._client.system_one(state=state, questions=self._questions)
        noul = float(response.nouls["statement_true"].noul)
        return SimpleNamespace(
            nouls={"statement_true": SimpleNamespace(noul=noul, confidence=noul)},
            confidence_kind="jev_noul",
        )


class LocalForecastClient:
    """Ask a local model whether a statement is true, and orient confidence as P(true)."""

    def __init__(
        self,
        model: str,
        *,
        endpoint: str = DEFAULT_LOCAL_LLM_ENDPOINT,
        timeout: float = DEFAULT_LOCAL_LLM_TIMEOUT,
        reasoning_effort: str | None = None,
        transport: object | None = None,
    ):
        """Bind a local model. Tests may inject a transport."""
        if not model.strip():
            raise ValueError("local model must not be empty")
        self.model = model
        self.endpoint = endpoint
        self.timeout = timeout
        self.reasoning_effort = reasoning_effort
        self._transport = transport or _post_json

    def system_one(self, state: object, questions: Mapping[str, object]) -> SimpleNamespace:
        """Map answer=true to P(true)=confidence and answer=false to 1-confidence."""
        del questions
        payload = {
            "model": self.model,
            "messages": [
                {
                    "role": "system",
                    "content": (
                        "Decide whether the experiment statement is true. "
                        'Return only JSON shaped as {"answer":true,"confidence":0.7}. '
                        "answer must be a JSON boolean. confidence is your self-reported "
                        "confidence from 0 to 1 and does not replace the boolean."
                    ),
                },
                {
                    "role": "user",
                    "content": json.dumps({"state": state}, sort_keys=True, separators=(",", ":")),
                },
            ],
            "stream": False,
            "temperature": 0.0,
        }
        if self.reasoning_effort is not None:
            payload["reasoning_effort"] = self.reasoning_effort
        headers = {"Content-Type": "application/json"}
        try:
            response = self._transport(self.endpoint, payload, headers, self.timeout)
        except LocalLLMError:
            raise
        except Exception as exc:
            raise LocalLLMError(f"Local LLM request to {self.endpoint} failed: {exc}") from exc
        decision = _content_json(_message_content(response))
        affirmed = _boolean_decision(decision.get("answer"), name="answer")
        confidence = _bounded_number(decision.get("confidence"), name="confidence", low=0.0, high=1.0)
        probability = confidence if affirmed else 1.0 - confidence
        return SimpleNamespace(
            nouls={"statement_true": SimpleNamespace(noul=probability, confidence=confidence)},
            confidence_kind="llm_self_report",
        )


def make_forecast_client(
    backend: str,
    seed: int,
    *,
    model: str | None = None,
    endpoint: str = DEFAULT_LOCAL_LLM_ENDPOINT,
    timeout: float = DEFAULT_LOCAL_LLM_TIMEOUT,
    reasoning_effort: str | None = None,
) -> object:
    """Build a random, Jev, or local forecast client."""
    if backend == BACKEND_RANDOM:
        return RandomForecastClient(seed)
    if backend == BACKEND_JEV:
        return JevForecastClient()
    if backend == BACKEND_LOCAL:
        if not model:
            raise ValueError("local backend requires a model")
        return LocalForecastClient(
            model,
            endpoint=endpoint,
            timeout=timeout,
            reasoning_effort=reasoning_effort,
        )
    raise ValueError(f"backend must be one of {BACKENDS}")


def score_forecasts(
    events: Sequence[Mapping[str, object]],
    *,
    minimum_events: int = MINIMUM_EVENTS,
) -> dict[str, object]:
    """Score resolved boolean labels. This catalog has no ties."""
    if not events:
        raise ValueError("forecast events must not be empty")
    probabilities = [float(event["probability"]) for event in events]
    labels = [bool(event["label"]) for event in events]
    positives = sum(labels)
    negatives = len(labels) - positives
    return {
        "record_type": "summary",
        "events": len(events),
        "resolved": len(events),
        "ties": 0,
        "positives": positives,
        "negatives": negatives,
        "brier_score": brier_score(probabilities, labels),
        "log_loss": binary_log_loss(probabilities, labels),
        "minimum_events": minimum_events,
        "calibration_claim_supported": len(events) >= minimum_events and positives > 0 and negatives > 0,
    }


def run_forecasts(
    client: object,
    propositions: Sequence[Mapping[str, object]],
    *,
    backend: str,
) -> list[dict[str, object]]:
    """Ask one client for every proposition. The label is attached after the call."""
    events = []
    for item in propositions:
        response = client.system_one(item["state"], {"statement_true": None})
        decision = response.nouls["statement_true"]
        probability = float(decision.noul)
        events.append(
            {
                "record_type": "event",
                "id": item["id"],
                "backend": backend,
                "state": item["state"],
                "probability": probability,
                "confidence": float(decision.confidence),
                "confidence_kind": str(getattr(response, "confidence_kind", "noul")),
                "label": bool(item["label"]),
                "decision": probability >= 0.5,
            }
        )
    return events


def write_jsonl(path: str | Path, events: Sequence[Mapping[str, object]], summary: Mapping[str, object]) -> Path:
    """Write forecast events and one summary."""
    destination = Path(path)
    records = [*events, summary]
    destination.write_text(
        "".join(json.dumps(record, sort_keys=True, separators=(",", ":")) + "\n" for record in records),
        encoding="utf-8",
    )
    return destination


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    """Parse forecast CLI flags."""
    parser = argparse.ArgumentParser(
        prog="python -m spatial_ipd.forecast",
        description="Score a backend against committed arena outcomes.",
    )
    parser.add_argument("--jsonl", required=True)
    parser.add_argument("--backend", choices=BACKENDS, default=BACKEND_RANDOM)
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--local-model")
    parser.add_argument("--local-endpoint", default=DEFAULT_LOCAL_LLM_ENDPOINT)
    parser.add_argument("--local-timeout", type=float, default=DEFAULT_LOCAL_LLM_TIMEOUT)
    parser.add_argument("--local-reasoning-effort", choices=LOCAL_LLM_REASONING_EFFORTS)
    args = parser.parse_args(argv)
    if args.backend == BACKEND_LOCAL and not args.local_model:
        parser.error("--local-model is required for --backend local")
    return args


def main(argv: Sequence[str] | None = None, *, client: object | None = None) -> int:
    """Score one backend and write JSONL."""
    load_dotenv()
    args = parse_args(argv)
    owns_client = client is None
    if client is None:
        client = make_forecast_client(
            args.backend,
            args.seed,
            model=args.local_model,
            endpoint=args.local_endpoint,
            timeout=args.local_timeout,
            reasoning_effort=args.local_reasoning_effort,
        )
    try:
        events = run_forecasts(client, committed_propositions(), backend=args.backend)
    finally:
        if owns_client and hasattr(client, "close"):
            client.close()
    summary = score_forecasts(events)
    summary["backend"] = args.backend
    write_jsonl(args.jsonl, events, summary)
    print(
        f"backend={args.backend} events={summary['events']} positives={summary['positives']} "
        f"negatives={summary['negatives']} brier={summary['brier_score']} "
        f"log_loss={summary['log_loss']} claim={summary['calibration_claim_supported']} jsonl={args.jsonl}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
