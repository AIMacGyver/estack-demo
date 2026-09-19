"""Machine-readable reliability reports from labeled calibration JSONL."""

from __future__ import annotations

import argparse
import json
from collections.abc import Mapping, Sequence
from pathlib import Path
from statistics import fmean

from spatial_ipd.calibrate import binary_log_loss, brier_score

RELIABILITY_REPORT_VERSION = 1


class ReliabilityError(ValueError):
    """Malformed, unlabeled, or insufficiently structured calibration data."""


def read_calibration_events(paths: Sequence[str | Path]) -> list[dict[str, object]]:
    """Read event records while ignoring per-file summary records."""
    events = []
    for path in paths:
        source = Path(path)
        for line_number, raw in enumerate(source.read_text(encoding="utf-8").splitlines(), start=1):
            if not raw.strip():
                continue
            try:
                record = json.loads(raw)
            except json.JSONDecodeError as exc:
                raise ReliabilityError(f"{source}:{line_number} is invalid JSON") from exc
            if not isinstance(record, dict):
                raise ReliabilityError(f"{source}:{line_number} must be a JSON object")
            record_type = record.get("record_type")
            if record_type == "summary":
                continue
            if record_type != "event":
                raise ReliabilityError(f"{source}:{line_number} has unknown record_type {record_type!r}")
            events.append(record)
    return events


def _probability(event: Mapping[str, object]) -> float:
    value = event.get("resist_probability")
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ReliabilityError("event resist_probability must be numeric")
    probability = float(value)
    if not 0.0 <= probability <= 1.0:
        raise ReliabilityError("event resist_probability must be in [0, 1]")
    return probability


def _resolved_label(event: Mapping[str, object]) -> bool | None:
    value = event.get("hold_better")
    if value is None or isinstance(value, bool):
        return value
    raise ReliabilityError("event hold_better must be true, false, or null")


def _reliability_bins(
    probabilities: Sequence[float],
    labels: Sequence[bool],
    *,
    count: int,
) -> list[dict[str, int | float | None]]:
    buckets: list[list[tuple[float, bool]]] = [[] for _ in range(count)]
    for probability, label in zip(probabilities, labels, strict=True):
        index = min(int(probability * count), count - 1)
        buckets[index].append((probability, label))
    output = []
    for index, bucket in enumerate(buckets):
        mean_probability = fmean(item[0] for item in bucket) if bucket else None
        observed_frequency = fmean(float(item[1]) for item in bucket) if bucket else None
        output.append(
            {
                "lower": index / count,
                "upper": (index + 1) / count,
                "count": len(bucket),
                "mean_probability": mean_probability,
                "observed_frequency": observed_frequency,
                "calibration_gap": (
                    abs(mean_probability - observed_frequency)
                    if mean_probability is not None and observed_frequency is not None
                    else None
                ),
            }
        )
    return output


def reliability_report(
    events: Sequence[Mapping[str, object]],
    *,
    bins: int = 10,
    confidence_threshold: float = 0.8,
    minimum_events: int = 50,
) -> dict[str, object]:
    """Compute calibration, selective prediction, and action consistency."""
    if bins < 1:
        raise ReliabilityError("bins must be positive")
    if not 0.5 <= confidence_threshold <= 1.0:
        raise ReliabilityError("confidence_threshold must be in [0.5, 1]")
    if minimum_events < 1:
        raise ReliabilityError("minimum_events must be positive")

    resolved_events = []
    probabilities = []
    labels = []
    for event in events:
        probability = _probability(event)
        label = _resolved_label(event)
        if label is not None:
            resolved_events.append(event)
            probabilities.append(probability)
            labels.append(label)
    if not resolved_events:
        raise ReliabilityError("at least one resolved event is required")

    reliability_bins = _reliability_bins(probabilities, labels, count=bins)
    expected_calibration_error = sum(
        bucket["count"] / len(resolved_events) * bucket["calibration_gap"]
        for bucket in reliability_bins
        if bucket["count"] and bucket["calibration_gap"] is not None
    )
    predicted = [probability >= 0.5 for probability in probabilities]
    correct = [prediction == label for prediction, label in zip(predicted, labels, strict=True)]
    confident = [max(probability, 1.0 - probability) >= confidence_threshold for probability in probabilities]
    covered_correct = [is_correct for is_correct, is_confident in zip(correct, confident, strict=True) if is_confident]

    action_matches = []
    for event, prediction in zip(resolved_events, predicted, strict=True):
        decision = event.get("resist_decision")
        if not isinstance(decision, bool):
            raise ReliabilityError("resolved event resist_decision must be boolean")
        action_matches.append(decision == prediction)

    positives = sum(labels)
    negatives = len(labels) - positives
    return {
        "schema_version": RELIABILITY_REPORT_VERSION,
        "events": len(events),
        "resolved": len(resolved_events),
        "ties": len(events) - len(resolved_events),
        "positives": positives,
        "negatives": negatives,
        "brier_score": brier_score(probabilities, labels),
        "log_loss": binary_log_loss(probabilities, labels),
        "expected_calibration_error": expected_calibration_error,
        "reliability_bins": reliability_bins,
        "accuracy": fmean(correct),
        "confidence_threshold": confidence_threshold,
        "covered": sum(confident),
        "coverage": fmean(confident),
        "selective_accuracy": fmean(covered_correct) if covered_correct else None,
        "action_probability_consistency": fmean(action_matches),
        "minimum_events": minimum_events,
        "calibration_claim_supported": (len(resolved_events) >= minimum_events and positives > 0 and negatives > 0),
    }


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    """Parse reliability report flags."""
    parser = argparse.ArgumentParser(
        prog="python -m spatial_ipd.reliability",
        description="Build a reliability report from calibration JSONL events.",
    )
    parser.add_argument("--jsonl", nargs="+", required=True, help="One or more calibration JSONL files.")
    parser.add_argument("--bins", type=int, default=10)
    parser.add_argument("--confidence-threshold", type=float, default=0.8)
    parser.add_argument("--minimum-events", type=int, default=50)
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    """Print one machine-readable reliability report."""
    args = parse_args(argv)
    events = read_calibration_events(args.jsonl)
    report = reliability_report(
        events,
        bins=args.bins,
        confidence_threshold=args.confidence_threshold,
        minimum_events=args.minimum_events,
    )
    print(json.dumps(report, sort_keys=True, separators=(",", ":")))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
